"""Offline batch=1 timing runner. Measurements do not certify numerical correctness."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from metrics import aggregate_metrics, distribution, validate_timing

ROOT = Path(__file__).resolve().parent
METRICS = ('prefill_ms', 'ttft_ms', 'tpot_ms', 'decode_ms',
           'decode_tokens_per_second', 'total_ms')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def capture(command):
    try:
        return subprocess.check_output(command, cwd=ROOT.parent, text=True,
                                       stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT.parent/'build-sharegpt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=('diverse100', 'fixed9', 'sharegpt100'), default='sharegpt100',
                        help='Frozen workload collection; default sharegpt100')
    parser.add_argument('--case', action='append', help='Select case ID; repeat to select several; default entire dataset')
    parser.add_argument('--quick', action='store_true', help='Six fixed ShareGPT cases, matching the quick logits check')
    parser.add_argument('--warmup', type=int, default=1, help='Global real-request warmups, not per case')
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup >= 0 and repeats >= 1 required')
    dataset_root = ROOT if args.dataset == 'fixed9' else ROOT/args.dataset
    manifest = json.loads((dataset_root/'manifest.json').read_text())
    if args.quick:
        if args.dataset != 'sharegpt100' or args.case:
            parser.error('--quick requires sharegpt100 and cannot be combined with --case')
        args.case = json.loads((ROOT.parent/'tests/sharegpt_reference.json').read_text())['quick_cases']
    cases = [c for c in manifest['cases'] if not args.case or c['id'] in args.case]
    if not cases or (args.case and set(args.case)-{c['id'] for c in cases}):
        parser.error('Unknown case; diverse100: d001..d100; sharegpt100: s001..s100; fixed9: e.g. p16_g16')
    print(f'Dataset: {manifest["dataset"]}; {len(cases)} cases; '
          f'{args.warmup} global warmups; {args.repeats} measurements per case', flush=True)
    probe = args.build_dir.resolve()/'bin/benchmark_probe'
    if not probe.is_file():
        parser.error(f'Missing {probe}; build with -DQWEN_BUILD_BENCHMARKS=ON')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; existing results will not be overwritten')
    # Validate all source hashes before launching the engine. No Python ML dependencies needed.
    print('Checking local model and input hashes...', flush=True)
    for name, expected in manifest['model']['sha256'].items():
        if sha(args.model/name) != expected:
            raise RuntimeError(f'Model hash mismatch: {name}')
    inputs = {i['name']: i for i in manifest['inputs']}
    for item in inputs.values():
        for key, rel in item['files'].items():
            if sha(dataset_root/rel) != item['sha256'][key]:
                raise RuntimeError(f'Input hash mismatch: {rel}')
        ids = list(map(int, (dataset_root/item['files']['ids']).read_text().split()))
        if len(ids) != item['prompt_tokens'] or any(t < 0 or t >= 151936 for t in ids):
            raise RuntimeError('Invalid input token IDs')
    for c in cases:
        if c['prompt_tokens'] != inputs[c['input']]['prompt_tokens'] or c['output_tokens'] < 2:
            raise RuntimeError('Invalid case lengths')
        if args.dataset != 'fixed9' and not (0 < c['prompt_tokens'] < 1024 and c['output_tokens'] < 1024):
            raise RuntimeError('Subset lengths must be strictly below 1024')
    output.mkdir(parents=True, exist_ok=True)
    plan = output/'plan.txt'
    representative = min(manifest['cases'], key=lambda c: (abs(c['prompt_tokens']-512)+abs(c['output_tokens']-512), c['id']))
    planned = ([dict(representative, id='__warmup__')] if args.warmup else []) + cases
    plan.write_text(''.join(f'{c["id"]} {json.dumps(str(dataset_root/inputs[c["input"]]["files"]["ids"]))} {c["output_tokens"]}\n' for c in planned))
    cache = args.build_dir.resolve()/'CMakeCache.txt'
    build_info = [line for line in cache.read_text().splitlines()
                  if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CUDA_ARCHITECTURES:',
                                      'CMAKE_CUDA_COMPILER:', 'CMAKE_CXX_COMPILER:'))] if cache.exists() else []
    summary = {'status': 'RUNNING', 'numerical_acceptance': 'PENDING',
               'official_v0_baseline': False, 'batch_size': 1, 'concurrent_requests': 1,
               'timing_protocol': 3, 'warmup': args.warmup,
               'warmup_scope': 'global', 'warmup_case': representative if args.warmup else None,
               'repeats': args.repeats, 'case_ids': [c['id'] for c in cases],
               'git_commit': capture(['git', 'rev-parse', 'HEAD']),
               'git_status': capture(['git', 'status', '--short']),
               'probe_sha256': sha(probe),
               'model_source_sha256': sha(ROOT.parent/'models/qwen_model.cuh'),
               'timer_source_sha256': sha(ROOT/'benchmark_probe.cu'), 'manifest_sha256': sha(dataset_root/'manifest.json'),
               'dataset': manifest['dataset'], 'dataset_directory': str(dataset_root),
               'build_configuration': build_info, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'nvcc': capture(['nvcc', '--version']),
               'gpu_before': capture(['nvidia-smi', '--query-gpu=index,name,driver_version,temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader']),
               'memory': {'method': 'nvidia-smi process used_gpu_memory polling',
                          'interval_seconds': 1, 'scope': 'whole run including load and warmups',
                          'observed_peak_mib': None, 'samples': 0}, 'cases': []}
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    stop = threading.Event()
    samples = []
    process = None
    def memory_monitor():
        with (output/'memory.jsonl').open('w') as log:
            while not stop.is_set():
                raw = capture(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'])
                values = []
                for line in (raw or '').splitlines():
                    try:
                        pid, memory = line.split(',')
                        if int(pid) == process.pid:
                            values.append(float(memory))
                    except ValueError:
                        pass
                if values:
                    row = {'monotonic_seconds': time.monotonic(), 'mib': sum(values)}
                    samples.append(row)
                    log.write(json.dumps(row)+'\n'); log.flush()
                stop.wait(1)
    try:
        with (output/'stderr.log').open('w') as err, (output/'raw.jsonl').open('w') as raw:
            process = subprocess.Popen([str(probe), str(args.model.resolve()/'model.safetensors'),
                                        str(plan), str(args.warmup), str(args.repeats)],
                                       stdout=subprocess.PIPE, stderr=err, text=True)
            monitor = threading.Thread(target=memory_monitor, daemon=True)
            monitor.start()
            rows = []
            warmup_rows = []
            print('case          P/G       run   prefill(ms)   TTFT(ms)   TPOT(ms)   decode(tok/s)   total(ms)', flush=True)
            for line in process.stdout:
                raw.write(line); raw.flush()
                row = json.loads(line)
                if row['event'] == 'loaded':
                    if row.get('protocol') != 3:
                        raise RuntimeError('Outdated probe: rebuild benchmark_probe for per-token ITL support (protocol 3)')
                    summary['model_load_ms'] = row['model_load_ms']; save()
                    print(f'Model loaded: {row["model_load_ms"]:.2f} ms', flush=True)
                elif row['event'] == 'request':
                    if row['warmup']:
                        if row['case'] != '__warmup__' or len(row['generated_ids']) != representative['output_tokens']:
                            raise RuntimeError('Invalid global warmup result')
                        warmup_rows.append(row)
                        print(f'{row["case"]}: warmup {row["iteration"]+1}/{args.warmup}', flush=True)
                        continue
                    expected = next(c for c in cases if c['id'] == row['case'])
                    if (row['prompt_tokens'] != expected['prompt_tokens'] or
                        row['output_tokens'] != expected['output_tokens'] or
                        len(row['generated_ids']) != expected['output_tokens']):
                        raise RuntimeError('Probe output workload mismatch')
                    if not all(math.isfinite(row[k]) and row[k] > 0 for k in METRICS):
                        raise RuntimeError('Invalid timing result')
                    if not math.isclose(row['total_ms'], row['ttft_ms']+row['decode_ms'], rel_tol=1e-8):
                        raise RuntimeError('Timing boundaries inconsistent')
                    validate_timing(row)
                    row['itl_mean_ms'] = statistics.mean(row['itl_ms'])
                    rows.append(row)
                    lengths = f'{row["prompt_tokens"]}/{row["output_tokens"]}'
                    print(f'{row["case"]:14} {lengths:9} {row["iteration"]+1:2} {row["prefill_ms"]:12.2f} {row["ttft_ms"]:10.2f} {row["tpot_ms"]:10.3f} {row["decode_tokens_per_second"]:15.2f} {row["total_ms"]:11.2f}', flush=True)
            if process.wait() != 0:
                raise RuntimeError('Engine failed; see stderr.log')
            if sorted(r['iteration'] for r in warmup_rows) != list(range(args.warmup)):
                raise RuntimeError('Missing or duplicate global warmups')
            for c in cases:
                selected = [r for r in rows if r['case'] == c['id']]
                if sorted(r['iteration'] for r in selected) != list(range(args.repeats)):
                    raise RuntimeError('Missing or duplicate measured iterations')
                summary['cases'].append({'case': c['id'], 'prompt_tokens': c['prompt_tokens'],
                    'output_tokens': c['output_tokens'], **{key: distribution([r[key] for r in selected])
                       for key in METRICS},
                    'itl_ms': distribution([x for r in selected for x in r['itl_ms']]), 'generated_ids_repeatable': all(
                    r['generated_ids'] == selected[0]['generated_ids'] for r in selected)})
            with (output/'results.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['case', 'iteration', 'prompt_tokens', 'output_tokens', *METRICS, 'itl_mean_ms'], extrasaction='ignore')
                writer.writeheader(); writer.writerows(rows)
            with (output/'itl.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['case', 'iteration', 'output_token_index', 'itl_ms'])
                for row in rows:
                    writer.writerows((row['case'], row['iteration'], i, value)
                                     for i, value in enumerate(row['itl_ms'], 1))
            summary['aggregate'] = aggregate_metrics(rows)
            (output/'aggregate.json').write_text(json.dumps(summary['aggregate'], indent=2)+'\n')
            summary['status'] = 'MEASUREMENTS_COMPLETE_NUMERICAL_REVIEW_REQUIRED'
    except BaseException as exc:
        summary['status'] = 'FAILED_OR_INTERRUPTED'; summary['error'] = str(exc)
        raise
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        stop.set()
        if 'monitor' in locals():
            monitor.join(timeout=12)
        summary['memory']['samples'] = len(samples)
        summary['memory']['observed_peak_mib'] = max((s['mib'] for s in samples), default=None)
        if not samples:
            summary['memory']['unavailable_reason'] = 'No process memory samples available; see driver/tool support or run duration'
        summary['gpu_after'] = capture(['nvidia-smi', '--query-gpu=index,name,driver_version,temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader'])
        save()
    aggregate = summary['aggregate']
    print(f'\nAggregate: {aggregate["measured_requests"]} measured requests (warmup excluded)')
    print('Latency (ms)                 mean          P50          P95          P99')
    for key, stats in aggregate['latency_distributions_ms'].items():
        print(f'{key:25} {stats["mean"]:12.3f} {stats["p50"]:12.3f} '
              f'{stats["p95"]:12.3f} {stats["p99"]:12.3f} (n={stats["count"]})')
    print(f'Token-weighted TPOT: {aggregate["token_weighted_tpot_ms"]:.3f} ms/token')
    for key in ('overall_decode_tokens_per_second', 'total_tokens_per_second',
                'output_tokens_per_second', 'serial_requests_per_second'):
        print(f'{key}: {aggregate[key]:.3f}')
    print('Throughput denominator: summed timed engine requests; concurrency=1; '
          'excludes warmup/load/I/O/gaps. Not concurrent service capacity.')
    print(f'Model load: {summary["model_load_ms"]:.2f} ms; observed process memory peak: '
          f'{summary["memory"]["observed_peak_mib"]} MiB')
    print(f'Results: {output}\nNumerical acceptance is pending; these are not official V0 scores.')


if __name__ == '__main__':
    main()
