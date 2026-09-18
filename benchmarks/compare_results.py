"""Compare two completed runs of the same fixed workload and timing protocol."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def validate_runs(old, new):
    for key in ['manifest_sha256', 'case_ids', 'batch_size', 'warmup', 'warmup_scope', 'repeats']:
        if old[key] != new[key]:
            raise ValueError('Incompatible runs: ' + key)
    if old.get('timing_protocol', 2) != new.get('timing_protocol', 2):
        raise ValueError('Different timing protocols; rerun both versions with the same timer')
    if any(x['status'] != 'MEASUREMENTS_COMPLETE_NUMERICAL_REVIEW_REQUIRED' for x in [old, new]):
        raise ValueError('Expected two completed runs')
    if old.get('timing_mode', 'full') != new.get('timing_mode', 'full'):
        raise ValueError('Different ITL modes; rerun both versions with the README benchmark command')
    if any(x.get('measurement_role', 'latency') != 'latency' for x in [old,new]):
        raise ValueError('Resource passes cannot supply formal latency comparisons')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v0', type=Path, required=True)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = [read(p / 'summary.json') for p in [args.v0, args.v1]]
    validate_runs(old, new)
    def records(path):
        with (path / 'results.csv').open() as f:
            return {(r['case'], int(r['iteration'])): r for r in csv.DictReader(f)}
    a, b = records(args.v0), records(args.v1)
    if a.keys() != b.keys():
        raise ValueError('Measured requests differ')
    rows = []
    for key, r0 in a.items():
        r1 = b[key]
        if any(r0[k] != r1[k] for k in ['prompt_tokens', 'output_tokens']):
            raise ValueError('Token budget differs')
        row = {'case': key[0], 'iteration': key[1], 'prompt_tokens': int(r0['prompt_tokens']),
               'output_tokens': int(r0['output_tokens'])}
        for metric in ['prefill_ms', 'ttft_ms', 'tpot_ms', 'total_ms']:
            v0, v1 = float(r0[metric]), float(r1[metric])
            if v0 <= 0 or v1 <= 0:
                raise ValueError('Nonpositive timing')
            row[metric] = {'v0': v0, 'v1': v1, 'speedup': v0 / v1}
        rows.append(row)
        print(key[0], 'TTFT %.3fx TPOT %.3fx total %.3fx' % tuple(
            row[m]['speedup'] for m in ['ttft_ms', 'tpot_ms', 'total_ms']))
    aggs = [read(p / 'aggregate.json') for p in [args.v0, args.v1]]
    means = {k: {'v0': aggs[0]['request_means'][k], 'v1': aggs[1]['request_means'][k],
                 'speedup': aggs[0]['request_means'][k] / aggs[1]['request_means'][k]}
             for k in aggs[0]['request_means']}
    result = {'scope': 'Current V0 vs V1 experiment; HF numerical acceptance remains pending',
              'v0_commit': old['git_commit'], 'v1_commit': new['git_commit'],
              'manifest_sha256': old['manifest_sha256'], 'request_means': means,
              'token_weighted_tpot_ms': {'v0': aggs[0]['token_weighted_tpot_ms'],
                                         'v1': aggs[1]['token_weighted_tpot_ms']},
              'overall_decode_tokens_per_second': {
                  'v0': aggs[0]['overall_decode_tokens_per_second'],
                  'v1': aggs[1]['overall_decode_tokens_per_second']}, 'requests': rows}
    if old.get('timing_protocol') in (3,4):
        result['timing_protocol'] = old['timing_protocol']
        result['throughput'] = {key: {'v0': aggs[0][key], 'v1': aggs[1][key],
            'speedup': aggs[1][key]/aggs[0][key]} for key in (
            'total_tokens_per_second', 'output_tokens_per_second',
            'overall_decode_tokens_per_second', 'serial_requests_per_second')}
        result['latency_distributions_ms'] = {
            'v0': aggs[0]['latency_distributions_ms'], 'v1': aggs[1]['latency_distributions_ms']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Mean TTFT %.3fx; token-weighted TPOT %.3fx' % (
        means['ttft_ms']['speedup'], aggs[0]['token_weighted_tpot_ms'] / aggs[1]['token_weighted_tpot_ms']))


if __name__ == '__main__':
    main()
