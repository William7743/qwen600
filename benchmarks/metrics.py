"""Protocol-3 metrics for serial, pretokenized engine requests (milliseconds)."""
import math
import statistics

LATENCIES = ('prefill_ms', 'ttft_ms', 'decode_ms', 'tpot_ms', 'total_ms')


def distribution(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Expected nonempty finite nonnegative measurements')
    def percentile(p):
        index = (len(values)-1)*p
        lo = math.floor(index)
        hi = math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return dict(count=len(values), mean=statistics.mean(values), min=values[0],
                max=values[-1], median=percentile(.5), p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99))


def validate_timing(row):
    count = row['output_tokens']-1
    if count < 1 or len(row['itl_ms']) != count:
        raise ValueError('Missing or extra ITL samples')
    values = [row[k] for k in LATENCIES]+row['itl_ms']
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Invalid latency')
    tail = row['decode_tail_ms']
    if not math.isfinite(tail) or tail < 0 or row['prefill_ms'] > row['ttft_ms']:
        raise ValueError('Invalid timing boundaries')
    for a, b in ((sum(row['itl_ms'])+tail, row['decode_ms']),
                 (row['ttft_ms']+row['decode_ms'], row['total_ms']),
                 (row['decode_ms']/count, row['tpot_ms']),
                 (1000*count/row['decode_ms'], row['decode_tokens_per_second'])):
        if not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError('Inconsistent timing or rate')


def aggregate_metrics(rows):
    if not rows:
        raise ValueError('No measured requests')
    for row in rows:
        validate_timing(row)
    decode_ms = sum(r['decode_ms'] for r in rows)
    total_ms = sum(r['total_ms'] for r in rows)
    prompt = sum(r['prompt_tokens'] for r in rows)
    output = sum(r['output_tokens'] for r in rows)
    decode_tokens = output-len(rows)
    distributions = {key: distribution([r[key] for r in rows]) for key in LATENCIES}
    distributions['itl_ms'] = distribution([x for r in rows for x in r['itl_ms']])
    return {
        'measured_requests': len(rows), 'total_prompt_tokens': prompt,
        'total_output_tokens': output, 'total_request_ms': total_ms,
        'total_decode_tokens': decode_tokens, 'total_decode_ms': decode_ms,
        'request_means': {key: distributions[key]['mean'] for key in LATENCIES},
        'latency_distributions_ms': distributions,
        'percentile_method': 'linear interpolation at (n-1)*p; request latencies across requests; ITL across all token intervals',
        'token_weighted_tpot_ms': decode_ms/decode_tokens,
        'overall_decode_tokens_per_second': 1000*decode_tokens/decode_ms,
        'total_tokens_per_second': 1000*(prompt+output)/total_ms,
        'output_tokens_per_second': 1000*output/total_ms,
        'serial_requests_per_second': 1000*len(rows)/total_ms,
        'throughput_scope': 'concurrency=1; denominator=sum timed engine request durations (decode metric uses sum decode durations); excludes load, warmup, I/O and inter-request gaps; not server wall-clock throughput',
    }
