"""Small arithmetic and malformed-output checks, no model/GPU required."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from metrics import aggregate_metrics, distribution, validate_timing


class MetricsTest(unittest.TestCase):
    def row(self, prompt, itl, ttft, tail):
        decode = sum(itl)+tail
        return dict(prompt_tokens=prompt, output_tokens=len(itl)+1,
                    prefill_ms=ttft-1, ttft_ms=ttft, decode_ms=decode,
                    total_ms=ttft+decode, tpot_ms=decode/len(itl),
                    decode_tokens_per_second=1000*len(itl)/decode,
                    itl_ms=itl, decode_tail_ms=tail)

    def test_rates_use_totals_and_exclude_first_token_for_decode(self):
        result = aggregate_metrics([self.row(10,[2,4],10,0), self.row(30,[10],20,2)])
        self.assertEqual(result['total_prompt_tokens'],40)
        self.assertEqual(result['total_output_tokens'],5)
        self.assertEqual(result['total_request_ms'],48)
        self.assertEqual(result['total_decode_tokens'],3)
        self.assertAlmostEqual(result['total_tokens_per_second'],937.5)
        self.assertAlmostEqual(result['output_tokens_per_second'],5000/48)
        self.assertAlmostEqual(result['serial_requests_per_second'],2000/48)
        self.assertAlmostEqual(result['overall_decode_tokens_per_second'],3000/18)
        itl = result['latency_distributions_ms']['itl_ms']
        self.assertEqual(itl['count'],3)
        self.assertEqual(itl['p50'],4)
        self.assertAlmostEqual(itl['p95'],9.4)
        self.assertAlmostEqual(itl['p99'],9.88)
        self.assertEqual(result['token_weighted_tpot_ms'],6)

    def test_percentile_singleton_and_interpolation(self):
        self.assertEqual(distribution([4])['p99'],4)
        self.assertEqual(distribution([0,100])['p95'],95)

    def test_reject_missing_intervals_nonfinite_and_inconsistent_boundaries(self):
        for key,value in [('itl_ms',[2]), ('itl_ms',[2,float('nan')]),
                          ('decode_tail_ms',-1), ('decode_ms',99), ('tpot_ms',99),
                          ('total_ms',0)]:
            row = self.row(10,[2,4],10,0)
            row[key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                validate_timing(row)


if __name__=='__main__':
    unittest.main()
