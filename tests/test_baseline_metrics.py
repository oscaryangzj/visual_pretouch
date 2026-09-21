import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import baseline_metrics


class BaselineMetricsTests(unittest.TestCase):
    def test_summary_uses_successful_rows_with_mm_errors(self):
        rows = [
            {'method': 'b1_crossing_projection', 'status': 'ok', 'error_mm': '5'},
            {'method': 'b1_crossing_projection', 'status': 'ok', 'error_mm': '15'},
            {'method': 'b1_crossing_projection', 'status': 'failed', 'error_mm': ''},
            {'method': 'b2_linear_stat_time', 'status': 'ok', 'error_mm': '25'},
        ]
        summary = baseline_metrics.summarize(rows)
        b1, b2 = summary
        self.assertEqual((b1['trials'], b1['successful_predictions'], b1['mm_samples']), (3, 2, 2))
        self.assertEqual((b1['mean_error_mm'], b1['median_error_mm']), (10, 10))
        self.assertEqual((b1['hit_at_10mm'], b1['hit_at_15mm'], b1['hit_at_20mm']), (.5, 1, 1))
        self.assertEqual((b2['mean_error_mm'], b2['hit_at_20mm']), (25, 0))


if __name__ == '__main__':
    unittest.main()
