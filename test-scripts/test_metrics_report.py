"""Tests for metrics_report utility."""
import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import metrics_report


class MetricsReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        payload = {
            "total": 5,
            "agreement_rate": 0.8,
            "accuracy_vs_ground_truth": 0.6,
            "avg_confidence": 0.75,
            "disagreements": [
                {
                    "document_id": "doc-2",
                    "agree": False,
                    "reason": "Judge prefers Medium",
                    "preferred_tier": "Medium",
                    "severity": "High",
                    "raw_output": {},
                }
            ],
        }
        json.dump(payload, open(self.tmp.name, "w", encoding="utf-8"))

    def tearDown(self) -> None:
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_load_metrics_summary(self) -> None:
        summary = metrics_report.load_metrics_summary(Path(self.tmp.name))
        self.assertEqual(summary.total, 5)
        self.assertEqual(summary.agreement_rate, 0.8)
        self.assertEqual(len(summary.disagreements), 1)

    def test_format_report_contains_key_stats(self) -> None:
        summary = metrics_report.load_metrics_summary(Path(self.tmp.name))
        report = metrics_report.format_report(summary, top_disagreements=2)
        self.assertIn("Total Cases: 5", report)
        self.assertIn("Agreement Rate: 80.00%", report)
        self.assertIn("Doc doc-2", report)


if __name__ == "__main__":
    unittest.main()
