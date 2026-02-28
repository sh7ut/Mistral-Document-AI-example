"""Tests for evaluation harness."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification import ClassificationResult
from evaluation import (
    EvaluationCase,
    LlmJudge,
    metrics_summary_from_dict,
    metrics_summary_to_dict,
)


class FakeJudgeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_args = None

    def chat(self, **kwargs):
        self.last_args = kwargs
        choice = SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))
        return SimpleNamespace(choices=[choice])


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        prediction = ClassificationResult(
            document_id="doc-1",
            risk_tier="High",
            rationale="Cash burn",
            confidence=0.9,
            raw_output={},
        )
        self.case = EvaluationCase(
            document_id="doc-1",
            prediction=prediction,
            ground_truth_tier="High",
        )

    def test_evaluate_case_parses_payload(self) -> None:
        client = FakeJudgeClient({"agree": True, "reason": "Matches", "preferred_tier": "High", "severity": "Low"})
        judge = LlmJudge(client=client)

        result = judge.evaluate_case(self.case)

        self.assertTrue(result.agree)
        self.assertEqual(result.preferred_tier, "High")
        self.assertEqual(client.last_args["model"], "mistral-large-latest")

    def test_summarize_metrics(self) -> None:
        judge = LlmJudge(client=FakeJudgeClient({"agree": True}))
        results = [judge.evaluate_case(self.case)]
        metrics = judge.summarize([self.case], results)
        self.assertEqual(metrics.total, 1)
        self.assertEqual(metrics.agreement_rate, 1.0)
        self.assertEqual(metrics.accuracy_vs_ground_truth, 1.0)
        self.assertAlmostEqual(metrics.avg_confidence, 0.9)

    def test_length_mismatch_raises(self) -> None:
        judge = LlmJudge(client=FakeJudgeClient({"agree": True}))
        with self.assertRaises(ValueError):
            judge.summarize([self.case], [])

    def test_metrics_summary_round_trip_serialization(self) -> None:
        judge = LlmJudge(client=FakeJudgeClient({"agree": True}))
        results = [judge.evaluate_case(self.case)]
        metrics = judge.summarize([self.case], results)
        payload = metrics_summary_to_dict(metrics)
        loaded = metrics_summary_from_dict(payload)
        self.assertEqual(loaded.total, metrics.total)
        self.assertEqual(len(loaded.disagreements), len(metrics.disagreements))


if __name__ == "__main__":
    unittest.main()
