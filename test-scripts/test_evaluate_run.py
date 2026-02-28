"""Tests for evaluate_run batch helper."""
import json
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification import ClassificationResult
from evaluation import EvaluationResult, MetricsSummary
from evaluate_run import load_manifest, run_batch
from utils import EntityFeature


class DummyFeatureStore:
    def __init__(self) -> None:
        feature = EntityFeature(
            entity_type="text_block",
            name="page_0_text",
            page_index=0,
            text_snippet="Cash flow",
        )
        self.store = {"doc-1": [feature], "doc-2": [feature]}

    def load_features(self, document_id: str):
        return self.store[document_id]


class DummyClassifier:
    def classify(self, document_id, features):  # noqa: D401
        return ClassificationResult(
            document_id=document_id,
            risk_tier="High" if document_id == "doc-1" else "Low",
            rationale="stub",
            confidence=0.8,
            raw_output={},
        )


class DummyJudge:
    def evaluate_case(self, case):  # noqa: D401
        return EvaluationResult(
            document_id=case.document_id,
            agree=True,
            reason="ok",
            preferred_tier=case.ground_truth_tier,
            severity="Low",
            raw_output={},
        )

    def summarize(self, cases, evaluations):  # noqa: D401
        return MetricsSummary(
            total=len(cases),
            agreement_rate=1.0,
            accuracy_vs_ground_truth=0.5,
            avg_confidence=sum(case.prediction.confidence for case in cases) / len(cases),
            disagreements=[],
        )


class EvaluateRunTests(unittest.TestCase):
    def test_run_batch_returns_metrics_and_results(self) -> None:
        manifest = [
            {"document_id": "doc-1", "ground_truth_tier": "High"},
            {"document_id": "doc-2", "ground_truth_tier": "Low"},
        ]
        payload = run_batch(
            manifest,
            classifier=DummyClassifier(),
            judge=DummyJudge(),
            feature_store=DummyFeatureStore(),
        )
        self.assertEqual(len(payload["classifications"]), 2)
        self.assertEqual(payload["classifications"][0]["risk_tier"], "High")
        self.assertEqual(len(payload["evaluations"]), 2)
        self.assertIn("metrics", payload)
        self.assertEqual(payload["metrics"]["total"], 2)

    def test_load_manifest_parses_json_list(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump([
                {"document_id": "doc-1", "ground_truth_tier": "High"}
            ], tmp)
            tmp_path = Path(tmp.name)
        try:
            manifest = load_manifest(tmp_path)
            self.assertEqual(manifest[0]["document_id"], "doc-1")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
