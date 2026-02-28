"""Integration-style tests for DocumentPipeline run stage."""
import sys
from pathlib import Path
from unittest import TestCase, mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline
from classification import ClassificationResult
from utils import EntityFeature

SAMPLE_PDF = ROOT / "sample-data" / "Amazon-2025-Proxy-Statement.pdf"


class DocumentPipelineRunTests(TestCase):
    def setUp(self) -> None:
        if not SAMPLE_PDF.exists():
            self.skipTest("Sample PDF not available")
        self.cfg = pipeline.PipelineConfig(input_path=SAMPLE_PDF)
        self.mock_client = mock.MagicMock()
        self.mock_extractor = mock.MagicMock()
        self.mock_store = mock.MagicMock()
        self.mock_classifier = mock.MagicMock()
        self.mock_judge = mock.MagicMock()
        self.mock_features = [
            EntityFeature(
                entity_type="text_block",
                name="page_0_text",
                page_index=0,
                text_snippet="Sample",
            )
        ]
        self.mock_extractor.build_features.return_value = self.mock_features
        self.mock_store.save_features.return_value = Path("/tmp/features.json")
        self.mock_classifier.classify.return_value = ClassificationResult(
            document_id="file_abc",
            risk_tier="Medium",
            rationale="Sample rationale",
            confidence=0.66,
            raw_output={},
        )
        self.mock_judge.evaluate_case.return_value = mock.MagicMock(
            agree=True,
            preferred_tier="Medium",
            severity="Low",
            reason="Matches",
        )

    def _build_pipeline(self) -> pipeline.DocumentPipeline:
        return pipeline.DocumentPipeline(
            client=self.mock_client,
            config=self.cfg,
            feature_extractor=self.mock_extractor,
            feature_store=self.mock_store,
            classifier=self.mock_classifier,
            judge=self.mock_judge,
        )

    def test_run_persists_features_and_returns_summary(self) -> None:
        doc_pipeline = self._build_pipeline()
        uploaded = pipeline.UploadedDocument(
            file_id="file_abc",
            file_name="sample.pdf",
            signed_url="https://signed",
        )
        doc_pipeline.upload_document = mock.MagicMock(return_value=uploaded)
        doc_pipeline.run_ocr = mock.MagicMock(
            return_value={
                "model": "mistral-ocr-latest",
                "pages": [{"index": 0, "markdown": "Hello"}],
                "usage_info": {"tokens": 10},
            }
        )
        doc_pipeline.cleanup_remote_file = mock.MagicMock()

        result = doc_pipeline.run(
            embed_features=True,
            expiry_hours=24,
            classify=True,
            ground_truth_tier="Medium",
        )

        doc_pipeline.upload_document.assert_called_once_with(expiry_hours=24)
        doc_pipeline.run_ocr.assert_called_once()
        self.mock_extractor.build_features.assert_called_once()
        self.mock_store.save_features.assert_called_once_with("file_abc", self.mock_features)
        doc_pipeline.cleanup_remote_file.assert_called_once()
        self.mock_classifier.classify.assert_called_once_with("file_abc", self.mock_features)
        self.assertEqual(result["feature_count"], 1)
        self.assertIn("feature_path", result)
        self.assertEqual(result["classification"]["risk_tier"], "Medium")
        self.mock_judge.evaluate_case.assert_called_once()
        self.assertTrue(result["evaluation"]["agree"])

    def test_run_always_cleans_up(self) -> None:
        doc_pipeline = self._build_pipeline()
        doc_pipeline.upload_document = mock.MagicMock(side_effect=RuntimeError("boom"))
        doc_pipeline.cleanup_remote_file = mock.MagicMock()
        with self.assertRaises(RuntimeError):
            doc_pipeline.run()
        doc_pipeline.cleanup_remote_file.assert_called_once()


if __name__ == "__main__":
    import unittest

    unittest.main()
