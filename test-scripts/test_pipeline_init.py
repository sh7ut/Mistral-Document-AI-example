"""Smoke tests for pipeline scaffolding."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline

SAMPLE_PDF = ROOT / "sample-data" / "Amazon-2025-Proxy-Statement.pdf"


class PipelineInitTests(unittest.TestCase):
    def test_config_validation_requires_existing_file(self) -> None:
        cfg = pipeline.PipelineConfig(input_path=Path("/tmp/does-not-exist.pdf"))
        with self.assertRaises(FileNotFoundError):
            cfg.validate()

    def setUp(self) -> None:
        if not SAMPLE_PDF.exists():
            self.skipTest("Sample PDF not found for tests.")

    def test_pipeline_initializes_with_valid_file(self) -> None:
        cfg = pipeline.PipelineConfig(input_path=SAMPLE_PDF)
        client = mock.MagicMock(name="MistralClient")
        pipeline.DocumentPipeline(client=client, config=cfg)

    def test_init_client_raises_without_env(self) -> None:
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": ""}):
            with self.assertRaises(ValueError):
                pipeline.DocumentPipeline.init_client()

    def test_upload_document_calls_mistral_apis(self) -> None:
        cfg = pipeline.PipelineConfig(input_path=SAMPLE_PDF)
        mock_client = mock.MagicMock()
        mock_client.files.upload.return_value.id = "file_123"
        mock_client.files.get_signed_url.return_value.url = "https://signed"
        doc_pipeline = pipeline.DocumentPipeline(client=mock_client, config=cfg)

        uploaded = doc_pipeline.upload_document(expiry_hours=12)

        mock_client.files.upload.assert_called_once()
        mock_client.files.get_signed_url.assert_called_once_with(
            file_id="file_123", expiry=12
        )
        self.assertEqual(uploaded.file_id, "file_123")
        self.assertEqual(uploaded.signed_url, "https://signed")

    def test_cleanup_remote_file_deletes_when_flag_enabled(self) -> None:
        cfg = pipeline.PipelineConfig(input_path=SAMPLE_PDF, delete_remote_file=True)
        mock_client = mock.MagicMock()
        mock_client.files.upload.return_value.id = "file_456"
        mock_client.files.get_signed_url.return_value.url = "https://signed"
        doc_pipeline = pipeline.DocumentPipeline(client=mock_client, config=cfg)
        doc_pipeline.upload_document()

        doc_pipeline.cleanup_remote_file()

        mock_client.files.delete.assert_called_once_with(file_id="file_456")

    def test_run_ocr_calls_client_with_signed_url(self) -> None:
        cfg = pipeline.PipelineConfig(input_path=SAMPLE_PDF)
        mock_client = mock.MagicMock()
        mock_client.files.upload.return_value.id = "file_789"
        mock_client.files.get_signed_url.return_value.url = "https://signed"
        mock_client.ocr.process.return_value = {"pages": []}
        doc_pipeline = pipeline.DocumentPipeline(client=mock_client, config=cfg)
        document = doc_pipeline.upload_document()

        response = doc_pipeline.run_ocr(document=document)

        mock_client.ocr.process.assert_called_once_with(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": "https://signed",
            },
            table_format="html",
            include_image_base64=True,
        )
        self.assertEqual(response, {"pages": []})

    def test_run_ocr_raises_without_uploaded_document(self) -> None:
        cfg = pipeline.PipelineConfig(input_path=SAMPLE_PDF)
        mock_client = mock.MagicMock()
        doc_pipeline = pipeline.DocumentPipeline(client=mock_client, config=cfg)
        with self.assertRaises(ValueError):
            doc_pipeline.run_ocr()


if __name__ == "__main__":
    unittest.main()
