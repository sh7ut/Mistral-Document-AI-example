"""Core pipeline orchestration for the Mistral Document AI workflow."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mistralai import Mistral as MistralClient

from classification import ClassificationResult, DocumentClassifier
from storage import LocalFeatureStore
from evaluation import EvaluationCase, LlmJudge
from utils import FeatureExtractor, parse_ocr_response

@dataclass
class PipelineConfig:
    """User-supplied runtime configuration."""

    input_path: Path
    delete_remote_file: bool = False
    table_format: str = "html"
    include_image_base64: bool = True

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if not self.input_path.is_file():
            raise ValueError(f"Input path must be a file: {self.input_path}")


class DocumentPipeline:
    """High-level pipeline driver (upload → OCR → parse → classify)."""

    def __init__(
        self,
        client: MistralClient,
        config: PipelineConfig,
        *,
        feature_extractor: Optional[FeatureExtractor] = None,
        feature_store: Optional[LocalFeatureStore] = None,
        classifier: Optional[DocumentClassifier] = None,
        judge: Optional[LlmJudge] = None,
    ):
        config.validate()
        self.client = client
        self.config = config
        self._uploaded_document: Optional[UploadedDocument] = None
        self.feature_extractor = feature_extractor or FeatureExtractor(client=self.client)
        self.feature_store = feature_store or LocalFeatureStore()
        self.classifier = classifier
        self.judge = judge

    @staticmethod
    def init_client() -> MistralClient:
        """Create a Mistral client using the required env var."""

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable not set. Please set it before running."
            )
        client = MistralClient(api_key=api_key)
        print("✓ Mistral client initialized")
        return client

    def run(
        self,
        *,
        embed_features: bool = False,
        expiry_hours: Optional[int] = None,
        classify: bool = False,
        ground_truth_tier: Optional[str] = None,
    ) -> dict:
        """End-to-end execution for upload → OCR → feature persistence (+ optional classification)."""

        uploaded: Optional[UploadedDocument] = None
        try:
            uploaded = self.upload_document(expiry_hours=expiry_hours)
            ocr_payload = self.run_ocr(uploaded)
            artifacts = parse_ocr_response(ocr_payload)
            features = self.feature_extractor.build_features(artifacts, embed=embed_features)
            output_path = self.feature_store.save_features(uploaded.file_id, features)
            result = {
                "uploaded_file_id": uploaded.file_id,
                "feature_count": len(features),
                "feature_path": str(output_path),
            }
            if classify:
                classifier = self.classifier or DocumentClassifier(client=self.client)
                classification = classifier.classify(uploaded.file_id, features)
                result["classification"] = {
                    "risk_tier": classification.risk_tier,
                    "confidence": classification.confidence,
                    "rationale": classification.rationale,
                }
                if ground_truth_tier:
                    judge = self.judge or LlmJudge(client=self.client)
                    eval_case = EvaluationCase(
                        document_id=uploaded.file_id,
                        prediction=classification,
                        ground_truth_tier=ground_truth_tier,
                    )
                    evaluation = judge.evaluate_case(eval_case)
                    result["evaluation"] = {
                        "agree": evaluation.agree,
                        "preferred_tier": evaluation.preferred_tier,
                        "severity": evaluation.severity,
                        "reason": evaluation.reason,
                    }
            return result
        finally:
            self.cleanup_remote_file()

    def upload_document(self, expiry_hours: Optional[int] = None) -> "UploadedDocument":
        """Upload local PDF to Mistral Cloud and return metadata."""

        input_path = self.config.input_path
        with input_path.open("rb") as file_handle:
            uploaded = self.client.files.upload(
                file={
                    "file_name": input_path.name,
                    "content": file_handle,
                },
                purpose="ocr",
            )

        signed_args = {"file_id": uploaded.id}
        if expiry_hours is not None:
            signed_args["expiry"] = expiry_hours
        signed_url = self.client.files.get_signed_url(**signed_args)

        document = UploadedDocument(
            file_id=uploaded.id,
            file_name=input_path.name,
            signed_url=signed_url.url,
            expiry_hours=expiry_hours,
        )
        self._uploaded_document = document
        return document

    def cleanup_remote_file(self) -> None:
        """Delete uploaded file from Mistral Cloud (if configured)."""

        if not self.config.delete_remote_file or not self._uploaded_document:
            return
        try:
            self.client.files.delete(file_id=self._uploaded_document.file_id)
        finally:
            self._uploaded_document = None

    def run_ocr(self, document: Optional["UploadedDocument"] = None) -> dict:
        """Invoke Document AI OCR using a signed URL."""

        target = document or self._uploaded_document
        if not target:
            raise ValueError("No uploaded document available. Call upload_document first.")

        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": target.signed_url,
            },
            table_format=self.config.table_format,
            include_image_base64=self.config.include_image_base64,
        )
        try:
            return response.model_dump()
        except AttributeError:
            return dict(response)


@dataclass
class UploadedDocument:
    """Metadata produced after uploading to Mistral Cloud."""

    file_id: str
    file_name: str
    signed_url: str
    expiry_hours: Optional[int] = None


__all__ = ["PipelineConfig", "DocumentPipeline", "UploadedDocument"]
