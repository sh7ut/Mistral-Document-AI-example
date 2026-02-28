"""Risk classification helpers powered by Mistral chat models."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mistralai import Mistral as MistralClient

from utils import EntityFeature

DEFAULT_PROMPT = (
    "You are a senior risk analyst. Review the extracted financial statement snippets and assign a risk tier. "
    "Respond strictly in JSON with keys risk_tier (Low/Medium/High), rationale (string), confidence (0-1 float)."
)


@dataclass
class ClassificationResult:
    document_id: str
    risk_tier: str
    rationale: str
    confidence: float
    raw_output: Dict[str, Any]


class DocumentClassifier:
    """Thin wrapper around a Mistral chat model for risk classification."""

    def __init__(
        self,
        client: Optional[MistralClient] = None,
        *,
        model: str = "mistral-large-latest",
        temperature: float = 0.2,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.client = client or self._init_client()
        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template

    @staticmethod
    def _init_client() -> MistralClient:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running.")
        return MistralClient(api_key=api_key)

    def classify(self, document_id: str, features: List[EntityFeature]) -> ClassificationResult:
        if not document_id:
            raise ValueError("document_id is required")
        if not features:
            raise ValueError("features list must not be empty")

        messages = self._build_messages(document_id, features)
        response = self._chat_completion(messages)
        payload = self._extract_json(response)
        return ClassificationResult(
            document_id=document_id,
            risk_tier=payload.get("risk_tier", "Unknown"),
            rationale=payload.get("rationale", ""),
            confidence=float(payload.get("confidence", 0.0)),
            raw_output=payload,
        )

    def _build_messages(self, document_id: str, features: List[EntityFeature]) -> List[Dict[str, Any]]:
        formatted = self._format_features(features)
        return [
            {
                "role": "system",
                "content": self.prompt_template,
            },
            {
                "role": "user",
                "content": f"Document ID: {document_id}\n\nExtracted Features:\n{formatted}",
            },
        ]

    @staticmethod
    def _format_features(features: List[EntityFeature]) -> str:
        parts = []
        for feature in features:
            snippet = (feature.text_snippet or "").strip().replace("\n", " ")
            snippet = snippet[:500]
            parts.append(
                f"[{feature.entity_type} page={feature.page_index}] {snippet}"
            )
        return "\n".join(parts)

    @staticmethod
    def _extract_json(response: Any) -> Dict[str, Any]:
        try:
            choice = response.choices[0]
            content = choice.message.content
            return json.loads(content)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to parse classifier response as JSON") from exc

    def _chat_completion(self, messages: List[Dict[str, Any]]):
        chat_api = getattr(self.client, "chat", None)
        if hasattr(chat_api, "complete"):
            return chat_api.complete(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        if callable(chat_api):
            return chat_api(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        raise ValueError("Client does not expose a chat completion method")


__all__ = ["DocumentClassifier", "ClassificationResult"]
