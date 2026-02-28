"""Risk classification helpers powered by Mistral chat models."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
import re
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
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        temperature: float = 0.2,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.client = client or self._init_client()
        env_model = os.getenv("CLASSIFIER_MODEL_ID")
        env_agent = os.getenv("CLASSIFIER_AGENT_ID")
        self.agent_id = agent_id or env_agent
        self.model = model or env_model or "mistral-large-latest"
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
        response = self._run_completion(messages)
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
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            text_parts.append(block["text"])
                        elif "content" in block:
                            text_parts.append(str(block["content"]))
                        elif "value" in block:
                            text_parts.append(str(block["value"]))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "".join(text_parts)
            text = DocumentClassifier._normalize_json_text(content)
            return json.loads(text, strict=False)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to parse classifier response as JSON") from exc

    def _run_completion(self, messages: List[Dict[str, Any]]):
        if self.agent_id:
            return self._agent_completion(messages)
        return self._chat_completion(messages)

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

    def _agent_completion(self, messages: List[Dict[str, Any]]):
        if not self.agent_id:
            raise ValueError("agent_id required for agent completion")
        agents_api = getattr(self.client, "agents", None)
        if hasattr(agents_api, "complete"):
            return agents_api.complete(
                agent_id=self.agent_id,
                messages=messages,
            )
        raise ValueError("Client does not expose an agents completion method")

    @staticmethod
    def _normalize_json_text(content: Any) -> str:
        text = str(content or "").strip()
        if not text:
            raise ValueError("Empty response content")
        # Strip ```json fences
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        if text and not text.lstrip().startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        return text.strip()


__all__ = ["DocumentClassifier", "ClassificationResult"]
