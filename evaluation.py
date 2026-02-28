"""Evaluation harness leveraging LLM-as-a-judge."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

from mistralai import Mistral as MistralClient

from classification import ClassificationResult

DEFAULT_EVAL_PROMPT = (
    "You are an impartial audit reviewer. Given a model-assigned risk tier and the ground truth tier, "
    "return JSON with keys: agree (true/false), reason (string), preferred_tier (string), severity (Low/Medium/High)."
)


@dataclass
class EvaluationCase:
    document_id: str
    prediction: ClassificationResult
    ground_truth_tier: str


@dataclass
class EvaluationResult:
    document_id: str
    agree: bool
    reason: str
    preferred_tier: str
    severity: str
    raw_output: Dict[str, Any]


@dataclass
class MetricsSummary:
    total: int
    agreement_rate: float
    accuracy_vs_ground_truth: float
    avg_confidence: float
    disagreements: List[EvaluationResult]


def evaluation_result_from_dict(payload: Dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        document_id=payload["document_id"],
        agree=bool(payload.get("agree", False)),
        reason=payload.get("reason", ""),
        preferred_tier=payload.get("preferred_tier", "Unknown"),
        severity=payload.get("severity", "Medium"),
        raw_output=payload.get("raw_output", {}),
    )


def evaluation_result_to_dict(result: EvaluationResult) -> Dict[str, Any]:
    data = asdict(result)
    return data


def metrics_summary_from_dict(payload: Dict[str, Any]) -> MetricsSummary:
    disagreements = [
        evaluation_result_from_dict(item) for item in payload.get("disagreements", [])
    ]
    return MetricsSummary(
        total=int(payload.get("total", 0)),
        agreement_rate=float(payload.get("agreement_rate", 0.0)),
        accuracy_vs_ground_truth=float(payload.get("accuracy_vs_ground_truth", 0.0)),
        avg_confidence=float(payload.get("avg_confidence", 0.0)),
        disagreements=disagreements,
    )


def metrics_summary_to_dict(summary: MetricsSummary) -> Dict[str, Any]:
    return {
        "total": summary.total,
        "agreement_rate": summary.agreement_rate,
        "accuracy_vs_ground_truth": summary.accuracy_vs_ground_truth,
        "avg_confidence": summary.avg_confidence,
        "disagreements": [
            evaluation_result_to_dict(result) for result in summary.disagreements
        ],
    }


class LlmJudge:
    def __init__(
        self,
        client: Optional[MistralClient] = None,
        *,
        model: str = "mistral-large-latest",
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_EVAL_PROMPT,
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

    def evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        messages = [
            {"role": "system", "content": self.prompt_template},
            {
                "role": "user",
                "content": self._format_case(case),
            },
        ]
        response = self._chat_completion(messages)
        payload = self._extract_json(response)
        return EvaluationResult(
            document_id=case.document_id,
            agree=bool(payload.get("agree", False)),
            reason=payload.get("reason", ""),
            preferred_tier=payload.get("preferred_tier", case.ground_truth_tier),
            severity=payload.get("severity", "Medium"),
            raw_output=payload,
        )

    def summarize(self, cases: Iterable[EvaluationCase], results: Iterable[EvaluationResult]) -> MetricsSummary:
        results_list = list(results)
        cases_list = list(cases)
        if len(results_list) != len(cases_list):
            raise ValueError("Mismatch between cases and results length")
        total = len(results_list)
        if total == 0:
            return MetricsSummary(total=0, agreement_rate=0.0, accuracy_vs_ground_truth=0.0, avg_confidence=0.0, disagreements=[])
        agreement_rate = sum(result.agree for result in results_list) / total
        accuracy = sum(
            1
            for case, result in zip(cases_list, results_list)
            if case.ground_truth_tier.lower() == case.prediction.risk_tier.lower()
        ) / total
        avg_confidence = sum(case.prediction.confidence for case in cases_list) / total
        disagreements = [result for result in results_list if not result.agree]
        return MetricsSummary(
            total=total,
            agreement_rate=agreement_rate,
            accuracy_vs_ground_truth=accuracy,
            avg_confidence=avg_confidence,
            disagreements=disagreements,
        )

    @staticmethod
    def _format_case(case: EvaluationCase) -> str:
        return (
            f"Document ID: {case.document_id}\n"
            f"Ground Truth Tier: {case.ground_truth_tier}\n"
            f"Model Prediction: {case.prediction.risk_tier} (confidence={case.prediction.confidence:.2f})\n"
            f"Model Rationale: {case.prediction.rationale}"
        )

    @staticmethod
    def _extract_json(response: Any) -> Dict[str, Any]:
        try:
            choice = response.choices[0]
            return json.loads(choice.message.content)
        except Exception as exc:  # pragma: no cover
            raise ValueError("Failed to parse judge response") from exc

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


__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "MetricsSummary",
    "LlmJudge",
    "evaluation_result_from_dict",
    "evaluation_result_to_dict",
    "metrics_summary_from_dict",
    "metrics_summary_to_dict",
]
