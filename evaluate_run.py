"""Batch evaluator that replays classifications and LLM judging from stored features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from classification import DocumentClassifier
from evaluation import (
    EvaluationCase,
    LlmJudge,
    metrics_summary_from_dict,
    metrics_summary_to_dict,
)
from metrics_report import format_report
from storage import LocalFeatureStore


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list")
    entries: List[Dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Manifest entries must be objects")
        if "document_id" not in entry or "ground_truth_tier" not in entry:
            raise ValueError("Each entry requires document_id and ground_truth_tier")
        entries.append(entry)
    return entries


def run_batch(
    manifest: Iterable[Dict[str, str]],
    *,
    classifier: DocumentClassifier,
    judge: LlmJudge,
    feature_store: LocalFeatureStore,
) -> Dict[str, object]:
    cases: List[EvaluationCase] = []
    evaluations = []
    classifications = []

    for entry in manifest:
        document_id = entry["document_id"]
        ground_truth = entry["ground_truth_tier"]
        features = feature_store.load_features(document_id)
        classification = classifier.classify(document_id, features)
        case = EvaluationCase(
            document_id=document_id,
            prediction=classification,
            ground_truth_tier=ground_truth,
        )
        evaluation = judge.evaluate_case(case)
        cases.append(case)
        evaluations.append(evaluation)
        classifications.append(
            {
                "document_id": document_id,
                "risk_tier": classification.risk_tier,
                "confidence": classification.confidence,
                "rationale": classification.rationale,
            }
        )

    metrics_obj = judge.summarize(cases, evaluations)
    return {
        "classifications": classifications,
        "evaluations": [
            {
                "document_id": result.document_id,
                "agree": result.agree,
                "preferred_tier": result.preferred_tier,
                "severity": result.severity,
                "reason": result.reason,
            }
            for result in evaluations
        ],
        "metrics": metrics_summary_to_dict(metrics_obj),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch classification + judging")
    parser.add_argument("manifest", type=Path, help="Path to manifest JSON")
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("demo_feature_store"),
        help="Directory containing stored feature JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("batch_results.json"),
        help="Where to write per-document classification/evaluation results",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("metrics_summary.json"),
        help="Where to write aggregate MetricsSummary JSON",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print a human-readable report using metrics_report.format_report",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    feature_store = LocalFeatureStore(root=args.feature_store)
    classifier = DocumentClassifier()
    judge = LlmJudge()
    batch_payload = run_batch(
        manifest,
        classifier=classifier,
        judge=judge,
        feature_store=feature_store,
    )

    args.output.write_text(json.dumps(batch_payload, indent=2), encoding="utf-8")
    metrics = batch_payload["metrics"]
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.print_report:
        summary = format_report(metrics_summary_from_dict(metrics))
        print(summary)


if __name__ == "__main__":
    main()
