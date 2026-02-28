"""CLI utility to render MetricsSummary artifacts as readable text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from evaluation import MetricsSummary, metrics_summary_from_dict


def load_metrics_summary(path: Path) -> MetricsSummary:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return metrics_summary_from_dict(payload)


def format_report(summary: MetricsSummary, *, top_disagreements: int = 3) -> str:
    lines = ["=== Evaluation Summary ==="]
    if summary.total == 0:
        lines.append("No evaluation cases found.")
        return "\n".join(lines)

    lines.append(f"Total Cases: {summary.total}")
    lines.append(f"Agreement Rate: {summary.agreement_rate:.2%}")
    lines.append(f"Accuracy vs Ground Truth: {summary.accuracy_vs_ground_truth:.2%}")
    lines.append(f"Average Confidence: {summary.avg_confidence:.2f}")

    if summary.disagreements:
        lines.append("")
        lines.append(
            f"Top Disagreements (showing up to {min(top_disagreements, len(summary.disagreements))}):"
        )
        for result in summary.disagreements[:top_disagreements]:
            lines.append(
                f"- Doc {result.document_id}: preferred={result.preferred_tier}, severity={result.severity}, reason={result.reason}"
            )
    else:
        lines.append("\nNo disagreements detected.")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render evaluation MetricsSummary")
    parser.add_argument("metrics_file", type=Path, help="Path to metrics_summary.json")
    parser.add_argument("--top", type=int, default=3, help="Number of disagreements to display")
    args = parser.parse_args(argv)

    summary = load_metrics_summary(args.metrics_file)
    report = format_report(summary, top_disagreements=args.top)
    print(report)


if __name__ == "__main__":
    main()
