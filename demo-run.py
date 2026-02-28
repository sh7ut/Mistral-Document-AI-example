"""CLI helper to run the document pipeline on a single PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from pipeline import DocumentPipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DocumentPipeline on a PDF.")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="sample-data/NVIDIA-2025-Annual-Report.pdf",
        help="Path to the PDF to process.",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="Do not delete the uploaded file from Mistral Cloud after processing.",
    )
    parser.add_argument(
        "--embed-features",
        action="store_true",
        help="Attach embeddings to extracted features.",
    )
    parser.add_argument(
        "--expiry-hours",
        type=int,
        default=24,
        help="Signed URL expiry duration (hours).",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run the classifier to produce risk tier outputs.",
    )
    parser.add_argument(
        "--ground-truth-tier",
        type=str,
        default=None,
        help="Optional ground-truth tier to trigger LLM judge evaluation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the pipeline result as JSON instead of a Python dict.",
    )
    return parser.parse_args()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args()
    cfg = PipelineConfig(
        input_path=Path(args.pdf_path),
        delete_remote_file=not args.keep_remote,
    )

    client = DocumentPipeline.init_client()
    pipeline = DocumentPipeline(client, cfg)
    result = pipeline.run(
        embed_features=args.embed_features,
        expiry_hours=args.expiry_hours,
        classify=args.classify,
        ground_truth_tier=args.ground_truth_tier,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
