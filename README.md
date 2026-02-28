# Mistral Document AI Demo

End-to-end sample showing how to ingest PDFs, run Mistral Document AI OCR, build features, classify risk, and judge outputs via LLM-as-a-judge.

## Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`
- `MISTRAL_API_KEY` exported in your shell (keys live in `~/.zshrc` per REQUIREMENT.md)

## Process a PDF
```bash
export MISTRAL_API_KEY=...  # ensure available in this terminal
python demo-run.py sample-data/Amazon-2025-Proxy-Statement.pdf --classify --ground-truth-tier High --json
```
Flags:
- `--embed-features` attaches embeddings via `mistral-embed-latest`.
- `--keep-remote` skips deleting the uploaded file from Mistral Cloud (useful for reuse).
- `--expiry-hours` controls signed URL lifetime.
- `--classify` and `--ground-truth-tier` enable classification + judging.

Outputs:
- Features saved under `demo_feature_store/<file_id>.json`.
- `classification` and `evaluation` keys appear in the CLI output when enabled.

## Evaluate Batches
1. Process each validation PDF with `demo-run.py --classify --ground-truth-tier …` so features and model predictions exist in `demo_feature_store/`.
2. Build a manifest JSON listing the stored document IDs and ground-truth tiers:
   ```json
   [
     {"document_id": "c672db8b-0180-4dcc-b09a-2a4cfd825bb1", "ground_truth_tier": "High"},
     {"document_id": "b8f4d9d9-c1ab-4abd-9849-10400f68b6f3", "ground_truth_tier": "Low"}
   ]
   ```
3. Run the batch evaluator to replay classification, judge each case, and emit metrics:
   ```bash
   python evaluate_run.py manifest.json --print-report
   ```
   - `batch_results.json`: per-document classification + evaluation info.
   - `metrics_summary.json`: aggregate agreement/accuracy payloads for dashboards.
4. Render or share the report anytime:
   ```bash
   python metrics_report.py metrics_summary.json --top 5
   ```

## Tests
```bash
python -m pytest test-scripts
```

## Next Steps
- Implement enriched entity extractors (accounts, ratios, counterparties).
- Ship batch evaluator script that replays stored features and emits `metrics_summary.json` automatically.
- Package orchestration/deployment (Prefect/Temporal flow) per architecture plan.
