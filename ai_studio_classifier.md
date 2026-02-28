# AI Studio Risk Classifier Blueprint

## 1. Agent + Deployment Plan
- **Objective**: map multi-page financial statement features to `risk_tier` (Low/Medium/High) with rationale + confidence.
- **Inputs**: serialized feature list from `utils.FeatureExtractor` (table summaries + key text snippets) plus metadata (doc id, industry, fiscal period when available).
- **Agent Instructions Skeleton**:
  ```
  You are a Tier-1 bank risk officer. Given structured excerpts from audited financial statements, assign a single risk tier.
  Respond as JSON: {"risk_tier": <Low|Medium|High>, "rationale": <string>, "confidence": <0..1>}.
  Consider liquidity, leverage, profitability, exposure flags, restatements, regulatory notes.
  ```
- **Few-Shot Examples**: curate 5–10 labeled exemplars per industry (banks, manufacturing, energy, tech growth). Each example includes feature snippets + ground truth tier.
- **AI Studio Steps**:
  1. Open **AI Studio → Agent Builder** and create a new Agent named `Risk Classifier`.
  2. Paste the instructions above, add the few-shot examples, and optionally attach tools if you want the Agent to fetch supporting context.
  3. In the Agent’s *Test* panel, feed feature payloads exported from `demo_feature_store/` until responses match expectation.
  4. Deploy the Agent as `risk-classifier-v1` (backed by `mistral-large-latest` or a fine-tuned model) and copy the deployment/endpoint ID (e.g., `astudio:endpoint:risk-classifier-v1`).
  5. (Optional) Use AI Studio’s **Classifiers/Fine-tuning** section to train on labeled pairs, then update the Agent to point at the fine-tuned model.

## 2. Batch Workflow
- **Synchronous usage**: `DocumentClassifier` reads `CLASSIFIER_MODEL_ID` env var and calls `client.chat.complete(model=...)` where `model` equals the AI Studio deployment ID.
- **Batch mode**: create an AI Studio job referencing the same prompt/fine-tuned artifact, uploading CSV of feature payloads and writing outputs to cloud storage. Use our `evaluate_run.py` to compare job outputs vs. ground truth.

## 3. Monitoring & Retraining
- Stream AI Studio usage metrics (latency, cost) to Grafana via Studio’s built-in monitoring.
- Schedule `evaluate_run.py` nightly on new docs; push `metrics_summary.json` to dashboards.
- Maintain a feedback dataset from human overrides; periodically update few-shots or re-run fine-tunes.
