# Execution Plan for Document AI Pipeline

## Phase 1: Setup and Environment Preparation
1. **Environment Setup**
   - Verify Python environment (3.8+)
   - Install required packages: `mistralai`, `python-dotenv`
   - Load Mistral API key from `~/.zshrc`

2. **API Key Validation**
   - Ensure `MISTRAL_API_KEY` is accessible via `os.getenv`
   - Test client initialization

## Phase 2: Target Architecture & Pipeline Design
1. **Architecture Blueprint**
   - Document ingestion-to-classification flow using Mistral Document AI + AI Studio
   - Define data stores, orchestration layer, monitoring hooks, and scaling strategy
   - Highlight security considerations (PII handling, key management)

2. **Judging Strategy Draft**
   - Specify quantitative metrics (precision/recall, calibration) and human-in-loop checkpoints
   - Describe batch evaluation workflow using AI Studio evaluation jobs

## Phase 3: Core Pipeline Implementation
1. **File Upload Module**
   - Implement PDF upload to Mistral Cloud
   - Handle file retrieval and signed URL generation
   - Add optional expiry for signed URLs

2. **OCR Processing Module**
   - Configure OCR endpoint with `mistral-ocr-latest`
   - Support table extraction (`table_format="html"`)
   - Enable image base64 inclusion

3. **Output Parsing Module**
   - Parse JSON response into structured data
   - Extract text, tables, images, and metadata
   - Handle pagination and multi-page documents

4. **Entity Extraction & Feature Layer**
   - Normalize entities (accounts, counterparties, risk factors)
   - Convert tables/images into model-ready embeddings or features

5. **Risk Classification Module**
   - Implement prompt- or fine-tuned classifier via AI Studio endpoints
   - Support synchronous (API) and asynchronous batch modes

## Phase 4: Quality Assurance
1. **Validation Layer**
   - Check for empty/malformed responses
   - Verify table structure integrity
   - Validate image base64 encoding

2. **Cleanup Module**
   - Implement file deletion from Mistral Cloud
   - Handle API errors gracefully

## Phase 5: Testing
1. **Unit Tests**
   - Mock API responses for offline testing
   - Validate error handling

2. **Integration Test**
   - Process sample financial statement PDF
   - Verify end-to-end flow

## Phase 6: Quality Judging & Monitoring
1. **Metric Computation**
   - Automate scoring vs. ground truth labels (precision/recall/F1, calibration curves)
   - Generate drift and anomaly reports

2. **Human Review Loop**
   - Route low-confidence cases to reviewers with structured UI payloads
   - Capture reviewer feedback to retrain classifier prompts/models

## Deliverables
- `architecture.md`: Target architecture + judging strategy
- `pipeline.py`: Main execution script
- `utils.py`: Helper functions (validation, parsing, feature building)
- `classification.py`: Risk classification + scoring logic
- `requirements.txt`: Dependencies
- Test suite with sample PDF + labelled baseline

## Constraints
- Use Mistral native stack only
- No hardcoded API keys
- Support multi-page PDFs with tables/images
