# RadGuard

**Low latency deterministic guardrail for catching measurement, laterality, and negation hallucinations in clinical LLM outputs.**

Python | FastAPI

## The Problem
Generative AI has transformed clinical documentation by drafting structured report summaries ("Impressions") directly from raw physician dictation ("Findings"). However, LLMs still introduce subtle, high-risk hallucinations:
* **Measurement & Unit Drift:** Turning a `3 mm` nodule in the findings into a `3 cm` mass in the summary.
* **Laterality Swaps:** Flipping `left` and `right` anatomical locations.
* **Negation Drops:** Summarizing *"no acute fracture"* as *"acute fracture."*

Using a second LLM ("LLM-as-a-Judge") to verify outputs adds 500ms–2,000ms of latency, increases API costs, and remains probabilistic. **RadGuard** solves this deterministically in under <50 milliseconds with zero external API calls.

## Architecture
RadGuard sits as a lightweight post-processing microservice between the LLM generator and the clinician's screen:

```text
[Raw Dictation] ──┐
                  ├──► [ RadGuard Deterministic Engine ] ──► [ Safe / Flagged JSON ]
[LLM Summary]   ──┘      ├── Measurement Normalizer (mm/cm/%)
                         ├── Laterality Cross-Checker (left/right/bilateral)
                         └── Negation Scope Detector
```

## Benchmark Results
Evaluated across 20 synthetic clinical dictation/summary pairs (13 valid edge cases including unit conversions like `10 mm` → `1 cm`, and 7 injected clinical hallucinations):

| Metric | Result |
| :--- | :--- |
| **Total Test Cases** | 20 |
| **Precision** | 100.0% |
| **Recall** | 100.0% |
| **F1-Score** | 1.00 |
| **Mean Latency** | ~1.4 ms |
| **P99 Latency** | ~3.2 ms |

*(Note: Run `python tests/benchmark.py` to reproduce these metrics locally.)*

## Quickstart

### 1. Clone & Install
```bash
git clone https://github.com
cd radguard
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Tests & Benchmark
```bash
pytest
python tests/benchmark.py
```

### 3. Start the API
```bash
uvicorn app.main:app --reload
```

### 4. Example Request
```bash
curl -X POST "http://127.0.0" \
     -H "Content-Type: application/json" \
     -d '{
           "findings_text": "There is a 4 mm non-obstructing calculus in the left kidney. No hydronephrosis.",
           "impression_text": "4 cm stone in the right kidney."
         }'
```

**Example Response:**
```json
{
  "is_safe": false,
  "latency_ms": 1.18,
  "flags": [
    {
      "category": "MEASUREMENT_MISMATCH",
      "severity": "CRITICAL",
      "source_evidence": "4 mm",
      "generated_evidence": "4 cm",
      "explanation": "Measurement unit/value mismatch: 40.0mm generated vs 4.0mm in source."
    },
    {
      "category": "LATERALITY_CONFLICT",
      "severity": "CRITICAL",
      "source_evidence": "left kidney",
      "generated_evidence": "right kidney",
      "explanation": "Laterality conflict detected: 'right' introduced without source support."
    }
  ]
}
```

## Design Trade-offs & Future Work
* **Precision vs. Recall:** Rule-based extractors excel at high-precision checks (numbers, units, explicit lateralities) without latency overhead, though complex multi-clause negations benefit from dependency parsing (`spaCy`).
* **Next Steps:** Add support for anatomical synonym normalization (e.g., mapping *renal* ↔ *kidney* and *hepatic* ↔ *liver*) and Prometheus latency metrics.
