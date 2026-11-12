# RadGuard

**Deterministic, sub-10ms hallucination guardrail between an LLM and a clinician in a radiology reporting workflow.**

Python 3.11+ · FastAPI · Pydantic v2 · pytest

RadGuard checks an LLM-drafted **Impression** against source **Findings** before the report reaches a radiologist. The validation path is 100% deterministic: pure Python, regex, and window-based rules. There are no external LLM APIs in the loop.

## Clinical risk

Generative models now draft structured impressions from dictated findings. The failures that matter in radiology are small, high-severity substitutions—not fluent nonsense:

- **Measurement & unit drift.** A `3 mm` nodule becomes a `3 cm` mass (10× error). Follow-up intervals, biopsy thresholds, and staging change.
- **Laterality swaps.** `Left kidney` becomes `right kidney`. The wrong organ is treated.
- **Negation drops.** `No acute fracture` becomes `acute fracture`. A ruled-out finding is asserted as present.

A second LLM used as a judge adds 500ms–2,000ms of latency, metered API cost, and residual non-determinism. That is the wrong control for a hard safety check sitting on the hot path to a clinician’s screen.

RadGuard is a post-generator microservice: same inputs, structured flags, sub-10ms CPU time, identical output for identical text.

```text
[Raw Findings / Dictation] ──┐
                             ├──► [ RadGuard Engine ] ──► { is_safe, flags[], latency_ms }
[LLM Impression]           ──┘      ├── Measurement normalizer (mm / cm / %)
                                    ├── Laterality cross-check (left / right / bilateral)
                                    └── Negation-scope detector
```

`is_safe` is true if and only if `flags` is empty. Flags are ranked CRITICAL before HIGH. Overlapping categories (unit error **and** a side swap) all surface.

## Why deterministic beats LLM-as-a-judge here

| | RadGuard | LLM-as-a-judge |
| --- | --- | --- |
| Latency | Sub-10ms, in-process | Hundreds of ms to seconds |
| Cost | CPU only | Token-metered every report |
| Consistency | Identical flags for identical text | Sampling / prompt drift |
| Failure mode | Misses phenomena outside the rule set | Confident, fluent misses |
| Deployment | Single container, no egress | Requires model endpoint + PHI review |

Numbers, units, laterality tokens, and explicit negation cues are closed classes. They do not need a language model. Keep the LLM on draft generation; keep the guardrail boring.

## Architecture

```
app/
├── main.py              # FastAPI: POST /validate, GET /health
├── schemas.py           # Request/response models, size limits
├── engine.py            # Normalize → validators → rank flags; never crash
├── normalize.py         # NFKC, unicode dashes/spaces
└── validators/
    ├── measurements.py  # Aliases, ranges, cm→mm
    ├── laterality.py    # Organs, lobes, unbound left-/right-sided
    └── negations.py     # Three-state polarity, sentence-bounded windows
```

- **Measurements.** `mm` / `cm` / `%`, spelled-out (`millimeter`, `centimetre`), compact (`10mm`), hyphenated (`10-mm`), decimals, and ranges (`8-10 mm`, `1.2 x 0.8 cm`). Lengths canonicalize to millimeters so `10 mm` vs `1 cm` is safe and `3 mm` vs `3 cm` is `CRITICAL`.
- **Laterality.** `left` / `right` / `bilateral`, `left-sided`, `on the left`, lobe abbreviations (RUL/RML/RLL/LUL/LLL), parenthetical `(L)`/`(R)`, and implied left for lingula. Synonyms such as kidney/renal are grouped. Bilateral in findings covers a unilateral impression.
- **Negations.** Entities: fracture, pneumothorax, effusion, mass, hemorrhage, stenosis. Cues: `no`, `without`, `absent`, `negative for`, `rule out`, `denies`, `unremarkable for`, `free of`. `cannot exclude` is **uncertain** (overstatement or false reassurance is flagged). Double negatives (`not without`) cancel to asserted. Cue scope stops at sentence boundaries.
- **Fail-closed.** Validator exceptions and truncated oversize input emit `PARSE_AMBIGUITY` (HIGH) instead of crashing. Empty/whitespace input is safe (no generated claims).

## Quickstart

```bash
git clone https://github.com/EarthlyAlien/RadGuard.git
cd RadGuard
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

```bash
pytest
python tests/benchmark.py
```

```bash
uvicorn app.main:app --reload
```

```bash
curl -s -X POST "http://127.0.0.1:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
        "findings_text": "There is a 4 mm non-obstructing calculus in the left kidney. No hydronephrosis.",
        "impression_text": "4 cm stone in the right kidney."
      }'
```

**Example response:**

```json
{
  "is_safe": false,
  "flags": [
    {
      "category": "MEASUREMENT_MISMATCH",
      "severity": "CRITICAL",
      "source_evidence": "4 mm",
      "generated_evidence": "4 cm",
      "explanation": "Measurement unit/value mismatch: 40mm generated vs 4mm in source."
    },
    {
      "category": "LATERALITY_CONFLICT",
      "severity": "CRITICAL",
      "source_evidence": "left kidney",
      "generated_evidence": "right kidney",
      "explanation": "Laterality conflict detected: 'right' kidney in the impression conflicts with 'left' kidney in the findings."
    }
  ],
  "latency_ms": 0.21
}
```

`GET /health` returns liveness plus readiness (`ready` is true only if a self-check `10 mm` ≡ `1 cm` evaluation succeeds):

```json
{"status": "ok", "ready": true, "version": "0.2.0", "validators": ["measurements", "laterality", "negations"]}
```

OpenAPI: `GET /openapi.json` or `/docs`. Requests are capped at 20,000 characters per field and 64 KiB bodies; malformed payloads return `{ "error": { "code", "message", "details" } }`.

### Docker

Non-root image (`uid 10001`):

```bash
docker build -t radguard .
docker run --rm -p 8000:8000 radguard
```

## Dataset

`data/synthetic_cases.json` holds **40** findings/impression pairs:

- 23 safe, including `10 mm` → `1 cm`, spelled-out units, compact/hyphenated forms, ranges, unicode NBSP, mixed case, `denies` / `unremarkable for` / `free of`, and double-negative `not without`.
- 17 injected hallucinations: unit errors (including spelled-out and ranges), laterality swaps (organs, lobes, unbound left-/right-sided, `on the left`), dropped negations, and `cannot exclude` polarity abuse.

Ground truth for the benchmark is `expected_safe`. Precision / recall / F1 are computed on the **unsafe** class.

## Benchmark results

Evaluated with `python tests/benchmark.py` (classification over 40 cases; latency over 100 repeats per case):

| Metric | Result |
| --- | --- |
| Total Cases | 40 |
| Precision | 100.0% |
| Recall | 100.0% |
| F1-Score | 1.00 |
| Mean Latency (ms) | 0.098 |
| P99 Latency (ms) | 0.201 |

No false positives or false negatives on this labeled set. That is **not** a claim of 100% recall in the wild; it is the score on the synthetic contract.

## Remaining known gaps

These are accepted limitations of a sub-10ms rule engine, not silent TODOs:

- **Paraphrase without a closed-class token.** `"dominant nodule enlarged"` with no digits, laterality word, or guarded entity will not flag.
- **Synonyms outside the lexicon.** Kidney/renal and lobe abbreviations are mapped; uncovered nicknames (`"the left kidney lower pole calyx"` vs an unseen synonym) can miss a bind.
- **Window, not a dependency parse.** Laterality attaches to organs in a character window; two organs near one `left`/`right` can share a side. Negation cues do not cross sentence boundaries, but nested clauses inside a sentence (`"it would be wrong to claim there is no fracture"`) can still fool a window.
- **Hedges beyond `cannot exclude` / `rule out`.** `"possible effusion"`, `"may represent a mass"`, `"likely fracture"` are not a third polarity yet.
- **Priors and implied math.** `"stable versus 8 mm on the last CT"` with an impression of `"12 mm"` only flags if both numbers+units are explicit in the two fields.

Failing closed on those would destroy precision. They belong in a slower second stage, not this hot path.

## License

MIT. See [LICENSE](LICENSE).
