# Check-In 3: Advanced Extension, Comparison, and Next Steps

## 1) Advanced Extension Implementation

### What advanced method was added
For Check-In 3, I added an advanced multimodal document-understanding pipeline based on `LayoutLMv3` (`scripts/layoutlmv3_model.py`) for invoice field extraction.

Compared to OCR-only baselines, this extension adds:

- token classification with text + layout features (word positions and page structure),
- weak-label dataset building from OCR tokens + invoice ground truth,
- fine-tuning/inference flow for `LayoutLMv3`,
- post-processing for entity-to-field mapping,
- fallback/resolution logic specifically for hard `seller_name` and `client_name` cases.

### What was implemented in pipeline/training terms

- **Data strategy**: OCR words and boxes converted to token labels (BIO-style weak supervision) for model training.
- **Training/inference setup**: `LayoutLMv3InvoiceTokenClassifier` loads processor/model, trains/reloads model artifacts, runs inference on held-out data.
- **Advanced post-processing**:
  - side-aware candidate ranking (left/right page priors),
  - inline seller/client pair resolver,
  - anchor-based fallback (`seller`, `client`, etc.),
  - bleed-truncation + plausibility checks to reduce name/address contamination.

---

## 2) Comparison to Earlier Baselines

To compare fairly, we use the same core extraction-evaluation framing (field-level exact match against held-out ground truth).

### Summary comparison table

| Model | Role | Avg. field test accuracy | Strengths | Main weaknesses |
|---|---|---:|---|---|
| `basic_model` | Check-In 2 baseline | Lower than top models (rule-based baseline) | Simple, interpretable, fast iteration | Sensitive to ROI layout drift and numeric formatting/OCR quirks |
| `pt_model` | Strong OCR baseline | **98.84%** | Best aggregate field accuracy; robust practical performance | Still heuristic-dependent; may degrade on unseen templates/noisy OCR |
| `layoutlmv3_model` | Advanced extension | **95.56%** | Better handling of spatial/entity ambiguity; extensible learned approach | More complex; still needs fallback tuning for edge-case names |

### Interpretation

- `pt_model` currently wins on aggregate accuracy and is the best primary model for current deliverables.
- `layoutlmv3_model` is still valuable as the advanced extension because it introduces a scalable learned pipeline and improved behavior on hard spatial/name patterns that rule systems struggle with.

---

## 3) Ablation / Controlled Comparison

I ran focused controlled changes on `layoutlmv3_model` while monitoring the same name metrics and error tables.

### Ablation target

Improve seller/client name extraction failures due to:

- cross-party bleeding,
- partial-name truncation,
- fallback overreach into address lines.

### What was changed

Key controlled modifications (one theme at a time):

1. **Enable/tune anchor fallback + side-aware scoring**  
2. **Add inline seller/client pair resolver** for `Seller: Client: Name1 Name2` patterns  
3. **Add bleed truncation + plausibility gating** (drop address-like suffixes/tails)  
4. **Expand name-tail reconstruction** to recover full multi-token organization names  

### Effect (observed trend)

| LayoutLMv3 tuning stage | Seller acc | Client acc | Key observation |
|---|---:|---:|---|
| Early split/debug stage | 58% | 45% | High missing/mismatch; fallback unstable |
| Mid tuning | 70% | 88% | Large client gain after resolver/fallback refinement |
| Latest reported stage | **84%** | **92%** | Stronger and more stable name extraction; remaining errors are harder edge cases |

This controlled progression shows that the post-processing components were causal drivers of the gains.

---

## 4) Failure Analysis + Interpretation

Even after improvements, the advanced method still fails on specific hard cases.

### Where it still fails

From mismatch debugging tables, recurrent patterns include:

- **cross-party confusion**: seller predicted as client-like name (right-side bleed),
- **address contamination**: organization name mixed with location/address token (e.g., "Village", "Trail", "Tunnel"),
- **partial spans**: predicts only suffix/tail (e.g., missing first company token),
- **nearby plausible distractors**: model/fallback picks a valid-looking person/org string but wrong party.

### Concrete qualitative examples observed during debugging

- Seller GT: `Mendoza and Sons` -> Pred: `Sons` (partial tail)
- Seller GT: `Ward-Day` -> Pred: `Ridge Young` (cross-party/distractor)
- Seller GT: `Lee, Young and Krause` -> Pred: `Krause Oconnor` (party overlap)
- Client GT: `Oconnor Inc` -> Pred: `Inc Johnston` (token shift/bleed)
- Client GT: `Harrison-Davis` -> Pred: `Davis Extensions` (address/business suffix contamination)

### Why these failures occur

- Dense invoice header regions place seller/client/address tokens very close in reading order.
- OCR tokenization noise alters boundaries before model post-processing runs.
- Fallback heuristics still occasionally over-prioritize local high-confidence spans that are semantically wrong for the target field.

### How this differs from earlier baselines

- Compared to simple zonal extraction, failures are now more about **disambiguation** than complete misses.
- Compared to OCR-only heuristics, the model is better at many ambiguous layouts, but still needs stronger confidence calibration and boundary control on edge cases.

---

## 5) Plan for Final Deliverable

Final project goal: ship a working **Donut** pipeline and a working **SmolVLM-Instruct** pipeline, then compare them to current baselines.

### Highest-priority next steps

1. **Deliver Donut end-to-end**
   - train/infer/evaluate on same held-out split and same metric protocol,
   - add failure analysis section parallel to current pipelines.
2. **Deliver SmolVLM-Instruct end-to-end**
   - define prompt/output schema for structured invoice fields,
   - run controlled prompt variants and evaluate exact-match metrics.
3. **Unify benchmark harness**
   - one evaluation table for all models (`basic_model`, `pt_model`, `layoutlmv3_model`, Donut, SmolVLM),
   - consistent normalization and field-level reporting.
4. **Model selection package**
   - finalize recommendation by aggregate accuracy + key-field robustness + runtime/complexity tradeoff.

### Top risks

- **Data/format mismatch risk**: generative models may output schema-inconsistent text without strict parsing.
- **Compute/time risk**: Donut/VLM training and prompt iteration can be slower than OCR pipelines.
- **Fairness risk in comparisons**: inconsistent splits or normalization would invalidate model ranking.
- **Generalization risk**: template-specific gains may not transfer to noisier or shifted invoice layouts.

