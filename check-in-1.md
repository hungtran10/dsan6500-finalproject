
## Problem Framing & Scope

Task definition: to extract structured invoice fields from each invoice image into a target JSON with the fields  already in the CSV files (e.g., client_name, seller_name, invoice_number, invoice_date, items, quantity, etc.).

Modality: Vision

I want to compare at least 3 pipelines:

1. Rule-based baseline: OCR (Tesseract/PaddleOCR) → regex/heuristics.

2. Hybrid supervised: OCR → LayoutLM (or similar) for field extraction.

3. End-to-end supervised: Donut (image → JSON).

Success criteria:

- Per-field metrics: precision / recall / F1 on extracted field values (treat extraction as a named-entity task).

- Normalized exact match (post-processing rules applied): whitespace/currency/date normalization before comparison.

- Numeric fields (values): numeric parsing + absolute/relative error (e.g., parsed value within ±$0.01 or relative tolerance).

- JSON/record-level metrics: exact JSON match (strict) and per-field average F1 (lenient).

- Table/line-item metrics: cell-level precision/recall and row-level correctness (exact match for all key columns).

Practical acceptance thresholds:

- Core fields (invoice number, invoice date, gross worth): target ≥ 90% normalized F1.

- Vendor name: ≥ 85% F1 (due to naming variations).

- Line items: ≥ 60–80% (harder; ok as stretch goal).

- Show improvements relative to the baseline (e.g., +X% F1 over rule-based).

Feasible scope for semester

- Core scope (recommended, strong finish):

    - Build baseline (Tesseract + heuristics) and evaluate.

    - Implement and fine-tune LayoutLMv3 (or LayoutLMv2) using OCR tokens + boxes for field extraction.

    - Implement/finetune Donut for end-to-end JSON extraction and compare.

    - Produce evaluation, error analysis, and a short report with example successes/failures.

- Stretch scope (only if time allows): add table extraction (CascadeTabNet / Table Transformer) and evaluate line-item accuracy; or add vendor-level generalization test (leave-one-vendor-out).


Risks & mitigation

Risk: insufficient labeled examples for fine-tuning. Mitigation: data augmentation (document transforms), synthetic invoice generation, or prompt-engineering for Donut to reduce labeling needs.

Risk: OCR token ↔ annotation alignment errors. Mitigation: build alignment debugging visuals; use robust tokenization and normalize annotations to OCR granularity.

## Data Access & Documentation

For this project I will be utilizing the collection of invoice images from this [Kaggle dataset](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr?resource=download).

There are a variety of invoice templates from multiple industries with different currencies, tax formats, and layouts that will make for an interesting project. 

Dataset License: [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

Dataset size: 1GB (zipped), 2.41GB (unzipped) 

Ways to Download the Dataset (instructions on the link above): kagglehub, Kaggle CLI, cURL, microissant, MCP, or manually downloading zip (easiest)

## EDA

Below are some high-level insights from my Exploratory Data Analysis. For more in-depth analysis, please visit eda.ipynb.

- 1413 unique invoices
- due_date column is missing values for all rows, meaning this field was not displayed in any of the invoices
- There are almost as many vendors as invoices in the dataset: 1361 unique vendors.
- The word token distrbution is relatively normal
- Image size, and currency is very consistent across all image files 
- The distribution of invoice total and tax are right skewed. 


## Evaluation Plan
Metrics:
- Field-level exact match accuracy: for each field prediction == ground_truth, so accuracy = $\frac{correct predictions}{total samples}$
  - Values will be normalized to maintain consistency
- Numeric error metrics: MAE, MSE for tax and total_amount
- OCR Alignment Sanity Metric: Since the dataset already includes OCR text, we can check if the annotated value appear in OCR text

Train / Validation / Test Strategy: split data using seller_name
70% vendors → training
15% vendors → validation
15% vendors → test

Alternative:  K-Fold Cross Validation

## Initial Baseline Representation

### Baseline 1 — Rule-Based Extraction (Classical) Pipeline:

OCR text (already provided) → regex / keyword rules → field extraction

Example regex patterns for fields 
- Invoice Number: Invoice\s*(No|Number|#)\s*[:\-]?\s*(\w+)
- Date: \d{2}/\d{2}/\d{4}
- Total Amount: Search near keyword "Total" and extract \$?\d+\.\d{2}

### Baseline 2 — Hybrid Supervised Model Pipeline: 

invoice image → OCR tokens + bounding boxes → LayoutLMv3 → token classification → field extraction

Each token is represented as (text embedding + position embedding + image features)

Example token: Total → next token likely TOTAL_AMOUNT

### Baseline 3 - End-to-End Model Pipeline:

invoice image → Vision Transformer → Transformer decoder → generated JSON

Example (Donut): Image → {invoice_number:..., total:...}

### Next steps:

- Implement rule-based extraction baseline using OCR text.

- Compute baseline metrics.

- Implement vendor-aware train/test split.

- Begin implementing LayoutLM input pipeline.