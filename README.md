# dsan6500-finalproject

## Data Access & Documentation

For this project I will be utilizing the collection of invoice images from this [Kaggle dataset](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr?resource=download).

There are a variety of invoice templates from multiple industries with different currencies, tax formats, and layouts that will make for an interesting project. 

Dataset License: [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

Dataset size: 1GB (zipped), 2.41GB (unzipped) 

Ways to Download the Dataset (instructions on the link above): kagglehub, Kaggle CLI, cURL, microissant, MCP, or manually downloading zip (easiest)

## Models Developed

1. `InvoiceZonalOCRPipeline` (`scripts/basic_model.py`) - zonal OCR baseline
2. `PytesseractInvoiceTextDetector` (`scripts/pt_model.py`) - full-page OCR + heuristics
3. `LayoutLMv3InvoiceTokenClassifier` (`scripts/layoutlmv3_model.py`) - weakly-supervised token classifier + fallback resolvers
4. Donut (`scripts/donut_model.py`, `Donut_model.ipynb`) - vision encoder–decoder fine-tuned for structured invoice text
5. SmolVLM (`scripts/smolvlm_model.py`, `SmolVLM_model.ipynb`) - compact vision-language model for invoice field extraction

### Notebook Map (Full Pipelines)

- `basic_models.ipynb`: contains both `InvoiceZonalOCRPipeline` and `PytesseractInvoiceTextDetector` end-to-end workflows.
- `layoutlmv3_model.ipynb`: full weak-supervision training + inference/evaluation pipeline for `LayoutLMv3InvoiceTokenClassifier`.
- `SmolVLM_model.ipynb`: full training + inference/evaluation pipeline for `SmolVLMInvoiceModel`.
- `Donut_model.ipynb`: full fine-tuning + inference/evaluation pipeline for `DonutInvoiceTextDetector`.

## Evaluation Metrics

The tables below summarize best **per-field** performance after merging model predictions with ground truth. **Accuracy** is correct predictions divided by the number of evaluated rows for that run. **Recall**, **precision**, and **F1** use the usual definitions. Sample counts differ by pipeline and notebook configuration; re-run the relevant notebook or script after training or data changes to refresh these numbers.

### Basic model (`InvoiceZonalOCRPipeline`)

| field | accuracy | recall | precision | f1 |
| --- | ---: | ---: | ---: | ---: |
| invoice_number | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| invoice_date | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| seller_name | 0.996390 | 0.996390 | 0.996390 | 0.996390 |
| client_name | 0.996390 | 0.996390 | 0.996390 | 0.996390 |
| net_worth | 0.126354 | 0.126354 | 0.246479 | 0.167064 |
| tax | 0.144404 | 0.144404 | 0.325203 | 0.200000 |
| total_amount | 0.093863 | 0.093863 | 0.168831 | 0.120650 |

### Pytesseract model (`PytesseractInvoiceTextDetector`)

| field | accuracy | recall | precision | f1 |
| --- | ---: | ---: | ---: | ---: |
| invoice_number | 0.992933 | 0.992933 | 0.992933 | 0.992933 |
| invoice_date | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| seller_name | 0.989399 | 0.989399 | 0.992908 | 0.991150 |
| client_name | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| net_worth | 0.968198 | 0.968198 | 0.975089 | 0.971631 |
| total_amount | 0.975265 | 0.975265 | 0.975265 | 0.975265 |
| tax | 0.992933 | 0.992933 | 0.992933 | 0.992933 |

### LayoutLMv3 (`LayoutLMv3InvoiceTokenClassifier`)

| field | accuracy | recall | precision | f1 |
| --- | ---: | ---: | ---: | ---: |
| invoice_number | 0.992933 | 0.992933 | 0.992933 | 0.992933 |
| invoice_date | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| seller_name | 0.844523 | 0.844523 | 0.898496 | 0.870674 |
| client_name | 0.918728 | 0.918728 | 0.945455 | 0.931900 |
| net_worth | 0.971731 | 0.971731 | 0.975177 | 0.973451 |
| tax | 0.982332 | 0.982332 | 0.985816 | 0.984071 |
| total_amount | 0.978799 | 0.978799 | 0.978799 | 0.978799 |

### SmolVLM (`SmolVLMInvoiceModel`)

| field | accuracy | recall | precision | f1 |
| --- | ---: | ---: | ---: | ---: |
| invoice_number | 0.915493 | 0.915493 | 0.935252 | 0.925267 |
| invoice_date | 0.943662 | 0.943662 | 0.964029 | 0.953737 |
| seller_name | 0.971831 | 0.971831 | 0.992806 | 0.982206 |
| client_name | 0.922535 | 0.922535 | 0.942446 | 0.932384 |
| net_worth | 0.823944 | 0.823944 | 0.841727 | 0.832740 |
| tax | 0.859155 | 0.859155 | 0.945736 | 0.900369 |
| total_amount | 0.845070 | 0.845070 | 0.863309 | 0.854093 |

### Donut (`DonutInvoiceTextDetector`)

| field | accuracy | recall | precision | f1 |
| --- | ---: | ---: | ---: | ---: |
| invoice_number | 0.750 | 0.750 | 0.750000 | 0.750000 |
| invoice_date | 0.975 | 0.975 | 1.000000 | 0.987342 |
| seller_name | 0.700 | 0.700 | 0.756757 | 0.727273 |
| client_name | 0.525 | 0.525 | 0.567568 | 0.545455 |
| net_worth | 0.825 | 0.825 | 0.891892 | 0.857143 |
| tax | 0.775 | 0.775 | 0.861111 | 0.815789 |
| total_amount | 0.775 | 0.775 | 0.837838 | 0.805195 |

## Getting Started

To get this pipeline running, follow these steps to set up your environment and process your first batch of invoices.

### 1. Prerequisites

Install the Python dependencies:

```bash
pip install opencv-python pytesseract transformers datasets torch scikit-learn matplotlib seaborn
```

Also make sure your Tesseract binary is installed and available on PATH.

### 2. Basic Usage (`InvoiceZonalOCRPipeline`)

The following snippet shows how to initialize each pipeline and process a single invoice.

```python
from scripts.basic_model import InvoiceZonalOCRPipeline

# Optional: tune these coordinates for your invoice template
zones = {
    "default": {
        "invoice_number": (0.21, 0.025, 0.15, 0.04),
        "date": (0.48, 0.05, 0.30, 0.08),
        "seller_name": (0.05, 0.205, 0.40, 0.025),
        "client_name": (0.50, 0.205, 0.40, 0.025),
        "net_worth": (0.50, 0.60, 0.15, 0.025),
        "tax": (0.66, 0.60, 0.12, 0.025),
        "total_amount": (0.79, 0.60, 0.15, 0.025),
    }
}

pipeline = InvoiceZonalOCRPipeline(
    template_zones=zones,
    output_dir="/path/to/data_dir/...output_images/basic_ocr",
)

sample_image = "/path/to/data_dir/...batch_1/batch1_1/batch1-0049.jpg"
pipeline.visualize_zones(sample_image, template_name="default")
result = pipeline.process_invoice(sample_image, template_name="default")

if result["success"]:
    print("Extracted Fields:")
    for field, value in result["fields"].items():
        print(f"{field}: {value}")
```

### 3. Processing a Batch (`InvoiceZonalOCRPipeline`)

If you have a folder full of invoices of the same type, you can process them all at once into a **Pandas DataFrame**:

```python
import pandas as pd

subfolders = ["batch1_1", "batch1_2", "batch1_3"]
base_path = "/path/to/data_dir/...batch_1/"

all_preds = [
    pipeline.process_folder(f"{base_path}{folder}", template_name="default", sample_frac=0.2)
    for folder in subfolders
]
df_results = pd.concat(all_preds, ignore_index=True)

# Save the results to a CSV
df_results.to_csv("extracted_data.csv", index=False)
print(df_results.head())
```

-----

### 4. Evaluation (`InvoiceZonalOCRPipeline`)

```python
import pandas as pd

ground_truth_df = pd.read_csv("/path/to/data_dir/...cleaned_invoices.csv")
metrics_df = pipeline.evaluate_against_ground_truth(df_results, ground_truth_df)
pipeline.visualize_evaluation_metrics(metrics_df)
```

### 5. CV + OCR Pipeline (`PytesseractInvoiceTextDetector`)

This pipeline uses preprocessed images from `scripts/preprocess.py` and then runs text extraction/evaluation from `scripts/pt_model.py`.

```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.preprocess import InvoiceImagePreprocessor
from scripts.pt_model import PytesseractInvoiceTextDetector
from scripts.visualize_util import create_analysis_dashboard, visualize_sample_results

output_image_path = "/path/to/data_dir/...processed_images"
csv_file_paths = [
    "/path/to/data_dir/...batch_1/batch1_1.csv",
    "/path/to/data_dir/...batch_1/batch1_2.csv",
    "/path/to/data_dir/...batch_1/batch1_3.csv",
]
image_folders = [
    "/path/to/data_dir/...batch_1/batch1_1",
    "/path/to/data_dir/...batch_1/batch1_2",
    "/path/to/data_dir/...batch_1/batch1_3",
]

preprocessor = InvoiceImagePreprocessor(output_dir=output_image_path)
all_results = []
for csv_file, image_folder in zip(csv_file_paths, image_folders):
    results_df = preprocessor.process_images(csv_file, image_folder)
    all_results.append(results_df)

combined_results = pd.concat(all_results, ignore_index=True)

ground_truth_df = pd.read_csv("/path/to/data_dir/...cleaned_invoices.csv")
gt_merged_df = pd.merge(
    ground_truth_df, combined_results, left_on="File Name", right_on="original_file"
).drop(columns=["File Name", "due_date"], errors="ignore")

train_df, test_df = train_test_split(gt_merged_df, test_size=0.2, random_state=42)

ocr_text_detector = PytesseractInvoiceTextDetector(
    output_dir="/path/to/data_dir/...output_images",
    debug_totals=True,
)
_ = ocr_text_detector.process_dataset(combined_results, sample_frac=None)
metrics_df, overall = ocr_text_detector.evaluate_against_ground_truth(test_df)
print(metrics_df)
print("Overall:", overall)

_ = create_analysis_dashboard(
    ocr_text_detector.full_results,
    metrics_df=metrics_df,
    fields=["invoice_number", "invoice_date", "seller_name", "client_name", "tax", "net_worth", "total_amount"],
    panel_model="ocr",
)
visualize_sample_results(
    ocr_text_detector.full_results,
    visualize_text_fn=ocr_text_detector.visualize_text_extraction,
    n_samples=3,
)
```

### 6. LayoutLMv3 Pipeline (`LayoutLMv3InvoiceTokenClassifier`)

This pipeline trains on weak labels generated from OCR tokens + ground truth, then evaluates on a held-out split.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.layoutlmv3_model import (
    LayoutLMv3InvoiceDatasetBuilder,
    LayoutLMv3InvoiceTokenClassifier,
)
from scripts.visualize_util import create_analysis_dashboard, visualize_sample_results

ground_truth_df = pd.read_csv("/path/to/data_dir/...cleaned_invoices.csv")
combined_results = pd.read_csv("/path/to/data_dir/...combined_results.csv")

gt_merged_df = pd.merge(
    ground_truth_df, combined_results, left_on="File Name", right_on="original_file"
).drop(columns=["File Name", "due_date"], errors="ignore")

train_df, test_df = train_test_split(gt_merged_df, test_size=0.2, random_state=42)

builder = LayoutLMv3InvoiceDatasetBuilder(output_dir="/path/to/data_dir/...layoutlmv3_data")
examples = builder.build_examples(
    train_df,
    image_col="processed_path",
    key_col="processed_file",
    fields=["invoice_number", "invoice_date", "seller_name", "client_name", "tax", "net_worth", "total_amount"],
    max_examples=None,
)
_ = builder.save_jsonl(examples)

train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)

layoutlm_output_dir = "/path/to/data_dir/...layoutlmv3_finetuned_model"
layoutlm_detector = LayoutLMv3InvoiceTokenClassifier()
layoutlm_detector.train(
    train_examples=train_examples,
    eval_examples=val_examples,
    output_dir=layoutlm_output_dir,
    fields=["invoice_number", "invoice_date", "seller_name", "client_name", "tax", "net_worth", "total_amount"],
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
)

# Inference + evaluation on held-out test split
layoutlm_detector.reload_model(layoutlm_output_dir)
layoutlm_detector.enable_seller_anchor_fallback = True
layoutlm_pred_df = layoutlm_detector.run_inference(
    test_df,
    image_col="processed_path",
    key_col="processed_file",
    sample_frac=None,
    debug_mode=True,
)
layoutlm_metrics_df, layoutlm_overall = layoutlm_detector.evaluate_against_ground_truth(
    ground_truth_df=test_df,
    pred_df=layoutlm_pred_df,
    fields=["invoice_number", "invoice_date", "seller_name", "client_name", "tax", "net_worth", "total_amount"],
    merge_key="processed_file",
)
print(layoutlm_metrics_df)
print("Overall:", layoutlm_overall)

_ = create_analysis_dashboard(
    layoutlm_detector.full_results,
    metrics_df=layoutlm_metrics_df,
    fields=["invoice_number", "invoice_date", "seller_name", "client_name", "tax", "net_worth", "total_amount"],
    title="LayoutLMv3 (weak supervision) Dashboard",
    panel_model="layoutlm",
)
visualize_sample_results(
    layoutlm_detector.full_results,
    visualize_text_fn=layoutlm_detector.visualize_text_extraction,
    n_samples=2,
    title="LayoutLM sample token-label overlays",
)
```

### 7. SmolVLM Pipeline (`SmolVLMInvoiceModel`)

This pipeline builds multimodal training samples, fine-tunes SmolVLM, and evaluates on a held-out test split.

```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.smolvlm_model import SmolVLMInvoiceModel, DEFAULT_FIELDS

# Paths (edit for your machine/data layout)
PROJECT_ROOT = Path("/path/to/data_dir").resolve()
OUTPUT_DIR = PROJECT_ROOT / "output_images" / "smolvlm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GT_CSV_PATH = PROJECT_ROOT / "cleaned_invoices.csv"
PROCESSED_IMAGES_CSV_PATH = PROJECT_ROOT / "combined_results.csv"

# Load prepared data artifacts
ground_truth_df = pd.read_csv(GT_CSV_PATH)
processed_images_df = pd.read_csv(PROCESSED_IMAGES_CSV_PATH)

# Initialize SmolVLM wrapper
smol = SmolVLMInvoiceModel(
    model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
    output_dir=OUTPUT_DIR,
)

samples = smol.build_samples(
    ground_truth_df=ground_truth_df,
    processed_images_df=processed_images_df,
    fields=DEFAULT_FIELDS,
)

# Train/validation/test split
train_samples, temp_samples = train_test_split(samples, test_size=0.2, random_state=42)
val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

test_keys = {s.processed_file for s in test_samples}
test_processed_images_df = processed_images_df[
    processed_images_df["processed_file"].astype(str).isin(test_keys)
].copy()

if "processed_file" in ground_truth_df.columns:
    test_ground_truth_df = ground_truth_df[
        ground_truth_df["processed_file"].astype(str).isin(test_keys)
    ].copy()
else:
    test_ground_truth_df = ground_truth_df.copy()

# Fine-tune SmolVLM (set FAST_MODE=False for a fuller run)
FAST_MODE = True
train_subset = train_samples
val_subset = val_samples

if FAST_MODE:
    train_subset, _ = train_test_split(
        train_samples,
        train_size=min(0.2, max(32 / len(train_samples), 0.05)),
        random_state=42,
    )
    val_subset, _ = train_test_split(
        val_samples,
        train_size=min(0.5, max(16 / len(val_samples), 0.1)),
        random_state=42,
    )

history = smol.train(
    train_samples=train_subset,
    val_samples=val_subset,
    epochs=1,
    batch_size=1 if FAST_MODE else 2,
    learning_rate=2e-5,
    max_label_length=128 if FAST_MODE else 256,
)
print(history)

model_dir = smol.save()
print("Saved model to", model_dir)

# Test-set inference + evaluation
pred_df = smol.predict_dataset(processed_images_df=test_processed_images_df)
metrics_df, overall = smol.evaluate_against_ground_truth(
    ground_truth_df=test_ground_truth_df,
    pred_df=pred_df,
    fields=DEFAULT_FIELDS,
)
print(metrics_df)
print("Test overall metrics:", overall)
```

### 8. Donut Pipeline (`DonutInvoiceTextDetector`)

This pipeline fine-tunes Donut on invoice images, reloads the trained checkpoint, and evaluates on a held-out test split.
Unlike the OCR/LayoutLM pipelines, Donut inference here reads from `original_path` image paths.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.donut_model import DonutInvoiceTextDetector
from scripts.donut_training_utils import train_donut_invoice_model
from scripts.visualize_util import create_analysis_dashboard, visualize_sample_results

ground_truth_df = pd.read_csv("/path/to/data_dir/...cleaned_invoices.csv")
combined_results = pd.read_csv("/path/to/data_dir/...combined_results.csv")

# Join labels with source image paths
full_df = ground_truth_df.merge(
    combined_results[["original_file", "original_path"]],
    left_on="File Name",
    right_on="original_file",
    how="inner",
)

# Optional subset for faster experiments
sample_df = full_df.iloc[:200, :]

# Train / val / test split
train_df, holdout_df = train_test_split(sample_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
test_df = holdout_df

# Fine-tune Donut
train_output = train_donut_invoice_model(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    output_dir="/path/to/data_dir/...donut_finetuned_model",
    image_col="original_path",
    augment_factor=1,
)
print("Validation metrics:", train_output["validation_metrics"])
print("Test metrics:", train_output["test_metrics"])

# Reload trained checkpoint
donut_detector = DonutInvoiceTextDetector(
    output_dir="/path/to/data_dir/...donut_output",
)
donut_detector.reload_model("/path/to/data_dir/...donut_finetuned_model")

# Test-set inference
summary_df = donut_detector.run_inference(
    test_df,
    image_path_col="original_path",
    sample_frac=1,
    batch_size=4,
)
print(summary_df.head())

# Evaluation
donut_metrics_df, donut_overall = donut_detector.evaluate_against_ground_truth(test_df)
print(donut_metrics_df)
print("Overall Metrics:", donut_overall)

# Dashboard + sample outputs
_ = create_analysis_dashboard(
    donut_detector.full_results,
    metrics_df=donut_metrics_df,
    fields=["invoice_number", "invoice_date", "seller_name", "client_name", "tax", "net_worth", "total_amount"],
)
visualize_sample_results(
    donut_detector.full_results,
    visualize_text_fn=None,  # Donut does not return OCR boxes
    n_samples=3,
)
```

## Notes

- `InvoiceZonalOCRPipeline` (baseline) is template-zone dependent: expect strong performance on fields with stable layout, but weaker transfer to unseen invoice designs unless ROI boxes are retuned with `visualize_zones(...)`.
- `PytesseractInvoiceTextDetector` is OCR-first and more layout-agnostic than zonal OCR, but still sensitive to image quality/preprocessing; low-quality scans can reduce numeric and name accuracy.
- `LayoutLMv3InvoiceTokenClassifier` uses weak supervision plus fallback heuristics: best results require good weak-label coverage and consistent OCR tokens; run the diagnostics/debug cells in `layoutlmv3_model.ipynb` before changing heuristics.
- `SmolVLMInvoiceModel` and `DonutInvoiceTextDetector` are heavier trainable VLMs/DocVLMs: first runs are slower and more resource-intensive than OCR pipelines, so start with smaller subsets/epochs, then scale up.
- Donut pipeline uses `original_path` images (not `processed_path`), while OCR/LayoutLM pipelines typically use processed-image artifacts; mixing these paths is a common first-run mistake.
- For fair comparisons across models, keep the same field list, train/test split seed, and evaluation table generation workflow in the notebooks.
