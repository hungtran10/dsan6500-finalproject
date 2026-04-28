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

## Notes

- `InvoiceZonalOCRPipeline` is template-zone dependent. Tune ROIs using `visualize_zones(...)`.
- For fair model comparison, keep the same train/test split and field list across `pt_model` and `layoutlmv3_model`.
- Numeric fields now support locale formats like `1 234,56` in the baseline parser/evaluator.
