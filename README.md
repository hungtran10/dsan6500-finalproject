# dsan6500-finalproject

## Data Access & Documentation

For this project I will be utilizing the collection of invoice images from this [Kaggle dataset](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr?resource=download).

There are a variety of invoice templates from multiple industries with different currencies, tax formats, and layouts that will make for an interesting project. 

Dataset License: [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

Dataset size: 1GB (zipped), 2.41GB (unzipped) 

Ways to Download the Dataset (instructions on the link above): kagglehub, Kaggle CLI, cURL, microissant, MCP, or manually downloading zip (easiest)

## Models Developed

1. Basic non-deep learning model (InvoiceZonalOCRPipeline)

## Getting Started

To get this pipeline running, follow these steps to set up your environment and process your first batch of invoices.

### 1\. Prerequisites

Install the Python dependencies:

```bash
pip install opencv-python pytesseract transformers 
```

### 2\. Basic Usage

The following snippet shows how to initialize each pipeline and process a single invoice.

#### InvoiceZonalOCRPipeline

```python
from invoice_pipeline import InvoiceZonalOCRPipeline

# 1. Initialize the pipeline
pipeline = InvoiceZonalOCRPipeline(output_dir="./results")

# 2. Process a single invoice
# Note: Ensure your template_zones in the class match your document layout
result = pipeline.process_invoice("path/to/invoice.png")

if result["success"]:
    print("Extracted Fields:")
    for field, value in result["fields"].items():
        print(f"{field}: {value}")
```

### 3\. Processing a Batch

If you have a folder full of invoices of the same type, you can process them all at once into a **Pandas DataFrame**:

```python
df_results = pipeline.process_folder("./invoices_folder")

# Save the results to a CSV
df_results.to_csv("extracted_data.csv", index=False)
print(df_results.head())
```

-----

### 4\. Tuning Your Zones (The "Spatial Map")

Since this is a template-based approach, the **accuracy depends on your coordinates**. Use the built-in visualization tool to "see" what the pipeline sees.

```python
# This will open a plot showing green boxes around your defined zones
pipeline.visualize_zones("path/to/invoice.png")
```

> **Pro-Tip:** The coordinates $(x, y, w, h)$ are **normalized** between $0.0$ and $1.0$.
>
>   * $(0.5, 0.5)$ represents the dead center of the page.
>   * If a box is too tight and cutting off text, increase the `pad` parameter in the `process_invoice` method or slightly expand the width/height in your `template_zones` dictionary.

### 5\. Evaluation

If you have a CSV of "Ground Truth" (correct) data, you can run a performance audit to see where the OCR is struggling:

```python
import pandas as pd

# Load your manual annotations
ground_truth = pd.read_csv("manual_annotations.csv")

# Run evaluation
metrics = pipeline.evaluate_against_ground_truth(df_results, ground_truth)
```
