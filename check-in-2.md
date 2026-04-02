## Baseline Model - Zonal Pipeline

My baseline (non-deep learning) model - `InvoiceZonalOCRPipeline` - utilizes the python package `cv2` to handle fixed-layout invoices. By using OpenCV for spatial mapping, you eliminate the "black box" nature of deep learning and gain full control over the extraction logic. Basically, the model pipeline has four steps
1. Image preprocessing: deskewing, adapative thresholding, and morphological operations
2. Zonal Mapping (through spatial extraction): ROI cropping
3. Intelligent OCR: It prioritizes Tesseract’s Legacy Engine (OEM 0), which uses non-deep-learning pattern matching. It applies specific Page Segmentation Modes (PSM) and character whitelists depending on the field (e.g., only allowing numbers for the "Total Amount" field).
4. Data Validation & Evaluation

The biggest limitation to this approach is the zonal mapping makes the text detection and extraction process rigid and inflexible. 

## CNN-based Model - Pytesseract Pipeline
This model - `PytesseractInvoiceTextDetector` uses Heuristic Layout Analysis. It basically reads the document to find anchor points and dynamically determines where regions begin and end. It is a structural parsing engine that converts unstructured OCR word-blobs into a hierarchical document model. It moves beyond simple coordinate cropping by using document anchors and arithmetic validation. 

This pipeline is a significant step up in complexity. While the previous version relied on fixed spatial coordinates, the `PytesseractInvoiceTextDetector` uses **Heuristic Layout Analysis**. It "reads" the document to find anchor points and dynamically determines where regions begin and end. It is a structural parsing engine that converts unstructured OCR word-blobs into a hierarchical document model. It moves beyond simple coordinate cropping by using document anchors and arithmetic validation.

### 1. Dynamic Region Assignment
Instead of using hardcoded boxes, the pipeline identifies key layout landmarks to segment the page into logical zones:
* **Header (Top-Left):** For metadata like Invoice Numbers and Dates.
* **Split-Middle:** A vertical split to separate **Seller** info (left) from **Client** info (right).
* **Table Body:** A central zone identified by row-item headers.
* **Summary (Bottom):** A region dedicated to financial totals and tax breakdowns.

### 2. Line Reconstruction & Clustering
OCR engines often return a "bag of words." This pipeline implements a custom clustering algorithm (`_cluster_words_by_line`) that:
* Groups words based on **vertical proximity** (y-coordinate tolerance).
* Sorts words horizontally to reconstruct natural reading order.
* Computes a data-driven tolerance based on the median word height of the document.

### 3. Arithmetic-Validated Totals
To ensure data integrity in the "Bottom" region, the pipeline doesn't just "read" numbers; it validates them using financial logic. It looks for monetary patterns and applies a scoring system:
* **Regex Filtering:** Identifies currency patterns (e.g., $1,234.56$).
* **Consistency Check:** It attempts to satisfy the equation $$Net Worth + Tax \approx Total Amount$$.
* **Rank-Based Assignment:** If labels are missing, it intelligently assigns the smallest value to Tax, the middle to Net, and the largest to Total.

### 4. Structural Table Parsing
One of the most complex features is the `extract_table_dataframe` method, which transforms line items into a structured `pandas.DataFrame`:
* **Header Detection:** Scans for keywords like "Description," "Qty," and "Unit Price."
* **Column Binning:** Calculates the horizontal center of header text to create "bins." Every word below the header is then assigned to a column based on which bin its x-coordinate falls into.


---

### Comparison 
| Feature | Zonal Pipeline (Previous) | Layout-Aware Pipeline (Current) |
| :--- | :--- | :--- |
| **Flexibility** | Rigid; requires exact template match. | Flexible; adapts to varying header lengths. |
| **Data Output** | Key-Value pairs only. | Key-Values + Structured DataFrames (Tables). |
| **Logic** | Purely Spatial. | Heuristic + Arithmetic Validation. |
| **OCR Level** | ROI-based (crops). | Full-page word-level analysis. |


## Results, Evaluation & Failure Analysis


### 1. Zonal Pipeline (Coordinate-Based)
* **Strengths:** Achieved near-perfect scores for `invoice_number` (1.0 F1) and `client_name` (0.99 F1). This confirms that header information in this dataset is highly standardized in its spatial positioning.
* **Weaknesses:** Failed critically on financial fields (`tax`, `total_amount`, `net_worth`), with recall scores dropping below **10%**.
* **Analysis:** Because the Zonal pipeline relies on static (x, y) coordinates, it cannot account for vertical shifts. If an invoice has more line items than the template expects, the "Totals" block is pushed down, falling outside the pre-defined extraction zone.

### 2. Layout-Aware Pipeline (Heuristic-Based)
* **Strengths:** Dramatically improved performance on financial totals, with F1 scores rising from **~0.15 to ~0.90**. It achieved an **Overall F1-Score of 0.946**.
* **Precision vs. Recall:** The pipeline maintains very high precision (~0.98), meaning when it extracts a value, it is almost certainly correct. The slight dip in recall for financial fields (indices 4-6) suggests that in some complex layouts, the anchor words (like "Total") were either missing or obscured.
* **Analysis:** By using "items" and "summary" as anchors to dynamically define regions, this pipeline successfully follows the "flow" of the document. It effectively "finds" the data rather than just "looking" at a spot.

---

### Comparison Summary
| Metric | Zonal Pipeline | Layout-Aware Pipeline |
| :--- | :--- | :--- |
| **Top-of-Page Fields** | Excellent (Static) | Excellent (Dynamic) |
| **Bottom-of-Page Fields** | **Fails** (Coordinate Shift) | **Succeeds** (Anchor Detection) |
| **Robustness** | Low | High |
| **Overall Accuracy** | ~45% (Estimated) | **91.18%** |

---
### Zonal Pipeline Metrics

| field          | accuracy | precision | recall   | f1       | 
| -------------- | -------- | --------- | -------- | -------- | 
| client_name    | 0.996390 | 0.996390  | 0.996390 | 0.996390 | 
| invoice_number | 1.000000 | 1.000000  | 1.000000 | 1.000000 | 
| tax            | 0.093863 | 1.000000  | 0.093863 | 0.171617 | 
| total_amount   | 0.086643 | 0.800000  | 0.086643 | 0.156352 | 
| net_worth      | 0.075812 | 0.913043  | 0.075812 | 0.140000 | 

**Overall Metrics (Layout-Aware)**

* **Accuracy:** 0.45054
* **Precision:** 0.94188
* **Recall:** 0.45054
* **F1:** 0.49288
---

### Layout-Aware Pipeline Metrics

| field          | accuracy | precision | recall   | f1       |
| -------------- | -------- | --------- | -------- | -------- |
| invoice_number | 0.992908 | 0.992908  | 0.992908 | 0.992908 |
| invoice_date   | 1.000000 | 1.000000  | 1.000000 | 1.000000 |
| seller_name    | 1.000000 | 1.000000  | 1.000000 | 1.000000 |
| client_name    | 1.000000 | 1.000000  | 1.000000 | 1.000000 |
| net_worth      | 0.765957 | 0.964286  | 0.765957 | 0.853755 |
| total_amount   | 0.787234 | 0.925000  | 0.787234 | 0.850575 |
| tax            | 0.836879 | 0.983333  | 0.836879 | 0.904215 |

**Overall Metrics (Layout-Aware)**

* **Accuracy:** 0.91185
* **Precision:** 0.98253
* **Recall:** 0.91185
* **F1:** 0.94587

## Drawbacks of the Layout-Aware Pipline

* Layout/format inconsistencies: PyTesseract extracts raw text without semantic understanding.
Example: Fields like Net Worth or Total Amount may appear in different positions across extracted text. PyTesseract captures text but cannot reliably map them to structured labels.
Pattern: Field misalignment and missing mappings cause systematic errors in field extraction.

* Failure on complex layouts: PyTesseract reads text linearly so it doesn’t handle multi-column or table structures natively.
Example: A table of line items with multiple columns (item, quantity, price) may get flattened, concatenated, or misaligned.
Pattern: Structured fields like tables, headers, and footers break, especially when alignment is inconsistent or fonts vary.

