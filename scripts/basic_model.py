import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import random
from tqdm import tqdm

@staticmethod
def clean_company_name(text):
    if text is None:
        return None

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

@staticmethod
def clean_amount(text):
    if text is None:
        return None

    text = str(text)
    text = re.sub(r"[$€£¥₹\s]", "", text)

    matches = re.findall(r"\d+[.,]\d{2}", text)
    if not matches:
        return None

    value = matches[-1].replace(",", ".")
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return None
    
class InvoiceZonalOCRPipeline:
    """
    OpenCV + Tesseract pipeline for template-based invoice extraction.

    Strategy:
    1. Deskew and clean the page.
    2. Crop fixed regions of interest (ROIs) based on invoice template layout.
    3. Run OCR only on each ROI.
    4. Use regex rules on the ROI text to extract fields.
    """

    def __init__(
        self,
        output_dir: str,
        template_zones: Optional[Dict[str, Dict[str, Tuple[float, float, float, float]]]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tesseract config tuned for invoice text.
        self.tesseract_config_legacy = (
            r"--oem 0 --psm 6 "
            r"-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$€£¥₹()/[]% "
        )

        self.tesseract_config_lstm = (
            r"--oem 1 --psm 6 "
            r"-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$€£¥₹()/[]% "
        )

        # Relative ROIs: (x, y, w, h) in normalized coordinates [0,1].
        # Tune these for your invoice template.
        self.template_zones = template_zones or {
            "default": {
            # Top-left: Invoice number
            "invoice_number": (0.21, 0.025, 0.12, 0.04),

            # Slightly below, shifted right: Date
            "date": (0.48, 0.05, 0.30, 0.08),

            # Middle section
            "vendor_name": (0.05, 0.215, 0.40, 0.02),   # Seller block (left)
            "client_name": (0.50, 0.215, 0.40, 0.02),   # Client block (right)

            # Bottom-right summary section
            "net_worth": (0.45, 0.79, 0.20, 0.12),      # Net worth
            "tax": (0.65, 0.79, 0.15, 0.12),           # VAT
            "total_amount": (0.80, 0.79, 0.18, 0.12),  # Gross worth
            }
        }

        self.FIELD_PSM = {
            "invoice_number": 7,
            "date": 7,
            "invoice_date": 7,
            "net_worth": 7,
            "tax": 7,
            "total_amount": 7,
            "vendor_name": 6,
            "client_name": 6,
        }

    
    # Preprocessing
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(str(image_path))
        return img

    def deskew(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Rotate image to reduce skew using the minimum-area rectangle of foreground pixels.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Invert threshold so text becomes foreground.
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image_bgr

        angle = cv2.minAreaRect(coords)[-1]

        # OpenCV's angle convention correction
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image_bgr.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def clean_image(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Stronger denoising
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive threshold (better for legacy OCR)
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )

        # Morphology to clean text blobs
        kernel = np.ones((2, 2), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        return th

    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """
        Full page preprocessing: load, deskew, and lightly clean.
        Returns the cleaned grayscale/binary image for ROI extraction.
        """
        img = self.load_image(image_path)
        if img is None:
            return None

        img = self.deskew(img)
        cleaned = self.clean_image(img)
        return cleaned

    
    # ROI extraction
    
    def get_template_zones(self, template_name: str = "default") -> Dict[str, Tuple[float, float, float, float]]:
        return self.template_zones.get(template_name, self.template_zones["default"])

    def crop_relative_roi(self, image: np.ndarray, box: Tuple[float, float, float, float], pad: int = 0) -> np.ndarray:
        """
        Crop a region specified in normalized coordinates, with optional pixel padding.
        """
        h, w = image.shape[:2]
        x_rel, y_rel, rw_rel, rh_rel = box

        x1 = max(0, int(x_rel * w) - pad)
        y1 = max(0, int(y_rel * h) - pad)
        x2 = min(w, int((x_rel + rw_rel) * w) + pad)
        y2 = min(h, int((y_rel + rh_rel) * h) + pad)

        return image[y1:y2, x1:x2]

    def visualize_zones(self, image_path: str, template_name: str = "default") -> None:
        """
        Draw the configured zones on top of the invoice so you can tune them.
        """
        img = self.load_image(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return

        zones = self.get_template_zones(template_name)
        vis = img.copy()
        h, w = vis.shape[:2]

        for field, box in zones.items():
            x_rel, y_rel, rw_rel, rh_rel = box
            x1 = int(x_rel * w)
            y1 = int(y_rel * h)
            x2 = int((x_rel + rw_rel) * w)
            y2 = int((y_rel + rh_rel) * h)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, field, (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 18))
        plt.imshow(vis_rgb)
        plt.axis("off")
        plt.title(f"Configured zones: {Path(image_path).name}")
        plt.show()

    
    # OCR
    
    def ocr_image(self, image: np.ndarray, psm: Optional[int] = None) -> str:
        """
        Run OCR using legacy Tesseract (OEM 0). Falls back to LSTM (OEM 1) if needed.
        """
        def run_ocr(config):
            return pytesseract.image_to_string(image, config=config).strip()

        # Adjust PSM dynamically
        if psm is not None:
            legacy_config = self.tesseract_config_legacy.replace("--psm 6", f"--psm {psm}")
            lstm_config = self.tesseract_config_lstm.replace("--psm 6", f"--psm {psm}")
        else:
            legacy_config = self.tesseract_config_legacy
            lstm_config = self.tesseract_config_lstm

        # Try legacy engine first
        try:
            text = run_ocr(legacy_config)

            # If OCR output is empty or garbage, fallback
            if not text or len(text.strip()) < 2:
                raise ValueError("Weak OCR output from legacy engine")

            return text

        except Exception:
            # Fallback to LSTM
            try:
                return run_ocr(lstm_config)
            except Exception as e:
                print(f"OCR failed completely: {str(e)}")
                return ""

    def ocr_roi(self, roi, field_name=None, psm=6):
        """
        OCR a region with field-specific configs and fallback attempts.
        """
        if roi is None or roi.size == 0:
            return ""

        numeric_fields = {"net_worth", "tax", "total_amount"}

        configs = []

        if field_name in numeric_fields:
            configs = [
                r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,$",
                r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,$",
                r"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,$",
            ]
        else:
            configs = [
                f"--oem 3 --psm {psm}",
                "--oem 3 --psm 6",
                "--oem 3 --psm 7",
            ]

        # Try original ROI first, then inverted ROI as a fallback
        roi_variants = [roi, cv2.bitwise_not(roi)]

        for img in roi_variants:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config).strip()
                    if text:
                        return text
                except Exception:
                    continue

        return ""

    # Field parsing
    def parse_field(self, field: str, text: str) -> Optional[str]:
        """
        Convert ROI OCR text into a field value using regex and cleanup rules.
        """
        if not text:
            return None

        t = re.sub(r"\s+", " ", str(text)).strip()

        # Direct field-specific parsing first
        if field in ["date", "invoice_date"]:
            m = re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b", t)
            return m.group(0) if m else None

        if field == "invoice_number":
            patterns = [
                r"invoice\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Z0-9\-]{4,})",
                r"inv(?:\.|oice)?\s*(?:no\.?|#)?\s*[:\-]?\s*([A-Z0-9\-]{4,})",
                r"\b([0-9]{6,})\b",
            ]
            for pat in patterns:
                m = re.search(pat, t, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return None

        if field == "vendor_name":
            # ROI often contains only the name, so use the full OCR text directly
            cleaned = clean_company_name(t)
            return cleaned if cleaned else None

        if field == "client_name":
            cleaned = clean_company_name(t)
            return cleaned if cleaned else None

        if field in ["net_worth", "tax", "total_amount"]:
            cleaned = clean_amount(t)
            return cleaned if cleaned else None

        # Fallback regex patterns if needed
        patterns = {
            "tax": [
                r"(?:vat|tax)\s*[:\-%]?\s*([0-9]{1,2}(?:\.[0-9]+)?%?)",
                r"\b([0-9]{1,2}(?:\.[0-9]+)?%)\b",
            ],
            "total_amount": [
                r"(?:gross\s*worth|grand\s*total|total|amount\s*due)\s*[:\-]?\s*\$?\s*([0-9,]+\.[0-9]{2})",
                r"\b\$?\s*([0-9,]+\.[0-9]{2})\b",
            ],
        }

        for pat in patterns.get(field, []):
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()

        return None

    # End-to-end extraction
    def process_invoice(self, image_path: str, template_name: str = "default") -> Dict[str, Any]:
        """
        Process one invoice image end-to-end.

        Returns a dictionary with OCR text by zone and final parsed fields.
        """
        cleaned = self.preprocess(image_path)
        if cleaned is None:
            return {
                "image_path": str(image_path),
                "success": False,
                "error": "Could not load image",
                "zone_text": {},
                "fields": {}
            }

        zones = self.get_template_zones(template_name)

        zone_text = {}
        fields = {}

        for field_name, box in zones.items():
            pad = 0
            if field_name in ["net_worth", "tax", "total_amount"]:
                pad = 10

            roi = self.crop_relative_roi(cleaned, box, pad=pad)

            if field_name in ["net_worth", "tax", "total_amount"]:

                # try OCR on the crop as-is first, then retry with a larger crop if empty
                psm = self.FIELD_PSM.get(field_name, 6)
                text = self.ocr_roi(roi, field_name=field_name, psm=psm)

                if not text.strip():
                    roi_big = self.crop_relative_roi(cleaned, box, pad=20)
                    roi_big = cv2.threshold(roi_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    text = self.ocr_roi(roi_big, field_name=field_name, psm=psm)
            else:
                psm = self.FIELD_PSM.get(field_name, 6)
                text = self.ocr_roi(roi, field_name=field_name, psm=psm)

            zone_text[field_name] = text
            parsed_value = self.parse_field(field_name, text)
            fields[field_name] = parsed_value

            # if not parsed_value:
            #     print(f"[DEBUG] Failed to extract {field_name}")
            #     print(f"OCR TEXT:\n{text}\n{'-'*40}")

        return {
            "image_path": str(image_path),
            "success": True,
            "zone_text": zone_text,
            "fields": fields,
        }

    

    def process_folder(self, folder_path, template_name="default", sample_size=None, sample_frac=None,
        random_state=42, file_extensions=("*.jpg", "*.png", "*.jpeg") ):
            """
            Process images in a folder with optional sampling and progress tracking.

            Parameters
            ----------
            folder_path : str or Path
                Directory containing invoice images
            template_name : str
                Template zone configuration
            sample_size : int, optional
                Number of images to randomly sample
            sample_frac : float, optional
                Fraction of dataset to sample (e.g., 0.1 for 10%)
            random_state : int
                Seed for reproducibility
            file_extensions : tuple
                File types to include

            Returns
            -------
            pd.DataFrame
            """
            folder_path = Path(folder_path)

            # Collect images
            image_paths = []
            for ext in file_extensions:
                image_paths.extend(folder_path.glob(ext))

            image_paths = sorted(image_paths)

            if len(image_paths) == 0:
                print("No images found.")
                return pd.DataFrame()

            # -------------------------
            # Sampling logic
            # -------------------------
            random.seed(random_state)

            if sample_frac is not None:
                k = int(len(image_paths) * sample_frac)
                image_paths = random.sample(image_paths, k)

            elif sample_size is not None:
                k = min(sample_size, len(image_paths))
                image_paths = random.sample(image_paths, k)

            print(f"Processing {len(image_paths)} images...")

            # -------------------------
            # Processing with tqdm
            # -------------------------
            results = []

            for img_path in tqdm(image_paths, desc="Processing invoices"):
                result = self.process_invoice(img_path, template_name=template_name)

                results.append({
                    "File Name": img_path.name,
                    "image_path": str(img_path),
                    "success": result["success"],
                    **result["fields"]
                })

            df = pd.DataFrame(results)

            return df
    
    def evaluate_against_ground_truth(self, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare extracted OCR results against ground truth annotations.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Output from process_folder or a similar DataFrame with extracted fields.
            Must contain a 'File Name' column.
        ground_truth_df : pd.DataFrame
            Ground truth DataFrame with columns:
            'File Name', 'client_name', 'seller_name', 'invoice_number',
            'invoice_date', 'due_date', 'tax', 'total_amount', 'net_worth'

        Returns
        -------
        pd.DataFrame
            Field-level metrics: accuracy, precision, recall, and f1.
        """

        fields = [
            "client_name",
            "seller_name",
            "invoice_number",
            "invoice_date",
            "due_date",
            "tax",
            "total_amount",
            "net_worth",
        ]

        def normalize_value(field, value):
            if pd.isna(value) or value is None:
                return None

            s = str(value).strip()

            if s == "":
                return None

            if field in ["tax", "total_amount", "net_worth"]:
                s = s.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace("₹", "")
                s = s.replace(" ", "")
                if "," in s and "." not in s:
                    s = s.replace(",", ".")
                elif s.count(",") > 0 and s.count(".") > 0:
                    s = s.replace(",", "")
                try:
                    return round(float(s), 2)
                except Exception:
                    return None

            if field in ["invoice_date", "due_date"]:
                dt = pd.to_datetime(s, errors="coerce")
                if pd.isna(dt):
                    return None
                return dt.strftime("%Y-%m-%d")

            s = re.sub(r"\s+", " ", s).lower().strip()
            s = re.sub(r"[^\w\s\-]", "", s)
            return s

        if "File Name" not in predictions_df.columns:
            raise ValueError("predictions_df must contain a 'File Name' column.")
        if "File Name" not in ground_truth_df.columns:
            raise ValueError("ground_truth_df must contain a 'File Name' column.")

        eval_df = predictions_df.merge(
            ground_truth_df[["File Name"] + fields],
            on="File Name",
            how="inner",
            suffixes=("_pred", "_gt")
        )

        rows = []

        for field in fields:
            pred_col = f"{field}_pred"
            gt_col = f"{field}_gt"

            if pred_col not in eval_df.columns or gt_col not in eval_df.columns:
                continue

            pred_vals = eval_df[pred_col].apply(lambda x: normalize_value(field, x))
            gt_vals = eval_df[gt_col].apply(lambda x: normalize_value(field, x))

            valid_gt = gt_vals.notna()
            valid_pred = pred_vals.notna()

            matches = valid_gt & valid_pred & (pred_vals == gt_vals)

            tp = int(matches.sum())
            fp = int((valid_pred & (~matches)).sum())
            fn = int((valid_gt & (~matches)).sum())

            accuracy = tp / int(valid_gt.sum()) if int(valid_gt.sum()) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            rows.append({
                "field": field,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(valid_gt.sum())
            })

        metrics_df = pd.DataFrame(rows)
        print(metrics_df.to_string(index=False))
        return metrics_df