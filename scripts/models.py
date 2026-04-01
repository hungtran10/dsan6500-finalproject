import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import cv2
from PIL import Image
import re
import json
from pathlib import Path
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

class InvoiceTextDetector:
    """
    Text detection and OCR extraction for preprocessed invoice images
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$€£¥₹()[] '

    def assign_regions(self, extracted_text, image_shape):
        h, w = image_shape[:2]

        regions = {
            "top_left": [],
            "seller": [],
            "client": [],
            "table": [],
            "bottom": []
        }

        item_y = self._find_anchor_y(extracted_text, r"items?")
        summary_y = self._find_anchor_y(extracted_text, r"summary")

        # Fallbacks if anchors are not found
        table_start = int(0.30 * h) if item_y is None else max(int(0.20 * h), item_y - int(0.03 * h))
        table_end = int(0.85 * h) if summary_y is None else min(int(0.95 * h), summary_y - int(0.02 * h))

        top_cut = int(0.16 * h)

        for item in extracted_text:
            x, y, bw, bh = item["bbox"]
            x_rel = x / w

            if y < top_cut:
                regions["top_left"].append(item)

            elif table_start <= y <= table_end:
                regions["table"].append(item)

            elif y > table_end:
                regions["bottom"].append(item)

            else:
                # Middle block: seller / client
                if x_rel < 0.5:
                    regions["seller"].append(item)
                else:
                    regions["client"].append(item)

        return regions
    
    def region_to_text(self, region_words):
        if not region_words:
            return ""

        lines = self._cluster_words_by_line(region_words)
        text_lines = []

        for line in lines:
            words = sorted(line["words"], key=lambda x: x["bbox"][0])
            text_lines.append(" ".join(w["text"] for w in words))

        return "\n".join(text_lines)

    def _ocr_words(self, image, confidence_threshold=30, psm=6, extra_config=""):
        """
        Run Tesseract OCR on a grayscale image array and return filtered word records.
        """
        config = f'--oem 3 --psm {psm} {extra_config}'.strip()

        ocr_data = pytesseract.image_to_data(
            image,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        extracted_text = []
        for i in range(len(ocr_data["text"])):
            text = str(ocr_data["text"][i]).strip()

            conf_raw = ocr_data["conf"][i]
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = -1

            if text and confidence > confidence_threshold:
                extracted_text.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": (
                        int(ocr_data["left"][i]),
                        int(ocr_data["top"][i]),
                        int(ocr_data["width"][i]),
                        int(ocr_data["height"][i]),
                    ),
                    "block_num": int(ocr_data["block_num"][i]),
                    "par_num": int(ocr_data["par_num"][i]),
                    "line_num": int(ocr_data["line_num"][i]),
                    "word_num": int(ocr_data["word_num"][i]),
                })

        return extracted_text
    
    def _cluster_words_by_line(self, words, y_tol=None):
        """
        Group OCR words into approximate text lines using y-position.
        Returns a list of dicts: [{"y": ..., "words": [...]}, ...]
        """
        if not words:
            return []

        heights = [w["bbox"][3] for w in words]
        if y_tol is None:
            y_tol = max(8, int(np.median(heights) * 0.7))

        sorted_words = sorted(words, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        lines = []

        for item in sorted_words:
            y = item["bbox"][1]
            if not lines or abs(y - lines[-1]["y"]) > y_tol:
                lines.append({"y": y, "words": [item]})
            else:
                lines[-1]["words"].append(item)

        return lines
    
    def extract_bottom_totals(self, bottom_words):
        """
        Extract tax and total_amount from the bottom region.
        tax corresponds to VAT value
        total_amount corresponds to Gross worth
        """
        if not bottom_words:
            return {}

        lines = self._cluster_words_by_line(bottom_words)
        line_texts = []
        for line in lines:
            txt = " ".join(w["text"] for w in sorted(line["words"], key=lambda x: x["bbox"][0]))
            line_texts.append((txt, line["words"]))

        money_pattern = re.compile(r"(?<!\w)\d[\d\s.,]*[.,]\d{2}(?!\w)")

        best_row_amounts = None

        for txt, _ in line_texts:
            low = txt.lower()
            amounts = []
            for m in money_pattern.finditer(txt):
                norm = self._normalize_money(m.group(0))
                if norm is not None:
                    amounts.append(float(norm))

            if len(amounts) >= 3 and ("vat" in low or "gross" in low or "summary" in low):
                best_row_amounts = amounts
                break

        if best_row_amounts is None:
            for txt, _ in line_texts:
                amounts = []
                for m in money_pattern.finditer(txt):
                    norm = self._normalize_money(m.group(0))
                    if norm is not None:
                        amounts.append(float(norm))
                if len(amounts) >= 3:
                    best_row_amounts = amounts
                    break

        result = {}

        if best_row_amounts and len(best_row_amounts) >= 3:
            result["tax"] = f"{best_row_amounts[1]:.2f}"
            result["total_amount"] = f"{best_row_amounts[2]:.2f}"
            return result

        all_amounts = []
        for txt, _ in line_texts:
            for m in money_pattern.finditer(txt):
                norm = self._normalize_money(m.group(0))
                if norm is not None:
                    all_amounts.append(float(norm))

        if all_amounts:
            uniq = sorted(set(all_amounts))
            result["tax"] = f"{uniq[0]:.2f}"
            result["total_amount"] = f"{uniq[-1]:.2f}"

        return result
    
    def extract_table_dataframe(self, table_words):
        """
        Convert the table region into a clean product-level DataFrame.
        Expected columns:
        No., Description, Qty, UM, Net price, Net worth, VAT[%], Gross Worth
        """
        if not table_words:
            return pd.DataFrame()

        lines = self._cluster_words_by_line(table_words)

        line_data = []
        for line in lines:
            words = sorted(line["words"], key=lambda x: x["bbox"][0])
            text = " ".join(w["text"] for w in words)
            line_data.append({
                "words": words,
                "text": text
            })

        # Find header row
        header_idx = None
        for i, row in enumerate(line_data):
            txt = row["text"].lower()
            if (
                "description" in txt and
                "qty" in txt and
                "um" in txt and
                "net" in txt and
                "gross" in txt
            ):
                header_idx = i
                break

        if header_idx is None:
            return pd.DataFrame()

        data_lines = line_data[header_idx + 1:]

        product_rows = []
        current_row = None

        item_pattern = re.compile(r"^\d+\.$")

        for row in data_lines:
            words = row["words"]
            text = row["text"].strip()

            if not words:
                continue

            first_word = words[0]["text"].strip()

            # New product row starts with "1.", "2.", etc.
            if item_pattern.match(first_word):
                if current_row:
                    product_rows.append(current_row)

                current_row = {
                    "item_no": first_word.replace(".", ""),
                    "words": words.copy()
                }
            else:
                # Continuation of previous product description
                if current_row:
                    current_row["words"].extend(words)

        if current_row:
            product_rows.append(current_row)

        structured_rows = []

        for row in product_rows:
            words = sorted(row["words"], key=lambda x: x["bbox"][0])
            text = " ".join(w["text"] for w in words)

            # Extract monetary values in order they appear
            money_values = []
            for m in re.finditer(r"\d[\d\s,]*[.,]\d{2}", text):
                norm = self._normalize_money(m.group(0))
                if norm is not None:
                    money_values.append(float(norm))

            # Expected order in a clean row:
            # qty, net_price, net_worth, gross_worth
            qty = None
            net_price = None
            net_worth = None
            gross_worth = None

            if len(money_values) >= 4:
                qty = money_values[0]
                net_price = money_values[1]
                net_worth = money_values[2]
                gross_worth = money_values[-1]

            vat_match = re.search(r"(\d{1,2})\s*%", text)
            vat_pct = vat_match.group(1) if vat_match else None

            # Description: remove item number and numeric tokens
            desc = re.sub(r"\d[\d\s,]*[.,]\d{2}", "", text)
            desc = re.sub(r"\b\d+\.\b", "", desc)
            desc = re.sub(r"\s+", " ", desc).strip()

            structured_rows.append({
                "item_no": row["item_no"],
                "description": desc,
                "qty": qty,
                "um": None,  # can be filled later if you want to extract units explicitly
                "net_price": net_price,
                "net_worth": net_worth,
                "vat_pct": vat_pct,
                "gross_worth": gross_worth,
                "raw_text": text
            })

        return pd.DataFrame(structured_rows)
    
    def _find_label_word(self, extracted_text, label_pattern):
        """
        Find the first OCR word matching a label pattern such as Seller or Client.
        """
        for item in extracted_text:
            if re.search(label_pattern, item["text"], re.IGNORECASE):
                return item
        return None
    
    def _normalize_date(self, value):
        """
        Convert dates like 12/20/2014 or 12-20-14 to YYYY-MM-DD.
        """
        if value is None or pd.isna(value):
            return None

        s = str(value).strip()
        if not s:
            return None

        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
        return None


    def _normalize_money(self, value):
        """
        Normalize money strings to a plain decimal string with 2 places.
        Examples:
        - 1 140,24 -> 1140.24
        - 1,140.24 -> 1140.24
        - 103,66   -> 103.66
        """
        if value is None or pd.isna(value):
            return None

        s = str(value).strip().replace(" ", "")
        if not s:
            return None

        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")

        try:
            return f"{float(s):.2f}"
        except ValueError:
            return None

    def extract_party_name(self, image, extracted_text, party="seller"):
        """
        Extract seller_name or client_name by:
        1) finding the label on the page
        2) cropping the correct half of the page
        3) stopping crop before the table starts
        4) OCR'ing the crop
        5) selecting the most likely name line
        """
        h, w = image.shape[:2]
        mid = w // 2

        label_regex = r"\bSeller\b" if party.lower() == "seller" else r"\bClient\b"
        label_word = self._find_label_word(extracted_text, label_regex)

        table_start_y = self._find_anchor_y(extracted_text, r"items?")
        if table_start_y is None:
            table_start_y = int(0.30 * h)

        if label_word:
            x, y, bw, bh = label_word["bbox"]

            x1, x2 = (0, mid) if party.lower() == "seller" else (mid, w)

            y1 = max(0, y - int(0.02 * h))
            y2 = min(h, table_start_y - int(0.01 * h))

            if y2 <= y1:
                y2 = min(h, y + int(0.18 * h))

            crop = image[y1:y2, x1:x2]
        else:
            x1, x2 = (0, mid) if party.lower() == "seller" else (mid, w)
            y1 = int(0.15 * h)
            y2 = min(h, table_start_y - int(0.01 * h))
            crop = image[y1:y2, x1:x2]

        block_text = ""
        for psm in (6, 11, 4):
            block_text = pytesseract.image_to_string(crop, config=f"--oem 3 --psm {psm}")
            name = self._extract_party_name_from_block_text(block_text, party)
            if name:
                return name, block_text

        return None, block_text
    
    def _extract_party_name_from_block_text(self, block_text, party):
        lines = [re.sub(r"\s+", " ", line).strip() for line in block_text.splitlines()]
        lines = [line for line in lines if line]

        label_pattern = rf"^{party}\s*:?\s*$"
        lines = [line for line in lines if not re.match(label_pattern, line, re.IGNORECASE)]

        skip_pattern = re.compile(
            r"(tax id|iban|vat|address|street|item|items|description|qty|um|net price|net worth|gross worth|summary|total|\bbox\b|\bpo\b|\bno\.\b|\d{4,})",
            re.IGNORECASE
        )

        for line in lines:
            if skip_pattern.search(line):
                continue
            if re.search(r"[A-Za-z]{2,}", line):
                return line

        return None
    
    def _first_match(self, text, patterns):
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return None

    def extract_invoice_fields_region_aware(self, image, extracted_text, regions):
        fields = {}

        top_left_text = self.region_to_text(regions["top_left"])

        fields["invoice_number"] = self._first_match(top_left_text, [
            r"invoice\s*(?:no\.?|#|number)?\s*:?\s*([A-Z0-9\-]{4,})",
            r"inv(?:\.|oice)?\s*(?:no\.?|#)?\s*:?\s*([A-Z0-9\-]{4,})",
            r"(?:^|\n)(?:invoice|inv).*?([0-9]{6,})",
            r"(?:^|\n)([0-9]{6,})\s*(?:\n|$)",
            r"no\.?\s*:?\s*([0-9]{6,})",
        ])

        date_raw = self._first_match(top_left_text, [
            r"date.*?([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
            r"issued?.*?([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})",
            r"([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{4})",
            r"([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2})",
        ])
        if date_raw:
            normalized_date = self._normalize_date(date_raw)
            if normalized_date:
                fields["invoice_date"] = normalized_date

        # Seller/client
        seller_name, _ = self.extract_party_name(image, extracted_text, party="seller")
        client_name, _ = self.extract_party_name(image, extracted_text, party="client")

        if seller_name:
            fields["seller_name"] = seller_name
        if client_name:
            fields["client_name"] = client_name

        # Bottom summary totals
        bottom_fields = self.extract_bottom_totals(regions["bottom"])
        fields.update(bottom_fields)

        return fields
    
    def process_single_image(self, image_path):
        result = {
            "image_path": str(image_path),
            "filename": Path(image_path).name,
            "success": False,
            "extracted_text": [],
            "invoice_fields": {},
            "table_df": pd.DataFrame(),
            "total_words": 0,
            "avg_confidence": 0
        }

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return result

            extracted_text = self._ocr_words(image, confidence_threshold=30, psm=6)
            if not extracted_text:
                return result

            result["extracted_text"] = extracted_text
            result["total_words"] = len(extracted_text)
            result["avg_confidence"] = float(np.mean([t["confidence"] for t in extracted_text]))

            regions = self.assign_regions(extracted_text, image.shape)
            result["invoice_fields"] = self.extract_invoice_fields_region_aware(image, extracted_text, regions)
            result["table_df"] = self.extract_table_dataframe(regions["table"])
            result["success"] = True

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

        return result
    
    def process_dataset(self, processed_images_df, batch_size=10, save_word_level=False, sample_frac=None, random_state=42):
        successful_images = processed_images_df[
            processed_images_df["status"] == "success"
        ]

        if sample_frac is not None:
            successful_images = successful_images.sample(frac=sample_frac, random_state=random_state)
            print(f"Processing {len(successful_images)} sampled images ({sample_frac*100:.1f}%)...")
        else:
            print(f"Processing {len(successful_images)} images...")

        results = []
        summary_rows = []
        word_rows = []
        table_rows = []

        for i in tqdm(range(0, len(successful_images), batch_size), desc="Processing OCR batches"):
            batch_df = successful_images.iloc[i:i + batch_size]

            for _, row in batch_df.iterrows():
                path = row["processed_path"]
                if not path or not Path(path).exists():
                    continue

                result = self.process_single_image(path)
                results.append(result)

                summary_row = {
                    "filename": result["filename"],
                    "image_path": result["image_path"],
                    "success": result["success"],
                    "total_words": result["total_words"],
                    "avg_confidence": result["avg_confidence"],
                    "has_invoice_fields": bool(result["invoice_fields"]),
                }
                summary_row.update(result["invoice_fields"])
                summary_rows.append(summary_row)

                if save_word_level and result["success"]:
                    for word in result["extracted_text"]:
                        word_rows.append({
                            "filename": result["filename"],
                            "text": word["text"],
                            "confidence": word["confidence"],
                            "bbox_x": word["bbox"][0],
                            "bbox_y": word["bbox"][1],
                            "bbox_width": word["bbox"][2],
                            "bbox_height": word["bbox"][3],
                            "block_num": word["block_num"],
                            "par_num": word["par_num"],
                            "line_num": word["line_num"],
                            "word_num": word["word_num"],
                        })

                if result["success"] and isinstance(result.get("table_df"), pd.DataFrame):
                    df = result["table_df"]
                    if not df.empty:
                        df = df.copy()
                        df.insert(0, "filename", result["filename"])
                        table_rows.append(df)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.output_dir / "ocr_invoice_results.csv", index=False)

        if table_rows:
            table_df = pd.concat(table_rows, ignore_index=True)
            table_df = table_df.dropna(how="all")
            table_df.to_csv(self.output_dir / "ocr_line_items.csv", index=False)

        if save_word_level and word_rows:
            word_df = pd.DataFrame(word_rows)
            word_df.to_csv(self.output_dir / "ocr_word_level.csv", index=False)

        self.full_results = results
        self._print_summary(results)

        return summary_df
    
    def _find_anchor_y(self, extracted_text, pattern):
        """
        Return the smallest y-position of any word matching the pattern.
        """
        ys = [
            item["bbox"][1]
            for item in extracted_text
            if re.fullmatch(pattern, item["text"].strip(), re.IGNORECASE)
        ]
        return min(ys) if ys else None
    
    def evaluate_against_ground_truth(self, ground_truth_df, merge_key="processed_file", restrict_to_matched=True):
        if not hasattr(self, "full_results"):
            raise ValueError("Run process_dataset() first.")

        pred_rows = []
        for r in self.full_results:
            row = {
                "processed_file": r["filename"],
            }
            row.update(r["invoice_fields"])
            pred_rows.append(row)

        pred_df = pd.DataFrame(pred_rows)

        # Basic diagnostics
        print("Prediction rows:", len(pred_df))
        print("Prediction unique keys:", pred_df[merge_key].nunique())
        print("Ground truth rows:", len(ground_truth_df))
        print("Ground truth unique keys:", ground_truth_df[merge_key].nunique())

        pred_keys = set(pred_df[merge_key].astype(str))
        gt_keys = set(ground_truth_df[merge_key].astype(str))
        overlap_keys = pred_keys & gt_keys

        print("Key overlap:", len(overlap_keys))

        if restrict_to_matched:
            ground_truth_df = ground_truth_df[ground_truth_df[merge_key].astype(str).isin(overlap_keys)].copy()
            pred_df = pred_df[pred_df[merge_key].astype(str).isin(overlap_keys)].copy()

        merged = ground_truth_df.merge(
            pred_df,
            on=merge_key,
            how="inner",
            suffixes=("_gt", "_pred")
        )

        def normalize(val, field):
            if pd.isna(val):
                return np.nan

            s = str(val).strip()
            if s.lower() in ["", "nan", "none"]:
                return np.nan

            if field in ["total_amount", "tax"]:
                norm = self._normalize_money(s)
                return norm if norm is not None else np.nan

            if field == "invoice_date":
                norm = self._normalize_date(s)
                return norm if norm is not None else np.nan

            s = re.sub(r"\s+", " ", s)
            return s.lower()

        fields = ["invoice_number", "invoice_date", "seller_name", "client_name", "total_amount", "tax"]
        results = []

        for field in fields:
            gt_col = f"{field}_gt"
            pred_col = f"{field}_pred"

            if gt_col not in merged.columns:
                continue

            gt = merged[gt_col].apply(lambda x: normalize(x, field))
            pred = merged[pred_col].apply(lambda x: normalize(x, field))

            valid_gt = gt.notna()
            valid_pred = pred.notna()
            correct = (gt == pred) & valid_gt & valid_pred

            gt_count = int(valid_gt.sum())
            pred_count = int(valid_pred.sum())
            correct_count = int(correct.sum())

            accuracy = correct_count / gt_count if gt_count else np.nan
            precision = correct_count / pred_count if pred_count else np.nan
            recall = correct_count / gt_count if gt_count else np.nan
            f1 = (
                2 * precision * recall / (precision + recall)
                if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0
                else np.nan
            )

            results.append({
                "field": field,
                "ground_truth_count": gt_count,
                "predicted_count": pred_count,
                "correct": correct_count,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        metrics_df = pd.DataFrame(results)

        total_gt = metrics_df["ground_truth_count"].sum()
        total_pred = metrics_df["predicted_count"].sum()
        total_correct = metrics_df["correct"].sum()

        overall = {
            "accuracy": total_correct / total_gt if total_gt else np.nan,
            "precision": total_correct / total_pred if total_pred else np.nan,
            "recall": total_correct / total_gt if total_gt else np.nan,
            "f1": (
                2 * (total_correct / total_pred) * (total_correct / total_gt)
                / ((total_correct / total_pred) + (total_correct / total_gt))
                if total_pred and total_gt else np.nan
            )
        }

        return metrics_df, overall
    
    def _print_summary(self, results):
        """
        Print a summary of OCR and invoice field extraction results.

        Parameters
        ----------
        results : list[dict]
            List of per-image result dictionaries produced by process_single_image function.
        """
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{'='*60}")
        print("TEXT DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful extractions: {len(successful)}")
        print(f"Failed extractions: {len(failed)}")
        
        if successful:
            avg_words = np.mean([r['total_words'] for r in successful])
            avg_confidence = np.mean([r['avg_confidence'] for r in successful])
            print(f"Average words per image: {avg_words:.1f}")
            print(f"Average confidence: {avg_confidence:.1f}%")
            
            # Count extracted fields
            field_counts = {}
            for result in successful:
                for field in result['invoice_fields'].keys():
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            if field_counts:
                print(f"\nExtracted invoice fields:")
                for field, count in field_counts.items():
                    print(f"  {field}: {count} images ({count/len(successful)*100:.1f}%)")
    
    def visualize_text_extraction(self, image_path, result):
        """
        Visualize OCR word boxes and extracted invoice fields for a single image.

        Parameters
        ----------
        image_path : str or Path
            Path to the invoice image.
        result : dict
            Result dictionary returned by process_single_image.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes around detected text
        for text_item in result['extracted_text']:
            x, y, w, h = text_item['bbox']
            confidence = text_item['confidence']
            
            # Color based on confidence
            if confidence > 80:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 50:
                color = (255, 255, 0)  # Yellow for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
            
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_rgb, f"{text_item['text'][:10]}...", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(image_rgb)
        plt.title(f'Text Detection Results: {Path(image_path).name}')
        plt.axis('off')
        plt.show()
        
        # Print extracted fields
        if result['invoice_fields']:
            print("\nExtracted Invoice Fields:")
            for field, value in result['invoice_fields'].items():
                print(f"  {field}: {value}")

    def create_analysis_dashboard(self):
        """
        Create a comprehensive analysis dashboard for OCR and invoice extraction.
        Requires process_dataset to have been run first so that self.full_results exists.

        Returns
        -------
        dict or None
            Summary statistics including:
            - total_processed
            - successful
            - field_extraction_rates
            - avg_confidence
            - avg_words

            Returns None if no results are available.
        """
        if not hasattr(self, 'full_results'):
            print("No results to analyze. Run process_dataset first.")
            return
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE INVOICE ANALYSIS DASHBOARD")
        print(f"{'='*80}")
        
        # Overall Processing Statistics
        successful_results = [r for r in self.full_results if r['success']]
        failed_results = [r for r in self.full_results if not r['success']]
        
        print(f"\nPROCESSING OVERVIEW")
        print(f"{'='*50}")
        print(f"Total images processed: {len(self.full_results):,}")
        print(f"Successful extractions: {len(successful_results):,} ({len(successful_results)/len(self.full_results)*100:.1f}%)")
        print(f"Failed extractions: {len(failed_results):,} ({len(failed_results)/len(self.full_results)*100:.1f}%)")
        
        if not successful_results:
            print("No successful results to analyze.")
            return
        
        # OCR Quality Analysis
        print(f"\nOCR QUALITY METRICS")
        print(f"{'='*50}")
        
        word_counts = [r['total_words'] for r in successful_results]
        confidences = [r['avg_confidence'] for r in successful_results]
        
        print(f"Average words per invoice: {np.mean(word_counts):.1f}")
        print(f"Median words per invoice: {np.median(word_counts):.0f}")
        print(f"Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"Average OCR confidence: {np.mean(confidences):.1f}%")
        print(f"Median OCR confidence: {np.median(confidences):.1f}%")
        print(f"Confidence range: {min(confidences):.1f}% - {max(confidences):.1f}%")
        
        # Field Extraction Analysis
        print(f"\nFIELD EXTRACTION ANALYSIS")
        print(f"{'='*50}")
        
        field_counts = {}
        all_fields = {}
        
        for result in successful_results:
            for field, value in result['invoice_fields'].items():
                if field not in field_counts:
                    field_counts[field] = 0
                    all_fields[field] = []
                field_counts[field] += 1
                all_fields[field].append(value)
        
        total_invoices = len(successful_results)
        print(f"Field extraction success rates:")

        # Calculate success % per field
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_invoices * 100
            print(f"  {field:15}: {count:3d} invoices ({percentage:5.1f}%)")
        
        # Data Quality Insights
        print(f"\nDATA QUALITY INSIGHTS")
        print(f"{'='*50}")
        
        if 'total_amount' in all_fields:
            amounts = []
            for amount_str in all_fields['total_amount']:
                try:
                    # Clean and convert to float
                    clean_amount = re.sub(r'[,$]', '', amount_str)
                    amount = float(clean_amount)
                    amounts.append(amount)
                except:
                    pass
            
            if amounts:
                print(f"Total amount statistics:")
                print(f"  Count: {len(amounts)} valid amounts")
                print(f"  Average: ${np.mean(amounts):.2f}")
                print(f"  Median: ${np.median(amounts):.2f}")
                print(f"  Range: ${min(amounts):.2f} - ${max(amounts):.2f}")
                print(f"  Total value: ${sum(amounts):,.2f}")
        
        if 'invoice_number' in all_fields:
            inv_numbers = all_fields['invoice_number']
            print(f"\nInvoice number patterns:")
            print(f"  Total extracted: {len(inv_numbers)}")
            
            # Analyze patterns
            numeric_only = [n for n in inv_numbers if n.isdigit()]
            alphanumeric = [n for n in inv_numbers if not n.isdigit()]
            
            print(f"  Numeric only: {len(numeric_only)} ({len(numeric_only)/len(inv_numbers)*100:.1f}%)")
            print(f"  Alphanumeric: {len(alphanumeric)} ({len(alphanumeric)/len(inv_numbers)*100:.1f}%)")
            
            if numeric_only:
                lengths = [len(n) for n in numeric_only]
                print(f"  Numeric length range: {min(lengths)} - {max(lengths)} digits")
        
        # Create visualizations
        self._create_analysis_plots(successful_results, all_fields)
        
        # Data Export Summary
        print(f"\nEXPORTED DATA FILES")
        print(f"{'='*50}")
        csv_files = list(self.output_dir.glob('*.csv'))
        for csv_file in csv_files:
            file_size = csv_file.stat().st_size / 1024  # KB
            print(f"  {csv_file.name}: {file_size:.1f} KB")
        
        return {
            'total_processed': len(self.full_results),
            'successful': len(successful_results),
            'field_extraction_rates': {k: v/total_invoices for k, v in field_counts.items()},
            'avg_confidence': np.mean(confidences),
            'avg_words': np.mean(word_counts)
        }
    
    def _create_analysis_plots(self, successful_results, all_fields):
        """
        Create diagnostic plots for OCR quality and invoice field extraction.

        Parameters
        ----------
        successful_results : list[dict]
            List of successful per-image OCR/extraction results.
        all_fields : dict
            Dictionary mapping field names to lists of extracted field values.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Invoice Processing Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: OCR Confidence Distribution
        confidences = [r['avg_confidence'] for r in successful_results]
        axes[0, 0].hist(confidences, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('OCR Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidences):.1f}%')
        axes[0, 0].legend()
        
        # Plot 2: Words per Invoice Distribution
        word_counts = [r['total_words'] for r in successful_results]
        axes[0, 1].hist(word_counts, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Words per Invoice Distribution')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(word_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(word_counts):.1f}')
        axes[0, 1].legend()
        
        # Plot 3: Field Extraction Success Rates
        field_counts = {}
        for result in successful_results:
            for field in result['invoice_fields'].keys():
                field_counts[field] = field_counts.get(field, 0) + 1
        
        if field_counts:
            fields = list(field_counts.keys())
            rates = [field_counts[f]/len(successful_results)*100 for f in fields]
            
            bars = axes[1, 0].bar(range(len(fields)), rates, color='orange', alpha=0.7)
            axes[1, 0].set_title('Field Extraction Success Rates')
            axes[1, 0].set_xlabel('Fields')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_xticks(range(len(fields)))
            axes[1, 0].set_xticklabels(fields, rotation=45, ha='right')
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{rate:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Amount Distribution (if available)
        if 'total_amount' in all_fields:
            amounts = []
            for amount_str in all_fields['total_amount']:
                try:
                    clean_amount = re.sub(r'[,$]', '', amount_str)
                    amount = float(clean_amount)
                    if amount > 0 and amount < 10000:  # Filter reasonable amounts
                        amounts.append(amount)
                except:
                    pass
            
            if amounts:
                axes[1, 1].hist(amounts, bins=15, color='purple', alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Invoice Amount Distribution')
                axes[1, 1].set_xlabel('Amount ($)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(np.mean(amounts), color='red', linestyle='--',
                                  label=f'Mean: ${np.mean(amounts):.2f}')
                axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No amount data\navailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Invoice Amount Distribution')
        
        plt.tight_layout()
        plt.show()
    
    

    def visualize_sample_results(self, n_samples=3):
        """
        Visualize a small set of sample OCR and extraction results.

        Parameters
        ----------
        n_samples : int, optional
            Number of successful invoice examples to display.
        """
        if not hasattr(self, 'full_results'):
            print("No results to visualize. Run process_dataset first.")
            return
        
        successful_results = [r for r in self.full_results if r['success']][:n_samples]
        
        for i, result in enumerate(successful_results):
            print(f"\n{'='*60}")
            print(f"Sample {i+1}: {result['filename']}")
            print(f"{'='*60}")
            
            # Show basic stats
            print(f"Total words detected: {result['total_words']}")
            print(f"Average confidence: {result['avg_confidence']:.1f}%")
            
            # Show extracted invoice fields
            if result['invoice_fields']:
                print("\nExtracted Invoice Fields:")
                for field, value in result['invoice_fields'].items():
                    print(f"  {field}: {value}")
            
            # Show sample text (first 10 words)
            if result['extracted_text']:
                print(f"\nSample extracted text (first 10 words):")
                sample_text = ' '.join([item['text'] for item in result['extracted_text'][:10]])
                print(f"  {sample_text}...")
            
            # Visualize on image
            if Path(result['image_path']).exists():
                self.visualize_text_extraction(result['image_path'], result)

    def debug_end_to_end(self, processed_images_df, ground_truth_df, n_samples=5):
        """
        Debug pipeline on a small sample of invoices and print full intermediate outputs.
        """

        # --- Filter valid images ---
        df = processed_images_df[
            (processed_images_df["status"] == "success") &
            (processed_images_df["processed_path"].notnull())
        ]

        sample_df = df.sample(min(n_samples, len(df)), random_state=42)

        for _, row in sample_df.iterrows():
            image_path = row["processed_path"]
            filename = Path(image_path).name

            print("\n" + "="*100)
            print(f"DEBUGGING FILE: {filename}")
            print("="*100)

            # --- Load image ---
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Could not load image")
                continue

            # =========================================================
            # 1. OCR WORDS (USING PIPELINE METHOD)
            # =========================================================
            extracted_text = self._ocr_words(image, confidence_threshold=30, psm=6)

            print(f"\nOCR WORDS ({len(extracted_text)} words):")
            for t in extracted_text[:50]:
                print(f"{t['text']:20} | conf={int(t['confidence']):3} | bbox={t['bbox']}")
            if len(extracted_text) > 50:
                print(f"... ({len(extracted_text)-50} more words)")

            # =========================================================
            # 2. REGION ASSIGNMENT
            # =========================================================
            regions = self.assign_regions(extracted_text, image.shape)

            print("\nREGION WORD COUNTS:")
            for region, words in regions.items():
                print(f"{region:10}: {len(words)} words")

            # --- WORDS PER REGION ---
            print("\nWORDS BY REGION:")
            for region, words in regions.items():
                print(f"\n--- {region.upper()} ---")
                preview = " ".join([w["text"] for w in words[:30]])
                print(preview if preview else "[EMPTY]")
                if len(words) > 30:
                    print(f"... ({len(words)-30} more words)")

            # =========================================================
            # 3. REGION TEXT RECONSTRUCTION
            # =========================================================
            print("\nRECONSTRUCTED REGION TEXT:")
            region_texts = {}

            for region, words in regions.items():
                text_block = self.region_to_text(words)
                region_texts[region] = text_block

                print(f"\n--- {region.upper()} ---")
                print(text_block if text_block else "[EMPTY]")

            # =========================================================
            # 4. LABEL-BASED SELLER / CLIENT EXTRACTION DEBUG
            # =========================================================
            print("\nSELLER / CLIENT BLOCK OCR DEBUG:")

            seller_name, seller_block = self.extract_party_name(image, extracted_text, party="seller")
            client_name, client_block = self.extract_party_name(image, extracted_text, party="client")

            print("\n--- SELLER BLOCK OCR TEXT ---")
            print(seller_block if seller_block else "[EMPTY]")

            print("\n--- CLIENT BLOCK OCR TEXT ---")
            print(client_block if client_block else "[EMPTY]")

            # =========================================================
            # 5. FIELD EXTRACTION (FULL PIPELINE)
            # =========================================================
            fields = self.extract_invoice_fields_region_aware(
                image,
                extracted_text,
                regions
            )

            print("\nEXTRACTED FIELDS:")
            print(f"Invoice Number : {fields.get('invoice_number')}")
            print(f"Invoice Date   : {fields.get('invoice_date')}")
            print(f"Seller Name    : {fields.get('seller_name')}")
            print(f"Client Name    : {fields.get('client_name')}")
            print(f"Total Amount   : {fields.get('total_amount')}")
            print(f"Tax            : {fields.get('tax')}")

            # =========================================================
            # 6. GROUND TRUTH COMPARISON
            # =========================================================
            gt_row = ground_truth_df[
                ground_truth_df["processed_path"] == image_path
            ]

            if not gt_row.empty:
                gt_row = gt_row.iloc[0]

                print("\nGROUND TRUTH:")
                print(f"Invoice Number : {gt_row.get('invoice_number')}")
                print(f"Invoice Date   : {gt_row.get('invoice_date')}")
                print(f"Seller Name    : {gt_row.get('seller_name')}")
                print(f"Client Name    : {gt_row.get('client_name')}")
                print(f"Total Amount   : {gt_row.get('total_amount')}")
                print(f"Tax            : {gt_row.get('tax')}")

            else:
                print("\nNo ground truth match found!")

            print("\n" + "="*100 + "\n")