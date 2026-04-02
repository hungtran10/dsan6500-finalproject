from __future__ import annotations
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

class PytesseractInvoiceTextDetector:
    """
    Text detection and OCR extraction for preprocessed invoice images
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-$€£¥₹()[] '

    def assign_regions(self, extracted_text, image_shape):
        """
        Assign OCR words to coarse invoice layout regions based on position.

        Parameters
        ----------
        extracted_text : list[dict]
            OCR word records, each containing at least a "bbox" key in (x, y, w, h) format.
        image_shape : tuple
            Shape of the image array, typically (height, width) or (height, width, channels).

        Returns
        -------
        dict
            Dictionary of region names mapped to lists of OCR word records.
            Regions include: top_left, seller, client, table, and bottom.
        """
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
        """
        Reconstruct text from OCR words within a region by grouping words into lines.

        Parameters
        ----------
        region_words : list[dict]
            OCR word records for a single region.

        Returns
        -------
        str
            Reconstructed multi-line text for the region, or an empty string if no words are provided.
        """
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
        Run Tesseract OCR on a grayscale image and return filtered word-level results.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale image array to OCR.
        confidence_threshold : int, optional
            Minimum confidence score required to keep a detected word.
        psm : int, optional
            Tesseract page segmentation mode.
        extra_config : str, optional
            Additional Tesseract configuration string.

        Returns
        -------
        list[dict]
            List of OCR word records containing text, confidence, bounding box, and OCR indices.
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
        Group OCR words into approximate text lines using vertical proximity.

        Parameters
        ----------
        words : list[dict]
            OCR word records, each containing a "bbox" key.
        y_tol : int or None, optional
            Maximum vertical distance between words to consider them part of the same line.
            If None, a data-driven tolerance is computed from median word height.

        Returns
        -------
        list[dict]
            List of line groups, where each group contains a y-position and the words assigned to that line.
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
        Extract tax, net_worth, and total_amount from the bottom region.

        Preference order:
        1) lines containing '$'
        2) lines containing summary-like labels
        3) arithmetic consistency: net_worth + tax ≈ total_amount

        Rule:
        - smallest numeric summary value -> tax
        - middle value                  -> net_worth
        - largest value                  -> total_amount

        Parameters
        ----------
        bottom_words : list[dict]
            OCR word records from the bottom region of the invoice.

        Returns
        -------
        dict
            Dictionary containing any of the following keys when detected:
            - tax
            - net_worth
            - total_amount
        """
        if not bottom_words:
            return {}

        lines = self._cluster_words_by_line(bottom_words)
        line_texts = []
        for line in lines:
            txt = " ".join(w["text"] for w in sorted(line["words"], key=lambda x: x["bbox"][0]))
            line_texts.append((txt, line["words"]))

        money_pattern = re.compile(r"(?<!\w)\d[\d\s.,]*[.,]\d{2}(?!\w)")

        def line_amounts(txt):
            amounts = []
            for m in money_pattern.finditer(txt):
                norm = self._normalize_money(m.group(0))
                if norm is not None:
                    amounts.append(float(norm))
            return amounts

        def looks_like_summary(txt):
            low = txt.lower()
            return ("summary" in low) or ("vat" in low) or ("gross" in low)

        def score_candidate(txt, amounts):
            score = 0

            # Prefer lines with $
            if "$" in txt:
                score += 5

            # Prefer summary-like lines
            if looks_like_summary(txt):
                score += 3

            # Prefer lines with at least 3 numeric values
            score += min(len(amounts), 5)

            # Extra bonus if the line has the exact expected shape
            if len(amounts) >= 3:
                svals = sorted(set(amounts))
                if len(svals) >= 3:
                    tax, net_worth, total = svals[0], svals[1], svals[-1]
                    if abs((net_worth + tax) - total) <= max(0.05 * total, 0.10):
                        score += 5

            return score

        candidates = []
        for txt, _ in line_texts:
            amounts = line_amounts(txt)
            if len(amounts) >= 2:
                candidates.append((score_candidate(txt, amounts), txt, amounts))

        # Choose best candidate if possible
        candidate_amounts = None
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_txt, best_amounts = candidates[0]
            if len(best_amounts) >= 2:
                candidate_amounts = best_amounts

        result = {}

        def assign_from_values(vals):
            vals = sorted(set(vals))
            if len(vals) >= 3:
                tax = vals[0]
                net_worth = vals[1]
                total_amount = vals[-1]
                return {
                    "tax": f"{tax:.2f}",
                    "net_worth": f"{net_worth:.2f}",
                    "total_amount": f"{total_amount:.2f}",
                }
            if len(vals) == 2:
                return {
                    "tax": f"{min(vals):.2f}",
                    "total_amount": f"{max(vals):.2f}",
                }
            return {}

        # First pass: best candidate line
        if candidate_amounts and len(candidate_amounts) >= 2:
            result = assign_from_values(candidate_amounts)

            # If we got 3 values, validate net_worth + tax ≈ total_amount
            if "tax" in result and "net_worth" in result and "total_amount" in result:
                tax = float(result["tax"])
                net_worth = float(result["net_worth"])
                total_amount = float(result["total_amount"])

                if abs((tax + net_worth) - total_amount) <= max(0.05 * total_amount, 0.10):
                    return result

        # Second pass: any line with 3+ amounts
        for txt, _ in line_texts:
            amounts = line_amounts(txt)
            if len(amounts) >= 3:
                candidate = assign_from_values(amounts)
                if candidate.get("tax") and candidate.get("net_worth") and candidate.get("total_amount"):
                    tax = float(candidate["tax"])
                    net_worth = float(candidate["net_worth"])
                    total_amount = float(candidate["total_amount"])
                    if abs((tax + net_worth) - total_amount) <= max(0.05 * total_amount, 0.10):
                        return candidate

        # Final fallback: use all bottom-region monetary tokens
        all_amounts = []
        for txt, _ in line_texts:
            for m in money_pattern.finditer(txt):
                norm = self._normalize_money(m.group(0))
                if norm is not None:
                    all_amounts.append(float(norm))

        if all_amounts:
            vals = sorted(set(all_amounts))
            if len(vals) >= 3:
                result["tax"] = f"{vals[0]:.2f}"
                result["net_worth"] = f"{vals[1]:.2f}"
                result["total_amount"] = f"{vals[-1]:.2f}"
            elif len(vals) == 2:
                result["tax"] = f"{vals[0]:.2f}"
                result["total_amount"] = f"{vals[1]:.2f}"

        return result
    
    def _find_label_word(self, extracted_text, label_pattern):
        """
        Find the first OCR word matching a label pattern such as Seller or Client.

        Parameters
        ----------
        extracted_text : list[dict]
            OCR word records for the full page.
        label_pattern : str
            Regular expression pattern used to identify the label word.

        Returns
        -------
        dict or None
            The first matching OCR word record, or None if no match is found.
        """
        for item in extracted_text:
            if re.search(label_pattern, item["text"], re.IGNORECASE):
                return item
        return None
    
    def _normalize_date(self, value):
        """
        Convert dates like 12/20/2014 or 12-20-14 to YYYY-MM-DD.

        Parameters
        ----------
        value : any
            Raw date value extracted from OCR or ground truth.

        Returns
        -------
        str or None
            Normalized date string in YYYY-MM-DD format, or None if parsing fails.
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

        Parameters
        ----------
        value : any
            Raw money value extracted from OCR or ground truth.

        Returns
        -------
        str or None
            Normalized numeric string such as '1140.24', or None if parsing fails.
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

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale invoice image.
        extracted_text : list[dict]
            OCR word records from the full page.
        party : str, optional
            Party to extract; expected values are "seller" or "client".

        Returns
        -------
        tuple or None
            A tuple of (extracted_name, crop_ocr_text) if a name is found, otherwise (None, crop_ocr_text).
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
        """
        Select the most likely seller or client name from OCR text produced by a cropped region.

        Parameters
        ----------
        block_text : str
            OCR text extracted from the cropped seller/client block.
        party : str
            Party being extracted; expected values are "seller" or "client".

        Returns
        -------
        str or None
            The extracted party name, or None if no plausible name line is found.
        """

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
        """
        Return the first regex capture match found in a text string.

        Parameters
        ----------
        text : str
            Text to search.
        patterns : list[str]
            Ordered list of regular expression patterns to try.

        Returns
        -------
        str or None
            The first matched capture group, or None if no pattern matches.
        """
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return None

    def extract_invoice_fields_region_aware(self, image, extracted_text, regions):
        """
        Extract structured invoice fields using layout-aware OCR regions.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale invoice image.
        extracted_text : list[dict]
            OCR word records from the full page.
        regions : dict
            Region dictionary produced by assign_regions().

        Returns
        -------
        dict
            Dictionary of extracted invoice fields such as:
            - invoice_number
            - invoice_date
            - seller_name
            - client_name
            - tax
            - net_worth
            - total_amount
        """
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
    
    def extract_table_dataframe(self, table_words):
        """
        Parse the invoice line-item table into a structured pandas DataFrame.

        Parameters
        ----------
        table_words : list[dict]
            OCR word records assigned to the table region.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing line-item rows and parsed columns such as item number,
            description, quantity, unit, net price, net worth, VAT percentage, and gross worth.
        """
        if not table_words:
            return pd.DataFrame()

        lines = self._cluster_words_by_line(table_words)
        if not lines:
            return pd.DataFrame()

        # Find the header line
        header_idx = None
        for i, line in enumerate(lines):
            text = " ".join(w["text"].lower() for w in line["words"])
            if "description" in text and "qty" in text and "um" in text and "vat" in text and "gross" in text:
                header_idx = i
                break

        if header_idx is None:
            # Fallback: return raw line text if header detection fails
            raw_rows = []
            for i, line in enumerate(lines):
                raw_rows.append({
                    "row_num": i,
                    "raw_text": " ".join(w["text"] for w in sorted(line["words"], key=lambda x: x["bbox"][0]))
                })
            return pd.DataFrame(raw_rows)

        header_words = sorted(lines[header_idx]["words"], key=lambda x: x["bbox"][0])

        def find_x_by_token(token_pattern, start_idx=0):
            for w in header_words[start_idx:]:
                if re.fullmatch(token_pattern, w["text"].strip(), re.IGNORECASE):
                    return w["bbox"][0]
            return None

        # Header x positions
        item_no_x = find_x_by_token(r"no\.?")
        desc_x = find_x_by_token(r"description")
        qty_x = find_x_by_token(r"qty")
        um_x = find_x_by_token(r"um")

        # After UM, headers often appear as Net price / Net worth / VAT / Gross worth
        net_price_x = None
        net_worth_x = None
        vat_x = None
        gross_x = None

        after_um = False
        net_seen = 0
        for w in header_words:
            txt = w["text"].strip().lower()
            if um_x is not None and w["bbox"][0] >= um_x:
                after_um = True

            if after_um and txt == "net":
                net_seen += 1
                if net_seen == 1 and net_price_x is None:
                    net_price_x = w["bbox"][0]
                elif net_seen == 2 and net_worth_x is None:
                    net_worth_x = w["bbox"][0]
            elif after_um and txt == "vat" and vat_x is None:
                vat_x = w["bbox"][0]
            elif after_um and txt == "gross" and gross_x is None:
                gross_x = w["bbox"][0]

        # Fallback boundaries if any header token was not found
        xs = [x for x in [item_no_x, desc_x, qty_x, um_x, net_price_x, net_worth_x, vat_x, gross_x] if x is not None]
        if len(xs) < 4:
            raw_rows = []
            for i, line in enumerate(lines[header_idx + 1:], start=1):
                raw_rows.append({
                    "row_num": i,
                    "raw_text": " ".join(w["text"] for w in sorted(line["words"], key=lambda x: x["bbox"][0]))
                })
            return pd.DataFrame(raw_rows)

        # Use the detected header x positions to create bins
        col_names = ["item_no", "description", "qty", "um", "net_price", "net_worth", "vat_pct", "gross_worth"]
        col_starts = [item_no_x, desc_x, qty_x, um_x, net_price_x, net_worth_x, vat_x, gross_x]

        # Keep only columns with valid starts
        cols = [(name, x) for name, x in zip(col_names, col_starts) if x is not None]
        cols = sorted(cols, key=lambda z: z[1])

        if len(cols) < 4:
            return pd.DataFrame()

        boundaries = []
        for i in range(len(cols) - 1):
            boundaries.append((cols[i][1] + cols[i + 1][1]) / 2)

        def assign_col(x_center):
            for i, boundary in enumerate(boundaries):
                if x_center < boundary:
                    return cols[i][0]
            return cols[-1][0]

        data_rows = []
        for row_idx, line in enumerate(lines[header_idx + 1:], start=1):
            words = sorted(line["words"], key=lambda x: x["bbox"][0])
            if not words:
                continue

            row_dict = {name: "" for name, _ in cols}
            row_dict["row_num"] = row_idx

            raw_text = " ".join(w["text"] for w in words)
            row_dict["raw_text"] = raw_text

            for w in words:
                x, y, bw, bh = w["bbox"]
                x_center = x + bw / 2
                col = assign_col(x_center)
                row_dict[col] = (row_dict[col] + " " + w["text"]).strip()

            data_rows.append(row_dict)

        df = pd.DataFrame(data_rows)

        # Optional cleanup of numeric columns
        for col in ["qty", "net_price", "net_worth", "vat_pct", "gross_worth"]:
            if col in df.columns:
                df[col] = df[col].str.replace(" ", "", regex=False)

        return df
    
    def process_single_image(self, image_path):
        """
        Run OCR, region assignment, field extraction, and table parsing for one invoice image.

        Parameters
        ----------
        image_path : str or Path
            Path to the preprocessed invoice image.

        Returns
        -------
        dict
            Per-image result dictionary containing image metadata, OCR words, extracted fields,
            parsed table data, and processing success status.
        """
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
        """
        Process a dataset of preprocessed invoice images and save aggregated OCR outputs.

        Parameters
        ----------
        processed_images_df : pandas.DataFrame
            DataFrame containing at least status and processed_path columns.
        batch_size : int, optional
            Number of images to process per batch.
        save_word_level : bool, optional
            Whether to save word-level OCR output to CSV.
        sample_frac : float or None, optional
            Fraction of successful images to sample for processing.
        random_state : int, optional
            Random seed used when sampling.

        Returns
        -------
        pandas.DataFrame
            Summary DataFrame with one row per processed image.
        """

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
        Find the vertical position of the first matching anchor word.

        Parameters
        ----------
        extracted_text : list[dict]
            OCR word records for the full page.
        pattern : str
            Regular expression pattern used to find the anchor word.

        Returns
        -------
        int or None
            The smallest y-position among matching words, or None if no match is found.
        """
        ys = [
            item["bbox"][1]
            for item in extracted_text
            if re.fullmatch(pattern, item["text"].strip(), re.IGNORECASE)
        ]
        return min(ys) if ys else None
    
    def evaluate_against_ground_truth(self, ground_truth_df, merge_key="processed_file", restrict_to_matched=True):
        """
        Compare extracted invoice fields against ground-truth values using exact-match metrics.

        Parameters
        ----------
        ground_truth_df : pandas.DataFrame
            Ground-truth dataset containing the labeled invoice fields.
        merge_key : str, optional
            Column used to align predictions with ground truth.
        restrict_to_matched : bool, optional
            Whether to score only rows present in both predictions and ground truth.

        Returns
        -------
        tuple
            A tuple of (metrics_df, overall_metrics), where metrics_df contains per-field
            accuracy, precision, recall, and F1, and overall_metrics contains micro-averaged scores.
        """
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

            if field in ["total_amount", "tax", "net_worth"]:
                norm = self._normalize_money(s)
                return norm if norm is not None else np.nan

            if field == "invoice_date":
                norm = self._normalize_date(s)
                return norm if norm is not None else np.nan

            s = re.sub(r"\s+", " ", s)
            return s.lower()

        fields = [
            "invoice_number",
            "invoice_date",
            "seller_name",
            "client_name",
            "net_worth",
            "total_amount",
            "tax"
        ]

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
        Print aggregate OCR and field extraction statistics for a batch of processed images.

        Parameters
        ----------
        results : list[dict]
            List of per-image result dictionaries returned by process_single_image().

        Returns
        -------
        None
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
        Visualize OCR word boxes and extracted fields for a single invoice image.

        Parameters
        ----------
        image_path : str or Path
            Path to the invoice image.
        result : dict
            Result dictionary returned by process_single_image().

        Returns
        -------
        None
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

    def debug_end_to_end(self, processed_images_df, ground_truth_df, sample_frac=None, n_samples=5, random_state=42):
        """
        Print end-to-end debugging output for a small sample of invoices.

        Parameters
        ----------
        processed_images_df : pandas.DataFrame
            DataFrame containing processed invoice image paths and status values.
        ground_truth_df : pandas.DataFrame
            Ground-truth labels for the same invoices.
        sample_frac : float or None, optional
            Fraction of eligible rows to sample for debugging.
        n_samples : int, optional
            Number of samples to inspect when sample_frac is not provided.
        random_state : int, optional
            Random seed used when sampling.

        Returns
        -------
        None
        """

        df = processed_images_df[
            (processed_images_df["status"] == "success") &
            (processed_images_df["processed_path"].notnull())
        ]

        if sample_frac is not None:
            sample_df = df.sample(frac=sample_frac, random_state=random_state)
            print(f"Debugging {len(sample_df)} sampled invoices ({sample_frac*100:.1f}%)")
        else:
            sample_df = df.sample(min(n_samples, len(df)), random_state=random_state)
            print(f"Debugging {len(sample_df)} invoices")

        for _, row in sample_df.iterrows():
            image_path = row["processed_path"]
            filename = Path(image_path).name

            print("\n" + "=" * 100)
            print(f"DEBUGGING FILE: {filename}")
            print("=" * 100)

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Could not load image")
                continue

            extracted_text = self._ocr_words(image, confidence_threshold=30, psm=6)

            print(f"\nOCR WORDS ({len(extracted_text)} words):")
            for t in extracted_text[:50]:
                print(f"{t['text']:20} | conf={int(t['confidence']):3} | bbox={t['bbox']}")
            if len(extracted_text) > 50:
                print(f"... ({len(extracted_text)-50} more words)")

            regions = self.assign_regions(extracted_text, image.shape)

            print("\nREGION WORD COUNTS:")
            for region, words in regions.items():
                print(f"{region:10}: {len(words)} words")

            print("\nWORDS BY REGION:")
            for region, words in regions.items():
                print(f"\n--- {region.upper()} ---")
                preview = " ".join([w["text"] for w in words[:30]])
                print(preview if preview else "[EMPTY]")
                if len(words) > 30:
                    print(f"... ({len(words)-30} more words)")

            print("\nRECONSTRUCTED REGION TEXT:")
            for region, words in regions.items():
                print(f"\n--- {region.upper()} ---")
                text_block = self.region_to_text(words)
                print(text_block if text_block else "[EMPTY]")

            seller_name, seller_block = self.extract_party_name(image, extracted_text, party="seller")
            client_name, client_block = self.extract_party_name(image, extracted_text, party="client")
            bottom_fields = self.extract_bottom_totals(regions["bottom"])

            fields = self.extract_invoice_fields_region_aware(image, extracted_text, regions)
            fields.update(bottom_fields)

            print("\nSELLER / CLIENT BLOCK OCR DEBUG:")
            print("\n--- SELLER BLOCK OCR TEXT ---")
            print(seller_block if seller_block else "[EMPTY]")

            print("\n--- CLIENT BLOCK OCR TEXT ---")
            print(client_block if client_block else "[EMPTY]")

            print("\nEXTRACTED FIELDS:")
            print(f"Invoice Number : {fields.get('invoice_number')}")
            print(f"Invoice Date   : {fields.get('invoice_date')}")
            print(f"Seller Name    : {fields.get('seller_name')}")
            print(f"Client Name    : {fields.get('client_name')}")
            print(f"Net Worth      : {fields.get('net_worth')}")
            print(f"Total Amount   : {fields.get('total_amount')}")
            print(f"Tax            : {fields.get('tax')}")

            gt_row = ground_truth_df[ground_truth_df["processed_path"] == image_path]

            if not gt_row.empty:
                gt_row = gt_row.iloc[0]

                print("\nGROUND TRUTH:")
                print(f"Invoice Number : {gt_row.get('invoice_number')}")
                print(f"Invoice Date   : {gt_row.get('invoice_date')}")
                print(f"Seller Name    : {gt_row.get('seller_name')}")
                print(f"Client Name    : {gt_row.get('client_name')}")
                print(f"Net Worth      : {gt_row.get('net_worth')}")
                print(f"Total Amount   : {gt_row.get('total_amount')}")
                print(f"Tax            : {gt_row.get('tax')}")
            else:
                print("\nNo ground truth match found!")

            print("\n" + "=" * 100 + "\n")

DEFAULT_FIELDS = [
    "invoice_number",
    "invoice_date",
    "seller_name",
    "client_name",
    "tax",
    "net_worth",
    "total_amount",
]


def _get_successful_results(results):
    return [r for r in results if r.get("success")]


def _field_display_name(field):
    rename_map = {
        "vendor_name": "seller_name",
    }
    return rename_map.get(field, field)


def _field_extraction_rates(results, fields=DEFAULT_FIELDS):
    successful = _get_successful_results(results)
    n = len(successful)
    if n == 0:
        return {f: 0.0 for f in fields}

    rates = {}
    for field in fields:
        count = 0
        for r in successful:
            invoice_fields = r.get("invoice_fields", {})
            if field in invoice_fields and invoice_fields[field] not in [None, "", np.nan]:
                count += 1
            elif field == "seller_name" and "vendor_name" in invoice_fields and invoice_fields["vendor_name"] not in [None, "", np.nan]:
                count += 1
        rates[field] = count / n
    return rates


def _field_accuracies(metrics_df, fields=DEFAULT_FIELDS):
    if metrics_df is None or metrics_df.empty:
        return {f: np.nan for f in fields}

    df = metrics_df.copy()
    df["field"] = df["field"].replace({"vendor_name": "seller_name"})

    acc = {}
    for field in fields:
        row = df[df["field"] == field]
        acc[field] = float(row["accuracy"].iloc[0]) if not row.empty else np.nan
    return acc


def _field_outcome_counts(metrics_df, fields=DEFAULT_FIELDS):
    """
    Returns counts needed for stacked bars:
    correct, incorrect, missing_pred
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(index=fields, columns=["correct", "incorrect", "missing_pred"]).fillna(0)

    df = metrics_df.copy()
    df["field"] = df["field"].replace({"vendor_name": "seller_name"})
    df = df.set_index("field")

    rows = []
    for field in fields:
        if field in df.index:
            gt_count = float(df.loc[field, "ground_truth_count"])
            pred_count = float(df.loc[field, "predicted_count"])
            correct = float(df.loc[field, "correct"])
            incorrect = max(pred_count - correct, 0.0)
            missing_pred = max(gt_count - pred_count, 0.0)
        else:
            correct = 0.0
            incorrect = 0.0
            missing_pred = 0.0

        rows.append({
            "field": field,
            "correct": correct,
            "incorrect": incorrect,
            "missing_pred": missing_pred
        })

    return pd.DataFrame(rows).set_index("field")

def create_analysis_dashboard(results, metrics_df=None, fields=DEFAULT_FIELDS, title="Invoice Processing Analysis Dashboard",
    save_path=None,show=True):
    """
    Standalone dashboard function.
    Works with any pipeline that returns:
      - results: list[dict] with keys like success, total_words, avg_confidence, invoice_fields
      - metrics_df: optional evaluation dataframe from evaluate_against_ground_truth()
    """
    successful = _get_successful_results(results)
    failed = [r for r in results if not r.get("success")]

    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    print(f"\nPROCESSING OVERVIEW")
    print(f"{'='*50}")
    print(f"Total images processed: {len(results):,}")
    print(f"Successful extractions: {len(successful):,} ({(len(successful)/len(results)*100 if results else 0):.1f}%)")
    print(f"Failed extractions: {len(failed):,} ({(len(failed)/len(results)*100 if results else 0):.1f}%)")

    if not successful:
        print("No successful results to analyze.")
        return None

    word_counts = [r.get("total_words", 0) for r in successful]
    confidences = [r.get("avg_confidence", 0) for r in successful]

    rates = _field_extraction_rates(results, fields=fields)
    acc = _field_accuracies(metrics_df, fields=fields) if metrics_df is not None else {f: np.nan for f in fields}
    outcome_df = _field_outcome_counts(metrics_df, fields=fields)

    print(f"\nOCR QUALITY METRICS")
    print(f"{'='*50}")
    print(f"Average words per invoice: {np.mean(word_counts):.1f}")
    print(f"Median words per invoice: {np.median(word_counts):.0f}")
    print(f"Word count range: {min(word_counts)} - {max(word_counts)}")
    print(f"Average OCR confidence: {np.mean(confidences):.1f}%")
    print(f"Median OCR confidence: {np.median(confidences):.1f}%")
    print(f"Confidence range: {min(confidences):.1f}% - {max(confidences):.1f}%")

    print(f"\nFIELD EXTRACTION SUCCESS RATES")
    print(f"{'='*50}")
    for field in fields:
        print(f"  {field:15}: {rates[field]*100:5.1f}%")

    if metrics_df is not None and not metrics_df.empty:
        print(f"\nFIELD-LEVEL EXACT MATCH ACCURACIES")
        print(f"{'='*50}")
        for field in fields:
            if pd.notna(acc[field]):
                print(f"  {field:15}: {acc[field]*100:5.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1) OCR confidence distribution
    axes[0, 0].hist(confidences, bins=20, alpha=0.8, edgecolor="black")
    axes[0, 0].set_title("OCR Confidence Distribution")
    axes[0, 0].set_xlabel("Confidence (%)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(np.mean(confidences), linestyle="--", label=f"Mean: {np.mean(confidences):.1f}%")
    axes[0, 0].legend()

    # 2) Accuracy for the requested fields
    if metrics_df is not None and not metrics_df.empty:
        acc_vals = [acc[field] for field in fields]
        bars = axes[0, 1].bar(range(len(fields)), acc_vals, alpha=0.85, edgecolor="black")
        axes[0, 1].set_title("Field-Level Exact Match Accuracy")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_xticks(range(len(fields)))
        axes[0, 1].set_xticklabels(fields, rotation=45, ha="right")
        axes[0, 1].set_ylim(0, 1.05)

        for bar, v in zip(bars, acc_vals):
            if pd.notna(v):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v*100:.1f}%",
                                ha="center", va="bottom", fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, "No evaluation metrics\na vailable", ha="center", va="center",
                        transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title("Field-Level Exact Match Accuracy")

    # 3) Field extraction success rates
    rates_vals = [rates[field] * 100 for field in fields]
    bars = axes[1, 0].bar(range(len(fields)), rates_vals, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Field Extraction Success Rates")
    axes[1, 0].set_ylabel("Success Rate (%)")
    axes[1, 0].set_xticks(range(len(fields)))
    axes[1, 0].set_xticklabels(fields, rotation=45, ha="right")
    axes[1, 0].set_ylim(0, 105)

    for bar, v in zip(bars, rates_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    # 4) Stacked outcome breakdown
    if metrics_df is not None and not metrics_df.empty:
        x = np.arange(len(fields))
        correct = outcome_df.loc[fields, "correct"].values
        incorrect = outcome_df.loc[fields, "incorrect"].values
        missing_pred = outcome_df.loc[fields, "missing_pred"].values

        axes[1, 1].bar(x, correct, label="Correct", alpha=0.85, edgecolor="black")
        axes[1, 1].bar(x, incorrect, bottom=correct, label="Incorrect", alpha=0.85, edgecolor="black")
        axes[1, 1].bar(x, missing_pred, bottom=correct + incorrect, label="Missing prediction", alpha=0.85, edgecolor="black")

        axes[1, 1].set_title("Prediction Outcome Breakdown")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(fields, rotation=45, ha="right")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No evaluation metrics\navailable", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title("Prediction Outcome Breakdown")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "total_processed": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "field_extraction_rates": rates,
        "field_accuracies": acc,
        "avg_confidence": float(np.mean(confidences)),
        "avg_words": float(np.mean(word_counts)),
    }


def visualize_sample_results(results, visualize_text_fn=None, n_samples=3, title="Sample OCR Results"):
    """
    Standalone sample visualizer.
    `visualize_text_fn` should be a callable like:
        visualize_text_fn(image_path, result_dict)
    """
    successful = _get_successful_results(results)[:n_samples]

    for i, result in enumerate(successful, start=1):
        print(f"\n{'='*60}")
        print(f"Sample {i}: {result.get('filename', 'unknown')}")
        print(f"{'='*60}")

        print(f"Total words detected: {result.get('total_words', 0)}")
        print(f"Average confidence: {result.get('avg_confidence', 0):.1f}%")

        invoice_fields = result.get("invoice_fields", {})
        if invoice_fields:
            print("\nExtracted Invoice Fields:")
            for field, value in invoice_fields.items():
                print(f"  {field}: {value}")

        if result.get("extracted_text"):
            sample_text = " ".join([item["text"] for item in result["extracted_text"][:10]])
            print(f"\nSample extracted text (first 10 words):")
            print(f"  {sample_text}...")

        if visualize_text_fn is not None and result.get("image_path") and Path(result["image_path"]).exists():
            visualize_text_fn(result["image_path"], result)