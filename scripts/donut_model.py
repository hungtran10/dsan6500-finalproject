"""Donut-based invoice extraction pipeline.

This module replaces OCR-first extraction (e.g., Tesseract) with a Donut
(Document Understanding Transformer) pipeline. It is designed to be adaptable
for future document pipelines that return per-image result dictionaries and an
optional evaluation dataframe.

Primary outputs
---------------
- results: list[dict] where each dict may contain keys such as:
  - image_path
  - filename
  - success
  - extracted_text
  - invoice_fields
  - total_words
  - avg_confidence
  - line_items_df

- metrics_df: optional pandas.DataFrame returned by evaluate_against_ground_truth()
  with per-field exact-match metrics.

Notes
-----
Donut is OCR-free and directly maps an image to structured text. In practice,
this pipeline works best with a fine-tuned invoice checkpoint or a prompt
format that the model has seen during training.

Official Hugging Face Donut docs show DonutProcessor + VisionEncoderDecoderModel,
with generation using decoder_input_ids built from a task prompt and then parsing
with processor.token2json().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


DEFAULT_FIELDS = [
    "invoice_number",
    "invoice_date",
    "seller_name",
    "client_name",
    "net_worth",
    "tax",
    "total_amount",
]


@dataclass
class DonutConfig:
    """Configuration for the Donut invoice pipeline."""

    model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    task_prompt: str = "<s_invoice>"
    max_length: int = 512
    device: Optional[str] = None
    use_fp16: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    sample_seed: int = 42


class DonutInvoiceTextDetector:
    """End-to-end invoice extractor powered by Donut."""

    def __init__(self, output_dir: str | Path, config: Optional[DonutConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or DonutConfig()
        self.device = self._resolve_device(self.config.device)
        self.processor, self.model = self._load_model(self.config.model_name)
        self.full_results: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------
    # Model loading / inference
    # ---------------------------------------------------------------------
    def _resolve_device(self, device: Optional[str]) -> torch.device:
        """Resolve the device used for inference."""
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_name: str) -> Tuple[DonutProcessor, VisionEncoderDecoderModel]:
        """Load the Donut processor and model from Hugging Face."""
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        return processor, model

    def _prepare_image(self, image_path: str | Path) -> Image.Image:
        """Load an image as a PIL RGB image."""
        image = Image.open(image_path).convert("RGB")
        return image

    def _build_decoder_input_ids(self) -> torch.Tensor:
        """Build Donut decoder input ids from the configured task prompt."""
        return self.processor.tokenizer(
            self.config.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

    def _generate_sequence(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Run Donut generation and return the raw generated sequence plus scores."""
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self._build_decoder_input_ids()

        gen_kwargs = dict(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.config.max_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        if self.config.num_beams and self.config.num_beams > 1:
            gen_kwargs["num_beams"] = self.config.num_beams
        if self.config.repetition_penalty and self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        if self.config.no_repeat_ngram_size and self.config.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.config.max_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,   # add this
            )

        # Donut docs show: batch_decode -> strip special tokens -> token2json.
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        # Compute a lightweight confidence proxy from generation scores when available.
        avg_confidence = self._sequence_confidence(outputs)

        return {
            "sequence": sequence,
            "avg_confidence": avg_confidence,
            "outputs": outputs,
        }

    def _sequence_confidence(self, outputs) -> float:
        """Estimate average token confidence from generation scores.

        Returns NaN if scores are unavailable or transition-score computation fails.
        """
        try:
            if not hasattr(outputs, "scores") or outputs.scores is None or len(outputs.scores) == 0:
                return float("nan")

            # compute_transition_scores is part of the generation mixin used by HF seq2seq models.
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
            # transition_scores are log-probs; convert to a geometric mean token probability.
            token_probs = torch.exp(transition_scores)
            return float(token_probs.mean().item())
        except Exception:
            return float("nan")

    def _parse_generated_text(self, generated_sequence: str) -> Dict[str, Any]:
        """Parse Donut-generated text into a JSON-like dictionary."""
        cleaned = generated_sequence.strip()

        # Remove common special tokens / task token prefix
        cleaned = cleaned.replace(self.processor.tokenizer.eos_token or "", "")
        cleaned = cleaned.replace(self.processor.tokenizer.pad_token or "", "")
        cleaned = re.sub(r"<.*?>", "", cleaned, count=1).strip()

        try:
            parsed = self.processor.token2json(cleaned)
            if isinstance(parsed, dict):
                return parsed
            return {"raw": parsed}
        except Exception:
            try:
                parsed = json.loads(cleaned)
                return parsed if isinstance(parsed, dict) else {"raw": parsed}
            except Exception:
                return {"raw_text": cleaned}

    def reload_model(self, model_name_or_path: str) -> None:
        """Reload the processor and model from a fine-tuned Donut checkpoint.

        Use this after training with donut_training_utils.train_donut_invoice_model(...)
        to point the inference pipeline at the saved checkpoint.
        """
        self.config.model_name = model_name_or_path
        self.processor = DonutProcessor.from_pretrained(model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    # ---------------------------------------------------------------------
    # Field normalization / evaluation helpers
    # ---------------------------------------------------------------------
    def _normalize_date(self, value: Any) -> Optional[str]:
        """Normalize a date-like value to YYYY-MM-DD."""
        if value is None or pd.isna(value):
            return None
        s = str(value).strip()
        if not s:
            return None
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
        return None

    def _normalize_money(self, value: Any) -> Optional[str]:
        """Normalize a money-like value to a fixed two-decimal string."""
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

    def _first_match(self, text: str, patterns: Sequence[str]) -> Optional[str]:
        """Return the first regex capture match found in the provided text."""
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return None

    def _coerce_field(self, value: Any, field: str) -> Any:
        """Normalize a field value for extraction or evaluation."""
        if pd.isna(value):
            return np.nan

        s = str(value).strip()
        if s.lower() in ["", "nan", "none"]:
            return np.nan

        if field in {"total_amount", "tax", "net_worth"}:
            norm = self._normalize_money(s)
            return norm if norm is not None else np.nan

        if field == "invoice_date":
            norm = self._normalize_date(s)
            return norm if norm is not None else np.nan

        return re.sub(r"\s+", " ", s).lower()

    # ---------------------------------------------------------------------
    # Public extraction methods
    # ---------------------------------------------------------------------
    def process_single_image(self, image_path: str | Path) -> Dict[str, Any]:
        """Run Donut inference on one invoice image and return a structured result."""
        result = {
            "image_path": str(image_path),
            "filename": Path(image_path).name,
            "success": False,
            "extracted_text": "",
            "raw_sequence": "",
            "parsed_payload": {},
            "invoice_fields": {},
            "table_df": pd.DataFrame(),
            "total_words": 0,
            "avg_confidence": float("nan"),
        }

        try:
            pil_image = self._prepare_image(image_path)
            gen = self._generate_sequence(pil_image)
            sequence = gen["sequence"]
            parsed = self._parse_generated_text(sequence)

            invoice_fields = self.extract_invoice_fields_from_json(parsed)
            line_items_df = self.extract_line_items_from_json(parsed)

            result["extracted_text"] = sequence
            result["raw_sequence"] = sequence
            result["parsed_payload"] = parsed
            result["invoice_fields"] = invoice_fields
            result["table_df"] = line_items_df
            result["total_words"] = len(re.findall(r"\w+", sequence))
            result["avg_confidence"] = gen["avg_confidence"]

            # Mark success only if something useful was extracted
            result["success"] = bool(invoice_fields)

            print("RAW DONUT SEQUENCE:", sequence[:500])
            print("PARSED PAYLOAD:", parsed)
            print("EXTRACTED FIELDS:", invoice_fields)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        return result

    def extract_invoice_fields_from_json(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Map Donut output into a flat invoice field dictionary."""
        fields: Dict[str, Any] = {}

        # Support common Donut-style structures and custom invoice schemas.
        candidate_sources = [parsed]
        for key in ["invoice", "header", "subtotal", "total", "summary"]:
            val = parsed.get(key)
            if isinstance(val, dict):
                candidate_sources.append(val)

        # Direct lookups first.
        lookup_keys = {
            "invoice_number": ["invoice_number", "inv_no", "no", "id"],
            "invoice_date": ["invoice_date", "date", "issue_date"],
            "seller_name": ["seller_name", "vendor_name", "supplier_name"],
            "client_name": ["client_name", "buyer_name", "customer_name"],
            "tax": ["tax", "vat", "vat_amount"],
            "net_worth": ["net_worth", "subtotal", "sub_total", "net_amount"],
            "total_amount": ["total_amount", "total", "grand_total", "gross_worth"],
        }

        for out_key, key_variants in lookup_keys.items():
            found = None
            for src in candidate_sources:
                for k in key_variants:
                    if k in src and src[k] not in [None, "", {}]:
                        found = src[k]
                        break
                if found is not None:
                    break

            if found is not None:
                fields[out_key] = found

        # Normalize canonical fields.
        if "invoice_date" in fields:
            norm = self._normalize_date(fields["invoice_date"])
            if norm is not None:
                fields["invoice_date"] = norm

        for money_field in ["tax", "net_worth", "total_amount"]:
            if money_field in fields:
                norm = self._normalize_money(fields[money_field])
                if norm is not None:
                    fields[money_field] = norm

        # If the parsed output contains a line_items-like list, keep it for table export.
        return fields

    def extract_line_items_from_json(self, parsed: Dict[str, Any]) -> pd.DataFrame:
        """Convert Donut line-item output into a DataFrame when available."""
        for key in ["line_items", "items", "menu", "products", "table"]:
            if key in parsed and isinstance(parsed[key], list):
                df = pd.DataFrame(parsed[key])
                if not df.empty:
                    df = df.copy()
                    df["source"] = key
                return df
        return pd.DataFrame()

    def process_dataset(
        self,
        processed_images_df: pd.DataFrame,
        batch_size: int = 10,
        save_word_level: bool = False,
        sample_frac: Optional[float] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Process a dataset of images and save aggregated Donut outputs."""
        successful_images = processed_images_df[processed_images_df["status"] == "success"].copy()

        if sample_frac is not None:
            successful_images = successful_images.sample(frac=sample_frac, random_state=random_state)
            print(f"Processing {len(successful_images)} sampled images ({sample_frac*100:.1f}%)...")
        else:
            print(f"Processing {len(successful_images)} images...")

        results: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        table_rows: List[pd.DataFrame] = []

        for start in range(0, len(successful_images), batch_size):
            batch_df = successful_images.iloc[start : start + batch_size]
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
                    "raw_sequence": result.get("raw_sequence", "")[:250],
                    "parsed_keys": ",".join(result.get("parsed_payload", {}).keys()) if isinstance(result.get("parsed_payload"), dict) else "",
                }
                summary_row.update(result["invoice_fields"])
                summary_rows.append(summary_row)

                if isinstance(result.get("table_df"), pd.DataFrame) and not result["table_df"].empty:
                    df = result["table_df"].copy()
                    df.insert(0, "filename", result["filename"])
                    table_rows.append(df)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.output_dir / "donut_invoice_results.csv", index=False)

        if table_rows:
            table_df = pd.concat(table_rows, ignore_index=True).dropna(how="all")
            table_df.to_csv(self.output_dir / "donut_line_items.csv", index=False)

        self.full_results = results
        self._print_summary(results)
        return summary_df

    def evaluate_against_ground_truth(
        self,
        ground_truth_df: pd.DataFrame,
        merge_key: str = "processed_file",
        restrict_to_matched: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Evaluate predicted invoice fields against ground truth using exact-match metrics."""
        if not hasattr(self, "full_results"):
            raise ValueError("Run process_dataset() first.")

        pred_rows = []
        for r in self.full_results:
            row = {"processed_file": r["filename"]}
            row.update(r.get("invoice_fields", {}))
            pred_rows.append(row)

        pred_df = pd.DataFrame(pred_rows)

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

        merged = ground_truth_df.merge(pred_df, on=merge_key, how="inner", suffixes=("_gt", "_pred"))

        print("Merged columns:", merged.columns.tolist())

        fields = ["invoice_number", "invoice_date", "seller_name", "client_name", "net_worth", "total_amount", "tax"]
        results = []

        for field in fields:
            gt_col = f"{field}_gt"
            pred_col = f"{field}_pred"
            if gt_col not in merged.columns:
                continue

            gt = merged[gt_col].apply(lambda x: self._coerce_field(x, field))
            pred = merged[pred_col].apply(lambda x: self._coerce_field(x, field))

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
                "f1": f1,
            })

        metrics_df = pd.DataFrame(results)

        if metrics_df.empty:
            raise ValueError(
                "No fields were evaluated. Check that ground_truth_df contains the expected "
                "label columns: invoice_number, invoice_date, seller_name, client_name, net_worth, "
                "total_amount, tax."
            )
        
        total_gt = metrics_df["ground_truth_count"].sum() if not metrics_df.empty else 0
        total_pred = metrics_df["predicted_count"].sum() if not metrics_df.empty else 0
        total_correct = metrics_df["correct"].sum() if not metrics_df.empty else 0

        overall = {
            "accuracy": total_correct / total_gt if total_gt else np.nan,
            "precision": total_correct / total_pred if total_pred else np.nan,
            "recall": total_correct / total_gt if total_gt else np.nan,
            "f1": (
                2 * (total_correct / total_pred) * (total_correct / total_gt)
                / ((total_correct / total_pred) + (total_correct / total_gt))
                if total_pred and total_gt else np.nan
            ),
        }

        return metrics_df, overall

    # ---------------------------------------------------------------------
    # Summary / sample inspection
    # ---------------------------------------------------------------------
    def _print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print a concise summary of Donut extraction results."""
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        print(f"\n{'='*60}")
        print("DONUT EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful extractions: {len(successful)}")
        print(f"Failed extractions: {len(failed)}")

        if successful:
            avg_words = np.mean([r.get("total_words", 0) for r in successful])
            avg_confidence = np.nanmean([r.get("avg_confidence", np.nan) for r in successful])
            print(f"Average words per image: {avg_words:.1f}")
            print(f"Average sequence confidence: {avg_confidence:.3f}")

            field_counts: Dict[str, int] = {}
            for result in successful:
                for field in result.get("invoice_fields", {}).keys():
                    field_counts[field] = field_counts.get(field, 0) + 1

            if field_counts:
                print("\nExtracted invoice fields:")
                for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {field}: {count} images ({count/len(successful)*100:.1f}%)")

    def visualize_sample_results(self, n_samples: int = 3) -> None:
        """Print a few sample Donut extraction results."""
        if not hasattr(self, "full_results"):
            print("No results to visualize. Run process_dataset first.")
            return

        successful_results = [r for r in self.full_results if r.get("success")][:n_samples]
        for i, result in enumerate(successful_results, start=1):
            print(f"\n{'='*60}")
            print(f"Sample {i}: {result.get('filename', 'unknown')}")
            print(f"{'='*60}")
            print(f"Total words detected: {result.get('total_words', 0)}")
            print(f"Average confidence: {result.get('avg_confidence', float('nan')):.3f}")
            if result.get("invoice_fields"):
                print("\nExtracted Invoice Fields:")
                for field, value in result["invoice_fields"].items():
                    print(f"  {field}: {value}")
            seq = result.get("extracted_text", "")
            if seq:
                print("\nGenerated text (truncated):")
                print(f"  {seq[:250]}{'...' if len(seq) > 250 else ''}")


# -------------------------------------------------------------------------
# Standalone analysis dashboard functions (pipeline-agnostic)
# -------------------------------------------------------------------------

def _get_successful_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if r.get("success")]


def _field_extraction_rates(results: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, float]:
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
        rates[field] = count / n
    return rates


def _field_accuracies(metrics_df: Optional[pd.DataFrame], fields: Sequence[str]) -> Dict[str, float]:
    if metrics_df is None or metrics_df.empty:
        return {f: np.nan for f in fields}

    df = metrics_df.copy()
    df = df.set_index("field")

    acc = {}
    for field in fields:
        acc[field] = float(df.loc[field, "accuracy"]) if field in df.index else np.nan
    return acc


def _field_outcome_counts(metrics_df: Optional[pd.DataFrame], fields: Sequence[str]) -> pd.DataFrame:
    """Build per-field counts for correct / incorrect / missing predictions."""
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(index=fields, columns=["correct", "incorrect", "missing_pred"]).fillna(0)

    df = metrics_df.copy().set_index("field")
    rows = []
    for field in fields:
        if field in df.index:
            gt_count = float(df.loc[field, "ground_truth_count"])
            pred_count = float(df.loc[field, "predicted_count"])
            correct = float(df.loc[field, "correct"])
            incorrect = max(pred_count - correct, 0.0)
            missing_pred = max(gt_count - pred_count, 0.0)
        else:
            correct = incorrect = missing_pred = 0.0

        rows.append({
            "field": field,
            "correct": correct,
            "incorrect": incorrect,
            "missing_pred": missing_pred,
        })
    return pd.DataFrame(rows).set_index("field")


def create_analysis_dashboard(
    results: List[Dict[str, Any]],
    metrics_df: Optional[pd.DataFrame] = None,
    fields: Sequence[str] = DEFAULT_FIELDS,
    title: str = "Invoice Processing Analysis Dashboard",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """Create a pipeline-agnostic dashboard from results and optional evaluation metrics.

    Parameters
    ----------
    results : list[dict]
        Per-image result dictionaries. Each dict should include keys such as
        success, total_words, avg_confidence, and invoice_fields.
    metrics_df : pandas.DataFrame, optional
        Evaluation dataframe returned by evaluate_against_ground_truth().
    fields : sequence[str], optional
        Fields to visualize.
    title : str, optional
        Figure title.
    save_path : str or Path, optional
        Path to save the generated figure.
    show : bool, optional
        Whether to display the figure immediately.

    Returns
    -------
    dict
        Summary statistics including total_processed, successful, failed,
        field_extraction_rates, field_accuracies, avg_confidence, and avg_words.
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
        return {}

    avg_words = float(np.mean([r.get("total_words", 0) for r in successful]))
    confidences = [r.get("avg_confidence", np.nan) for r in successful]
    avg_confidence = float(np.nanmean(confidences))

    rates = _field_extraction_rates(results, fields=fields)
    acc = _field_accuracies(metrics_df, fields=fields)
    outcome_df = _field_outcome_counts(metrics_df, fields=fields)

    print(f"\nFIELD EXTRACTION SUCCESS RATES")
    print(f"{'='*50}")
    for field in fields:
        print(f"  {field:15}: {rates[field]*100:5.1f}%")

    if metrics_df is not None and not metrics_df.empty:
        print(f"\nFIELD-LEVEL EXACT MATCH ACCURACIES")
        print(f"{'='*50}")
        for field in fields:
            if pd.notna(acc.get(field, np.nan)):
                print(f"  {field:15}: {acc[field]*100:5.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1) Confidence distribution
    axes[0, 0].hist(confidences, bins=20, alpha=0.8, edgecolor="black")
    axes[0, 0].set_title("Donut Confidence Distribution")
    axes[0, 0].set_xlabel("Confidence proxy")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(np.nanmean(confidences), linestyle="--", label=f"Mean: {np.nanmean(confidences):.3f}")
    axes[0, 0].legend()

    # 2) Accuracy per field
    if metrics_df is not None and not metrics_df.empty:
        acc_vals = [acc.get(field, np.nan) for field in fields]
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
        axes[0, 1].text(0.5, 0.5, "No evaluation metrics\navailable", ha="center", va="center",
                        transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title("Field-Level Exact Match Accuracy")

    # 3) Success rates
    rates_vals = [rates[field] * 100 for field in fields]
    bars = axes[1, 0].bar(range(len(fields)), rates_vals, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Field Extraction Success Rates")
    axes[1, 0].set_ylabel("Success Rate (%)")
    axes[1, 0].set_xticks(range(len(fields)))
    axes[1, 0].set_xticklabels(fields, rotation=45, ha="right")
    axes[1, 0].set_ylim(0, 105)
    for bar, v in zip(bars, rates_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=9)

    # 4) Outcome breakdown
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
        "avg_confidence": avg_confidence,
        "avg_words": avg_words,
    }


def visualize_sample_results(
    results: List[Dict[str, Any]],
    n_samples: int = 3,
) -> None:
    """Print a small sample of Donut OCR/extraction outputs.

    Parameters
    ----------
    results : list[dict]
        Per-image extraction results.
    n_samples : int, optional
        Number of successful examples to print.
    """
    successful = _get_successful_results(results)[:n_samples]
    for i, result in enumerate(successful, start=1):
        print(f"\n{'='*60}")
        print(f"Sample {i}: {result.get('filename', 'unknown')}")
        print(f"{'='*60}")
        print(f"Total words detected: {result.get('total_words', 0)}")
        print(f"Average confidence: {result.get('avg_confidence', float('nan')):.3f}")
        if result.get("invoice_fields"):
            print("\nExtracted Invoice Fields:")
            for field, value in result["invoice_fields"].items():
                print(f"  {field}: {value}")
        seq = result.get("extracted_text", "")
        if seq:
            print("\nGenerated text (truncated):")
            print(f"  {seq[:250]}{'...' if len(seq) > 250 else ''}")


# Donut fine-tuning now lives in donut_training_utils.py.
#
# Typical usage:
#   from donut_training_utils import DonutFineTuningConfig, train_donut_invoice_model
#   fit = train_donut_invoice_model(ground_truth_df, output_dir="./donut_model")
#   donut_detector.reload_model(fit["save_path"])
