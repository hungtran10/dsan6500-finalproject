from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FIELDS: Sequence[str] = (
    "invoice_number",
    "invoice_date",
    "seller_name",
    "client_name",
    "tax",
    "net_worth",
    "total_amount",
)


def _get_successful_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if r.get("success")]


def _field_extraction_rates(results: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, float]:
    successful = _get_successful_results(results)
    n = len(successful)
    if n == 0:
        return {f: 0.0 for f in fields}

    rates: Dict[str, float] = {}
    for field in fields:
        count = 0
        for r in successful:
            invoice_fields = r.get("invoice_fields", {})
            if field in invoice_fields and invoice_fields[field] not in [None, "", np.nan]:
                count += 1
        rates[field] = count / n
    return rates


def _field_metrics(metrics_df: Optional[pd.DataFrame], fields: Sequence[str], metric_col: str) -> Dict[str, float]:
    if metrics_df is None or metrics_df.empty or metric_col not in metrics_df.columns:
        return {f: np.nan for f in fields}
    df = metrics_df.copy().set_index("field")
    out: Dict[str, float] = {}
    for field in fields:
        out[field] = float(df.loc[field, metric_col]) if field in df.index else np.nan
    return out


def _field_outcome_counts(metrics_df: Optional[pd.DataFrame], fields: Sequence[str]) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(index=fields, columns=["correct", "incorrect", "missing_pred"]).fillna(0)

    df = metrics_df.copy().set_index("field")
    rows = []
    for field in fields:
        if field in df.index and {"ground_truth_count", "predicted_count", "correct"}.issubset(df.columns):
            gt_count = float(df.loc[field, "ground_truth_count"])
            pred_count = float(df.loc[field, "predicted_count"])
            correct = float(df.loc[field, "correct"])
            incorrect = max(pred_count - correct, 0.0)
            missing_pred = max(gt_count - pred_count, 0.0)
        else:
            correct = incorrect = missing_pred = 0.0

        rows.append(
            {
                "field": field,
                "correct": correct,
                "incorrect": incorrect,
                "missing_pred": missing_pred,
            }
        )
    return pd.DataFrame(rows).set_index("field")


def create_analysis_dashboard(
    results: List[Dict[str, Any]],
    metrics_df: Optional[pd.DataFrame] = None,
    fields: Sequence[str] = DEFAULT_FIELDS,
    title: str = "Invoice Processing Analysis Dashboard",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> Dict[str, Any]:
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

    word_counts = [r.get("total_words", 0) for r in successful]
    confidences = [r.get("avg_confidence", np.nan) for r in successful]
    valid_confidences = [c for c in confidences if pd.notna(c)]

    rates = _field_extraction_rates(results, fields=fields)
    accuracy = _field_metrics(metrics_df, fields, "accuracy")
    precision = _field_metrics(metrics_df, fields, "precision")
    # Prefer standard recall when present, fall back to donut's field_recall.
    recall = _field_metrics(metrics_df, fields, "recall")
    if all(pd.isna(v) for v in recall.values()):
        recall = _field_metrics(metrics_df, fields, "field_recall")
    outcome_df = _field_outcome_counts(metrics_df, fields=fields)

    print(f"\nFIELD EXTRACTION SUCCESS RATES")
    print(f"{'='*50}")
    for field in fields:
        print(f"  {field:15}: {rates[field]*100:5.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1) Confidence distribution
    if valid_confidences:
        axes[0, 0].hist(valid_confidences, bins=20, alpha=0.8, edgecolor="black")
        axes[0, 0].set_title("Confidence Distribution")
        axes[0, 0].set_xlabel("Confidence")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].axvline(np.mean(valid_confidences), linestyle="--", label=f"Mean: {np.mean(valid_confidences):.3f}")
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "No confidence scores\navailable", ha="center", va="center", transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title("Confidence Distribution")

    # 2) Accuracy per field
    if metrics_df is not None and not metrics_df.empty:
        acc_vals = [accuracy.get(field, np.nan) for field in fields]
        bars = axes[0, 1].bar(range(len(fields)), acc_vals, alpha=0.85, edgecolor="black")
        axes[0, 1].set_title("Field-Level Accuracy (Exact Match)")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_xticks(range(len(fields)))
        axes[0, 1].set_xticklabels(fields, rotation=45, ha="right")
        axes[0, 1].set_ylim(0, 1.05)
        for bar, v in zip(bars, acc_vals):
            if pd.notna(v):
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, "No evaluation metrics\navailable", ha="center", va="center", transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title("Field-Level Accuracy (Exact Match)")

    # 3) Extraction rates
    rates_vals = [rates[field] * 100 for field in fields]
    bars = axes[1, 0].bar(range(len(fields)), rates_vals, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Field Extraction Success Rates")
    axes[1, 0].set_ylabel("Success Rate (%)")
    axes[1, 0].set_xticks(range(len(fields)))
    axes[1, 0].set_xticklabels(fields, rotation=45, ha="right")
    axes[1, 0].set_ylim(0, 105)
    for bar, v in zip(bars, rates_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

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
        axes[1, 1].text(0.5, 0.5, "No evaluation metrics\navailable", ha="center", va="center", transform=axes[1, 1].transAxes, fontsize=12)
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
        "field_accuracies": accuracy,
        "field_recalls": recall,
        "field_precisions": precision,
        "avg_confidence": float(np.nanmean(confidences)),
        "avg_words": float(np.mean(word_counts)),
    }


def visualize_sample_results(
    results: List[Dict[str, Any]],
    visualize_text_fn=None,
    n_samples: int = 3,
    title: str = "Sample OCR Results",
) -> None:
    _ = title
    successful = _get_successful_results(results)[:n_samples]

    for i, result in enumerate(successful, start=1):
        print(f"\n{'='*60}")
        print(f"Sample {i}: {result.get('filename', 'unknown')}")
        print(f"{'='*60}")
        print(f"Total words detected: {result.get('total_words', 0)}")

        avg_conf = result.get("avg_confidence", np.nan)
        if pd.notna(avg_conf):
            print(f"Average confidence: {float(avg_conf):.3f}")
        else:
            print("Average confidence: N/A")

        invoice_fields = result.get("invoice_fields", {})
        if invoice_fields:
            print("\nExtracted Invoice Fields:")
            for field, value in invoice_fields.items():
                print(f"  {field}: {value}")

        extracted_text = result.get("extracted_text")
        if isinstance(extracted_text, list) and extracted_text:
            # Pytesseract format: list of dicts with "text"
            sample_text = " ".join([item.get("text", "") for item in extracted_text[:10]]).strip()
            if sample_text:
                print("\nSample extracted text (first 10 words):")
                print(f"  {sample_text}...")
        elif isinstance(extracted_text, str) and extracted_text:
            # Donut format: generated string
            print("\nGenerated text (truncated):")
            print(f"  {extracted_text[:250]}{'...' if len(extracted_text) > 250 else ''}")

        if visualize_text_fn is not None and result.get("image_path") and Path(result["image_path"]).exists():
            visualize_text_fn(result["image_path"], result)

