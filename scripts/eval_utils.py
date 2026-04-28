from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


MONEY_FIELDS = {"total_amount", "tax", "net_worth"}
DATE_FIELDS = {"invoice_date"}


def normalize_date(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return dt.strftime("%Y-%m-%d")
    return None


def normalize_money(value) -> str | None:
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


def normalize_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if s.lower() in {"", "nan", "none"}:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if s else None


def default_field_normalizer(field: str) -> Callable[[object], object]:
    if field in MONEY_FIELDS:
        return lambda v: normalize_money(v)
    if field in DATE_FIELDS:
        return lambda v: normalize_date(v)
    return lambda v: normalize_text(v)


@dataclass(frozen=True)
class ExactMatchOverall:
    accuracy: float | np.floating | None
    precision: float | np.floating | None
    recall: float | np.floating | None
    f1: float | np.floating | None

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def evaluate_exact_match(
    *,
    ground_truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    fields: Iterable[str],
    merge_key: str = "processed_file",
    restrict_to_matched: bool = True,
    field_normalizer_factory: Callable[[str], Callable[[object], object]] = default_field_normalizer,
) -> tuple[pd.DataFrame, dict]:
    """
    Shared exact-match evaluator used across pipelines.

    Expected schema:
    - `ground_truth_df` contains ground-truth columns for each field.
    - `pred_df` contains predicted columns for each field.
    - Both contain `merge_key`.
    - After merge, columns are suffixed with `_gt` and `_pred`.
    """

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
        suffixes=("_gt", "_pred"),
    )

    results: list[dict] = []

    for field in fields:
        gt_col = f"{field}_gt"
        pred_col = f"{field}_pred"

        if gt_col not in merged.columns:
            continue

        norm = field_normalizer_factory(field)
        gt = merged[gt_col].apply(norm)
        pred = merged[pred_col].apply(norm)

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

        results.append(
            {
                "field": field,
                "ground_truth_count": gt_count,
                "predicted_count": pred_count,
                "correct": correct_count,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    metrics_df = pd.DataFrame(results)

    total_gt = metrics_df["ground_truth_count"].sum() if not metrics_df.empty else 0
    total_pred = metrics_df["predicted_count"].sum() if not metrics_df.empty else 0
    total_correct = metrics_df["correct"].sum() if not metrics_df.empty else 0

    overall = ExactMatchOverall(
        accuracy=total_correct / total_gt if total_gt else np.nan,
        precision=total_correct / total_pred if total_pred else np.nan,
        recall=total_correct / total_gt if total_gt else np.nan,
        f1=(
            2 * (total_correct / total_pred) * (total_correct / total_gt)
            / ((total_correct / total_pred) + (total_correct / total_gt))
            if total_pred and total_gt
            else np.nan
        ),
    )

    return metrics_df, overall.as_dict()


def summarize_field_prediction_gaps(
    *,
    ground_truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    fields: Iterable[str],
    merge_key: str = "processed_file",
    field_normalizer_factory: Callable[[str], Callable[[object], object]] = default_field_normalizer,
    max_examples: int = 8,
) -> pd.DataFrame:
    """
    For each field, count rows where GT is non-empty but prediction is missing (normalized).

    Use this to separate "wrong string" errors from empty predictions when accuracy drops.
    """
    merged = ground_truth_df.merge(
        pred_df,
        on=merge_key,
        how="inner",
        suffixes=("_gt", "_pred"),
    )
    rows_out: list[dict[str, Any]] = []

    for field in fields:
        gt_col = f"{field}_gt"
        pred_col = f"{field}_pred"
        if gt_col not in merged.columns:
            continue

        norm = field_normalizer_factory(field)
        gt = merged[gt_col].apply(norm)
        pred = merged[pred_col].apply(norm) if pred_col in merged.columns else pd.Series([np.nan] * len(merged))

        valid_gt = gt.notna()
        missing_pred = valid_gt & pred.isna()
        mismatch = valid_gt & pred.notna() & (gt != pred)

        gap_idx = merged.index[missing_pred]
        sample_keys: list[str] = []
        for ix in gap_idx[:max_examples]:
            sample_keys.append(str(merged.loc[ix, merge_key]))

        rows_out.append(
            {
                "field": field,
                "gt_nonempty": int(valid_gt.sum()),
                "pred_missing_given_gt": int(missing_pred.sum()),
                "mismatch_given_both": int(mismatch.sum()),
                "sample_keys_missing_pred": sample_keys,
            }
        )

    return pd.DataFrame(rows_out)

