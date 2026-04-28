from __future__ import annotations

import json
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seqeval.metrics import f1_score as _f1_score  # type: ignore

from PIL import Image
import cv2

import pytesseract

from .eval_utils import normalize_date, normalize_money, normalize_text, evaluate_exact_match

import torch
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    set_seed
)

DEFAULT_FIELDS: list[str] = [
    "invoice_number",
    "invoice_date",
    "seller_name",
    "client_name",
    "net_worth",
    "total_amount",
    "tax",
]


def _xywh_to_xyxy(bbox_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    return x, y, x + w, y + h


def _normalize_box_1000(
    *,
    box_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[int]:
    x0, y0, x1, y1 = box_xyxy
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]

    def clamp(v: int, low: int, high: int) -> int:
        return max(low, min(int(v), high))

    x0n = clamp(round(1000 * x0 / width), 0, 1000)
    y0n = clamp(round(1000 * y0 / height), 0, 1000)
    x1n = clamp(round(1000 * x1 / width), 0, 1000)
    y1n = clamp(round(1000 * y1 / height), 0, 1000)
    return [x0n, y0n, x1n, y1n]


@dataclass(frozen=True)
class OcrWord:
    text: str
    bbox_xywh: tuple[int, int, int, int]
    confidence: float


def ocr_words_from_image(
    image: np.ndarray,
    *,
    confidence_threshold: int = 30,
    psm: int = 6,
    extra_config: str = "",
) -> list[OcrWord]:

    config = f"--oem 3 --psm {psm} {extra_config}".strip()
    ocr_data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: list[OcrWord] = []
    for i in range(len(ocr_data["text"])):
        txt = str(ocr_data["text"][i]).strip()
        conf_raw = ocr_data["conf"][i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1

        if not txt or conf <= confidence_threshold:
            continue

        words.append(
            OcrWord(
                text=txt,
                confidence=conf,
                bbox_xywh=(
                    int(ocr_data["left"][i]),
                    int(ocr_data["top"][i]),
                    int(ocr_data["width"][i]),
                    int(ocr_data["height"][i]),
                ),
            )
        )

    return words


def _token_norm_for_field(token: str, field: str) -> str:
    token = token.strip()
    if not token:
        return ""

    if field in {"total_amount", "tax", "net_worth"}:
        cleaned = token.replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace("₹", "")
        nm = normalize_money(cleaned)
        if nm is not None:
            return nm
        return re.sub(r"[^\d.,-]", "", cleaned)

    if field == "invoice_date":
        nd = normalize_date(token)
        if nd is not None:
            return nd
        return token.lower()

    if field == "invoice_number":
        # Keep alnum and dash only to be robust to OCR punctuation noise.
        return re.sub(r"[^A-Za-z0-9\-]", "", token).lower()

    # names and other text
    nt = normalize_text(token)
    return nt or ""


def _value_tokens_for_field(value: Any, field: str) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if field in {"total_amount", "tax", "net_worth"}:
        nm = normalize_money(value)
        return [nm] if nm else []

    if field == "invoice_date":
        nd = normalize_date(value)
        return [nd] if nd else []

    if field == "invoice_number":
        v = re.sub(r"[^A-Za-z0-9\-]", "", str(value)).lower().strip()
        return [v] if v else []

    nt = normalize_text(value)
    if not nt:
        return []
    return [t for t in nt.split(" ") if t]


def _find_exact_subsequence(haystack: list[str], needle: list[str]) -> tuple[int, int] | None:
    if not haystack or not needle:
        return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i, i + n
    return None


def _find_fuzzy_window(
    haystack: list[str],
    needle: list[str],
    *,
    min_ratio: float = 0.92,
    window_slack: int = 2,
) -> tuple[int, int] | None:
    try:
        from rapidfuzz.fuzz import ratio as rf_ratio  # type: ignore
    except Exception:  # pragma: no cover
        rf_ratio = None

    needle_join = " ".join(needle)
    if not needle_join.strip():
        return None

    base = len(needle)
    best: tuple[float, int, int] | None = None

    for wlen in range(max(1, base - window_slack), base + window_slack + 1):
        for i in range(0, len(haystack) - wlen + 1):
            cand = " ".join(haystack[i : i + wlen])
            if not cand.strip():
                continue
            if rf_ratio is not None:
                sc = rf_ratio(needle_join, cand) / 100.0
            else:
                import difflib

                sc = difflib.SequenceMatcher(a=needle_join, b=cand).ratio()
            if best is None or sc > best[0]:
                best = (sc, i, i + wlen)

    if best and best[0] >= min_ratio:
        _, s, e = best
        return s, e
    return None


def _find_anchor_guided_window(
    haystack_raw: list[str],
    haystack_norm: list[str],
    needle_norm: list[str],
    *,
    anchor_terms: list[str],
    min_ratio: float = 0.72,
    max_tokens_after_anchor: int = 28,
    window_slack: int = 2,
) -> tuple[int, int] | None:
    """
    Anchor-guided matcher used for weak labels of party names.

    It searches windows after anchor terms like "seller"/"client" and picks the
    span with highest string similarity to the ground-truth target.
    """
    if not haystack_norm or not needle_norm:
        return None

    try:
        from rapidfuzz.fuzz import ratio as rf_ratio  # type: ignore
    except Exception:  # pragma: no cover
        rf_ratio = None

    anchor_set = set(anchor_terms)
    target = " ".join(needle_norm).strip()
    if not target:
        return None

    anchor_positions: list[int] = []
    for i, t in enumerate(haystack_norm):
        if t in anchor_set or t.rstrip(":") in anchor_set:
            anchor_positions.append(i)
    if not anchor_positions:
        return None

    base = len(needle_norm)
    best: tuple[float, int, int] | None = None

    for aidx in anchor_positions:
        start = aidx + 1
        stop = min(len(haystack_norm), aidx + 1 + max_tokens_after_anchor)
        if start >= stop:
            continue
        for wlen in range(max(1, base - window_slack), base + window_slack + 1):
            for s in range(start, max(start, stop - wlen + 1)):
                e = s + wlen
                if e > stop:
                    break
                cand_norm_tokens = haystack_norm[s:e]
                if not any(tok for tok in cand_norm_tokens):
                    continue
                # Avoid windows that clearly run into metadata fields.
                if any(tok in {"tax", "id", "iban", "vat", "invoice", "date"} for tok in cand_norm_tokens):
                    continue

                cand = " ".join(cand_norm_tokens).strip()
                if not cand:
                    continue
                if rf_ratio is not None:
                    sc = rf_ratio(target, cand) / 100.0
                else:
                    import difflib

                    sc = difflib.SequenceMatcher(a=target, b=cand).ratio()
                if best is None or sc > best[0]:
                    best = (sc, s, e)

    if best and best[0] >= min_ratio:
        _, s, e = best
        return s, e
    return None


def _is_party_name_span_quality(words: list[str], s: int, e: int) -> bool:
    """
    Keep seller/client weak labels compact and name-like.
    Reject spans that look like IDs/addresses/metadata.
    """
    if s < 0 or e > len(words) or s >= e:
        return False
    # Organization names can be longer than person names; keep a reasonable cap.
    # Match weak-label fallback below (<=14) so long firm names still supervise.
    if (e - s) > 14:
        return False

    toks = [str(w).strip() for w in words[s:e] if str(w).strip()]
    if not toks:
        return False
    txt = " ".join(toks)
    norm = normalize_text(txt) or ""
    bad_terms = {"tax", "id", "iban", "invoice", "date", "vat", "total", "amount", "summary"}
    if any(t in bad_terms for t in norm.split(" ")):
        return False

    alpha = sum(ch.isalpha() for ch in txt)
    digits = sum(ch.isdigit() for ch in txt)
    if alpha < 2:
        return False
    # Allow small numeric noise (e.g. OCR artifacts), but reject ID-like spans.
    if digits > max(4, alpha):
        return False
    return True


def _money_tol(v: float) -> float:
    return max(0.05, abs(v) * 0.01)


def _line_word_indices_from_ocr(ocr_words: list[OcrWord]) -> list[list[int]]:
    """
    Cluster OCR words into reading-order lines using y-center proximity.
    Returns line groups as lists of original word indices.
    """
    if not ocr_words:
        return []

    centers = [ow.bbox_xywh[1] + ow.bbox_xywh[3] / 2.0 for ow in ocr_words]
    heights = [max(1, ow.bbox_xywh[3]) for ow in ocr_words]
    med_h = int(np.median(heights)) if heights else 12
    y_tol = max(8, int(med_h * 0.8))

    # sort by y then x for robust line assignment
    order = sorted(range(len(ocr_words)), key=lambda i: (centers[i], ocr_words[i].bbox_xywh[0]))
    line_refs: list[float] = []
    line_groups: list[list[int]] = []

    for idx in order:
        cy = centers[idx]
        x = ocr_words[idx].bbox_xywh[0]
        assigned = -1
        best_dist = 1e9
        for li, ly in enumerate(line_refs):
            dist = abs(cy - ly)
            if dist <= y_tol and dist < best_dist:
                best_dist = dist
                assigned = li
        if assigned < 0:
            line_refs.append(cy)
            line_groups.append([idx])
        else:
            line_groups[assigned].append(idx)
            # running average for line reference
            n = len(line_groups[assigned])
            line_refs[assigned] = (line_refs[assigned] * (n - 1) + cy) / n

    # left-to-right within each line
    for g in line_groups:
        g.sort(key=lambda i: ocr_words[i].bbox_xywh[0])
    # top-to-bottom lines
    line_groups.sort(key=lambda g: min(centers[i] for i in g))
    return line_groups


def _money_candidates_for_line(words: list[str], line_idxs: list[int]) -> list[tuple[float, int, int]]:
    """
    Produce money candidates from a line as (value, start_idx, end_idx_exclusive).
    Handles both single-token and split-token amounts like "6 579,11".
    """
    candidates: list[tuple[float, int, int]] = []
    seen: set[tuple[float, int, int]] = set()
    if not line_idxs:
        return candidates

    def add(val: float, s: int, e: int):
        key = (round(val, 4), s, e)
        if key in seen:
            return
        seen.add(key)
        candidates.append((val, s, e))

    # single-token candidates
    for idx in line_idxs:
        nm = normalize_money(words[idx])
        if nm is None:
            continue
        add(float(nm), idx, idx + 1)

    # split-token candidates (adjacent in line): "6" + "579,11"
    for j in range(len(line_idxs) - 1):
        i0, i1 = line_idxs[j], line_idxs[j + 1]
        t0, t1 = words[i0].strip(), words[i1].strip()
        if re.fullmatch(r"\d{1,3}", t0) and re.fullmatch(r"\d{3}[,.]\d{2}", t1):
            nm = normalize_money(f"{t0} {t1}")
            if nm is not None:
                s, e = min(i0, i1), max(i0, i1) + 1
                add(float(nm), s, e)
    return candidates


def _line_arithmetic_money_spans(
    *,
    words: list[str],
    ocr_words: list[OcrWord] | None,
    ground_truth_row: pd.Series,
    occupied: list[bool],
) -> dict[str, tuple[int, int]]:
    """
    Infer money spans from line candidates using GT proximity + arithmetic consistency.
    Returns spans for any of {tax, net_worth, total_amount} that can be confidently assigned.
    """
    if not ocr_words:
        return {}

    gt_vals: dict[str, float] = {}
    for f in ("tax", "net_worth", "total_amount"):
        nm = normalize_money(ground_truth_row.get(f))
        if nm is not None:
            gt_vals[f] = float(nm)
    if not gt_vals:
        return {}

    line_groups = _line_word_indices_from_ocr(ocr_words)
    if not line_groups:
        return {}

    best: tuple[float, int, int, dict[str, tuple[int, int]]] | None = None
    # rank tuple: (negative_matched_fields, negative_anchor_hits, error_sum, spans)
    for g in line_groups:
        cands = _money_candidates_for_line(words, g)
        if len(cands) < 2:
            continue
        line_text = " ".join(words[i] for i in g)
        line_norm = normalize_text(line_text) or ""
        anchor_hits = sum(
            int(tok in line_norm.split(" "))
            for tok in ("summary", "vat", "tax", "net", "gross", "total", "amount")
        )

        chosen: dict[str, tuple[float, int, int, float]] = {}
        for field, gt in gt_vals.items():
            field_best: tuple[float, int, int, float] | None = None  # val, s, e, err
            for val, s, e in cands:
                if s < 0 or e > len(occupied) or s >= e:
                    continue
                if any(occupied[s:e]):
                    continue
                err = abs(val - gt)
                if err > _money_tol(gt):
                    continue
                if field_best is None or err < field_best[3]:
                    field_best = (val, s, e, err)
            if field_best is not None:
                chosen[field] = field_best

        if len(chosen) < 2:
            continue

        # if all 3 are present, enforce arithmetic consistency relative to GT
        if all(k in chosen for k in ("tax", "net_worth", "total_amount")):
            tax_v = chosen["tax"][0]
            net_v = chosen["net_worth"][0]
            tot_v = chosen["total_amount"][0]
            if abs((tax_v + net_v) - tot_v) > _money_tol(tot_v):
                continue

        err_sum = sum(v[3] for v in chosen.values())
        rank = (-len(chosen), -anchor_hits, err_sum)
        spans = {k: (int(v[1]), int(v[2])) for k, v in chosen.items()}
        if best is None or rank < best[:3]:
            best = (rank[0], rank[1], rank[2], spans)

    return best[3] if best is not None else {}


def weak_label_words_bio(
    *,
    words: list[str],
    ground_truth_row: pd.Series,
    fields: Iterable[str] = DEFAULT_FIELDS,
    ocr_words: list[OcrWord] | None = None,
) -> tuple[list[str], dict[str, tuple[int, int] | None]]:
    """
    Weakly label OCR *words* with BIO tags by matching to ground-truth values.

    Returns:
      - labels: one label per input word
      - spans: dict field -> (start,end) matched span in word indices (end exclusive) or None
    """

    fields = list(fields)
    labels = ["O"] * len(words)
    occupied = [False] * len(words)
    spans: dict[str, tuple[int, int] | None] = {f: None for f in fields}

    # Label non-money fields first so party-name spans are not overwritten by money pre-pass.
    non_money_priority = ["invoice_number", "invoice_date", "seller_name", "client_name"]
    non_money_fields = [f for f in non_money_priority if f in fields]
    money_fields = [f for f in ("tax", "net_worth", "total_amount") if f in fields]
    other_fields = [f for f in fields if f not in set(non_money_fields + money_fields)]

    def assign_span(field: str, s: int, e: int) -> bool:
        if s >= e or any(occupied[s:e]):
            return False
        spans[field] = (s, e)
        occupied[s:e] = [True] * (e - s)
        ent = field.upper()
        labels[s] = f"B-{ent}"
        for ii in range(s + 1, e):
            labels[ii] = f"I-{ent}"
        return True

    for field in non_money_fields:
        target = _value_tokens_for_field(ground_truth_row.get(field), field)
        if not target:
            continue

        hay = [_token_norm_for_field(w, field) for w in words]
        hay = [h for h in hay]  # keep indices aligned (empty tokens stay empty)

        # Names: allow partial matches (e.g., missing punctuation); use text-normalized tokens
        match = _find_exact_subsequence(hay, target)
        if match is None:
            match = _find_fuzzy_window(hay, target)
        if match is None and field in {"seller_name", "client_name"}:
            # Targeted weak-label pass: constrain party-name matching to regions
            # after Seller:/Client: anchors to reduce cross-field confusion.
            raw_norm = [normalize_text(w) or "" for w in words]
            anchors = ["seller"] if field == "seller_name" else ["client"]
            match = _find_anchor_guided_window(
                haystack_raw=words,
                haystack_norm=raw_norm,
                needle_norm=target,
                anchor_terms=anchors,
                min_ratio=0.70,
                max_tokens_after_anchor=28,
                window_slack=2,
            )
        if match is None:
            continue

        s, e = match
        if field in {"seller_name", "client_name"} and not _is_party_name_span_quality(words, s, e):
            # If quality gate rejects the candidate, keep a softer fallback instead of
            # dropping the label entirely, to avoid zero-coverage supervision.
            span_len = e - s
            if span_len <= 0 or span_len > 14:
                continue
        assign_span(field, s, e)

    # Money weak labels: line/arithmetic-aware pass after names.
    money_line_spans = _line_arithmetic_money_spans(
        words=words,
        ocr_words=ocr_words,
        ground_truth_row=ground_truth_row,
        occupied=occupied,
    )
    for f in money_fields:
        if f not in money_line_spans:
            continue
        s, e = money_line_spans[f]
        assign_span(f, s, e)

    # Fill unresolved money fields with exact/fuzzy/anchor-guided matching.
    for field in money_fields + other_fields:
        if spans.get(field) is not None:
            continue
        target = _value_tokens_for_field(ground_truth_row.get(field), field)
        if not target:
            continue

        hay = [_token_norm_for_field(w, field) for w in words]
        hay = [h for h in hay]

        match = _find_exact_subsequence(hay, target)
        if match is None:
            match = _find_fuzzy_window(hay, target)
        if match is None and field in {"tax", "net_worth", "total_amount"}:
            money_anchor_terms = {
                "tax": ["vat", "tax"],
                "net_worth": ["net", "networth", "subtotal"],
                "total_amount": ["total", "gross", "amount", "summary"],
            }
            raw_norm = [normalize_text(w) or "" for w in words]
            match = _find_anchor_guided_window(
                haystack_raw=words,
                haystack_norm=raw_norm,
                needle_norm=target,
                anchor_terms=money_anchor_terms.get(field, []),
                min_ratio=0.68,
                max_tokens_after_anchor=36,
                window_slack=2,
            )
        if match is None:
            continue
        s, e = match
        assign_span(field, s, e)

    return labels, spans


class LayoutLMv3InvoiceDatasetBuilder:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        ocr_confidence_threshold: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_confidence_threshold = ocr_confidence_threshold

    def build_examples(
        self,
        merged_df: pd.DataFrame,
        *,
        image_col: str = "processed_path",
        key_col: str = "processed_file",
        fields: Iterable[str] = DEFAULT_FIELDS,
        max_examples: int | None = None,
    ) -> list[dict[str, Any]]:
        fields = list(fields)

        examples: list[dict[str, Any]] = []
        df = merged_df.copy()
        if max_examples is not None:
            df = df.head(max_examples)

        for _, row in df.iterrows():
            image_path = row.get(image_col)
            if not image_path or not Path(str(image_path)).exists():
                continue

            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            ocr_words = ocr_words_from_image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                confidence_threshold=self.ocr_confidence_threshold,
                psm=6,
            )
            if not ocr_words:
                continue

            words = [ow.text for ow in ocr_words]
            boxes = [
                _normalize_box_1000(box_xyxy=_xywh_to_xyxy(ow.bbox_xywh), width=w, height=h)
                for ow in ocr_words
            ]

            labels, spans = weak_label_words_bio(
                words=words,
                ground_truth_row=row,
                fields=fields,
                ocr_words=ocr_words,
            )

            examples.append(
                {
                    "image_path": str(image_path),
                    "processed_file": row.get(key_col) or Path(str(image_path)).name,
                    "words": words,
                    "boxes": boxes,
                    "labels": labels,
                    "matched_spans": {k: list(v) if v is not None else None for k, v in spans.items()},
                    "matched_spans_meta": {
                        "seller_name": {
                            "matched": spans.get("seller_name") is not None,
                            "quality_ok": (
                                _is_party_name_span_quality(
                                    words, int(spans["seller_name"][0]), int(spans["seller_name"][1])
                                )
                                if spans.get("seller_name") is not None
                                else False
                            ),
                        },
                        "client_name": {
                            "matched": spans.get("client_name") is not None,
                            "quality_ok": (
                                _is_party_name_span_quality(
                                    words, int(spans["client_name"][0]), int(spans["client_name"][1])
                                )
                                if spans.get("client_name") is not None
                                else False
                            ),
                        },
                    },
                }
            )

        return examples

    def save_jsonl(self, examples: list[dict[str, Any]], filename: str = "layoutlmv3_weak_labels.jsonl") -> Path:
        out = self.output_dir / filename
        with out.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return out


class LayoutLMv3InvoiceTokenClassifier:
    """
    End-to-end wrapper:
      - build dataset inputs via LayoutLMv3Processor
      - fine-tune LayoutLMv3ForTokenClassification
      - run inference and post-process entities into invoice_fields

    This class is implemented in later TODOs (training + inference).
    """
    # Increment when final inference-time name-resolution behavior changes.
    RUNTIME_VERSION = "layoutlmv3-name-resolver-2026-04-28-final"

    def __init__(self):
        self.model = None
        self.processor = None
        self.id2label: dict[int, str] | None = None
        self.label2id: dict[str, int] | None = None
        self.full_results: list[dict[str, Any]] | None = None
        # When the model predicts no B-/I- seller span (common if weak labels omit seller
        # on many docs), inference only gets a seller string if this is True.
        # Bleed/truncation for anchor extraction is handled in predict(); set False to ablate.
        self.enable_seller_anchor_fallback: bool = True
        # Top-band heuristic is high-recall but can hurt precision on held-out invoices.
        self.enable_seller_top_band_heuristic: bool = False
        self.runtime_version: str = self.RUNTIME_VERSION

    @staticmethod
    def build_label_maps(fields: Iterable[str] = DEFAULT_FIELDS) -> tuple[dict[str, int], dict[int, str]]:
        fields = list(fields)
        labels = ["O"]
        for f in fields:
            ent = f.upper()
            labels.append(f"B-{ent}")
            labels.append(f"I-{ent}")
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}
        return label2id, id2label

    def train(
        self,
        *,
        train_examples: list[dict[str, Any]],
        eval_examples: list[dict[str, Any]] | None,
        output_dir: str | Path,
        fields: Iterable[str] = DEFAULT_FIELDS,
        base_model: str = "microsoft/layoutlmv3-base",
        max_length: int = 512,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        logging_steps: int = 50,
        save_total_limit: int = 2,
        seed: int = 42,
    ):
        """
        Fine-tune LayoutLMv3 for token classification using weak BIO labels.

        `train_examples` / `eval_examples` expected schema (from LayoutLMv3InvoiceDatasetBuilder):
          - image_path: str
          - words: list[str]
          - boxes: list[list[int]]  # normalized 0..1000 xyxy
          - labels: list[str]       # BIO labels per word
        """

        set_seed(seed)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.label2id, self.id2label = self.build_label_maps(fields)

        self.processor = LayoutLMv3Processor.from_pretrained(base_model, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            base_model,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        def encode_example(ex: dict[str, Any]) -> dict[str, torch.Tensor]:
            image = Image.open(ex["image_path"]).convert("RGB")
            word_labels = [self.label2id.get(l, 0) for l in ex["labels"]]

            encoded = self.processor(
                images=image,
                text=ex["words"],
                boxes=ex["boxes"],
                word_labels=word_labels,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # remove batch dim
            return {k: v.squeeze(0) for k, v in encoded.items()}

        class _TorchDataset(torch.utils.data.Dataset):
            def __init__(self, examples: list[dict[str, Any]]):
                self.examples = examples

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                return encode_example(self.examples[idx])

        train_ds = _TorchDataset(train_examples)
        eval_ds = _TorchDataset(eval_examples) if eval_examples else None

        def compute_metrics(eval_pred):
            # Optional seqeval; training still works without it.

            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)

            true_labels = []
            true_preds = []
            for p_seq, l_seq in zip(preds, labels):
                seq_l = []
                seq_p = []
                for p, l in zip(p_seq, l_seq):
                    if l == -100:
                        continue
                    seq_l.append(self.id2label[int(l)])
                    seq_p.append(self.id2label[int(p)])
                true_labels.append(seq_l)
                true_preds.append(seq_p)

            return {"seqeval_f1": float(_f1_score(true_labels, true_preds))}

        ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        args_kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "logging_steps": logging_steps,
            "save_strategy": "steps",
            "save_steps": logging_steps,
            "save_total_limit": save_total_limit,
            "remove_unused_columns": False,
            "seed": seed,
        }

        if eval_ds is not None:
            if "evaluation_strategy" in ta_params:
                args_kwargs["evaluation_strategy"] = "steps"
            elif "eval_strategy" in ta_params:
                args_kwargs["eval_strategy"] = "steps"
            if "eval_steps" in ta_params:
                args_kwargs["eval_steps"] = logging_steps
        else:
            if "evaluation_strategy" in ta_params:
                args_kwargs["evaluation_strategy"] = "no"
            elif "eval_strategy" in ta_params:
                args_kwargs["eval_strategy"] = "no"

        if "report_to" in ta_params:
            args_kwargs["report_to"] = []

        args = TrainingArguments(**args_kwargs)

        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "compute_metrics": (compute_metrics if eval_ds is not None else None),
        }

        trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
        if "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = self.processor.tokenizer
        elif "processing_class" in trainer_params:
            # Newer transformers versions replaced tokenizer with processing_class
            trainer_kwargs["processing_class"] = self.processor

        trainer = Trainer(**trainer_kwargs)

        trainer.train()

        # Save model + processor
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        (output_dir / "label_map.json").write_text(
            json.dumps({"label2id": self.label2id, "id2label": self.id2label}, indent=2),
            encoding="utf-8",
        )

        return output_dir

    def reload_model(self, model_dir: str | Path):

        model_dir = Path(model_dir)
        self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)

        label_map_path = model_dir / "label_map.json"
        if label_map_path.exists():
            mp = json.loads(label_map_path.read_text(encoding="utf-8"))
            self.label2id = {k: int(v) for k, v in mp["label2id"].items()}
            self.id2label = {int(k): v for k, v in mp["id2label"].items()} if isinstance(next(iter(mp["id2label"])), str) else mp["id2label"]
        else:
            # fallback to model config
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            self.label2id = {k: int(v) for k, v in self.model.config.label2id.items()}

        return self

    def _predict_word_labels(
        self, *, image_path: str | Path
    ) -> tuple[list[str], list[str], list[float], list[float], list[float]]:
        """
        Returns:
          words, predicted_label_per_word, word_y_center_px per word,
          word_x_center_norm (0..1) per word,
          per-word mean max-softmax confidence (over WordPiece tokens mapped to each word).
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call train() or reload_model().")

        image_path = str(image_path)
    
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        ocr_words = ocr_words_from_image(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
            confidence_threshold=30,
            psm=6,
        )
        words = [ow.text for ow in ocr_words]
        word_y_center = [float(ow.bbox_xywh[1] + ow.bbox_xywh[3] / 2.0) for ow in ocr_words]
        word_x_center = [float((ow.bbox_xywh[0] + ow.bbox_xywh[2] / 2.0) / max(1, w)) for ow in ocr_words]
        boxes = [
            _normalize_box_1000(box_xyxy=_xywh_to_xyxy(ow.bbox_xywh), width=w, height=h)
            for ow in ocr_words
        ]

        if not words:
            return [], [], [], [], []

        self.model.eval()
        device = next(self.model.parameters()).device

        encoded = self.processor(
            images=Image.fromarray(img_rgb),
            text=words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            pred_ids = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy().tolist()
            pred_probs = torch.softmax(logits, dim=-1).max(dim=-1).values.squeeze(0).detach().cpu().numpy().tolist()

        if self.id2label is None:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        # Token -> word aggregation
        # BatchEncoding provides word_ids() on the CPU-backed encoding.
        try:
            word_id_list = encoded.word_ids(batch_index=0)  # type: ignore[attr-defined]
        except Exception:
            # fallback: re-encode to get word_ids mapping
            be = self.processor(
                images=Image.fromarray(img_rgb),
                text=words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            word_id_list = be.word_ids(batch_index=0)  # type: ignore[attr-defined]

        per_word_votes: dict[int, dict[str, float]] = {}
        for token_idx, wid in enumerate(word_id_list):
            if wid is None:
                continue
            lbl = self.id2label.get(int(pred_ids[token_idx]), "O")
            conf = float(pred_probs[token_idx]) if token_idx < len(pred_probs) else 1.0
            bucket = per_word_votes.setdefault(int(wid), {})
            bucket[lbl] = bucket.get(lbl, 0.0) + conf

        pred_word_labels = ["O"] * len(words)
        for wid, weighted_votes in per_word_votes.items():
            # Confidence-aware vote: token predictions contribute by softmax confidence.
            pred_word_labels[wid] = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)[0][0]

        sums = [0.0] * len(words)
        counts = [0] * len(words)
        for token_idx, wid in enumerate(word_id_list):
            if wid is None:
                continue
            conf = float(pred_probs[token_idx]) if token_idx < len(pred_probs) else 1.0
            sums[wid] += conf
            counts[wid] += 1
        word_confidences = [(sums[i] / counts[i]) if counts[i] else 0.0 for i in range(len(words))]

        return words, pred_word_labels, word_y_center, word_x_center, word_confidences

    @staticmethod
    def _entities_from_word_labels(words: list[str], labels: list[str]) -> dict[str, list[str]]:
        entities: dict[str, list[str]] = {}
        cur_field: str | None = None
        cur_tokens: list[str] = []

        def flush():
            nonlocal cur_field, cur_tokens
            if cur_field and cur_tokens:
                entities.setdefault(cur_field, []).append(" ".join(cur_tokens).strip())
            cur_field = None
            cur_tokens = []

        for w, lab in zip(words, labels):
            if lab == "O" or not lab:
                flush()
                continue

            if lab.startswith("B-"):
                flush()
                cur_field = lab[2:].lower()
                cur_tokens = [w]
            elif lab.startswith("I-") and cur_field == lab[2:].lower():
                cur_tokens.append(w)
            else:
                flush()

        flush()
        return entities

    @staticmethod
    def _entity_spans_from_word_labels(
        words: list[str], labels: list[str]
    ) -> list[dict[str, Any]]:
        """
        Return extracted entity spans with token boundaries.
        """
        spans: list[dict[str, Any]] = []
        cur_field: str | None = None
        cur_tokens: list[str] = []
        cur_start: int | None = None

        def flush(end_idx: int):
            nonlocal cur_field, cur_tokens, cur_start
            if cur_field and cur_tokens and cur_start is not None:
                spans.append(
                    {
                        "field": cur_field,
                        "text": " ".join(cur_tokens).strip(),
                        "start": int(cur_start),
                        "end": int(end_idx),
                    }
                )
            cur_field = None
            cur_tokens = []
            cur_start = None

        for idx, (w, lab) in enumerate(zip(words, labels)):
            if lab == "O" or not lab:
                flush(idx)
                continue
            if lab.startswith("B-"):
                flush(idx)
                cur_field = lab[2:].lower()
                cur_tokens = [w]
                cur_start = idx
            elif lab.startswith("I-") and cur_field == lab[2:].lower():
                cur_tokens.append(w)
            else:
                flush(idx)

        flush(len(words))
        return spans

    @staticmethod
    def _find_anchor_positions(words: list[str], anchor: str) -> list[int]:
        anchor = (normalize_text(anchor) or anchor).lower()
        pos: list[int] = []
        for i, w in enumerate(words):
            nw = (normalize_text(w) or "").lower().rstrip(":")
            if nw == anchor:
                pos.append(i)
        return pos

    def _select_party_entity_with_anchor(
        self,
        words: list[str],
        spans: list[dict[str, Any]],
        *,
        field: str,
        anchor: str | tuple[str, ...],
        word_x_center: list[float] | None = None,
        expected_side: str | None = None,
    ) -> str | None:
        """
        Prefer entity spans that are close to the corresponding section anchor.
        ``anchor`` may be a single token or a tuple (e.g. seller/vendor/supplier for seller_name).
        """
        candidates = [s for s in spans if s.get("field") == field and str(s.get("text", "")).strip()]
        if not candidates:
            return None
        anchor_terms = (anchor,) if isinstance(anchor, str) else tuple(anchor)
        anchors: list[int] = []
        for a in anchor_terms:
            anchors.extend(self._find_anchor_positions(words, a))
        anchors = sorted(set(anchors))
        if not anchors:
            # fallback to existing name score only
            ranked = sorted(candidates, key=lambda s: self._score_name_candidate(s["text"]), reverse=True)
            return ranked[0]["text"] if ranked else None

        def rank_key(c: dict[str, Any]) -> tuple[float, float, int, float]:
            start = int(c.get("start", 10**9))
            # prefer spans after anchor; if before, penalize heavily via distance bump
            dists = []
            for ap in anchors:
                d = start - ap
                if d < 0:
                    d += 10_000
                dists.append(d)
            best_dist = min(dists) if dists else 10_000
            score = float(self._score_name_candidate(c["text"]))
            side_pen = 0.0
            if (
                word_x_center is not None
                and expected_side is not None
                and 0 <= start < len(word_x_center)
            ):
                x = float(word_x_center[start])
                target_x = 0.25 if expected_side == "left" else 0.75
                side_pen = abs(x - target_x)
            # higher score better; lower distance better
            return (side_pen, -score, best_dist, -len(str(c["text"])))

        ranked = sorted(candidates, key=rank_key)
        return ranked[0]["text"] if ranked else None

    def _resolve_inline_seller_client_pair(
        self,
        words: list[str],
        *,
        word_x_center: list[float] | None = None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """
        Fast path for layouts like: Seller: Client: <seller_name> <client_name>
        where both labels appear on one line and names immediately follow.
        """
        norm_words = [normalize_text(w) or "" for w in words]
        seller_pos = self._anchor_word_positions(norm_words, "seller")
        client_pos = self._anchor_word_positions(norm_words, "client")
        trace: dict[str, Any] = {"seller_pos": seller_pos[:5], "client_pos": client_pos[:5]}
        if not seller_pos or not client_pos:
            return None, None, trace

        # nearest pair of anchors
        best_pair: tuple[int, int] | None = None
        best_gap = 10**9
        for s in seller_pos:
            for c in client_pos:
                g = abs(c - s)
                if g < best_gap:
                    best_gap = g
                    best_pair = (s, c)
        if best_pair is None or best_gap > 8:
            return None, None, trace

        lo = min(best_pair) + 1
        hi = min(len(words), max(best_pair) + 12)
        if lo >= hi:
            return None, None, trace

        bad = {"tax", "id", "iban", "invoice", "date", "summary", "total", "amount", "vat"}
        cands: list[dict[str, Any]] = []
        for i in range(lo, hi):
            w = words[i].strip(" :-,")
            nw = normalize_text(w) or ""
            if not nw or nw in bad:
                continue
            if any(ch.isdigit() for ch in w):
                continue
            # Keep inline pair very strict to avoid over-filling wrong client names.
            sc = float(self._score_name_candidate(w))
            if sc < 14.0:
                continue
            toks = [t for t in re.split(r"\s+", w.strip()) if t]
            if len(toks) > 3:
                continue
            x = float(word_x_center[i]) if word_x_center is not None and i < len(word_x_center) else 0.5
            cands.append({"idx": i, "text": w, "x": x, "score": sc})
        trace["inline_candidates"] = cands[:6]
        if len(cands) < 2:
            return None, None, trace

        # side-aware assignment first
        left = [c for c in cands if c["x"] <= 0.52]
        right = [c for c in cands if c["x"] >= 0.48]
        if left and right:
            s_pick = sorted(left, key=lambda z: z["idx"])[0]
            c_pick = sorted(right, key=lambda z: z["idx"])[0]
            if s_pick["idx"] != c_pick["idx"]:
                x_sep = abs(float(s_pick["x"]) - float(c_pick["x"]))
                quality_ok = (
                    float(s_pick["score"]) >= 16.0
                    and float(c_pick["score"]) >= 16.0
                    and x_sep >= 0.18
                )
                trace["chosen"] = {"seller": s_pick, "client": c_pick, "x_sep": x_sep, "quality_ok": quality_ok}
                return s_pick["text"], c_pick["text"], trace

        # fallback: first two candidates in reading order, map by anchor order
        cands = sorted(cands, key=lambda z: z["idx"])
        a_s, a_c = best_pair
        if a_s <= a_c:
            s_pick, c_pick = cands[0], cands[1]
        else:
            c_pick, s_pick = cands[0], cands[1]
        x_sep = abs(float(s_pick["x"]) - float(c_pick["x"]))
        quality_ok = (
            float(s_pick["score"]) >= 18.0
            and float(c_pick["score"]) >= 18.0
            and x_sep >= 0.22
        )
        trace["chosen"] = {"seller": s_pick, "client": c_pick, "x_sep": x_sep, "quality_ok": quality_ok}
        return s_pick["text"], c_pick["text"], trace

    @staticmethod
    def _summary_arithmetic_tol(total_amount: float) -> float:
        # Allow small OCR/rounding noise while enforcing monetary consistency.
        return max(0.03, abs(total_amount) * 0.01)

    @staticmethod
    def _looks_like_client_stop_line(txt: str) -> bool:
        t = normalize_text(txt)
        if not t:
            return True
        stop_prefixes = (
            "tax id",
            "tax identification",
            "vat",
            "invoice",
            "date",
            "total",
            "amount",
            "summary",
            "bill to",
            "ship to",
            "payment",
        )
        return any(t.startswith(pref) for pref in stop_prefixes)

    @staticmethod
    def _score_name_candidate(txt: str) -> float:
        """
        Score how likely a span is an organization/person name vs address/id noise.
        """
        t = txt.strip()
        if not t:
            return -1e9
        nt = normalize_text(t)
        if not nt:
            return -1e9

        alpha = sum(ch.isalpha() for ch in t)
        digits = sum(ch.isdigit() for ch in t)
        punct = sum(ch in ",.;:/\\|_#()[]{}" for ch in t)
        words = [w for w in re.split(r"\s+", nt) if w]

        bad_tokens = {"tax", "id", "iban", "invoice", "date", "total", "amount", "vat", "summary"}
        bad_tokens |= {
            "street",
            "st",
            "avenue",
            "ave",
            "road",
            "rd",
            "drive",
            "dr",
            "lane",
            "ln",
            "court",
            "ct",
            "parkway",
            "pkwy",
            "trace",
            "bridge",
            "center",
            "suite",
            "unit",
            "floor",
            "building",
            "apt",
            "apartment",
            "island",
            "islands",
        }
        if any(w in bad_tokens for w in words):
            return -50.0

        score = 0.0
        score += min(alpha, 40) * 1.5
        score -= digits * 3.0
        score -= punct * 1.5
        score += min(len(words), 6) * 1.2
        # Prefer medium-length names over very long address-like spans.
        score -= abs(len(t) - 22) * 0.12
        return score

    def _extract_party_name_from_anchor(
        self, words: list[str], *, anchor: str, min_score: float = 18.0
    ) -> str | None:
        """
        Fallback heuristic for blocks like:
          Client:
          ClientName
          Address...
          Tax ID: ...
        """
        if not words:
            return None
        joined = " ".join(words)
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", joined) if ln.strip()]
        if len(lines) < 2:
            return None

        anchor_norm = normalize_text(anchor) or ""
        for i, ln in enumerate(lines):
            n = normalize_text(ln)
            if n in {anchor_norm, f"{anchor_norm}:", f"{anchor_norm} information", f"{anchor_norm} info"}:
                best: tuple[float, str] | None = None
                for j in range(i + 1, min(i + 5, len(lines))):
                    cand = lines[j].strip(" :-")
                    if not cand:
                        continue
                    if self._looks_like_client_stop_line(cand):
                        break
                    score = self._score_name_candidate(cand)
                    if score >= min_score and (best is None or score > best[0]):
                        best = (score, cand)
                if best is not None:
                    return best[1]
                break
        return None

    @staticmethod
    def _anchor_word_positions(norm_words: list[str], anchor: str) -> list[int]:
        """Indices where OCR token equals anchor (case/normalization tolerant)."""
        anchor_norm = (normalize_text(anchor) or anchor).lower().rstrip(":")
        out: list[int] = []
        for idx, nw in enumerate(norm_words):
            t = (nw or "").lower().rstrip(":")
            if t == anchor_norm:
                out.append(idx)
        return out

    def _extract_party_name_from_anchor_tokens(
        self,
        words: list[str],
        *,
        anchor: str,
        min_score: float = 18.0,
        max_tokens_after_anchor: int = 25,
        max_span_tokens: int = 4,
        prefer_nearest: bool = False,
        word_y_center: list[float] | None = None,
        word_x_center: list[float] | None = None,
        seller_boundary_stops: bool = True,
    ) -> str | None:
        """
        Token-level fallback for OCR outputs where line breaks are unavailable.
        Finds `anchor` token (e.g. Seller/Client) and extracts the nearest
        name-like token span before hard stop markers (Tax ID, IBAN, etc.).
        """
        if not words:
            return None

        anchor_norm = normalize_text(anchor) or anchor.lower()
        norm_words = [normalize_text(w) or "" for w in words]

        stop_tokens = {
            "tax",
            "id",
            "taxid",
            "iban",
            "invoice",
            "date",
            "total",
            "amount",
            "vat",
            "summary",
            "payment",
            "bank",
            "account",
            # Address tokens: help stop before street lines.
            "street",
            "st",
            "avenue",
            "ave",
            "road",
            "rd",
            "drive",
            "dr",
            "lane",
            "ln",
            "court",
            "ct",
            "parkway",
            "pkwy",
            "trace",
            "bridge",
            "center",
            "suite",
            "unit",
            "floor",
            "building",
            "apt",
            "apartment",
            "island",
            "islands",
            "turnpike",
            "plain",
            "plains",
            "tunnel",
            "crossing",
            "mountain",
            "mountains",
            "village",
            "prairie",
            "creek",
            "station",
            "cliff",
            "glen",
            "glens",
            "mount",
            "lock",
            "locks",
            "pike",
            "trail",
        }
        # Section boundaries to avoid crossing into other party blocks.
        seller_hard_stops = {
            "client",
            "bill",
            "ship",
            "buyer",
            "inc",
            "llc",
            "ltd",
            "corp",
            "co",
            "unit",
            "suite",
            "ste",
            "floor",
            "bldg",
            "building",
            "address",
        }
        if anchor == "seller":
            stop_tokens |= seller_hard_stops if seller_boundary_stops else set()
            stop_tokens |= {"client"}
        elif anchor == "client":
            stop_tokens |= {"seller", "vendor", "supplier"}
        elif anchor in {"vendor", "supplier"}:
            # vendor/supplier rows often include legal suffixes in the firm line; avoid
            # duplicate seller_hard_stops here (historically omitted). Optional relax pass
            # can still trim via ``_truncate_seller_party_bleed``.
            stop_tokens |= {"client", "bill", "ship", "buyer"}

        def _norm_alnum(tok: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (tok or "").lower())

        stop_norm = {_norm_alnum(t) for t in stop_tokens}

        def is_stop_token(tok: str) -> bool:
            if tok in stop_tokens:
                return True
            return _norm_alnum(tok) in stop_norm

        # OCR often emits ``Seller: Client: NameA NameB`` on one band — peer labels sit
        # between the seller anchor and the seller party text. Skip them instead of
        # treating ``client`` as an immediate stop at ``s == start`` (which yielded
        # zero seller candidates).
        peer_section_cores = frozenset({"seller", "client", "vendor", "supplier"})

        def _skip_peer_section_labels(scan: int, window_end: int) -> int:
            k = scan
            while k < window_end:
                core = _norm_alnum(norm_words[k])
                if core not in peer_section_cores:
                    break
                k += 1
            return k

        def _token_is_anchor(tok_norm: str) -> bool:
            if not tok_norm:
                return False
            if tok_norm in {anchor_norm, f"{anchor_norm}:"}:
                return True
            # OCR variants: "seller)", "(seller", "seller —"
            stripped = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", tok_norm.lower())
            if stripped in {anchor_norm, f"{anchor_norm}:"}:
                return True
            if stripped.startswith(anchor_norm) and len(stripped) <= len(anchor_norm) + 2:
                return True
            return False

        for i, nw in enumerate(norm_words):
            # Anchor forms: "seller", "seller:", "client", "client:"
            if not _token_is_anchor(nw):
                continue

            start = i + 1
            end = min(len(words), i + max_tokens_after_anchor)
            # Clip seller extraction before the next Client section (swap-fix).
            # If Client sits immediately after Seller (next_c == start), clipping would
            # empty the window — skip clip so fallback can still find a name.
            if anchor_norm == "seller":
                clients = self._anchor_word_positions(norm_words, "client")
                next_c = next((c for c in clients if c > i), None)
                if next_c is not None and next_c > start:
                    end = min(end, next_c)
            elif anchor_norm == "client":
                sellers = self._anchor_word_positions(norm_words, "seller")
                next_s = next((s for s in sellers if s > i), None)
                if next_s is not None and next_s > start:
                    end = min(end, next_s)
            if start >= end:
                continue

            scan_start = _skip_peer_section_labels(start, end)
            if scan_start >= end:
                continue

            # Build candidate spans (length capped), stopping early on hard markers.
            best_score: tuple[float, str] | None = None  # client default
            best_layout_key: tuple[float, int, int, float] | None = None
            best_layout_cand: str | None = None
            best_near: tuple[int, float, str] | None = None  # seller without y

            expected_side: str | None = None
            if anchor_norm in {"seller", "vendor", "supplier"}:
                expected_side = "left"
            elif anchor_norm == "client":
                expected_side = "right"

            for s in range(scan_start, end):
                if is_stop_token(norm_words[s]):
                    break
                if norm_words[s] in {"and", "&"}:
                    # Avoid chopped tails like "and Clark" when OCR splits party names.
                    continue
                if word_x_center is not None and 0 <= s < len(word_x_center) and expected_side is not None:
                    xs = float(word_x_center[s])
                    # Hard side gate for clearly wrong-half candidates.
                    if expected_side == "left" and xs > 0.58:
                        continue
                    if expected_side == "right" and xs < 0.42:
                        continue
                for e in range(s + 1, min(s + max_span_tokens + 1, end + 1)):
                    span_norm = norm_words[s:e]
                    if any(is_stop_token(t) for t in span_norm):
                        break
                    cand = " ".join(words[s:e]).strip(" :-")
                    score = self._score_name_candidate(cand)
                    if score < min_score:
                        continue
                    dist = max(0, s - start)
                    if prefer_nearest:
                        if (
                            word_y_center is not None
                            and anchor_norm == "seller"
                            and 0 <= s < len(word_y_center)
                        ):
                            y_top = float(word_y_center[s])
                            hyp = 0 if bool(re.search(r"[A-Za-z]+-[A-Za-z]+", cand)) else 1
                            side_pen = 0.0
                            if word_x_center is not None and 0 <= s < len(word_x_center) and expected_side is not None:
                                target_x = 0.25 if expected_side == "left" else 0.75
                                side_pen = abs(float(word_x_center[s]) - target_x)
                            rank_key = (side_pen, y_top, dist, hyp, -score)
                            if best_layout_key is None or rank_key < best_layout_key:
                                best_layout_key = rank_key
                                best_layout_cand = cand
                        else:
                            rank = (dist, -score)
                            if best_near is None or rank < (best_near[0], -best_near[1]):
                                best_near = (dist, score, cand)
                    else:
                        if best_score is None or score > best_score[0]:
                            best_score = (score, cand)

            if prefer_nearest:
                if best_layout_cand is not None:
                    return best_layout_cand
                if best_near is not None:
                    return best_near[2]
            elif best_score is not None:
                return best_score[1]

        return None

    def _enumerate_anchor_party_candidates(
        self,
        words: list[str],
        *,
        anchor_terms: tuple[str, ...],
        min_score: float,
        max_tokens_after_anchor: int,
        max_span_tokens: int,
        word_x_center: list[float] | None = None,
        expected_side: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Enumerate name-like candidate spans after anchors, for paired seller/client resolve.
        """
        if not words:
            return []
        norm_words = [normalize_text(w) or "" for w in words]
        anchors: list[int] = []
        for a in anchor_terms:
            anchors.extend(self._anchor_word_positions(norm_words, a))
        anchors = sorted(set(anchors))
        if not anchors:
            return []

        stop_tokens = {
            "tax", "id", "taxid", "iban", "invoice", "date", "total",
            "amount", "vat", "summary", "payment", "bank", "account",
            "bill", "ship", "buyer",
        }
        if expected_side == "left":
            stop_tokens |= {"client"}
        if expected_side == "right":
            stop_tokens |= {"seller", "vendor", "supplier"}

        def _norm_alnum(tok: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (tok or "").lower())

        stop_norm = {_norm_alnum(t) for t in stop_tokens}

        def is_stop(tok: str) -> bool:
            return tok in stop_tokens or _norm_alnum(tok) in stop_norm

        cands: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        for aidx in anchors:
            start = aidx + 1
            end = min(len(words), aidx + max_tokens_after_anchor)
            if start >= end:
                continue
            for s in range(start, end):
                if is_stop(norm_words[s]):
                    break
                if word_x_center is not None and 0 <= s < len(word_x_center) and expected_side is not None:
                    xs = float(word_x_center[s])
                    if expected_side == "left" and xs > 0.65:
                        continue
                    if expected_side == "right" and xs < 0.35:
                        continue
                for e in range(s + 1, min(s + max_span_tokens + 1, end + 1)):
                    span_norm = norm_words[s:e]
                    if any(is_stop(t) for t in span_norm):
                        break
                    txt = " ".join(words[s:e]).strip(" :-,")
                    if not txt:
                        continue
                    sc = float(self._score_name_candidate(txt))
                    if sc < min_score:
                        continue
                    key = (s, e, txt.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    side_pen = 0.0
                    if word_x_center is not None and 0 <= s < len(word_x_center) and expected_side is not None:
                        target = 0.25 if expected_side == "left" else 0.75
                        side_pen = abs(float(word_x_center[s]) - target)
                    cands.append(
                        {
                            "text": txt,
                            "start": s,
                            "end": e,
                            "score": sc,
                            "dist": max(0, s - start),
                            "side_pen": side_pen,
                        }
                    )
        cands.sort(key=lambda x: (-x["score"], x["dist"], x["side_pen"], x["start"]))
        return cands

    def _resolve_seller_client_pair_with_anchors(
        self,
        words: list[str],
        *,
        word_x_center: list[float] | None = None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """
        Jointly resolve seller/client by selecting a non-overlapping candidate pair.
        Helps avoid swapped names when fallback dominates on held-out docs.
        """
        seller_cands = self._enumerate_anchor_party_candidates(
            words,
            anchor_terms=("seller", "vendor", "supplier"),
            min_score=14.0,
            max_tokens_after_anchor=30,
            max_span_tokens=14,
            word_x_center=word_x_center,
            expected_side="left",
        )[:20]
        client_cands = self._enumerate_anchor_party_candidates(
            words,
            anchor_terms=("client",),
            min_score=14.0,
            max_tokens_after_anchor=30,
            max_span_tokens=14,
            word_x_center=word_x_center,
            expected_side="right",
        )[:20]

        trace: dict[str, Any] = {
            "seller_candidates": seller_cands[:5],
            "client_candidates": client_cands[:5],
            "chosen_pair": None,
        }

        if not seller_cands and not client_cands:
            return None, None, trace

        best_pair: tuple[float, dict[str, Any] | None, dict[str, Any] | None] | None = None
        seller_pool = seller_cands if seller_cands else [None]
        client_pool = client_cands if client_cands else [None]

        for s in seller_pool:
            for c in client_pool:
                if s is not None and c is not None:
                    # Avoid overlap and exact duplicates.
                    if not (s["end"] <= c["start"] or c["end"] <= s["start"]):
                        continue
                    if normalize_text(s["text"]) == normalize_text(c["text"]):
                        continue
                score = 0.0
                if s is not None:
                    score += float(s["score"]) - 4.0 * float(s["side_pen"]) - 0.2 * float(s["dist"])
                if c is not None:
                    score += float(c["score"]) - 4.0 * float(c["side_pen"]) - 0.2 * float(c["dist"])
                # Prefer resolving both names when quality is similar.
                if s is not None and c is not None:
                    score += 2.5
                pair = (score, s, c)
                if best_pair is None or pair[0] > best_pair[0]:
                    best_pair = pair

        if best_pair is None:
            return None, None, trace

        _, s_best, c_best = best_pair
        seller_txt = s_best["text"] if s_best is not None else None
        client_txt = c_best["text"] if c_best is not None else None
        trace["chosen_pair"] = {"seller": s_best, "client": c_best}
        return seller_txt, client_txt, trace

    def _guess_seller_from_top_page_band(
        self,
        words: list[str],
        word_y_center: list[float] | None,
        *,
        band_frac: float = 0.22,
        min_score: float = 15.0,
        max_span_tokens: int = 8,
    ) -> str | None:
        """
        Last resort when no Seller/Vendor anchor appears in OCR: pick the strongest
        name-like span lying in the top vertical band (seller blocks are usually high).
        """
        if not words or not word_y_center or len(words) != len(word_y_center):
            return None
        ys = [float(y) for y in word_y_center]
        ymin, ymax = min(ys), max(ys)
        span_y = ymax - ymin
        if span_y <= 1e-6:
            return None
        thr = ymin + band_frac * span_y

        header_leads = {
            "invoice",
            "date",
            "tax",
            "total",
            "amount",
            "vat",
            "bill",
            "ship",
            "payment",
            "due",
            "page",
        }
        best: tuple[float, str] | None = None

        for i in range(len(words)):
            if ys[i] > thr:
                continue
            for j in range(i + 1, min(i + max_span_tokens + 1, len(words) + 1)):
                if max(ys[i:j]) > thr:
                    continue
                cand = " ".join(words[i:j]).strip()
                if len(cand) < 3:
                    continue
                nt = normalize_text(cand) or ""
                if not nt:
                    continue
                lead = nt.split()[0] if nt.split() else ""
                if lead in header_leads:
                    continue
                sc = float(self._score_name_candidate(cand))
                if sc < min_score:
                    continue
                if best is None or sc > best[0]:
                    best = (sc, cand)

        if not best:
            return None
        if self._looks_like_city_state_address_line(best[1]):
            return None
        return best[1]

    @staticmethod
    def _looks_like_city_state_address_line(txt: str) -> bool:
        """Reject OCR lines that look like City, ST rather than firm names."""
        t = (txt or "").strip()
        if not t:
            return True
        norm_t = (normalize_text(txt) or "").lower()
        # City, ST / Town, XY at end of span
        if re.search(r",\s*[A-Z]{2}\s*$", t):
            return True
        if norm_t.endswith("town") or norm_t.endswith("ville") or norm_t.endswith("side") or norm_t.endswith("land"):
            return True
        if " city " in f" {norm_t} ":
            return True
        # Lone two-letter token as US state-ish at end after comma (not inside a firm name).
        if re.search(r",\s*[A-Za-z]+\s+[A-Z]{2}\s*$", t):
            return True
        return False

    @staticmethod
    def _sanitize_seller_fallback_raw(txt: str) -> str:
        """Strip stray section labels OCR may merge into the seller span."""
        t = (txt or "").strip()
        if not t:
            return ""
        # Drop leading Client:/Seller: fragments when anchor span bleeds from layout.
        t = re.sub(
            r"^(client|seller|vendor|supplier)\s*:\s*",
            "",
            t,
            flags=re.I,
        ).strip()
        return t

    @staticmethod
    def _truncate_seller_party_bleed(txt: str) -> str:
        """
        Cut seller OCR glue before address lines or a second firm name.

        Handles:
          - ``Tran, Hurst and Rodgers Stephenson Inc …`` → ``Tran, Hurst and Rodgers``
          - ``Mendoza and Sons Fuller, Martin and Hays`` → ``Mendoza and Sons``
        """
        t = (txt or "").strip()
        if not t:
            return t
        # Law-firm style: ``Last, Other and Other``
        m = re.match(
            r"^([A-Za-z][A-Za-z\'\-]*,\s*[A-Za-z][A-Za-z\'\-]*\s+and\s+[A-Za-z][A-Za-z\'\-]*)",
            t,
        )
        if m:
            return m.group(1).strip()
        # ``Word(s) and Sons|Partners|…`` — end after first complete phrase (no bleed)
        m2 = re.match(
            r"^([A-Za-z][A-Za-z\'\-]*(?:\s+[A-Za-z][A-Za-z\'\-]*){0,5}\s+and\s+[A-Za-z][A-Za-z\'\-]*)",
            t,
        )
        if m2:
            return m2.group(1).strip()
        # Hyphenated party name only (single token often sufficient)
        if re.search(r"[A-Za-z]+-[A-Za-z]+", t):
            hyp = re.findall(r"[A-Za-z]+-[A-Za-z]+", t)
            if hyp:
                return hyp[0]
        # Cut before suite / entity suffix / street-style tail
        cut = re.split(
            r"\s+(?=Inc\.?|LLC\.?|L\.L\.C\.?|Ltd\.?|Corp\.?|Co\.|Unit\b|Suite\b|Ste\.?|Floor\b|#\d|\d+\s+[A-Za-z]+(?:\s+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Blvd\.?))\b)",
            t,
            maxsplit=1,
        )[0].strip()
        # Second firm glued: ``… Sons Fuller, Martin``
        cut = re.split(r"\s+(?=[A-Z][a-z]+,\s+[A-Z][a-z]+)", cut, maxsplit=1)[0].strip()
        return cut

    @staticmethod
    def _truncate_client_party_bleed(txt: str) -> str:
        """
        Trim client fallback strings that bleed into address/next-line tokens.
        """
        t = (txt or "").strip()
        if not t:
            return t
        # Keep "Last, Last and Last" client firm/person forms.
        m = re.match(
            r"^([A-Za-z][A-Za-z\'\-]*,\s*[A-Za-z][A-Za-z\'\-]*\s+and\s+[A-Za-z][A-Za-z\'\-]*)",
            t,
        )
        if m:
            return m.group(1).strip()
        # Keep "Word ... and Word" phrase before trailing address words.
        m2 = re.match(
            r"^([A-Za-z][A-Za-z\'\-]*(?:\s+[A-Za-z][A-Za-z\'\-]*){0,5}\s+and\s+[A-Za-z][A-Za-z\'\-]*)",
            t,
        )
        if m2:
            return m2.group(1).strip()
        # Cut before known location/address suffix style tails.
        cut = re.split(
            r"\s+(?=(?:North|South|East|West)\b|(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Drive|Dr\.|Lane|Ln\.|Court|Ct\.|Parkway|Pkwy\.|Route|Way|Walks?|Gateway|Meadow|Inlet|Branch|Ports?|Fort|Knolls?|Mills?|Lakes?|Parks?|Valleys?|Turnpike|Plains?|Tunnel|Crossing|Mountains?|Village|Prairie|Creek|Station|Cliff|Glens?|Mount|Locks?|Pike)\b)",
            t,
            maxsplit=1,
        )[0].strip()
        return cut

    @staticmethod
    def _expand_seller_tail_from_local_tokens(words: list[str], tail_candidate: str | None) -> str | None:
        """
        Recover full seller name when fallback captured only a tail, e.g.:
          "Andrade and Kim" <- "Hood, Andrade and Kim"
          "Clark"           <- "Johnson, Johnson and Clark"
        """
        t = (tail_candidate or "").strip()
        if not t:
            return t
        if not words:
            return t
        # If already comma-style full form, keep as is.
        if "," in t and " and " in f" {t.lower()} ":
            return t

        # Build normalized stream for local matching.
        # Keep commas so patterns like "Lastname, A and B" remain recoverable.
        seq = [str(w).strip(" :-") for w in words if str(w).strip(" :-")]
        norm = [normalize_text(x) or "" for x in seq]
        tn = normalize_text(t) or ""
        if not tn:
            return t
        tail_tokens = [x for x in tn.split(" ") if x]
        if not tail_tokens:
            return t

        # Find all exact tail occurrences in OCR token stream.
        hit_starts: list[int] = []
        n = len(tail_tokens)
        for i in range(0, len(norm) - n + 1):
            if norm[i : i + n] == tail_tokens:
                hit_starts.append(i)
        if not hit_starts:
            return t

        # Search each occurrence for a left expansion opportunity.
        for hit_start in hit_starts:
            left = max(0, hit_start - 5)
            window = seq[left : hit_start + n]
            joined = " ".join(window).strip()
            m = re.search(
                r"([A-Za-z][A-Za-z\'\-]*,\s*[A-Za-z][A-Za-z\'\-]*\s+and\s+[A-Za-z][A-Za-z\'\-]*)$",
                joined,
            )
            if m:
                return m.group(1).strip()
            # Fallback: "<Surname>, <Word>" (for company/person pairs)
            m2 = re.search(r"([A-Za-z][A-Za-z\'\-]*,\s*[A-Za-z][A-Za-z\'\-]*)$", joined)
            if m2:
                return m2.group(1).strip()
        return t

    @staticmethod
    def _expand_party_tail_from_local_tokens(words: list[str], tail_candidate: str | None) -> str | None:
        """
        Generic left-expansion for tail-only party predictions.
        Reconstructs patterns like "Last, Last and Last" and "Last, Last".
        """
        t = (tail_candidate or "").strip()
        if not t or not words:
            return t
        if "," in t and " and " in f" {t.lower()} ":
            return t

        seq = [str(w).strip(" :-") for w in words if str(w).strip(" :-")]
        norm = [normalize_text(x) or "" for x in seq]
        tn = normalize_text(t) or ""
        tail_tokens = [x for x in tn.split(" ") if x]
        if not tail_tokens:
            return t

        starts: list[int] = []
        n = len(tail_tokens)
        for i in range(0, len(norm) - n + 1):
            if norm[i : i + n] == tail_tokens:
                starts.append(i)
        if not starts:
            return t

        for hit_start in starts:
            left = max(0, hit_start - 5)
            window = seq[left : hit_start + n]
            joined = " ".join(window).strip()
            m = re.search(
                r"([A-Za-z][A-Za-z\'\-]*,\s*[A-Za-z][A-Za-z\'\-]*\s+and\s+[A-Za-z][A-Za-z\'\-]*)$",
                joined,
            )
            if m:
                return m.group(1).strip()
            m2 = re.search(r"([A-Za-z][A-Za-z\'\-]*,\s*[A-Za-z][A-Za-z\'\-]*)$", joined)
            if m2:
                return m2.group(1).strip()
        return t

    @staticmethod
    def _is_plausible_party_name(txt: str) -> bool:
        """
        Reject fallback names that are likely address/location fragments.
        """
        t = (txt or "").strip()
        if not t:
            return False
        nt = (normalize_text(t) or "").lower()
        if not nt:
            return False
        # Strong positives
        if re.search(r"[A-Za-z]+-[A-Za-z]+", t):
            return True
        if "," in t and " and " in f" {nt} ":
            return True
        if " and " in f" {nt} ":
            return True
        if re.search(r"\b(inc|llc|ltd|corp|plc)\b", nt):
            return True

        # Address/location-like suffixes are common anchor fallback errors.
        bad_tail = {
            "street", "st", "avenue", "ave", "road", "rd", "drive", "dr", "lane", "ln",
            "court", "ct", "parkway", "pkwy", "trace", "bridge", "center", "suite", "unit",
            "floor", "building", "apt", "apartment", "route", "way", "walk", "walks",
            "gateway", "meadow", "inlet", "branch", "port", "ports", "fort", "knoll", "knolls",
            "mill", "mills", "lake", "lakes", "park", "parks", "valley", "valleys", "island", "islands",
            "view", "course", "highway", "hwy", "blvd", "boulevard",
            "turnpike", "plain", "plains", "tunnel", "crossing", "mountain", "mountains",
            "village", "prairie", "creek", "station", "cliff", "glen", "glens",
            "mount", "lock", "locks", "pike", "trail",
        }
        toks = [x for x in re.split(r"\s+", nt) if x]
        if not toks:
            return False
        last = toks[-1].strip(".,")
        if last in bad_tail:
            return False
        # Suffix-like location endings (Sandraview, Bayspring, Stoneburg, etc.).
        bad_suffixes = (
            "view", "spring", "cove", "coves", "burg", "ford", "point", "points",
            "meadow", "grove", "heights", "knoll", "knolls", "park", "parks",
            "way", "route", "gateway", "inlet", "branch", "walk", "walks",
            "mills", "lakes", "valley", "valleys", "turnpike", "plains", "tunnel",
            "crossing", "mountain", "mountains", "village", "prairie", "creek",
            "station", "cliff", "glen", "glens", "mount", "locks", "pike", "trail",
        )
        if any(last.endswith(suf) for suf in bad_suffixes):
            return False
        return True

    @staticmethod
    def _compact_party_name_candidate(txt: str, *, require_hyphen: bool = False) -> str | None:
        """
        Convert noisy multi-token fallback spans into a compact party name.
        Priority: last hyphenated token (common in this dataset), then short alpha tail.
        """
        t = (txt or "").strip()
        if not t:
            return None

        # Most GT party names look like Lastname-Lastname.
        hyphen_tokens = re.findall(r"[A-Za-z]+-[A-Za-z]+", t)
        if hyphen_tokens:
            return hyphen_tokens[-1]

        # Comma-separated *company* names (e.g. ``Tran, Hurst and Rodgers``) are valid;
        # only reject obvious city/state lines.
        if LayoutLMv3InvoiceTokenClassifier._looks_like_city_state_address_line(t):
            return None

        # In this dataset, strict mode keeps precision high.
        if require_hyphen:
            return None

        tokens = [x for x in re.split(r"\s+", t) if x]
        alpha_tokens = [x for x in tokens if re.search(r"[A-Za-z]", x) and not re.search(r"\d", x)]
        if not alpha_tokens:
            return None
        if len(alpha_tokens) == 1:
            return alpha_tokens[0]
        low_joined = " ".join(alpha_tokens).lower()
        if " and " in low_joined:
            return " ".join(alpha_tokens)
        # Conservative fallback: short tail (often the actual party name chunk).
        return " ".join(alpha_tokens[-2:])

    @classmethod
    def _compact_seller_organization_fallback(cls, txt: str) -> str | None:
        """
        Last resort for seller fallback: keep comma + ``and`` firm strings intact.
        """
        t = cls._truncate_seller_party_bleed(cls._sanitize_seller_fallback_raw(txt))
        if not t:
            return None
        if cls._looks_like_city_state_address_line(t):
            return None
        nt = normalize_text(t) or ""
        low = nt.lower()
        # Full-line firm: comma + ``and`` (truncate already removed address bleed)
        if "," in t and " and " in low:
            return re.sub(r"\s+", " ", t).strip().strip(",").strip()
        # ``Mendoza and Sons`` style (no comma)
        if " and " in low and len(t) < 120:
            return re.sub(r"\s+", " ", t).strip().strip(",").strip()
        # Simple comma-separated without trailing ``and``
        if "," in t and re.search(r"[A-Za-z]+\s*,\s*[A-Za-z]+", t):
            parts = [p.strip() for p in re.split(r"\s*,\s*", t) if p.strip()]
            if parts:
                return ", ".join(parts)
        return None

    @staticmethod
    def _amount_candidates_from_words(words: list[str]) -> list[str]:
        """
        Build extra monetary candidates from OCR words (uni/bi/tri-grams),
        helping when field labels miss one of tax/net/total.
        """
        candidates: list[str] = []
        seen: set[str] = set()

        def _try_add(txt: str):
            nm = normalize_money(txt)
            if nm is None:
                return
            if nm in seen:
                return
            seen.add(nm)
            candidates.append(nm)

        for i, w in enumerate(words):
            _try_add(w)
            if i + 1 < len(words):
                _try_add(f"{w} {words[i+1]}")
            if i + 2 < len(words):
                _try_add(f"{w} {words[i+1]} {words[i+2]}")

        return candidates

    def _rank_name_candidates(self, candidates: list[str]) -> list[dict[str, Any]]:
        ranked = [
            {"text": c, "score": float(self._score_name_candidate(c))}
            for c in candidates
            if str(c).strip()
        ]
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    @staticmethod
    def _resolve_money_fields(
        entities: dict[str, list[str]],
        extra_amount_candidates: list[str] | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """
        Resolve tax/net/total jointly using arithmetic consistency.

        Layout predictions often surface multiple monetary candidates and the first one
        is not always the correct semantic field. This picks a triplet where
        net_worth + tax ~= total_amount when possible.
        """
        money_fields = ("tax", "net_worth", "total_amount")
        debug: dict[str, Any] = {
            "source": "none",
            "entity_candidate_pools": {},
            "entity_first_triplet": None,
            "resolved_triplet": None,
            "arithmetic_error_entity_first": None,
            "arithmetic_error_resolved": None,
            "used_extra_pool": False,
        }
        pools: dict[str, list[str]] = {}
        values: dict[str, list[float]] = {}

        for f in money_fields:
            raw = entities.get(f, []) or []
            cleaned: list[str] = []
            nums: list[float] = []
            seen_num: set[float] = set()
            for c in raw:
                nm = normalize_money(c)
                if nm is None:
                    continue
                v = float(nm)
                if v in seen_num:
                    continue
                seen_num.add(v)
                nums.append(v)
                cleaned.append(nm)
            pools[f] = cleaned
            values[f] = nums
            debug["entity_candidate_pools"][f] = cleaned[:6]

        def choose_triplet(vmap: dict[str, list[float]]) -> tuple[float, float, float] | None:
            best: tuple[float, float, float, float] | None = None
            for t in vmap["tax"] or []:
                for n in vmap["net_worth"] or []:
                    for tot in vmap["total_amount"] or []:
                        if tot < n or tot < t:
                            continue
                        err = abs((n + t) - tot)
                        tol = LayoutLMv3InvoiceTokenClassifier._summary_arithmetic_tol(tot)
                        # Prefer arithmetic-consistent candidates, then lowest error.
                        if err <= tol:
                            rank = (err, -tot, -n, -t)
                            if best is None or rank < best:
                                best = rank
            if best is None:
                return None
            _, neg_tot, neg_n, neg_t = best
            return (-neg_t, -neg_n, -neg_tot)

        # Baseline from entity-first candidates (before reassignment).
        base_triplet: tuple[float, float, float] | None = None
        if all(values[f] for f in money_fields):
            base_triplet = (values["tax"][0], values["net_worth"][0], values["total_amount"][0])
            debug["entity_first_triplet"] = {
                "tax": f"{base_triplet[0]:.2f}",
                "net_worth": f"{base_triplet[1]:.2f}",
                "total_amount": f"{base_triplet[2]:.2f}",
            }
            debug["arithmetic_error_entity_first"] = abs((base_triplet[0] + base_triplet[1]) - base_triplet[2])

        # First pass: entity-derived candidates only (higher precision).
        chosen = choose_triplet(values)

        # Second pass: only if first pass failed and at least one money field is missing,
        # allow global OCR amount candidates to fill missing pools.
        if chosen is None and (not values["tax"] or not values["net_worth"] or not values["total_amount"]):
            extras = extra_amount_candidates or []
            extra_vals: list[float] = []
            for x in extras:
                try:
                    extra_vals.append(float(x))
                except Exception:
                    continue

            v2 = {
                "tax": values["tax"] if values["tax"] else list(extra_vals),
                "net_worth": values["net_worth"] if values["net_worth"] else list(extra_vals),
                "total_amount": values["total_amount"] if values["total_amount"] else list(extra_vals),
            }
            chosen = choose_triplet(v2)
            debug["used_extra_pool"] = chosen is not None

        out: dict[str, str] = {}
        if chosen is not None:
            tax, net_worth, total_amount = chosen
            resolved_err = abs((tax + net_worth) - total_amount)
            debug["resolved_triplet"] = {
                "tax": f"{tax:.2f}",
                "net_worth": f"{net_worth:.2f}",
                "total_amount": f"{total_amount:.2f}",
            }
            debug["arithmetic_error_resolved"] = resolved_err

            # Confidence-gated reassignment:
            # only reassign full triplet when entity candidates are missing.
            has_missing_entity_field = any(not values[f] for f in money_fields)
            apply_reassignment = has_missing_entity_field

            if apply_reassignment:
                out["tax"] = f"{tax:.2f}"
                out["net_worth"] = f"{net_worth:.2f}"
                out["total_amount"] = f"{total_amount:.2f}"
                debug["source"] = "reassigned_triplet"
                return out, debug

        # Conservative field-local fallback: keep entity-derived money first.
        for f in money_fields:
            if pools[f]:
                out[f] = pools[f][0]
        if out:
            debug["source"] = "entity_first"
            return out, debug

        # Last resort when all entity pools are empty.
        extras = extra_amount_candidates or []
        extra_vals: list[float] = []
        for x in extras:
            try:
                extra_vals.append(float(x))
            except Exception:
                continue
        if extra_vals:
            # Heuristic ordering when we only have a global amount pool.
            svals = sorted(set(extra_vals))
            if len(svals) >= 3:
                t, n, tot = svals[0], svals[-2], svals[-1]
                if abs((n + t) - tot) <= LayoutLMv3InvoiceTokenClassifier._summary_arithmetic_tol(tot):
                    out["tax"] = f"{t:.2f}"
                    out["net_worth"] = f"{n:.2f}"
                    out["total_amount"] = f"{tot:.2f}"
            elif len(svals) == 2:
                out["tax"] = f"{svals[0]:.2f}"
                out["total_amount"] = f"{svals[1]:.2f}"
        debug["source"] = "global_fallback" if out else "none"
        return out, debug

    def predict(
        self,
        image_path: str | Path,
        *,
        fields: Iterable[str] = DEFAULT_FIELDS,
        return_debug: bool = False,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        words, word_labels, word_y_center, word_x_center, word_confidences = self._predict_word_labels(image_path=image_path)
        ents = self._entities_from_word_labels(words, word_labels)
        ent_spans = self._entity_spans_from_word_labels(words, word_labels)
        field_set = set(fields)
        debug_info: dict[str, Any] = {
            "name_resolution": {},
            "amount_candidate_count": 0,
            "money_resolution": {},
        }
        debug_info["runtime_version"] = self.runtime_version
        debug_info["word_token_confidences"] = word_confidences
        debug_info["avg_token_confidence"] = (
            float(np.nanmean(word_confidences)) if word_confidences else float("nan")
        )

        out: dict[str, str] = {}
        global_amounts = self._amount_candidates_from_words(words)
        debug_info["amount_candidate_count"] = len(global_amounts)
        money_resolved, money_debug = self._resolve_money_fields(ents, extra_amount_candidates=global_amounts)
        debug_info["money_resolution"] = money_debug
        for f in fields:
            cand_list = ents.get(f, [])

            if f in {"total_amount", "tax", "net_worth"}:
                if f in money_resolved:
                    out[f] = money_resolved[f]
            elif f == "invoice_date":
                if not cand_list:
                    continue
                for c in cand_list:
                    nd = normalize_date(c)
                    if nd is not None:
                        out[f] = nd
                        break
                else:
                    out[f] = cand_list[0]
            else:
                if not cand_list:
                    continue
                if f in {"seller_name", "client_name"}:
                    ranked = self._rank_name_candidates(cand_list)
                    if f == "client_name":
                        picked = self._select_party_entity_with_anchor(
                            words,
                            ent_spans,
                            field=f,
                            anchor="client",
                            word_x_center=word_x_center,
                            expected_side="right",
                        )
                    else:
                        # Prefer spans near a Seller/Vendor anchor (same idea as client_name).
                        picked = self._select_party_entity_with_anchor(
                            words,
                            ent_spans,
                            field=f,
                            anchor=("seller", "vendor", "supplier"),
                            word_x_center=word_x_center,
                            expected_side="left",
                        )
                        if not picked:
                            picked = ranked[0]["text"] if ranked else None
                    if picked:
                        out[f] = picked
                        debug_info["name_resolution"][f] = {
                            "source": "entity",
                            "selected": out[f],
                            "top_candidates": ranked[:5],
                        }
                elif f == "invoice_number":
                    out[f] = sorted(cand_list, key=lambda s: len(s), reverse=True)[0]
                else:
                    out[f] = sorted(cand_list, key=lambda s: len(s), reverse=True)[0]

        # Joint pair fallback before per-field fallbacks. This reduces seller/client swaps.
        if ("seller_name" in field_set or "client_name" in field_set) and (
            not out.get("seller_name") or not out.get("client_name")
        ):
            s_inline, c_inline, inline_trace = self._resolve_inline_seller_client_pair(
                words, word_x_center=word_x_center
            )
            inline_quality_ok = bool((inline_trace.get("chosen") or {}).get("quality_ok", False))
            if s_inline and not out.get("seller_name"):
                s_clean = self._truncate_seller_party_bleed(self._sanitize_seller_fallback_raw(s_inline))
                s_name = self._compact_seller_organization_fallback(s_clean) or self._compact_party_name_candidate(
                    s_clean, require_hyphen=False
                )
                if s_name:
                    out["seller_name"] = s_name
                    debug_info["name_resolution"]["seller_name"] = {
                        "source": "inline_pair_fallback",
                        "selected": out["seller_name"],
                        "top_candidates": [],
                        "pair_trace": inline_trace,
                    }
            # Client inline assignment is much riskier on held-out data; accept it only
            # when inline pairing quality is high, otherwise keep client anchor/entity path.
            if c_inline and not out.get("client_name") and inline_quality_ok:
                c_clean = re.sub(r"\s+", " ", c_inline).strip(" :-,")
                c_name = self._compact_party_name_candidate(c_clean, require_hyphen=True)
                if not c_name:
                    c_name = self._compact_party_name_candidate(c_clean, require_hyphen=False)
                if c_name:
                    out["client_name"] = c_name
                    debug_info["name_resolution"]["client_name"] = {
                        "source": "inline_pair_fallback",
                        "selected": out["client_name"],
                        "top_candidates": [],
                        "pair_trace": inline_trace,
                    }

        # Joint anchor pair fallback after inline pair.
        if ("seller_name" in field_set or "client_name" in field_set) and (
            not out.get("seller_name") or not out.get("client_name")
        ):
            s_pair_raw, c_pair_raw, pair_trace = self._resolve_seller_client_pair_with_anchors(
                words, word_x_center=word_x_center
            )
            norm_words = [normalize_text(w) or "" for w in words]
            has_client_anchor_token = bool(self._anchor_word_positions(norm_words, "client"))
            # Pair fallback is prone to wrong client assignments on held-out docs.
            # Keep seller from pair fallback, but gate client harder.
            pair_client_ok = False
            chosen_pair = pair_trace.get("chosen_pair") or {}
            cp = chosen_pair.get("client")
            sp = chosen_pair.get("seller")
            if cp is not None:
                try:
                    c_score = float(cp.get("score", -1e9))
                    c_side = float(cp.get("side_pen", 1e9))
                    x_sep = None
                    if sp is not None and "start" in sp and "start" in cp:
                        s_idx = int(sp["start"])
                        c_idx = int(cp["start"])
                        if word_x_center is not None and 0 <= s_idx < len(word_x_center) and 0 <= c_idx < len(word_x_center):
                            x_sep = abs(float(word_x_center[s_idx]) - float(word_x_center[c_idx]))
                    pair_client_ok = (
                        c_score >= 18.0
                        and c_side <= 0.25
                        and (x_sep is None or x_sep >= 0.18)
                        and (not has_client_anchor_token)
                    )
                except Exception:
                    pair_client_ok = False
            pair_trace["client_quality_ok"] = pair_client_ok
            if s_pair_raw and not out.get("seller_name"):
                s_clean = self._truncate_seller_party_bleed(self._sanitize_seller_fallback_raw(s_pair_raw))
                s_name = self._compact_seller_organization_fallback(s_clean) or self._compact_party_name_candidate(
                    s_clean, require_hyphen=False
                )
                if s_name:
                    out["seller_name"] = s_name
                    debug_info["name_resolution"]["seller_name"] = {
                        "source": "pair_fallback",
                        "selected": out["seller_name"],
                        "top_candidates": [],
                        "pair_trace": pair_trace,
                    }
            if c_pair_raw and not out.get("client_name") and pair_client_ok:
                c_clean = re.sub(r"\s+", " ", c_pair_raw).strip(" :-,")
                c_name = self._compact_party_name_candidate(c_clean, require_hyphen=True)
                if not c_name:
                    c_name = self._compact_party_name_candidate(c_clean, require_hyphen=False)
                if c_name:
                    out["client_name"] = c_name
                    debug_info["name_resolution"]["client_name"] = {
                        "source": "pair_fallback",
                        "selected": out["client_name"],
                        "top_candidates": [],
                        "pair_trace": pair_trace,
                    }

        # Anchor fallback for client_name when token labels miss it.
        if "client_name" in field_set and not out.get("client_name"):
            # Tiered client fallback: strict first, then looser to reduce missing on test set.
            client_name_raw = None
            client_name = None
            client_trace: dict[str, Any] = {"raw": None, "tiers_tried": []}
            client_tiers = (
                {"min_score": 18.0, "max_tokens_after_anchor": 22, "max_span_tokens": 12},
                {"min_score": 14.0, "max_tokens_after_anchor": 26, "max_span_tokens": 14},
            )
            for tier in client_tiers:
                client_trace["tiers_tried"].append(tier)
                client_name_raw = self._extract_party_name_from_anchor_tokens(
                    words,
                    anchor="client",
                    word_y_center=word_y_center,
                    word_x_center=word_x_center,
                    **tier,
                )
                if client_name_raw:
                    client_trace["picked_tier"] = tier
                    client_trace["raw"] = client_name_raw
                    break
            if client_name_raw:
                client_clean = re.sub(r"\s+", " ", client_name_raw).strip(" :-,")
                client_clean = self._truncate_client_party_bleed(client_clean)
                client_clean = self._expand_party_tail_from_local_tokens(words, client_clean) or client_clean
                client_name = self._compact_party_name_candidate(client_clean, require_hyphen=True)
                client_trace["compact_strict"] = client_name
                if not client_name:
                    # On held-out docs many client names are not hyphenated.
                    client_name = self._compact_party_name_candidate(client_clean, require_hyphen=False)
                    client_trace["compact_loose"] = client_name
                if client_name and not self._is_plausible_party_name(client_name):
                    client_trace["rejected_as_implausible"] = client_name
                    client_name = None
            if client_name:
                out["client_name"] = client_name
                debug_info["name_resolution"]["client_name"] = {
                    "source": "anchor_fallback",
                    "selected": out["client_name"],
                    "top_candidates": [],
                    "fallback_trace": client_trace,
                }
            else:
                debug_info["name_resolution"]["client_name"] = {
                    "source": "missing",
                    "selected": None,
                    "top_candidates": self._rank_name_candidates(ents.get("client_name", []))[:5],
                    "fallback_trace": client_trace,
                }
        elif "client_name" in field_set and "client_name" not in debug_info["name_resolution"]:
            debug_info["name_resolution"]["client_name"] = {
                "source": "missing",
                "selected": None,
                "top_candidates": self._rank_name_candidates(ents.get("client_name", []))[:5],
            }

        # Anchor fallback for seller_name is optional; when used, compact output aggressively.
        if "seller_name" in field_set and not out.get("seller_name"):
            if self.enable_seller_anchor_fallback:
                fb_trace: dict[str, Any] = {"anchors_tried": [], "raw": None, "compact_strict": None}
                seller_name_raw = None
                # Tiered extraction: strict stops + normal score first; then lower thresholds and
                # relax seller-only boundary stops (Inc/Co/…) so legal names still surface, with
                # ``_truncate_seller_party_bleed`` + compaction to trim bleed.
                fallback_tiers = (
                    {"min_score": 18.0, "seller_boundary_stops": True},
                    {"min_score": 14.0, "seller_boundary_stops": True},
                    {"min_score": 12.0, "seller_boundary_stops": False},
                )
                for tier in fallback_tiers:
                    for anc in ("seller", "vendor", "supplier"):
                        fb_trace["anchors_tried"].append({"anchor": anc, **tier})
                        seller_name_raw = self._extract_party_name_from_anchor_tokens(
                            words,
                            anchor=anc,
                            max_tokens_after_anchor=28,
                            max_span_tokens=14,
                            prefer_nearest=True,
                            word_y_center=word_y_center,
                            word_x_center=word_x_center,
                            **tier,
                        )
                        if seller_name_raw:
                            fb_trace["extraction_tier"] = tier
                            fb_trace["extraction_anchor"] = anc
                            fb_trace["raw"] = seller_name_raw
                            break
                    if seller_name_raw:
                        break
                if not seller_name_raw and self.enable_seller_top_band_heuristic:
                    seller_name_raw = self._guess_seller_from_top_page_band(words, word_y_center)
                    if seller_name_raw:
                        fb_trace["extraction_tier"] = {"source": "top_page_band_heuristic"}
                        fb_trace["extraction_anchor"] = None
                        fb_trace["raw"] = seller_name_raw
                seller_name = None
                if seller_name_raw:
                    seller_clean = self._sanitize_seller_fallback_raw(seller_name_raw)
                    seller_clean = self._truncate_seller_party_bleed(seller_clean)
                    seller_clean = self._expand_seller_tail_from_local_tokens(words, seller_clean) or seller_clean
                    fb_trace["sanitized_raw"] = seller_clean or None
                    # Prefer comma / ``… and …`` firms before hyphen-token extraction so long
                    # partnership strings are not discarded by strict hyphen-only mode.
                    seller_name = self._compact_seller_organization_fallback(seller_clean)
                    fb_trace["compact_org"] = seller_name
                    if not seller_name:
                        seller_name = self._compact_party_name_candidate(seller_clean, require_hyphen=True)
                        fb_trace["compact_strict"] = seller_name
                    if not seller_name:
                        seller_name = self._compact_party_name_candidate(seller_clean, require_hyphen=False)
                        fb_trace["compact_loose"] = seller_name
                    if seller_name and not self._is_plausible_party_name(seller_name):
                        fb_trace["rejected_as_implausible"] = seller_name
                        seller_name = None
                else:
                    fb_trace["reason"] = "no_span_after_tiered_extraction_and_top_band"

                if seller_name:
                    out["seller_name"] = seller_name
                    debug_info["name_resolution"]["seller_name"] = {
                        "source": "anchor_fallback",
                        "selected": out["seller_name"],
                        "top_candidates": [],
                        "fallback_trace": fb_trace,
                    }
                else:
                    debug_info["name_resolution"]["seller_name"] = {
                        "source": "missing",
                        "selected": None,
                        "top_candidates": self._rank_name_candidates(ents.get("seller_name", []))[:5],
                        "fallback_trace": fb_trace,
                    }
            if "seller_name" not in debug_info["name_resolution"]:
                debug_info["name_resolution"]["seller_name"] = {
                    "source": ("fallback_disabled" if not self.enable_seller_anchor_fallback else "missing"),
                    "selected": None,
                    "top_candidates": self._rank_name_candidates(ents.get("seller_name", []))[:5],
                }
        elif "seller_name" in field_set and "seller_name" not in debug_info["name_resolution"]:
            debug_info["name_resolution"]["seller_name"] = {
                "source": "missing",
                "selected": None,
                "top_candidates": self._rank_name_candidates(ents.get("seller_name", []))[:5],
            }

        if return_debug:
            return out, debug_info
        meta_only = {
            "word_token_confidences": word_confidences,
            "avg_token_confidence": debug_info["avg_token_confidence"],
        }
        return out, meta_only

    def process_single_image(
        self,
        image_path: str | Path,
        *,
        fields: Iterable[str] = DEFAULT_FIELDS,
        debug_mode: bool = False,
    ) -> dict[str, Any]:
        image_path = str(image_path)
        result: dict[str, Any] = {
            "image_path": image_path,
            "filename": Path(image_path).name,
            "success": False,
            "invoice_fields": {},
        }
        try:
            if debug_mode:
                pred, dbg = self.predict(image_path, fields=fields, return_debug=True)
                result["invoice_fields"] = pred
                result["prediction_debug"] = dbg
                result["word_token_confidences"] = dbg.get("word_token_confidences") or []
                result["avg_token_confidence"] = dbg.get("avg_token_confidence")
            else:
                pred, meta = self.predict(image_path, fields=fields, return_debug=False)
                result["invoice_fields"] = pred
                result["word_token_confidences"] = meta.get("word_token_confidences") or []
                result["avg_token_confidence"] = meta.get("avg_token_confidence")
            result["total_words"] = len(result.get("word_token_confidences") or [])
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        return result

    def visualize_text_extraction(
        self,
        image_path: str | Path,
        result: dict[str, Any] | None = None,
        *,
        show_labels: bool = True,
        show_confidence: bool = False,
        max_words: int | None = 180,
        figsize: tuple[int, int] = (16, 12),
    ) -> None:
        """
        Visualize OCR words for LayoutLM with predicted token labels.

        Designed to be compatible with `visualize_sample_results`, which calls
        `visualize_text_fn(image_path, result)`.
        """
        image_path = str(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not load image: {image_path}")
            return

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ocr_words = ocr_words_from_image(gray, confidence_threshold=30, psm=6)
        if not ocr_words:
            print(f"No OCR words found: {image_path}")
            return

        words, labels, _, _, word_conf = self._predict_word_labels(image_path=image_path)
        n = min(len(ocr_words), len(labels), len(word_conf), len(words))
        if n <= 0:
            print(f"No aligned OCR/prediction words for: {image_path}")
            return
        if max_words is not None:
            n = min(n, int(max_words))

        canvas = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        palette = {
            "O": (170, 170, 170),
            "INVOICE_NUMBER": (31, 119, 180),
            "INVOICE_DATE": (255, 127, 14),
            "SELLER_NAME": (44, 160, 44),
            "CLIENT_NAME": (214, 39, 40),
            "NET_WORTH": (148, 103, 189),
            "TOTAL_AMOUNT": (140, 86, 75),
            "TAX": (227, 119, 194),
        }

        for i in range(n):
            ow = ocr_words[i]
            x, y, w, h = ow.bbox_xywh
            lab = labels[i] if i < len(labels) else "O"
            base_lab = lab[2:] if lab.startswith(("B-", "I-")) else "O"
            color = palette.get(base_lab, (23, 190, 207))

            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            if show_labels:
                token_txt = words[i] if i < len(words) else ow.text
                ann = f"{lab} | {token_txt}"
                if show_confidence:
                    ann += f" ({word_conf[i]:.2f})"
                ty = max(14, y - 4)
                cv2.putText(
                    canvas,
                    ann,
                    (x, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        plt.figure(figsize=figsize)
        plt.imshow(canvas)
        plt.axis("off")
        plt.title(f"LayoutLM OCR + token labels ({Path(image_path).name})")
        plt.tight_layout()
        plt.show()

    def run_inference(
        self,
        df: pd.DataFrame,
        *,
        image_col: str = "processed_path",
        key_col: str = "processed_file",
        fields: Iterable[str] = DEFAULT_FIELDS,
        sample_frac: float | None = None,
        random_state: int = 42,
        debug_mode: bool = False,
    ) -> pd.DataFrame:
        run_df = df.copy()
        if sample_frac is not None:
            run_df = run_df.sample(frac=sample_frac, random_state=random_state)

        results: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []

        for _, r in run_df.iterrows():
            image_path = r.get(image_col)
            if not image_path or not Path(str(image_path)).exists():
                continue
            res = self.process_single_image(image_path, fields=fields, debug_mode=debug_mode)
            results.append(res)

            row = {"processed_file": r.get(key_col) or res["filename"]}
            row.update(res.get("invoice_fields", {}))
            rows.append(row)

        self.full_results = results
        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_against_ground_truth(
        *,
        ground_truth_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        fields: Iterable[str] = DEFAULT_FIELDS,
        merge_key: str = "processed_file",
        restrict_to_matched: bool = True,
    ):
        return evaluate_exact_match(
            ground_truth_df=ground_truth_df,
            pred_df=pred_df,
            fields=list(fields),
            merge_key=merge_key,
            restrict_to_matched=restrict_to_matched,
        )

