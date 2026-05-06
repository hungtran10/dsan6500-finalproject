"""Training utilities for invoice-only Donut fine-tuning.

This module is intentionally separate from the inference pipeline.
It provides:
- canonical invoice payload construction
- mild document-safe augmentation
- dataset + collator
- weighted seq2seq training
- validation metrics
- device / batch-size helpers

The training target format is a single line of bracket fields (see build_structured_invoice_text), e.g.:
    <s_invoice>[inv_no]=... | [inv_dt]=... | [seller]=... | [client]=... | [net]=... | [tax]=... | [amt]=...</s>

The canonical schema is fixed and ordered. The same parser is used for
training metrics and validation/inference-style parsing.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from transformers import (
    DonutProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    GenerationConfig
)
from transformers.models.bart.modeling_bart import shift_tokens_right


CANONICAL_INVOICE_FIELDS = [
    "invoice_number",
    "invoice_date",
    "seller_name",
    "client_name",
    "net_worth",
    "tax",
    "total_amount",
]

@dataclass
class DonutFineTuningConfig:
    model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    task_prompt_invoice: str = "<s_invoice>"
    label_max_length: int = 256
    generation_max_new_tokens: int = 256
    num_train_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 3
    numeric_loss_weight: float = 1.5
    augment_factor: int = 1
    random_state: int = 42
    device: Optional[str] = None

# Device helpers
def resolve_donut_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve the best available device for Donut training/inference.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        device (Optional[str]): Input parameter. Defaults to None.
    Outputs:
        torch.device: Function output value.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def recommend_donut_batch_sizes(device: torch.device, model_name: str = "") -> Tuple[int, int, int]:
    """
    Recommend train/eval batch sizes and gradient accumulation for the device.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        device (torch.device): Input parameter.
        model_name (str): Input parameter. Defaults to ''.
    Outputs:
        Tuple[int, int, int]: Function output value.
    """
    if device.type == "mps":
        return 1, 1, 8
    if device.type == "cuda":
        base = 2 if "base" in model_name else 1
        return base, base, 4
    return 1, 1, 1



# Normalization and parsing helpers
def normalize_money(value: Any) -> Optional[str]:
    """
    Normalize money strings to a plain decimal string with two places.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        value (Any): Input parameter.
    Outputs:
        Optional[str]: Function output value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    s = str(value).strip().replace(" ", "")
    if not s:
        return None

    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = s.replace("USD", "").replace("EUR", "").replace("GBP", "")

    # Handle parentheses as negatives.
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

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
        val = float(s)
        if negative:
            val = -val
        return f"{val:.2f}"
    except ValueError:
        return None


def normalize_date(value: Any) -> Optional[str]:
    """
    Normalize date-like values to ISO format (YYYY-MM-DD).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        value (Any): Input parameter.
    Outputs:
        Optional[str]: Function output value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    s = str(value).strip()
    if not s:
        return None

    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return dt.strftime("%Y-%m-%d")
    return None


def normalize_invoice_field(value: Any, field_name: str) -> Any:
    """
    Normalize one invoice field for training/evaluation.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        value (Any): Input parameter.
        field_name (str): Input parameter.
    Outputs:
        Any: Function output value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    s = str(value).strip()
    if s.lower() in {"", "nan", "none"}:
        return None

    if field_name in {"tax", "net_worth", "total_amount"}:
        return normalize_money(s)
    if field_name == "invoice_date":
        return normalize_date(s)

    return re.sub(r"\s+", " ", s).lower()


def _strip_task_token(sequence: str) -> str:
    """
    Remove the leading Donut task token from a decoded sequence.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        sequence (str): Input parameter.
    Outputs:
        str: Function output value.
    """
    return re.sub(r"^<[^>]+>", "", sequence).strip()


def safe_json_loads(sequence: str) -> Dict[str, Any]:
    """
    Parse a Donut-like sequence into a dictionary when possible.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        sequence (str): Input parameter.
    Outputs:
        Dict[str, Any]: Function output value.
    """
    cleaned = sequence.replace("<pad>", "").replace("</s>", "").strip()
    cleaned = _strip_task_token(cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except Exception:
        try:
            parsed = json.loads(cleaned.replace("'", '"'))
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception:
            return {"raw_text": cleaned}

def _sanitize_invoice_number_value(val: Any) -> str:
    """
    Strip merged ISO tails like 56014042-10-01 → 56014042; avoid digit-concat bugs.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        val (Any): Input parameter.
    Outputs:
        str: Function output value.
    """
    s = str(val).strip().lower()
    if not s:
        return ""
    # If the entire token is a date (2012-09-22), it is not an invoice number.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return ""
    # Merged bracket: "[inv_no]=60866416]=2021-02-17" → digits before inner "]="
    m = re.match(r"^(\d+)\]\s*=", s)
    if m and len(m.group(1)) >= 6:
        return m.group(1)
    m = re.match(r"^(\d{6,})(?:-\d{2}-\d{2})(?:\D|$)", s)
    if m:
        return m.group(1)
    # Prefer longest leading digit run (avoid stopping at digit-] word boundary too early)
    m = re.match(r"^(\d{6,})", s)
    if m:
        lead = m.group(1)
        # Degenerate decoder repetition (e.g. 508466232323...) → keep plausible width (6–9 digits).
        if len(lead) > 12:
            m9 = re.match(r"^(\d{6,9})", lead)
            return m9.group(1) if m9 else lead[:8]
        return lead
    d = re.sub(r"\D+", "", s)
    if len(d) > 14:
        m9 = re.match(r"^(\d{6,9})", d)
        return m9.group(1) if m9 else d[:8]
    return d


def _repair_invoice_number_from_merged_inv_no(raw: str, fields: Dict[str, Any]) -> None:
    """
    Recover full invoice # when model emits [inv_no]=608664]=date (truncated in value field).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    m = re.search(r"\[inv_no\]\s*=\s*(\d+)\]\s*=", raw, flags=re.IGNORECASE)
    if not m:
        return
    cand = m.group(1)
    if len(cand) < 6:
        return
    cur = str(fields.get("invoice_number") or "")
    cur_d = re.sub(r"\D", "", cur)
    if not cur_d or len(cand) > len(cur_d):
        fields["invoice_number"] = cand


def _collect_plausible_isos(text: str) -> List[str]:
    """
    ISO-like tokens with calendar-valid years in a sane range (reject 4042-… artifacts).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        text (str): Input parameter.
    Outputs:
        List[str]: Function output value.
    """
    found: List[str] = []
    for m in re.finditer(r"\d{4}-\d{2}-\d{2}", text):
        cand = m.group(0)
        dt = pd.to_datetime(cand, errors="coerce")
        if pd.notna(dt) and 1990 <= int(dt.year) <= 2035:
            found.append(cand)
    return found


def _money_decimal_is_glued_date_glitch(raw: str, m: Any) -> bool:
    """
    True for '71.96' in '2014-71.96' or '11.24' in '7-11-11.24' — not invoice amounts.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        m (Any): Input parameter.
    Outputs:
        bool: Function output value.
    """
    start = m.start()
    # "2014-71.96" — fake month segment (>12) glued to a decimal
    if start >= 5 and start + 5 <= len(raw):
        window = raw[start - 5 : start + 5]
        mx = re.match(r"^(\d{4})-(\d{2})\.(\d{2})", window)
        if mx and int(mx.group(2)) > 12:
            return True
    if start > 0 and raw[start - 1] == "-":
        i = start - 2
        digits = 0
        while i >= 0 and raw[i].isdigit():
            digits += 1
            i -= 1
        if digits >= 2 and i >= 0 and raw[i] == "-":
            return True
    return False


def _normalize_invoice_date_blob(blob: str) -> str:
    """
    Normalize OCR-glued date fragments before ISO extraction.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        blob (str): Input parameter.
    Outputs:
        str: Function output value.
    """
    b = str(blob or "").strip()
    if not b:
        return ""
    # Two ISO dates glued: "2017-10-2017-11-20" -> separate tokens.
    b = re.sub(r"(\d{4}-\d{2}-\d{2})(\d{4}-\d{2}-\d{2})", r"\1 \2", b)
    # Trailing "-YY" after full ISO (often duplicated day digit noise).
    b = re.sub(r"(\d{4}-\d{2}-\d{2})-\d{2}(?=\D|$)", r"\1", b)
    # Pattern YYYY-MM-DD-DD (middle fragment wrong): prefer last DD as day.
    m = re.fullmatch(r"(\d{4}-\d{2})-(\d{2})-(\d{2})", b)
    if m:
        ym, mid, last = m.group(1), int(m.group(2)), int(m.group(3))
        cand_b = f"{ym}-{last:02d}"
        dt_b = pd.to_datetime(cand_b, errors="coerce")
        if pd.notna(dt_b) and 1 <= last <= 31:
            b = cand_b
    return b


def _best_invoice_date_iso(raw: str, fields: Dict[str, Any]) -> Optional[str]:
    # Handle common decoder artifact: "YYYY-MM-DD-20" -> "YYYY-MM-DD"
    """
    Best invoice date iso.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        Optional[str]: Function output value.
    """
    raw = re.sub(r"(\d{4}-\d{2}-\d{2})-\d{2}(?=\D|$)", r"\1", raw)
    tag_pos = raw.find("[inv_dt]")
    if tag_pos >= 0:
        chunk = raw[tag_pos : tag_pos + 160]
        chunk = _normalize_invoice_date_blob(chunk)
        m_eq = re.search(r"\]\s*=\s*(\d{4}-\d{2}-\d{2})\b", chunk)
        if m_eq:
            tail = chunk[m_eq.end() : m_eq.end() + 20]
            if not re.match(r"^\s*-\d{2}\)", tail):
                return m_eq.group(1)
        vals = _collect_plausible_isos(chunk)
        if vals:
            # Prefer the later candidate near the tag to handle strings like
            # "2017-09-11-20)=2017-11-20" where the first ISO is corrupted.
            return vals[-1]
    if "invoice_date" in fields:
        blob = _normalize_invoice_date_blob(str(fields["invoice_date"]))
        vals = _collect_plausible_isos(blob)
        if vals:
            return vals[-1] if len(vals) > 1 else vals[0]
    vals = _collect_plausible_isos(_normalize_invoice_date_blob(raw))
    return vals[-1] if len(vals) > 1 else (vals[0] if vals else None)


def _safe_float(x: Any) -> Optional[float]:
    """
    Safe float.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        x (Any): Input parameter.
    Outputs:
        Optional[float]: Function output value.
    """
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _looks_like_money_only(s: str) -> bool:
    """
    Looks like money only.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        s (str): Input parameter.
    Outputs:
        bool: Function output value.
    """
    t = str(s).strip().replace(",", "").replace("$", "").replace(" ", "")
    return bool(re.match(r"^\d+\.\d{2}$", t))


def _clean_party_name_text(s: Any) -> str:
    """
    Clean seller/client name text while keeping normal punctuation.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        s (Any): Input parameter.
    Outputs:
        str: Function output value.
    """
    t = str(s or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    # Trailing dash noise (ASCII hyphen, en/em dash, horizontal bar, minus)
    t = re.sub(r"[\s\u002d\u2010\u2011\u2012\u2013\u2014\u2015\u2212]+$", "", t)
    t = re.sub(r"(?i)^client!\s*", "", t)
    t = re.sub(r"(?i)^client!=\s*", "", t)
    t = re.sub(r"(?i)\|\s*client!\s*", " | ", t)
    t = re.sub(r"(?i)\|\s*client!=\s*", " | ", t)
    # Remove obvious bracket/label debris and trailing unmatched symbols.
    t = re.sub(r"\s*\]\]+\s*$", "", t)
    t = re.sub(r"\s*[\[\]\|<>]+\s*$", "", t)
    # Remove malformed net/tax tails that leaked into names (e.g., "jnet]=59.50").
    t = re.sub(r"\s+[a-z]?net\]\s*=.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+[a-z]?tax\]\s*=.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?i)\s+inv_dt\]\s*=.*$", "", t)
    t = re.sub(r"(?i)\s+inv_dt\s*$", "", t)
    t = re.sub(r"\s*\|?\s*[a-z]{0,3}client\]?\s*=.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\|?\s*[a-z]{0,3}buyer\]?\s*=.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*[=\-–—]+\s*$", "", t)
    t = re.sub(r"\s+\|\s*$", "", t)
    t = re.sub(r"\|\s*$", "", t)
    # Only strip numeric bracket assignments at end (e.g. "[36]=61719.34"), not "[net]=".
    t = re.sub(r"\s*\[\d{2,}\]\s*=\s*[\d.,]+\s*$", "", t)
    # "arnold plc plc" → "arnold plc" (repeated legal suffix)
    t = re.sub(r"(?i)\b(plc|llc|ltd|inc|corp|co)(?:\s+\1)+\b", r"\1", t)
    t = re.sub(r"(?i)\s+and\s+\d+\.\d{2}\s*$", "", t)
    t = re.sub(r"(?i)(plc|ltd|inc)net\.\s*\d+(?:\.\d{2})?\s*$", r"\1", t)
    return t.strip(" ,")


def _trim_seller_value_at_client_spill(sn: str) -> str:
    """
    Decoder often uses '(client]=' / '(client)=' instead of '[client]='; seller capture runs until the next '[' tag only, so cut those spills.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        sn (str): Input parameter.
    Outputs:
        str: Function output value.
    """
    t = (sn or "").strip()
    if not t:
        return t
    for pat in (
        r"\s*\|\s*\(\s*client\]?\s*=",
        r"\s*\|\s*\(\s*client\s*=",
        r"(?<!\()\(\s*client\]?\s*=",
        r"\s+\.client\]",
        r"(?i)\s+\[tax\]\s*=",
        r"['\u2018\u2019]\s*net\]\s*=",
    ):
        m = re.search(pat, t, flags=re.I)
        if m:
            t = t[: m.start()].rstrip(" |")
    t = re.sub(r"\s*\|\s*\(\s*$", "", t)
    t = re.sub(r"\s*\|\s*$", "", t)
    return t.strip(" ,")


def _client_pipe_segment_is_junk(seg: str) -> bool:
    """
    Client pipe segment is junk.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        seg (str): Input parameter.
    Outputs:
        bool: Function output value.
    """
    s = (seg or "").strip()
    if not s:
        return True
    if re.match(r"(?i)^\(?net\)?\.?$", s):
        return True
    if re.match(r"(?i)^\(?net\)?\s*$", s):
        return True
    if re.match(r"(?i)^(net|tax)\]?\s*=", s):
        return True
    return False


def _strip_client_name_money_leak(t: str) -> str:
    """
    Remove trailing indexed-bracket or broken bracket junk from client only (after [net] extraction).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        t (str): Input parameter.
    Outputs:
        str: Function output value.
    """
    if not t:
        return t
    # Corrupted client tag: "[1ient]=graham-martinez" or decoder drops "[" → "1ient]=..."
    t = re.sub(r"(?i)^\s*\[\d*[a-z]*ient\]\s*=\s*", "", t)
    t = re.sub(r"(?i)^\s*\d+[a-z]*ient\]\s*=\s*", "", t)
    t = re.sub(r"(?i)\s+items\s*$", "", t)
    t = re.sub(r"(?i)\s*\|\s*[\s{]*net.*$", "", t)
    t = re.sub(r"\s*\[\d{2,}\]\s*=\s*[\d.,]+\s*$", "", t)
    t = re.sub(r"\s*\[\d{2,}\]\s*\d+\s*$", "", t)
    # Pipe + net/tax fragments leaked into client (e.g. "| (net)=7200 |tax]=720")
    if "|" in t:
        parts = [p.strip() for p in t.split("|")]
        if parts:
            tail_pat = re.compile(
                r"(?i)^(\(?net\)?|net\]?|tax\]?|t\s*ax)\s*=",
            )
            if any(tail_pat.match(p) for p in parts[1:]):
                t = parts[0]
            elif any(_client_pipe_segment_is_junk(p) for p in parts[1:]):
                t = parts[0]
    t = re.sub(r"(?i)\s*\|\s*\(?net\)?\s*=.*$", "", t)
    t = re.sub(r"(?i)\s*\|\s*\(?net\)?\.?\s*$", "", t)
    t = re.sub(r"(?i)\s*\|\s*net\]\s*=.*$", "", t)
    t = re.sub(r"(?i)\s*\|\s*tax\]\s*=.*$", "", t)
    t = re.sub(r"(?i)\s*\(net\)\s*=.*$", "", t)
    t = re.sub(r"(?i)\s*\(net\)\.?\s*$", "", t)
    t = re.sub(r"(?i)\s*tax\]\s*=.*$", "", t)
    # "=258.96" money glued to name
    t = re.sub(r"\s*=\s*\d+\.\d{2}\s*$", "", t)
    # Common dangling OCR tails after valid names.
    t = re.sub(r"(?i)\s+lttemt\s*$", "", t)
    t = re.sub(r"(?i)\s+lt\s*$", "", t)
    t = re.sub(r"[\s\u002d\u2010\u2011\u2012\u2013\u2014\u2015\u2212]+$", "", t)
    t = re.sub(r"(?i)\s*\|\s*\{?\s*net\b.*$", "", t)
    return t.strip(" ,")


def _split_seller_embedded_client(fields: Dict[str, Any]) -> None:
    """
    When seller text contains a unicode dash + client fragment, split into seller / client.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    sn = fields.get("seller_name")
    if not isinstance(sn, str) or not sn.strip():
        return
    parts = re.split(r"\s*[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]\s*", sn, maxsplit=1)
    if len(parts) < 2:
        parts = re.split(r"\s+-\s+", sn, maxsplit=1)
        if len(parts) < 2:
            return
        _left, _right = parts[0].strip(), parts[1].strip()
        if not re.search(r"-", _right, re.I) and not re.search(r"\s", _right.strip()):
            return
        left, right = _left, _right
    else:
        left, right = parts[0].strip(), parts[1].strip()
    if len(right) < 3 or not re.search(r"[a-z]", right, re.I):
        return
    fields["seller_name"] = _clean_party_name_text(left)
    if not fields.get("client_name"):
        fields["client_name"] = _clean_party_name_text(right)


def _normalize_leading_money_token(s: str) -> str:
    """
    Keep first dd.dd money token; strip glued junk like '70.09.09.09'.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        s (str): Input parameter.
    Outputs:
        str: Function output value.
    """
    t = (s or "").strip().replace(" ", "")
    # "9.9.00" (OCR duplicate middle segment) → "9.00" when last segment is cents.
    m3 = re.match(r"^(\d+)\.(\d)\.(\d{2})$", t)
    if m3:
        return f"{m3.group(1)}.{m3.group(3)}"
    m = re.match(r"^(\d+\.\d{2})", t)
    return m.group(1) if m else t


def _extract_tax_field(raw: str, fields: Dict[str, Any]) -> None:
    """
    Best-effort tax from bracket line; handles [tax]=, typos, and [n]xx.xx glue before total.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    if fields.get("tax"):
        return

    tm = re.search(r"\[tax\]\s*=\s*([\d.,]+)", raw)
    if tm:
        fields["tax"] = _normalize_leading_money_token(tm.group(1))
        return

    tm = re.search(r"tax\]\s*=\s*([\d.,]+)", raw)
    if tm:
        fields["tax"] = _normalize_leading_money_token(tm.group(1))
        return

    tm = re.search(r"(?:^|\|)\s*t\s*ax\s*=\s*([\d.,]+)", raw)
    if tm:
        fields["tax"] = _normalize_leading_money_token(tm.group(1))
        return

    tm = re.search(r"(?:^|\|)\s*tax\s*=\s*([\d.,]+)", raw)
    if tm:
        fields["tax"] = _normalize_leading_money_token(tm.group(1))
        return

    tm = re.search(r"ttax\]\s*=\s*([\d.,]+)", raw)
    if tm:
        fields["tax"] = _normalize_leading_money_token(tm.group(1))
        return

    glued = []
    for m in re.finditer(r"\[(\d)\](\d{2}\.\d{2})\b", raw):
        try:
            glued.append((m.start(), float(m.group(1) + m.group(2))))
        except ValueError:
            continue

    if glued:
        nw = _safe_float(fields.get("net_worth"))
        ta = _safe_float(fields.get("total_amount"))
        chosen: Optional[float] = None
        for _, val in glued:
            if nw is not None and ta is not None and abs(nw + val - ta) <= max(0.05, 0.005 * abs(ta)):
                chosen = val
                break
        if chosen is None and nw is not None:
            # Tax should be smaller than net in almost all invoice patterns.
            subs = [v for _, v in glued if 0 < v < nw - 1e-6 and v <= 0.5 * max(nw, 1.0)]
            if subs:
                chosen = max(subs)
        if chosen is None and nw is None and glued:
            # Avoid selecting implausibly large "tax" values from corrupted date/totals.
            cand = [v for _, v in glued if v < 5000]
            chosen = (cand[-1] if cand else None)
        if chosen is not None:
            fields["tax"] = f"{chosen:.2f}"

    if not fields.get("tax"):
        tm = re.search(r"\$\s*(\d+\.\d{2})\b", raw)
        if tm:
            fields["tax"] = tm.group(1).strip()

    if fields.get("tax"):
        fields["tax"] = _normalize_leading_money_token(str(fields["tax"]))


def _refine_total_from_brackets(fields: Dict[str, Any], bracket_money: List[str], raw: str) -> None:
    """
    Avoid picking a tax-sized [n]= amount as grand total when a larger bracket exists.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
        bracket_money (List[str]): Input parameter.
        raw (str): Input parameter.
    Outputs:
        None: Function output value.
    """
    if not bracket_money or not fields.get("total_amount"):
        return
    try:
        amounts = [float(x) for x in bracket_money]
    except ValueError:
        return
    last = amounts[-1]
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    explicit_amt = bool(re.search(r"\[amt\]\s*=", raw))

    if explicit_amt:
        return

    if nw is not None and last + 1e-6 < nw:
        fields["total_amount"] = f"{max(amounts):.2f}"
        return

    if nw is not None and tx is not None:
        expected = nw + tx
        close_to_net_plus_tax = any(abs(a - expected) <= max(0.05, 0.005 * expected) for a in amounts)
        if close_to_net_plus_tax and abs(last - expected) > max(0.05, 0.005 * expected):
            best = min(amounts, key=lambda a: abs(a - expected))
            fields["total_amount"] = f"{best:.2f}"


def _fix_total_if_leading_digit_in_raw(raw: str, fields: Dict[str, Any]) -> None:
    """
    Recover 1140.24 when model emits [11]= as [1]=140.24 and raw still contains the full token.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    m2 = re.search(r"\[\d{2,}\]\s*=\s*(\d+\.\d{2})\b", raw)
    if m2:
        try:
            v2 = float(m2.group(1))
            cur0 = _safe_float(fields.get("total_amount"))
            if cur0 is None or v2 > cur0 + 1.0:
                fields["total_amount"] = f"{v2:.2f}"
                return
        except ValueError:
            pass

    # Disabled generic prefix inflation (e.g., 65.45 -> 165.45) because it
    # caused more regressions than recoveries. Keep only explicit [11]= style fix above.
    return


def _extract_net_dollar_tax_adjacent(raw: str, fields: Dict[str, Any]) -> None:
    """
    Pattern '61563.04 $6156.30' (subtotal then tax) common when [net]/[tax] tags are missing.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    if fields.get("net_worth") and fields.get("tax"):
        return
    matches = list(re.finditer(r"(\d+\.\d{2})\s*\$\s*(\d+\.\d{2})\b", raw))
    if not matches:
        return
    ta = _safe_float(fields.get("total_amount"))
    tol = max(0.05, 0.005 * ta) if ta and ta > 0 else None
    chosen: Optional[Tuple[float, float]] = None
    if ta is not None and tol is not None:
        for m in matches:
            a, b = float(m.group(1)), float(m.group(2))
            if abs(a + b - ta) <= tol:
                chosen = (a, b)
                break
    if chosen is None:
        m0 = matches[0]
        chosen = (float(m0.group(1)), float(m0.group(2)))
    if not fields.get("net_worth"):
        fields["net_worth"] = f"{chosen[0]:.2f}"
    if not fields.get("tax"):
        fields["tax"] = f"{chosen[1]:.2f}"


def _infer_net_tax_from_balance(fields: Dict[str, Any], raw: str) -> None:
    """
    Fill missing net or tax from total = net + tax; then try two decimals in raw summing to total.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
        raw (str): Input parameter.
    Outputs:
        None: Function output value.
    """
    ta = _safe_float(fields.get("total_amount"))
    if ta is None or ta <= 0:
        return

    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))

    if nw is None and tx is not None:
        n2 = ta - tx
        if n2 > 0.01:
            fields["net_worth"] = f"{n2:.2f}"
        nw = _safe_float(fields.get("net_worth"))

    if tx is None and nw is not None:
        t2 = ta - nw
        if t2 > 0.01:
            fields["tax"] = f"{t2:.2f}"

    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    if nw is not None and tx is not None:
        return

    decs: List[float] = []
    for v in _raw_money_decimals(raw):
        if 0 < v < ta - 0.01:
            decs.append(v)
    decs_u = sorted(set(decs), reverse=True)
    tol = max(0.05, 0.005 * ta)
    best: Optional[Tuple[float, float, float]] = None
    for i in range(len(decs_u)):
        for j in range(i + 1, len(decs_u)):
            a, b = decs_u[i], decs_u[j]
            if abs(a + b - ta) <= tol:
                hi, lo = max(a, b), min(a, b)
                if hi <= 0:
                    continue
                ratio = lo / hi
                err = abs(a + b - ta) / ta
                pri = 0.0 if 0.03 <= ratio <= 0.35 else 2.0
                score = pri + err
                if best is None or score < best[2]:
                    best = (hi, lo, score)
    if best is not None:
        hi, lo = best[0], best[1]
        if not fields.get("net_worth"):
            fields["net_worth"] = f"{hi:.2f}"
        if not fields.get("tax"):
            fields["tax"] = f"{lo:.2f}"


def _recover_total_from_raw_decimals(raw: str, fields: Dict[str, Any]) -> None:
    """
    Recover a likely grand total when parsed total is implausibly small/corrupted.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    ta = _safe_float(fields.get("total_amount"))
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))

    # Trigger only when current total is clearly inconsistent with known amounts.
    suspicious = False
    if ta is None:
        suspicious = True
    elif nw is not None and ta + 1e-6 < nw:
        suspicious = True
    elif tx is not None and ta + 1e-6 < tx:
        suspicious = True
    if not suspicious:
        return

    candidates: List[float] = [v for v in _raw_money_decimals(raw) if v > 0]
    if not candidates:
        return

    # Prefer values that satisfy total ~= net + tax; else choose largest decimal.
    if nw is not None and tx is not None:
        target = nw + tx
        tol = max(0.05, 0.005 * max(target, 1.0))
        close = [v for v in candidates if abs(v - target) <= tol]
        if close:
            fields["total_amount"] = f"{max(close):.2f}"
            return

    fields["total_amount"] = f"{max(candidates):.2f}"


def _maybe_swap_net_tax(fields: Dict[str, Any]) -> None:
    """
    Fix inverted net/tax when values sum to total but tax dominates unrealistically.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or tx is None or ta is None or ta <= 0:
        return
    tol = max(0.05, 0.005 * ta)
    if abs((nw + tx) - ta) > tol:
        return
    if tx > nw and (tx / ta) > 0.5 and (nw / ta) < 0.5:
        fields["net_worth"], fields["tax"] = f"{tx:.2f}", f"{nw:.2f}"


def _rebalance_amount_triplet(raw: str, fields: Dict[str, Any]) -> None:
    """
    Choose the most plausible (net, tax, total) triple from available decimals.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    vals: List[float] = []
    for k in ("net_worth", "tax", "total_amount"):
        v = _safe_float(fields.get(k))
        if v is not None and v > 0:
            vals.append(v)
    for v in _raw_money_decimals(raw):
        if 0 < v < 1_000_000:
            vals.append(v)
    uniq = sorted(set(vals))
    if len(uniq) < 3:
        return

    def _score(net: float, tax: float, total: float) -> Optional[float]:
        """
        Score.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            net (float): Input parameter.
            tax (float): Input parameter.
            total (float): Input parameter.
        Outputs:
            Optional[float]: Function output value.
        """
        if not (0 < tax < net < total):
            return None
        ratio = tax / net
        if not (0.005 <= ratio <= 0.35):
            return None
        tol = max(0.05, 0.005 * total)
        if abs((net + tax) - total) > tol:
            return None
        s = abs((net + tax) - total) / total
        # Prefer tax ratios around common VAT bands.
        s += min(abs(ratio - 0.1), abs(ratio - 0.08), abs(ratio - 0.2))
        return s

    best: Optional[Tuple[float, float, float, float]] = None
    for total in uniq:
        for net in uniq:
            if net >= total:
                continue
            for tax in uniq:
                if tax >= net:
                    continue
                sc = _score(net, tax, total)
                if sc is None:
                    continue
                if best is None or sc < best[3]:
                    best = (net, tax, total, sc)
    if best is None:
        return

    fields["net_worth"] = f"{best[0]:.2f}"
    fields["tax"] = f"{best[1]:.2f}"
    fields["total_amount"] = f"{best[2]:.2f}"


def _repair_amounts_from_raw_extremes(raw: str, fields: Dict[str, Any]) -> None:
    """
    Final repair: use raw decimal extrema when total/net/tax are collapsed.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    decs = list(_raw_money_decimals(raw))
    if not decs:
        return
    uniq = sorted(set(v for v in decs if v > 0))
    if not uniq:
        return

    mx = max(uniq)
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))

    # If total is missing or collapsed to tax/net while a clearly larger amount exists, promote max decimal.
    if ta is None or (tx is not None and abs(ta - tx) <= 1e-6) or (nw is not None and abs(ta - nw) <= 1e-6):
        if ta is None or mx > ta + 0.5:
            fields["total_amount"] = f"{mx:.2f}"
            ta = mx

    # If net missing but tax+total available, infer net.
    if nw is None and ta is not None and tx is not None:
        n2 = ta - tx
        if n2 > 0.01:
            fields["net_worth"] = f"{n2:.2f}"
            nw = n2

    # If tax missing but total+net available, infer tax.
    if tx is None and ta is not None and nw is not None:
        t2 = ta - nw
        if t2 > 0.01:
            fields["tax"] = f"{t2:.2f}"
            tx = t2

    # If both missing and total known, find a pair from raw that sums to total.
    if ta is not None and (nw is None or tx is None):
        tol = max(0.05, 0.005 * ta)
        cands = [v for v in uniq if 0 < v < ta - 0.01]
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if abs((a + b) - ta) <= tol:
                    hi, lo = max(a, b), min(a, b)
                    if nw is None:
                        fields["net_worth"] = f"{hi:.2f}"
                    if tx is None:
                        fields["tax"] = f"{lo:.2f}"
                    return


def _snap_total_to_net_plus_tax(fields: Dict[str, Any]) -> None:
    """
    If net+tax is coherent and total is clearly wrong (e.g. equals tax only), fix total.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or tx is None or ta is None:
        return
    s = nw + tx
    tol = max(0.05, 0.005 * max(s, ta, 1.0))
    if abs(s - ta) <= tol:
        fields["total_amount"] = f"{s:.2f}"
        return
    if ta + tol < s or abs(ta - tx) <= 1e-6 or abs(ta - nw) <= 1e-6 or ta < nw - 1e-6:
        fields["total_amount"] = f"{s:.2f}"


def _lift_total_when_collapsed_to_net(raw: str, fields: Dict[str, Any]) -> None:
    """
    When total wrongly equals net and tax is missing, infer tax+total from a larger decimal in raw.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or ta is None or tx is not None:
        return
    tol = max(0.05, 0.005 * nw)
    if abs(ta - nw) > tol:
        return
    for T in sorted(set(_raw_money_decimals(raw)), reverse=True):
        if T <= nw + tol:
            continue
        t2 = T - nw
        if t2 <= 0.01:
            continue
        r = t2 / nw
        if 0.005 <= r <= 0.40:
            fields["tax"] = f"{t2:.2f}"
            fields["total_amount"] = f"{T:.2f}"
            return


def _raw_money_decimals(raw: str) -> List[float]:
    """
    Raw money decimals.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
    Outputs:
        List[float]: Function output value.
    """
    out: List[float] = []
    for m in re.finditer(r"(?<![\d.])(\d+\.\d{2})(?!\d)", raw):
        if _money_decimal_is_glued_date_glitch(raw, m):
            continue
        v = float(m.group(1))
        if 0 < v < 1_000_000:
            out.append(v)
    return out


def _harmonize_money_triplet(raw: str, fields: Dict[str, Any]) -> None:
    """
    Fix inflated/inconsistent net/tax/total using decimals that satisfy net+tax≈total.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    decs = sorted(set(_raw_money_decimals(raw)))
    if len(decs) < 2:
        return

    nw0 = _safe_float(fields.get("net_worth"))
    tx0 = _safe_float(fields.get("tax"))
    hint_sum: Optional[float] = None
    if nw0 is not None and tx0 is not None:
        hint_sum = nw0 + tx0

    best: Optional[Tuple[float, float, float, float]] = None
    for T in sorted(decs, reverse=True):
        if hint_sum is not None:
            tol_h = max(0.05, 0.005 * max(hint_sum, T, 1.0))
            if T + tol_h < hint_sum:
                continue
        for N in decs:
            if N >= T:
                continue
            X = T - N
            if X <= 0:
                continue
            ratio = X / N if N > 0 else 999
            if not (0.005 <= ratio <= 0.35):
                continue
            err = abs((N + X) - T)
            score = err / T + abs(ratio - 0.1)
            if nw0 is not None:
                score += abs(N - nw0) / max(nw0, 1.0) * 0.05
            if tx0 is not None:
                score += abs(X - tx0) / max(tx0, 1.0) * 0.05
            if best is None or score < best[3]:
                best = (N, X, T, score)

    if best is None:
        return

    N, X, T = best[0], best[1], best[2]
    fields["net_worth"] = f"{N:.2f}"
    fields["tax"] = f"{X:.2f}"
    fields["total_amount"] = f"{T:.2f}"


def _repair_glued_hyphen_total_suffix(raw: str, fields: Dict[str, Any]) -> None:
    """
    Recover total like 284.86 when decoder emits '-1284.86' (spurious leading 1 before hundreds).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    if fields.get("total_amount"):
        return
    m = re.search(r"-1(\d{3}\.\d{2})\s*$", raw.strip())
    if m:
        fields["total_amount"] = m.group(1)


def _snap_net_to_implied_total_minus_tax(fields: Dict[str, Any]) -> None:
    """
    When net is OCR-collapsed (e.g. 136.58 vs 1036.58) but tax+total are sane, set net = total - tax.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    ta = _safe_float(fields.get("total_amount"))
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    if ta is None or ta < 150:
        return
    if tx is None:
        return
    implied = ta - tx
    if implied <= 50:
        return
    if nw is None:
        fields["net_worth"] = f"{implied:.2f}"
        return
    if implied > 300 and (implied - nw) > 200 and nw < implied * 0.35:
        fields["net_worth"] = f"{implied:.2f}"


def _reapply_coherent_bracket_amounts(raw: str, fields: Dict[str, Any]) -> None:
    """
    After harmonize triplet heuristics, prefer explicit [net]/[tax]/[amt] when they balance.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    m_net = re.search(r"\[net\]\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
    m_tax = re.search(r"\[tax\]\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
    m_amt = re.search(r"\[amt\]\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
    nw_e = _safe_float(_normalize_leading_money_token(m_net.group(1))) if m_net else None
    tx_e = _safe_float(_normalize_leading_money_token(m_tax.group(1))) if m_tax else None
    amt_e = _safe_float(_normalize_leading_money_token(m_amt.group(1))) if m_amt else None
    if nw_e is None or tx_e is None or amt_e is None:
        return
    tol = max(0.06, 0.005 * amt_e)
    if abs((nw_e + tx_e) - amt_e) > tol:
        return
    nw_cur = _safe_float(fields.get("net_worth"))
    tx_cur = _safe_float(fields.get("tax"))
    ta_cur = _safe_float(fields.get("total_amount"))
    cur_balanced = (
        nw_cur is not None
        and tx_cur is not None
        and ta_cur is not None
        and abs((nw_cur + tx_cur) - ta_cur) <= max(0.06, 0.005 * ta_cur)
    )
    inflated_net = (
        nw_cur is not None
        and nw_e is not None
        and nw_cur > 500
        and nw_e < nw_cur / 30
    )
    if cur_balanced and not inflated_net:
        return
    fields["net_worth"] = f"{nw_e:.2f}"
    fields["tax"] = f"{tx_e:.2f}"
    fields["total_amount"] = f"{amt_e:.2f}"


def _fix_centifold_money_totals(fields: Dict[str, Any]) -> None:
    """
    Decoder typo: total ~100× (nw+tax), e.g. 35353.60 vs net+tax 353.66.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or tx is None or ta is None:
        return
    implied = nw + tx
    if implied <= 1:
        return
    r = ta / implied
    if 40 <= r <= 160:
        fields["total_amount"] = f"{implied:.2f}"


def _prefer_explicit_net_when_harmonize_inflated(raw: str, fields: Dict[str, Any]) -> None:
    """
    When triplet repair chose a huge net but raw [net]= is small and matches total−tax, prefer raw.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    m = re.search(r"\[net\]\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
    if not m:
        return
    nw_e = _safe_float(_normalize_leading_money_token(m.group(1)))
    nw_c = _safe_float(fields.get("net_worth"))
    tx_c = _safe_float(fields.get("tax"))
    ta_c = _safe_float(fields.get("total_amount"))
    if nw_e is None or nw_c is None or ta_c is None:
        return
    if nw_c <= 400 or nw_e >= nw_c / 6:
        return
    tx_use = tx_c if tx_c is not None else 0.0
    tol = max(0.15, 0.02 * ta_c)
    if abs((nw_e + tx_use) - ta_c) > tol:
        return
    fields["net_worth"] = f"{nw_e:.2f}"
    if tx_c is None and nw_e + tx_use > 0:
        fields["tax"] = f"{max(ta_c - nw_e, 0):.2f}"


def _fix_centifold_using_explicit_net(raw: str, fields: Dict[str, Any]) -> None:
    """
    Total ~100× too large vs explicit [net]+observed tax (harmonize inflated net to match junk total).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    m = re.search(r"\[net\]\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
    if not m:
        return
    nw_e = _safe_float(_normalize_leading_money_token(m.group(1)))
    tx_c = _safe_float(fields.get("tax"))
    ta_c = _safe_float(fields.get("total_amount"))
    nw_c = _safe_float(fields.get("net_worth"))
    if nw_e is None or tx_c is None or ta_c is None:
        return
    implied = nw_e + tx_c
    if implied <= 1:
        return
    r = ta_c / implied
    if not (35 <= r <= 170):
        return
    if nw_c is None or nw_c <= nw_e * 5:
        return
    fields["net_worth"] = f"{nw_e:.2f}"
    fields["total_amount"] = f"{implied:.2f}"


def _infer_net_tax_from_inclusive_total(fields: Dict[str, Any], raw: str) -> None:
    """
    When only grand total exists (~10% VAT style), split net/tax (training bracket invoices).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
        raw (str): Input parameter.
    Outputs:
        None: Function output value.
    """
    if fields.get("net_worth") or fields.get("tax"):
        return
    ta = _safe_float(fields.get("total_amount"))
    if ta is None or ta < 30 or ta > 500000:
        return
    if re.search(r"\[net\]\s*=", raw, flags=re.IGNORECASE) or re.search(
        r"\[tax\]\s*=", raw, flags=re.IGNORECASE
    ):
        return
    n_est = ta / 1.1
    t_est = ta - n_est
    if t_est <= 0:
        return
    fields["net_worth"] = f"{n_est:.2f}"
    fields["tax"] = f"{t_est:.2f}"


def _repair_total_minus_century_when_matches_net_plus_tax(fields: Dict[str, Any]) -> None:
    """
    Fix totals like 171.96 when net+tax=71.96 (spurious leading 1).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    ta = _safe_float(fields.get("total_amount"))
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    if ta is None or nw is None or tx is None:
        return
    implied = nw + tx
    tol = max(0.05, 0.005 * max(abs(implied), abs(ta), 1.0))
    if abs(implied - ta) <= tol:
        return
    for shift in (100, 1000):
        t2 = ta - shift
        if t2 > 0 and abs(implied - t2) <= tol:
            fields["total_amount"] = f"{t2:.2f}"
            return


def _infer_tax_total_from_net_only(fields: Dict[str, Any], raw: str) -> None:
    """
    Fallback when raw has [net] but tax/total tags are missing (10% VAT-style invoices).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
        raw (str): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or tx is not None or ta is not None:
        return
    if nw <= 0:
        return
    if not re.search(r"\[(?:1net|net)\]\s*=", raw, flags=re.IGNORECASE):
        return
    if re.search(r"(?:^|\||\s)tax\]\s*=", raw, flags=re.IGNORECASE) or re.search(
        r"\[tax\]\s*=", raw, flags=re.IGNORECASE
    ):
        return
    t = round(nw * 0.10, 2)
    fields["tax"] = f"{t:.2f}"
    fields["total_amount"] = f"{(nw + t):.2f}"


def _unpack_client_glued_amounts(fields: Dict[str, Any]) -> None:
    """
    Split 'name and 12.34' or 'name =12.34' glued into invclient/client values.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    cn = fields.get("client_name")
    if not isinstance(cn, str) or not cn.strip():
        return
    t = cn.strip()
    m = re.match(r"^(.+?)\s+and\s+(\d+\.\d{2})\s*$", t, flags=re.IGNORECASE)
    if m:
        fields["client_name"] = _clean_party_name_text(m.group(1))
        if not fields.get("net_worth"):
            fields["net_worth"] = m.group(2)
        return
    m2 = re.match(r"^(.+?)\s+=\s*(\d+\.\d{2})\s*$", t)
    if m2 and not re.search(r"\d+\.\d{2}", m2.group(1)):
        fields["client_name"] = _clean_party_name_text(m2.group(1))
        if not fields.get("net_worth"):
            fields["net_worth"] = m2.group(2)


def _recover_client_if_invclient_was_iso_date(fields: Dict[str, Any], raw: str) -> None:
    """
    When model puts the invoice date in [invclient]=, recover real client from llcient]= etc.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
        raw (str): Input parameter.
    Outputs:
        None: Function output value.
    """
    cn = fields.get("client_name")
    idt = fields.get("invoice_date")
    if not isinstance(cn, str) or not idt:
        return
    cn_s, idt_s = cn.strip(), str(idt).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cn_s) or cn_s != idt_s:
        return
    fields["client_name"] = ""
    m = re.search(r"(?i)l{1,2}cient\]\s*=\s*([^|\[]+)", raw)
    if not m:
        m = re.search(r"(?i)\[1client\]\s*=\s*([^|]+)", raw)
    if m:
        frag = re.split(r"\]\s*=\s*", m.group(1).strip())[0]
        fields["client_name"] = _clean_party_name_text(frag)


def _repair_tax_when_ocr_dropped_two_digits(fields: Dict[str, Any]) -> None:
    """
    tax]=84.48 style when net is huge and ~10%% tax should be ~8494.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    if nw is None or tx is None or nw < 500:
        return
    if tx >= nw * 0.03:
        return
    target = round(nw * 0.1, 2)
    tx100 = round(tx * 100, 2)
    if abs(tx100 - target) <= max(20.0, 0.02 * nw):
        fields["tax"] = f"{tx100:.2f}"


def _repair_total_when_sum_is_ten_x_parsed_total(fields: Dict[str, Any]) -> None:
    """
    Fix [amt]=2398.86 when net+tax≈23988.86 (missing digit in total).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or tx is None or ta is None:
        return
    exp = nw + tx
    if exp <= 0 or ta <= 0 or exp < ta * 4:
        return
    if abs(ta * 10 - exp) <= max(2.0, 0.005 * exp):
        fields["total_amount"] = f"{exp:.2f}"


def _repair_tax_outlier_from_total(fields: Dict[str, Any]) -> None:
    """
    If tax is implausibly large, recover tax from total-net.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is None or tx is None or ta is None:
        return
    if tx <= 0.35 * max(nw, 1.0):
        return
    implied = ta - nw
    if implied <= 0:
        return
    if implied <= 0.35 * max(nw, 1.0):
        fields["tax"] = f"{implied:.2f}"


def _demote_scaled_total(raw: str, fields: Dict[str, Any]) -> None:
    """
    Fix totals like 711.96 when raw clearly contains 71.96 (extra leading digit).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        raw (str): Input parameter.
        fields (Dict[str, Any]): Input parameter.
    Outputs:
        None: Function output value.
    """
    ta = _safe_float(fields.get("total_amount"))
    if ta is None or ta >= 500:
        return
    for v in sorted(set(_raw_money_decimals(raw)), reverse=True):
        if v <= 0 or v >= ta:
            continue
        r = ta / v
        if 9.5 <= r <= 10.5 or 1.95 <= r <= 2.05:
            fields["total_amount"] = f"{v:.2f}"
            return


def parse_structured_invoice_text(text: str) -> Dict[str, Any]:
    """
    Parse Donut / training-style bracket fields and tolerant corruptions (e.g. [d]=, Lno]=).

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        text (str): Input parameter.
    Outputs:
        Dict[str, Any]: Function output value.
    """
    raw = re.sub(r"<.*?>", "", text or "").lower().strip()
    if not raw:
        return {}

    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # Common decoder glitches: dotted keys, broken open-bracket, doubled pipes
    raw = raw.replace("[inv.dt]", "[inv_dt]")
    raw = raw.replace("[inv-dt]", "[inv_dt]")
    raw = raw.replace("[invdt]", "[inv_dt]")
    raw = raw.replace("[injdt]", "[inv_dt]")
    raw = raw.replace("[inj_dt]", "[inv_dt]")
    raw = raw.replace("[s_dt]", "[inv_dt]")
    raw = raw.replace("[instey_dt]", "[inv_dt]")
    raw = re.sub(r"\(net\]", "[net]", raw)
    raw = re.sub(r"<\s*seller\]", "[seller]", raw)
    raw = re.sub(r"<\s*amt\]", "[amt]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\|\s*\|+", "|", raw)
    # Decoder glitch: "| (client]=" instead of "| [client]=" — normalize so keys parse as [client].
    raw = re.sub(r"\|\s*\(\s*client\]?\s*=", "| [client]=", raw, flags=re.IGNORECASE)
    # "[inv.2014-12-20" / broken inv.dt glue → proper tag + date (before bracket scan).
    raw = re.sub(r"\[inv\.\s*(\d{4}-\d{2}-\d{2})", r"[inv_dt]=\1", raw, flags=re.IGNORECASE)
    raw = re.sub(r"cclient", "client", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\{net\]", "[net]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?<!\[)jnet\]\s*=", "[net]=", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[1\.net\]", "[net]", raw, flags=re.IGNORECASE)
    # Decoder glued ISOs: "2012-09-2012-07-22" → split so _collect_plausible_isos finds the first ISO.
    raw = re.sub(r"(\d{4}-\d{2}-\d{2})(?=\d{4})", r"\1 ", raw)
    # Stray " -2014" after a full [inv_dt]=YYYY-MM-DD (EOS garbage)
    raw = re.sub(
        r"(\[inv_dt\]\s*=\s*\d{4}-\d{2}-\d{2})\s+-\d{4}(?=\s|$|\||\[)",
        r"\1",
        raw,
        flags=re.IGNORECASE,
    )
    # "[inv.dt]=2014-seller]=..." merged field → seller only (re-pick date from ISOs in raw)
    raw = re.sub(r"\[inv_?dt\]\s*=\s*\d{4}-seller\]\s*=\s*", r"[seller]=", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\|\s*Client\]\s*=", "| [client]=", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\(\s*amt\]\s*=", "[amt]=", raw, flags=re.IGNORECASE)
    # Broken "[net]" open bracket: "| | [136.58 |" → "| [net]=136.58 |"
    raw = re.sub(r"\|\s*\|\s*\[\s*(\d+\.\d{2})\s*\|", r"| [net]=\1 |", raw)
    raw = re.sub(r"\|\s*\[\s*(\d+\.\d{2})\s*\|(?!\s*\])", r"| [net]=\1 |", raw)
    # "...2021-02-11-1284.86" → repair uses trailing -1284.86; also isolate ISO before '/' glue.
    raw = re.sub(r"(\d{4}-\d{2}-\d{2})/", r"\1 ", raw)

    # Additional mergers / OCR bracket noise seen on larger-sample inference.
    raw = re.sub(r"\s+\.client\]", " | [client]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?<!\[)\blclient\]", "[client]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?<!\[)\blnet\]", "[net]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?<!\[)\bltax\]", "[tax]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?<!\[)\blamt\]", "[amt]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"client\s*=\s*client\s*=", "[client]=", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[1amt\]", "[amt]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[1tax\]", "[tax]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[1net\]", "[net]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[1ers\]", "[net]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[1en\]", "[net]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?i)ftax\]\s*=\s*", "| [tax]=", raw)
    raw = re.sub(
        r"(?i)\|\s*lt\]\s*=\s*(\d+\.\d{2})\b",
        r"| [amt]=\1",
        raw,
    )
    raw = re.sub(
        r"(?i)tax\]\s*=\s*(\d+\.\d{2})\s*\]\s*=\s*(\d+\.\d{2})\b",
        r"[tax]=\1 | [amt]=\2",
        raw,
    )
    raw = re.sub(
        r"(?i)\[tax\]\s*=\s*(\d+\.\d{2})\s*t\]\s*=\s*(\d+\.\d{2})\b",
        r"[tax]=\1 | [amt]=\2",
        raw,
    )
    raw = re.sub(
        r"(?i)\[tax\]\s*=\s*(\d+\.\d{2})(\d+\.\d{2})\b",
        r"[tax]=\1 | [amt]=\2",
        raw,
    )
    raw = re.sub(r"(?i)\band\]\s*=\s*(\d+\.\d{2})\b", r"[amt]=\1", raw)
    raw = re.sub(
        r"\[client\]\s*=\s*([^|\[\]]+?)-net\]\s*=\s*([\d.,]+)",
        r"[client]=\1 | [net]=\2",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(
        r"\[client\]\s*=\s*([^|\[\]]+?)\]\s*=\s*(\d+\.\d{2})\b",
        r"[client]=\1 | [amt]=\2",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"(?i)\[invacia\]", "[inv_dt]", raw)
    raw = re.sub(r"(?i)\[inva_dt\]", "[inv_dt]", raw)
    raw = re.sub(r"(?i)(?<!\[)\|tax\]\s*=", "|[tax]=", raw)
    raw = re.sub(r"(?i)tax\]\s*-\s*", "tax]=", raw)
    raw = re.sub(r"(?i)\[inamt\]", "[amt]", raw)
    raw = re.sub(r"(?i)inamt\]", "[amt]", raw)
    raw = re.sub(r"(?i)inamt\]\s*=", "[amt]=", raw)
    raw = re.sub(r"(?i)llcient\]", "[client]", raw)
    raw = re.sub(r"(?i)(?<!\[)lcient\]", "[client]", raw)
    raw = re.sub(r"(?i)(?<!\[)eclient\]", "[client]", raw)
    raw = re.sub(r"(?i)\[1client\|", "[client]=", raw)
    # '... lt]=94.47' / 'ltdt]=449.90' / 'lttemt]=5634.90' are usually net captures.
    raw = re.sub(r"(?i)\|\s*lt\]\s*=\s*(\d+\.\d{2})\b", r"| [net]=\1", raw)
    raw = re.sub(r"(?i)\bltdt\]\s*=\s*(\d+\.\d{2})\b", r"[net]=\1", raw)
    raw = re.sub(r"(?i)\blttemt\]\s*=\s*(\d+\.\d{2})\b", r"[net]=\1", raw)
    raw = re.sub(r"(?i)<\s*client\]", "[client]", raw)
    raw = re.sub(r"(?i)\s+inamt\]\s*=", " [amt]=", raw)
    raw = re.sub(r"(?i)\[1\]\]\s*=\s*(\d+\.\d{2})\b", r"[amt]=\1", raw)

    fields: Dict[str, Any] = {}

    # Keys: letter-first then letters/digits/underscore/dot (normalize dots → underscore).
    # Value ends before the next [key]= **or** common decoder glitches: | (client]=, 'net]=, etc.
    _val_stop = (
        r"(?="
        r"\|\s*\[|"
        r"\s*\[(?:[a-z][a-z0-9_.]*|\d+[a-z][a-z0-9_]*)\]\s*=|"
        r"\|\s*\(\s*client|"
        r"\|\s*\(?\s*client\]?\s*=|"
        r"['\u2018\u2019]\s*net\]\s*=|"
        r"$"
        r")"
    )
    matches = [
        (m.group(1), m.group(2))
        for m in re.finditer(
            r"\[\s*((?:[a-z][a-z0-9_.]*|\d+[a-z][a-z0-9_]*))\s*\]\s*=\s*(.*?)\s*" + _val_stop,
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]

    FIELD_MAP = {
        "inv_no": "invoice_number",
        "invi_no": "invoice_number",
        "intv_no": "invoice_number",
        "inv_dt": "invoice_date",
        "instey_dt": "invoice_date",
        "seller": "seller_name",
        "vendor": "seller_name",
        "client": "client_name",
        "cluent": "client_name",
        "cllent": "client_name",
        "c1ient": "client_name",
        "cclient": "client_name",
        "gradu": "client_name",
        "buyer": "client_name",
        "net": "net_worth",
        "net_worth": "net_worth",
        "tax": "tax",
        "amt": "total_amount",
    }

    def _map_key(key: str, value: str = "") -> Optional[str]:
        """
        Map key.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            key (str): Input parameter.
            value (str): Input parameter. Defaults to ''.
        Outputs:
            Optional[str]: Function output value.
        """
        key_ns = key.replace(".", "_")
        stripped = re.sub(r"^\d+", "", key_ns)
        if stripped == "inclient":
            head = (value or "").strip().split()
            v0 = head[0] if head else ""
            if v0 and re.match(r"^\d{4}-\d{2}-\d{2}", v0):
                return "invoice_date"
            return "client_name"
        if stripped in ("invclient", "invcl"):
            head = (value or "").strip()
            vm = re.match(r"^(\d{4}-\d{2}-\d{2})\b", head)
            if vm:
                return "invoice_date"
            v0 = head.split()[0] if head.split() else ""
            if v0 and not re.fullmatch(r"\d+", v0):
                return "client_name"
            return None
        if key_ns in FIELD_MAP:
            return FIELD_MAP[key_ns]
        if stripped in FIELD_MAP:
            return FIELD_MAP[stripped]
        # Fuzzy key recovery for OCR-noisy date labels like invdt, instey_dt, inv_date.
        if "inv" in stripped and ("dt" in stripped or "date" in stripped):
            return "invoice_date"
        # OCR-noisy client tags: cluent, cllent, c1ient, etc.
        if ("ient" in stripped or "buyer" in stripped) and len(stripped) <= 12:
            return "client_name"
        return None

    for key, value in matches:
        value = value.strip()
        if value.lower() in {"null", "none", "n/a", ""}:
            continue
        canon = _map_key(key, value)
        if canon:
            fields[canon] = value
        elif key == "inv":
            iso_list = _collect_plausible_isos(value)
            if iso_list:
                fields.setdefault("invoice_date", iso_list[0])
            else:
                if re.search(r"[a-z]", value):
                    cl = re.search(r"\[client\]\s*=\s*([^|]+)", value, flags=re.IGNORECASE)
                    if cl and not fields.get("client_name"):
                        fields["client_name"] = _clean_party_name_text(cl.group(1))
                    seller_guess = re.split(r"\[client\]\s*=", value, flags=re.IGNORECASE)[0].strip()
                    seller_guess = _clean_party_name_text(seller_guess)
                    if seller_guess and not fields.get("seller_name"):
                        fields["seller_name"] = seller_guess
                digits = re.sub(r"\D", "", value)
                if len(digits) >= 6:
                    fields.setdefault(
                        "invoice_number",
                        _sanitize_invoice_number_value(value),
                    )
        # Corrupted short key from [inv_dt] -> [d] after generation drops tokens
        elif key == "d" and re.search(r"\d{4}-\d{2}-\d{2}", value):
            fields.setdefault("invoice_date", value.strip())
        elif key.startswith("inv") and re.search(r"\d{4}-\d{2}-\d{2}", value):
            # OCR-noisy inv* key carrying date, e.g. [invcl]=2014-10-01
            fields.setdefault("invoice_date", re.search(r"\d{4}-\d{2}-\d{2}", value).group(0))

    # Missing opening "[" before label: "[1]=seller]=..." → seller]=...
    _CORRUPT_CLOSING = (
        (re.compile(r"\|\s*seller\]\s*=\s*([^|]+)"), "seller_name"),
        (re.compile(r"seller\]\s*=\s*([^|]+)"), "seller_name"),
        (re.compile(r"\|\s*client\]\s*=\s*([^|]+)"), "client_name"),
        (re.compile(r"client\]\s*=\s*([^|]+)"), "client_name"),
        (re.compile(r"net\]\s*=\s*([^|]+)"), "net_worth"),
        (re.compile(r"tax\]\s*=\s*([^|]+)"), "tax"),
        (re.compile(r"amt\]\s*=\s*([^|]+)"), "total_amount"),
        (re.compile(r"inv_dt\]\s*=\s*([^|]+)"), "invoice_date"),
        (re.compile(r"inv_no\]\s*=\s*([^|]+)"), "invoice_number"),
    )
    for rx, canon in _CORRUPT_CLOSING:
        if fields.get(canon) not in [None, ""]:
            continue
        m = rx.search(raw)
        if m:
            val = m.group(1).strip()
            if val and val.lower() not in {"null", "none", "n/a"}:
                tgt = canon
                if tgt == "invoice_number":
                    val = _sanitize_invoice_number_value(val) or val
                fields[tgt] = val

    if fields.get("invoice_date"):
        fields["invoice_date"] = _normalize_invoice_date_blob(str(fields["invoice_date"]))
    inv_best = _best_invoice_date_iso(raw, fields)
    if inv_best:
        fields["invoice_date"] = inv_best

    # Explicit [net]= anywhere (decoder sometimes drops preceding "[" elsewhere)
    nm = re.search(r"\[net\]\s*=\s*([\d.,]+)", raw)
    if nm:
        fields["net_worth"] = _normalize_leading_money_token(nm.group(1))
    if "net_worth" not in fields:
        nm2 = re.search(r"(?:^|\||\s)[=\-–—]*\s*\[?net\]?\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
        if nm2:
            fields["net_worth"] = _normalize_leading_money_token(nm2.group(1))

    explicit_amt_tag = bool(re.search(r"\[amt\]\s*=", raw))

    if "total_amount" not in fields:
        m = re.search(r"\[amt\]\s*=\s*([\d.,]+)", raw)
        if m:
            fields["total_amount"] = _normalize_leading_money_token(m.group(1))
        else:
            m = re.search(r"\|\s*(\d+\.\d{2})\s*$", raw)
            if m:
                fields["total_amount"] = m.group(1)
            else:
                dec_amts = []
                for mm in re.finditer(r"(?<![\d.])(\d+\.\d{2})(?!\d)", raw):
                    if _money_decimal_is_glued_date_glitch(raw, mm):
                        continue
                    pos = mm.start()
                    before = raw[max(0, pos - 7) : pos]
                    if re.search(r"\d{4}-\d{2}-$", before):
                        continue
                    dec_amts.append(mm.group(1))
                if dec_amts:
                    fields["total_amount"] = dec_amts[-1]

    # [11]=1140.24 / [1]=284.86 when "[amt]=" missing or corrupted as [1amt]= (prefer last bracket total)
    bracket_money = re.findall(r"\[\d+\]\s*=\s*(\d+\.\d{2})\b", raw)
    if bracket_money:
        last_bm = bracket_money[-1]
        if not explicit_amt_tag:
            fields["total_amount"] = last_bm
        elif fields.get("total_amount"):
            try:
                cur = float(str(fields["total_amount"]).replace(",", "").replace("$", ""))
                if float(last_bm) > cur:
                    fields["total_amount"] = last_bm
            except ValueError:
                pass

    # Comma-as-decimal in the [amt] capture only (do not let a trailing "| 56779, 34" override a good [amt])
    if "total_amount" in fields:
        tv = str(fields["total_amount"]).strip().replace(" ", "")
        if re.match(r"^\d{1,3}(?:\.\d{3})*,\d{2}$", tv) or re.match(r"^\d+,\d{2}$", tv):
            if "," in tv and "." not in tv:
                fields["total_amount"] = tv.replace(",", ".")
            elif tv.rfind(",") > tv.rfind("."):
                fields["total_amount"] = tv.replace(".", "").replace(",", ".")

    if "invoice_number" not in fields:
        m = re.search(r"\b\d{6,}\b", raw)
        if m:
            fields["invoice_number"] = m.group(0)
    # Broken bracket: "lno]=56014042" or "lno]=..."
    if "invoice_number" not in fields:
        m = re.search(r"(?:\[)?[a-z]*no\]?\s*=\s*(\d{6,})\b", raw, re.IGNORECASE)
        if m:
            fields["invoice_number"] = m.group(1)

    if fields.get("invoice_number"):
        fields["invoice_number"] = _sanitize_invoice_number_value(fields["invoice_number"])
        if not fields["invoice_number"]:
            fields.pop("invoice_number", None)

    if "invoice_number" not in fields:
        date_digits = None
        if fields.get("invoice_date"):
            date_digits = re.sub(r"\D", "", str(fields["invoice_date"]))
        nums = re.findall(r"\b\d{6,}\b", raw)
        nums = [n for n in nums if (date_digits is None or n != date_digits)]
        if nums:
            fields["invoice_number"] = nums[0]

    _repair_invoice_number_from_merged_inv_no(raw, fields)
    if fields.get("invoice_number"):
        fields["invoice_number"] = _sanitize_invoice_number_value(fields["invoice_number"])

    # "... 61563.04 $6156.30 ..." without [net]/[tax] brackets
    if "net_worth" not in fields:
        nm = re.search(r"(\d+\.\d{2})\s*\$", raw)
        if nm:
            fields["net_worth"] = nm.group(1)

    if "net_worth" not in fields:
        rm = re.search(r"reft\]\s*=\s*([\d.,]+)", raw)
        if rm:
            fields["net_worth"] = rm.group(1).strip().replace(" ", "")

    _extract_tax_field(raw, fields)
    if not fields.get("tax"):
        tx2 = re.search(r"(?:^|\||\s)[=\-–—]*\s*\[?tax\]?\s*=\s*([\d.,]+)", raw, flags=re.IGNORECASE)
        if tx2:
            fields["tax"] = _normalize_leading_money_token(tx2.group(1))
    _refine_total_from_brackets(fields, bracket_money, raw)

    sn = fields.get("seller_name")
    if isinstance(sn, str) and sn:
        for sep in (
            " , [client]",
            "| [client]",
            " .client]",
            ".client]",
            " .client]=",
            ".client]=",
            " [client]=",
            "[client]=",
            "| (client]",
            "| (client]=",
            "| (client)=",
            "(client]=",
            "(client)=",
        ):
            if sep in sn:
                fields["seller_name"] = sn.split(sep)[0].strip()
                break
        fields["seller_name"] = re.split(r"\s*\|\s*\[client\]", fields["seller_name"])[0].strip()
        fields["seller_name"] = _trim_seller_value_at_client_spill(fields["seller_name"])
        # OCR-corrupted inline client tags often appear inside seller, e.g.:
        # "nichols ... cluent]=mckay plc [net]=..."
        sm = re.search(
            r"\b(?:\[[^\]]*ient\]|cl[iu]ent|c1ient|client!=\s*|client!\s*|client|buyer|gradu)\]?\s*=\s*([^|]+)",
            fields["seller_name"],
            flags=re.IGNORECASE,
        )
        if sm:
            cand = sm.group(1).strip()
            cand = re.sub(r"\s*\[(?:net|tax|amt|inv_[a-z0-9_]+)\].*$", "", cand, flags=re.IGNORECASE)
            cand = _clean_party_name_text(cand)
            if cand and not fields.get("client_name"):
                fields["client_name"] = cand
            fields["seller_name"] = re.split(
                r"\b(?:\[[^\]]*ient\]|[a-z]{0,3}client|cl[iu]ent|c1ient|client!=\s*|client!\s*|client|buyer|gradu)\]?\s*=",
                fields["seller_name"],
                flags=re.IGNORECASE,
            )[0].strip()
        fields["seller_name"] = _clean_party_name_text(fields["seller_name"])
        _split_seller_embedded_client(fields)

    if not fields.get("seller_name"):
        sm2 = re.search(r"(?:^|\|)\s*(?:\[[a-z0-9_]*seller\]?|[a-z0-9_]*seller\])\s*=\s*([^|]+)", raw, flags=re.IGNORECASE)
        if sm2:
            cand = _clean_party_name_text(sm2.group(1))
            if cand:
                fields["seller_name"] = cand

    cn = fields.get("client_name")
    if isinstance(cn, str) and cn:
        nm_c = re.search(r"\[net\]\s*=\s*([\d.,]+)", cn)
        if nm_c and not fields.get("net_worth"):
            fields["net_worth"] = nm_c.group(1).strip().replace(" ", "")
        nm_pipe = re.search(r"(?i)(?:\||^)\s*\(?net\)?\s*=\s*([\d.,]+)", cn)
        if nm_pipe and not fields.get("net_worth"):
            fields["net_worth"] = nm_pipe.group(1).strip().replace(" ", "")
        nm_leak = re.search(r"(?i)net\]\s*=\s*([\d.,]+)", cn)
        if nm_leak and not fields.get("net_worth"):
            fields["net_worth"] = nm_leak.group(1).strip().replace(" ", "")
        tx_c = re.search(r"\$\s*(\d+\.\d{2})\b", cn)
        if tx_c and not fields.get("tax"):
            fields["tax"] = tx_c.group(1).strip()
        tx_leak = re.search(r"(?i)tax\]\s*=\s*([\d.,]+)", cn)
        if tx_leak and not fields.get("tax"):
            fields["tax"] = tx_leak.group(1).strip().replace(" ", "")
        cn = re.sub(r"\s+items\s*,.*$", "", cn, flags=re.I)
        cn = re.sub(r"\s+items\s+.*$", "", cn, flags=re.I)
        cn = re.sub(r"\s+items\s*$", "", cn, flags=re.I)
        cn = re.sub(r"\s*,\s*\[net\].*$", "", cn, flags=re.I)
        cn = re.sub(r"\s*\[net\]\s*=.*$", "", cn)
        cn = re.sub(r"\s*\$\s*[\d.]+\s*$", "", cn)
        cn = re.sub(r"\s+\d{5,}\s+.*$", "", cn)
        cn = re.sub(r"\s+int\]\s*=.*$", "", cn, flags=re.I)
        cn = re.sub(r"\s+reft\]\s*=.*$", "", cn, flags=re.I)
        cn = re.sub(r"\s+ttax\]\s*=.*$", "", cn, flags=re.I)
        cn = re.sub(r"\s+and\s*$", "", cn)
        cn = _strip_client_name_money_leak(cn)
        fields["client_name"] = _clean_party_name_text(cn)

    # If client tag is dropped, grab next pipe segment after [seller]=... when it looks like a name.
    if not fields.get("client_name"):
        seg = re.search(r"\[seller\]\s*=\s*[^|]+\|\s*([^|]+)", raw, flags=re.IGNORECASE)
        if seg:
            cand = _clean_party_name_text(seg.group(1))
            if cand and not re.search(r"\d+\.\d{2}", cand):
                fields["client_name"] = cand

    # If client is still missing, recover from corrupted tags in full raw sequence:
    # cluent]=..., c1ient]=..., 근ient]=..., etc.
    if not fields.get("client_name"):
        cm = re.search(
            r"\b(?:cl[iu]ent|c1ient|client!=\s*|client!\s*|client|buyer|gradu)\]?\s*=\s*([^|]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if cm:
            cand = cm.group(1).strip()
            cand = re.sub(r"\s*\[(?:net|tax|amt|inv_[a-z0-9_]+)\].*$", "", cand, flags=re.IGNORECASE)
            cand = _clean_party_name_text(cand)
            if cand:
                fields["client_name"] = cand
    if not fields.get("client_name"):
        cm2 = re.search(r"\[[^\]]*ient\]\s*=\s*([^|]+)", raw, flags=re.IGNORECASE)
        if cm2:
            cand = cm2.group(1).strip()
            cand = _strip_client_name_money_leak(cand)
            cand = _clean_party_name_text(cand)
            if cand and not re.fullmatch(r"\d+(?:\.\d+)?", cand.strip()):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cand.strip()):
                    fields["client_name"] = cand
    if not fields.get("client_name"):
        pcr = re.search(r"\(\s*client\]?\s*=\s*([^|]+)", raw, flags=re.IGNORECASE)
        if pcr:
            cand = pcr.group(1).strip()
            cand = re.sub(r"['\u2018\u2019]\s*net\].*$", "", cand, flags=re.I)
            cand = re.sub(r"\s*=\s*[\d.,]+\s*$", "", cand)
            cand = _strip_client_name_money_leak(cand)
            cand = _clean_party_name_text(cand)
            if cand and len(cand) >= 2 and not re.fullmatch(r"\d+(?:\.\d+)?", cand.strip()):
                fields["client_name"] = cand

    _extract_tax_field(raw, fields)

    cn2 = fields.get("client_name")
    if isinstance(cn2, str) and _looks_like_money_only(cn2):
        mv = _safe_float(cn2)
        if mv is not None:
            ta0 = _safe_float(fields.get("total_amount"))
            if ta0 is None or abs(ta0 - mv) < 1e-6:
                fields["total_amount"] = f"{mv:.2f}"
            fields["client_name"] = ""

    nw = _safe_float(fields.get("net_worth"))
    tx = _safe_float(fields.get("tax"))
    ta = _safe_float(fields.get("total_amount"))
    if nw is not None and tx is not None and ta is not None and nw > 1.0:
        if abs(ta - tx) <= 1e-6 and abs((nw + tx) - ta) > max(0.05, 0.005 * max(nw, tx)):
            fields["total_amount"] = f"{nw + tx:.2f}"
        # Total should not be smaller than net or much smaller than net+tax.
        if ta + 1e-6 < nw or abs((nw + tx) - ta) > max(1.0, 0.02 * max(nw + tx, 1.0)):
            fields["total_amount"] = f"{nw + tx:.2f}"

    _repair_glued_hyphen_total_suffix(raw, fields)
    _fix_total_if_leading_digit_in_raw(raw, fields)
    _recover_total_from_raw_decimals(raw, fields)
    _extract_net_dollar_tax_adjacent(raw, fields)
    _infer_net_tax_from_balance(fields, raw)
    _maybe_swap_net_tax(fields)
    _rebalance_amount_triplet(raw, fields)
    _repair_amounts_from_raw_extremes(raw, fields)

    nw_f = _safe_float(fields.get("net_worth"))
    tx_f = _safe_float(fields.get("tax"))
    ta_f = _safe_float(fields.get("total_amount"))
    if nw_f is not None and tx_f is not None and ta_f is not None:
        tol_bal = max(0.05, 0.005 * ta_f)
        if abs((nw_f + tx_f) - ta_f) > tol_bal:
            _harmonize_money_triplet(raw, fields)

    _reapply_coherent_bracket_amounts(raw, fields)
    _fix_centifold_money_totals(fields)
    _fix_centifold_using_explicit_net(raw, fields)
    _prefer_explicit_net_when_harmonize_inflated(raw, fields)

    _demote_scaled_total(raw, fields)
    _repair_total_minus_century_when_matches_net_plus_tax(fields)

    nw2 = _safe_float(fields.get("net_worth"))
    tx2 = _safe_float(fields.get("tax"))
    ta2 = _safe_float(fields.get("total_amount"))
    if nw2 is not None and ta2 is not None and tx2 is None:
        t_infer = ta2 - nw2
        if t_infer > 0.01:
            fields["tax"] = f"{t_infer:.2f}"
            tx2 = t_infer
    if ta2 is not None and tx2 is not None and nw2 is None:
        n_infer = ta2 - tx2
        if n_infer > 0.01:
            fields["net_worth"] = f"{n_infer:.2f}"

    _lift_total_when_collapsed_to_net(raw, fields)
    _snap_total_to_net_plus_tax(fields)
    _snap_net_to_implied_total_minus_tax(fields)

    # Infer grand total when line cuts off after tax/net but GT expects net+tax≈total (weak fallback)
    if fields.get("tax") and fields.get("net_worth") and not fields.get("total_amount"):
        try:
            t = float(str(fields["tax"]).replace(",", ""))
            n = float(str(fields["net_worth"]).replace(",", ""))
            fields["total_amount"] = f"{n + t:.2f}"
        except ValueError:
            pass

    # Last pass: clip '(client]=' spills again after downstream edits.
    sn_last = fields.get("seller_name")
    if isinstance(sn_last, str) and sn_last.strip():
        fields["seller_name"] = _clean_party_name_text(_trim_seller_value_at_client_spill(sn_last))
    cn_last = fields.get("client_name")
    if isinstance(cn_last, str) and cn_last.strip():
        fields["client_name"] = _clean_party_name_text(_strip_client_name_money_leak(cn_last))
    if isinstance(fields.get("seller_name"), str) and fields["seller_name"] and not fields.get("client_name"):
        _split_seller_embedded_client(fields)

    _unpack_client_glued_amounts(fields)
    _recover_client_if_invclient_was_iso_date(fields, raw)
    if isinstance(fields.get("client_name"), str):
        fields["client_name"] = _clean_party_name_text(fields["client_name"])

    _fix_centifold_money_totals(fields)
    _repair_tax_when_ocr_dropped_two_digits(fields)
    _repair_total_when_sum_is_ten_x_parsed_total(fields)
    _repair_tax_outlier_from_total(fields)
    _infer_net_tax_from_inclusive_total(fields, raw)
    _infer_tax_total_from_net_only(fields, raw)

    return fields

def _normalize_eval_value(value: Any, field: str) -> Optional[str]:
    """
    Normalize eval value.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        value (Any): Input parameter.
        field (str): Input parameter.
    Outputs:
        Optional[str]: Function output value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    if field == "invoice_date":
        dt = pd.to_datetime(s, errors="coerce")
        return dt.strftime("%Y-%m-%d") if pd.notna(dt) else None

    if field in {"total_amount", "tax", "net_worth"}:
        s = s.replace("$", "").replace(",", "").strip()
        try:
            return f"{float(s):.2f}"
        except ValueError:
            return None

    if field == "invoice_number":
        return re.sub(r"\D+", "", s)

    return re.sub(r"\s+", " ", s).lower()


def _field_equal_tolerant(pred_val: Any, ref_val: Any, field: str) -> bool:
    """
    Field equal tolerant.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        pred_val (Any): Input parameter.
        ref_val (Any): Input parameter.
        field (str): Input parameter.
    Outputs:
        bool: Function output value.
    """
    pred_norm = _normalize_eval_value(pred_val, field)
    ref_norm = _normalize_eval_value(ref_val, field)

    if pred_norm is None or ref_norm is None:
        return False

    # Numeric tolerance
    if field in {"total_amount", "tax", "net_worth"}:
        try:
            p = float(pred_norm)
            r = float(ref_norm)
            return abs(p - r) <= max(0.01, 0.01 * abs(r))
        except ValueError:
            return False

    # Date tolerance (string normalized already)
    if field == "invoice_date":
        return pred_norm == ref_norm

    # String tolerance (ignore minor noise)
    return pred_norm == ref_norm


def flatten_invoice_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a payload into the canonical invoice schema in a fixed order.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        parsed (Dict[str, Any]): Input parameter.
    Outputs:
        Dict[str, Any]: Function output value.
    """
    if not isinstance(parsed, dict):
        return {field: None for field in CANONICAL_INVOICE_FIELDS}

    source = parsed
    if isinstance(parsed.get("invoice"), dict):
        source = parsed["invoice"]
    elif isinstance(parsed.get("header"), dict):
        source = parsed["header"]
    elif isinstance(parsed.get("summary"), dict):
        source = parsed["summary"]

    flattened = {
        "invoice_number": normalize_invoice_field(
            source.get("invoice_number") or source.get("inv_no") or source.get("no") or source.get("id"),
            "invoice_number",
        ),
        "invoice_date": normalize_invoice_field(
            source.get("invoice_date") or source.get("date") or source.get("issue_date"),
            "invoice_date",
        ),
        "seller_name": normalize_invoice_field(
            source.get("seller_name") or source.get("vendor_name") or source.get("supplier_name"),
            "seller_name",
        ),
        "client_name": normalize_invoice_field(
            source.get("client_name") or source.get("buyer_name") or source.get("customer_name"),
            "client_name",
        ),
        "net_worth": normalize_invoice_field(
            source.get("net_worth") or source.get("subtotal") or source.get("sub_total") or source.get("net_amount"),
            "net_worth",
        ),
        "tax": normalize_invoice_field(
            source.get("tax") or source.get("vat") or source.get("vat_amount"),
            "tax",
        ),
        "total_amount": normalize_invoice_field(
            source.get("total_amount") or source.get("total") or source.get("grand_total") or source.get("gross_worth"),
            "total_amount",
        ),
    }

    # Guarantee the same field order everywhere.
    return {field: flattened.get(field) for field in CANONICAL_INVOICE_FIELDS}


def safe_val(x):
    """
    Safe val.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        x (Any): Input parameter.
    Outputs:
        Any: Function output value.
    """
    return x if x not in [None, ""] else "NULL"

def build_structured_invoice_text(invoice_payload):
    """
    Training/inference target: fixed bracket order matching CANONICAL_INVOICE_FIELDS.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        invoice_payload (Any): Input parameter.
    Outputs:
        Any: Function output value.
    """
    return (
        "<s_invoice>"
        f"[inv_no]={safe_val(invoice_payload['invoice_number'])} | "
        f"[inv_dt]={safe_val(invoice_payload['invoice_date'])} | "
        f"[seller]={safe_val(invoice_payload['seller_name'])} | "
        f"[client]={safe_val(invoice_payload['client_name'])} | "
        f"[net]={safe_val(invoice_payload['net_worth'])} | "
        f"[tax]={safe_val(invoice_payload['tax'])} | "
        f"[amt]={safe_val(invoice_payload['total_amount'])}"
        "</s>"
    )

def build_canonical_invoice_payload(row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the canonical invoice payload in a fixed field order.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        row (pd.Series | Dict[str, Any]): Input parameter.
    Outputs:
        Dict[str, Any]: Function output value.
    """
    payload = {
        "invoice_number": normalize_invoice_field(row.get("invoice_number"), "invoice_number"),
        "invoice_date": normalize_invoice_field(row.get("invoice_date"), "invoice_date"),
        "seller_name": normalize_invoice_field(row.get("seller_name"), "seller_name"),
        "client_name": normalize_invoice_field(row.get("client_name"), "client_name"),
        "net_worth": normalize_invoice_field(row.get("net_worth"), "net_worth"),
        "tax": normalize_invoice_field(row.get("tax"), "tax"),
        "total_amount": normalize_invoice_field(row.get("total_amount"), "total_amount"),
    }

    return {field: payload.get(field) for field in CANONICAL_INVOICE_FIELDS}



# Training-frame builder
def build_donut_pretraining_frame(
    ground_truth_df: pd.DataFrame,
    image_col: str = "original_path",
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    augment_factor: int = 1,
) -> pd.DataFrame:
    """
    Convert labeled data into invoice-only Donut training examples.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        ground_truth_df (pd.DataFrame): Input parameter.
        image_col (str): Input parameter. Defaults to 'original_path'.
        sample_frac (Optional[float]): Input parameter. Defaults to None.
        random_state (int): Input parameter. Defaults to 42.
        augment_factor (int): Input parameter. Defaults to 1.
    Outputs:
        pd.DataFrame: Function output value.
    """
    df = ground_truth_df.copy()
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        image_path = row.get(image_col)
        if pd.isna(image_path) or not str(image_path).strip():
            continue

        invoice_payload = build_canonical_invoice_payload(row)
        invoice_text = build_structured_invoice_text(invoice_payload)

        filled_fields = sum(
            1 for k in CANONICAL_INVOICE_FIELDS if invoice_payload.get(k) not in [None, ""]
        )
        invoice_weight = 1.0 + 0.12 * filled_fields

        for augment_id in range(max(1, augment_factor)):
            rows.append(
                {
                    "image_path": str(image_path),
                    "target_text": invoice_text,
                    "loss_weight": invoice_weight,
                    "source_idx": idx,
                    "augment_id": augment_id,
                }
            )

    return pd.DataFrame(rows)

def augment_document_image(image: Image.Image) -> Image.Image:
    """
    No-op augmentation for the first original-image Donut run.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        image (Image.Image): Input parameter.
    Outputs:
        Image.Image: Function output value.
    """
    return image

class DonutInvoiceDataset(torch.utils.data.Dataset):
    """PyTorch dataset for invoice-only Donut fine-tuning examples."""

    def __init__(self, df: pd.DataFrame, processor: DonutProcessor, max_length: int, augment: bool = False):
        """
        Init.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            df (pd.DataFrame): Input parameter.
            processor (DonutProcessor): Input parameter.
            max_length (int): Input parameter.
            augment (bool): Input parameter. Defaults to False.
        Outputs:
            Any: Function output value.
        """
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        """
        Len.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            None.
        Outputs:
            Any: Function output value.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Getitem.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            idx (Any): Input parameter.
        Outputs:
            Any: Function output value.
        """
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((960, 960), Image.Resampling.LANCZOS)

        if self.augment:
            image = augment_document_image(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row["target_text"],
            add_special_tokens=False,
            max_length=self.max_length - 1,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # FORCE EOS
        if labels[-1] != self.processor.tokenizer.eos_token_id:
            labels = torch.cat([
                labels,
                torch.tensor([self.processor.tokenizer.eos_token_id])
            ])

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "loss_weight": torch.tensor(float(row.get("loss_weight", 1.0)), dtype=torch.float32),
        }


class DonutDataCollator:
    def __call__(self, features):
        """
        Call.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            features (Any): Input parameter.
        Outputs:
            Any: Function output value.
        """
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        labels = [f["labels"] for f in features]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        loss_weight = torch.stack([f["loss_weight"] for f in features])

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "loss_weight": loss_weight,
        }


class WeightedDonutTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with per-sample loss weighting for numeric-heavy examples."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            model (Any): Input parameter.
            inputs (Any): Input parameter.
            return_outputs (Any): Input parameter. Defaults to False.
            num_items_in_batch (Any): Input parameter. Defaults to None.
        Outputs:
            Any: Function output value.
        """
        loss_weight = inputs.pop("loss_weight", None)
        pixel_values = inputs.pop("pixel_values")
        labels = inputs.pop("labels")

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss

        if loss_weight is not None:
            loss = loss * loss_weight.to(loss.device).mean()

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Prediction step.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            model (Any): Input parameter.
            inputs (Any): Input parameter.
            prediction_loss_only (Any): Input parameter.
            ignore_keys (Any): Input parameter. Defaults to None.
        Outputs:
            Any: Function output value.
        """
        inputs = inputs.copy()
        inputs.pop("loss_weight", None)

        has_labels = "labels" in inputs

        if self.args.predict_with_generate and not has_labels:
            batch_size = inputs["pixel_values"].size(0)

            decoder_input_ids = self.task_prompt_ids.repeat(batch_size, 1).to(
                inputs["pixel_values"].device
            )

            inputs["decoder_input_ids"] = decoder_input_ids

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# Validation metrics
def build_donut_compute_metrics(processor: DonutProcessor):
    """
    Create a compute_metrics function for Donut validation.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        processor (DonutProcessor): Input parameter.
    Outputs:
        Any: Function output value.
    """

    def compute_metrics(eval_pred):
        """
        Compute metrics.

        Notes:
            Standardized docstring style for Donut training utilities.
        Inputs:
            eval_pred (Any): Input parameter.
        Outputs:
            Any: Function output value.
        """
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # HF can return logits (B, T, V) or generated ids (B, T). Decode only valid token ids.
        preds = np.asarray(preds)
        vocab_size = int(getattr(processor.tokenizer, "vocab_size", 0) or 0)
        logits_threshold = max(3000, (vocab_size // 2) if vocab_size else 3000)
        if preds.ndim == 3:
            if preds.shape[-1] >= logits_threshold:
                preds = preds.argmax(axis=-1)
            else:
                preds = preds[:, 0, :]
        if preds.ndim == 1:
            preds = preds[None, :]
        preds = np.nan_to_num(preds, nan=processor.tokenizer.pad_token_id, posinf=processor.tokenizer.pad_token_id, neginf=0)
        preds = np.rint(preds).astype(np.int64, copy=False)
        vocab_hi = max(int(processor.tokenizer.vocab_size) - 1, 0)
        preds = np.clip(preds, 0, vocab_hi)

        labels = np.asarray(labels)
        if labels.ndim == 3:
            if labels.shape[-1] >= logits_threshold:
                labels = labels.argmax(axis=-1)
            else:
                labels = labels[:, 0, :]
        if labels.ndim == 1:
            labels = labels[None, :]
        labels = np.nan_to_num(labels, nan=-100, posinf=-100, neginf=-100)
        labels = np.rint(labels).astype(np.int64, copy=False)
        label_ids = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        label_ids = np.clip(label_ids, 0, vocab_hi)

        pred_texts = processor.batch_decode(preds.tolist(), skip_special_tokens=False)
        label_texts = processor.batch_decode(label_ids.tolist(), skip_special_tokens=False)

        pred_payloads = [parse_structured_invoice_text(t) for t in pred_texts]
        label_payloads = [parse_structured_invoice_text(t) for t in label_texts]

        pred_payload = pred_payloads[0]
        label_payload = label_payloads[0]

        print("PRED PAYLOAD:", pred_payload)
        print("LABEL PAYLOAD:", label_payload)
        print(
            "DATE CHECK:",
            pred_payload.get("invoice_date"),
            label_payload.get("invoice_date"),
            _field_equal_tolerant(
                pred_payload.get("invoice_date"),
                label_payload.get("invoice_date"),
                "invoice_date"
            )
        )
        
        parse_hits = 0

        stats = {
            field: {"correct": 0, "pred": 0, "gt": 0}
            for field in CANONICAL_INVOICE_FIELDS
        }

        for pred, ref in zip(pred_payloads, label_payloads):

            # parse_rate: did we get at least one real field?
            if any(pred.get(field) not in [None, ""] for field in CANONICAL_INVOICE_FIELDS):
                parse_hits += 1

            for field in CANONICAL_INVOICE_FIELDS:
                pred_val = pred.get(field)
                ref_val = ref.get(field)

                pred_present = pred_val not in [None, ""]
                ref_present = ref_val not in [None, ""]

                if pred_present:
                    stats[field]["pred"] += 1
                if ref_present:
                    stats[field]["gt"] += 1
                if pred_present and ref_present and _field_equal_tolerant(pred_val, ref_val, field):
                    stats[field]["correct"] += 1

        metrics = {
            "parse_rate": parse_hits / len(pred_payloads) if pred_payloads else 0.0,
        }

        for field, s in stats.items():
            accuracy = s["correct"] / len(pred_payloads) if pred_payloads else np.nan
            precision = s["correct"] / s["pred"] if s["pred"] else np.nan
            recall = s["correct"] / s["gt"] if s["gt"] else np.nan

            metrics[f"{field}_accuracy"] = accuracy
            metrics[f"{field}_precision"] = precision
            metrics[f"{field}_recall"] = recall

        return metrics

    return compute_metrics

# Training entry point
def train_donut_invoice_model(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str | Path = "./donut_model",
        config: Optional[DonutFineTuningConfig] = None,
        image_col: str = "original_path",
        augment_factor: int = 1,
    ):
    """
    Fine-tune a Donut model on invoice data.

    Notes:
        Standardized docstring style for Donut training utilities.
    Inputs:
        train_df (pd.DataFrame): Input parameter.
        val_df (pd.DataFrame): Input parameter.
        test_df (pd.DataFrame): Input parameter.
        output_dir (str | Path): Input parameter. Defaults to './donut_model'.
        config (Optional[DonutFineTuningConfig]): Input parameter. Defaults to None.
        image_col (str): Input parameter. Defaults to 'original_path'.
        augment_factor (int): Input parameter. Defaults to 1.
    Outputs:
        Any: Function output value.
    """
    config = config or DonutFineTuningConfig()
    device = resolve_donut_device(config.device)
    train_bs, eval_bs, grad_accum = recommend_donut_batch_sizes(device, config.model_name)

    processor = DonutProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    model.to(device)

    task_prompt_ids = processor.tokenizer(
        config.task_prompt_invoice,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids

    tokenizer = processor.tokenizer
    
    tokenizer.model_max_length = config.label_max_length

    # model side 
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<s_invoice>")
    # model.config.decoder_start_token_id = tokenizer.bos_token_id

    # generation side: fresh config, no inherited max_length=20
    model.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        max_new_tokens=config.generation_max_new_tokens,
        num_beams=1,
        do_sample=False,
        repetition_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=False
    )
    
    model.config.use_cache = False
    model.config.is_encoder_decoder = True
    

    if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
        model.decoder.config.use_cache = False

    train_frame = build_donut_pretraining_frame(
        train_df,
        image_col=image_col,
        augment_factor=augment_factor
    )

    sample_text = train_frame.iloc[0]["target_text"]
    sample_ids = tokenizer(sample_text, add_special_tokens=True).input_ids

    print("=== TRAIN SAMPLE CHECK ===")
    print(sample_text)
    print(tokenizer.decode(sample_ids, skip_special_tokens=False))
    print("EOS present:", tokenizer.eos_token_id in sample_ids)
    assert tokenizer.eos_token_id in sample_ids, "EOS token is missing from the training label."

    val_frame = build_donut_pretraining_frame(
        val_df,
        image_col=image_col,
        augment_factor=1
    )
    test_frame = build_donut_pretraining_frame(
        test_df,
        image_col=image_col,
        augment_factor=1
    )

    # labels
    max_length = config.label_max_length
    train_dataset = DonutInvoiceDataset(train_frame, processor, max_length=max_length, augment=True)
    val_dataset = DonutInvoiceDataset(val_frame, processor, max_length=max_length, augment=False)
    test_dataset = DonutInvoiceDataset(test_frame, processor, max_length=max_length, augment=False)

    compute_metrics = build_donut_compute_metrics(processor)

    fp16 = device.type == "cuda"
    bf16 = False

    total_steps = (
        len(train_dataset) // (train_bs * grad_accum)
    ) * config.num_train_epochs

    warmup_steps = int(0.05 * total_steps)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,

        eval_strategy="no",
        save_strategy="no",

        logging_strategy="steps",
        logging_steps=25,
        
        generation_num_beams=3,

        predict_with_generate=True,
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to="none",

        fp16=False,
        bf16=False,

        dataloader_num_workers=0 if device.type == "mps" else 2,
        eval_accumulation_steps=1 if device.type == "mps" else 4,
    )

    trainer = WeightedDonutTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DonutDataCollator(),
        processing_class=processor,
        compute_metrics=compute_metrics,
    )
    trainer.task_prompt_ids = task_prompt_ids

    trainer.train()
    # === DEBUG: Inspect one prediction vs label ===
    model.eval()

    sample = val_dataset[0]

    pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

    # bad_words = [
    #         "s_number",
    #         "number",
    #         "date=",
    #         "[date]",
    #         "[s_number]"
    #     ]

    # bad_words_ids = [
    #     tokenizer(bw, add_special_tokens=False).input_ids
    #     for bw in bad_words
    # ]

    # Generate prediction
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=trainer.task_prompt_ids.to(model.device),
        max_new_tokens=config.generation_max_new_tokens,
        # bad_words_ids=bad_words_ids,
    )

    pred_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]

    # Decode label
    label_ids = sample["labels"]
    label_ids = torch.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    label_text = processor.tokenizer.decode(label_ids, skip_special_tokens=False)

    print("\n=== DEBUG SAMPLE ===")
    print("PRED TEXT:\n", pred_text)
    print("\nLABEL TEXT:\n", label_text)
    print("====================\n")


    val_metrics = trainer.evaluate()
    test_metrics = trainer.predict(test_dataset).metrics

    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    processor.save_pretrained(str(save_path))

    return {
        "trainer": trainer,
        "processor": processor,
        "model": model,
        "train_frame": train_frame,
        "val_frame": val_frame,
        "test_frame": test_frame,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "save_path": str(save_path),
        "device": str(device),
        "batch_sizes": {
            "train": train_bs,
            "eval": eval_bs,
            "grad_accum": grad_accum,
        },
    }
