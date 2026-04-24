from __future__ import annotations

import json
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def weak_label_words_bio(
    *,
    words: list[str],
    ground_truth_row: pd.Series,
    fields: Iterable[str] = DEFAULT_FIELDS,
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

    # Prefer matchable/structured fields first to reduce overlaps.
    priority = [
        "invoice_number",
        "invoice_date",
        "total_amount",
        "tax",
        "net_worth",
        "seller_name",
        "client_name",
    ]
    ordered_fields = [f for f in priority if f in fields] + [f for f in fields if f not in priority]

    for field in ordered_fields:
        target = _value_tokens_for_field(ground_truth_row.get(field), field)
        if not target:
            continue

        hay = [_token_norm_for_field(w, field) for w in words]
        hay = [h for h in hay]  # keep indices aligned (empty tokens stay empty)

        # Names: allow partial matches (e.g., missing punctuation); use text-normalized tokens
        match = _find_exact_subsequence(hay, target)
        if match is None:
            match = _find_fuzzy_window(hay, target)
        if match is None:
            continue

        s, e = match
        if any(occupied[s:e]):
            continue

        spans[field] = (s, e)
        occupied[s:e] = [True] * (e - s)

        ent = field.upper()
        labels[s] = f"B-{ent}"
        for i in range(s + 1, e):
            labels[i] = f"I-{ent}"

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

            labels, spans = weak_label_words_bio(words=words, ground_truth_row=row, fields=fields)

            examples.append(
                {
                    "image_path": str(image_path),
                    "processed_file": row.get(key_col) or Path(str(image_path)).name,
                    "words": words,
                    "boxes": boxes,
                    "labels": labels,
                    "matched_spans": {k: list(v) if v is not None else None for k, v in spans.items()},
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

    def __init__(self):
        self.model = None
        self.processor = None
        self.id2label: dict[int, str] | None = None
        self.label2id: dict[str, int] | None = None
        self.full_results: list[dict[str, Any]] | None = None

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

    def _predict_word_labels(self, *, image_path: str | Path) -> tuple[list[str], list[str]]:
        """
        Returns: (words, predicted_label_per_word)
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
        boxes = [
            _normalize_box_1000(box_xyxy=_xywh_to_xyxy(ow.bbox_xywh), width=w, height=h)
            for ow in ocr_words
        ]

        if not words:
            return [], []

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
            pred_ids = outputs.logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy().tolist()

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

        per_word_votes: dict[int, list[str]] = {}
        for token_idx, wid in enumerate(word_id_list):
            if wid is None:
                continue
            lbl = self.id2label.get(int(pred_ids[token_idx]), "O")
            per_word_votes.setdefault(int(wid), []).append(lbl)

        pred_word_labels = ["O"] * len(words)
        for wid, votes in per_word_votes.items():
            # simple majority vote
            counts: dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            pred_word_labels[wid] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]

        return words, pred_word_labels

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

    def predict(self, image_path: str | Path, *, fields: Iterable[str] = DEFAULT_FIELDS) -> dict[str, str]:
        words, word_labels = self._predict_word_labels(image_path=image_path)
        ents = self._entities_from_word_labels(words, word_labels)

        out: dict[str, str] = {}
        for f in fields:
            cand_list = ents.get(f, [])
            if not cand_list:
                continue

            # Pick first candidate that normalizes cleanly for structured fields.
            if f in {"total_amount", "tax", "net_worth"}:
                for c in cand_list:
                    nm = normalize_money(c)
                    if nm is not None:
                        out[f] = nm
                        break
                else:
                    out[f] = cand_list[0]
            elif f == "invoice_date":
                for c in cand_list:
                    nd = normalize_date(c)
                    if nd is not None:
                        out[f] = nd
                        break
                else:
                    out[f] = cand_list[0]
            else:
                # Names / invoice_number: prefer longer span
                out[f] = sorted(cand_list, key=lambda s: len(s), reverse=True)[0]

        return out

    def process_single_image(self, image_path: str | Path, *, fields: Iterable[str] = DEFAULT_FIELDS) -> dict[str, Any]:
        image_path = str(image_path)
        result: dict[str, Any] = {
            "image_path": image_path,
            "filename": Path(image_path).name,
            "success": False,
            "invoice_fields": {},
        }
        try:
            pred = self.predict(image_path, fields=fields)
            result["invoice_fields"] = pred
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        return result

    def run_inference(
        self,
        df: pd.DataFrame,
        *,
        image_col: str = "processed_path",
        key_col: str = "processed_file",
        fields: Iterable[str] = DEFAULT_FIELDS,
        sample_frac: float | None = None,
        random_state: int = 42,
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
            res = self.process_single_image(image_path, fields=fields)
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

