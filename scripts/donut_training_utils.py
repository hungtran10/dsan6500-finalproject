"""Training utilities for Donut invoice fine-tuning.

This module is intentionally separate from the inference pipeline. It provides:
- dataset preparation in Donut pretraining-style format
- mild document-safe augmentation
- early stopping + validation metrics
- numeric-heavy sample loss weighting
- optional table-extraction fine-tuning
- batch-size defaults that are conservative for Apple silicon / MPS (including M4)

The training examples are precomputed as text targets in the format:
    <s_invoice>{...json...}
    <s_table>{...json...}

That mirrors Donut's task-prompted generation workflow.
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
)


# -----------------------------------------------------------------------------
# Configuration / device helpers
# -----------------------------------------------------------------------------


@dataclass
class DonutFineTuningConfig:
    """Configuration for Donut fine-tuning on invoice documents."""

    model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    task_prompt_invoice: str = "<s_invoice>"
    task_prompt_table: str = "<s_table>"
    max_length_invoice: int = 512
    max_length_table: int = 768
    num_train_epochs: int = 15
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 3
    numeric_loss_weight: float = 1.5
    table_loss_weight: float = 2.0
    augment_factor: int = 1
    random_state: int = 42
    device: Optional[str] = None


def resolve_donut_device(device: Optional[str] = None) -> torch.device:
    """Resolve the best available device for Donut training/inference."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def recommend_donut_batch_sizes(device: torch.device, model_name: str = "") -> Tuple[int, int, int]:
    """Recommend train/eval batch sizes and gradient accumulation for the current device.

    For Apple silicon / MPS (including M4), a conservative starting point is
    batch size 1 with gradient accumulation. Increase only after confirming
    memory stability.
    """
    if device.type == "mps":
        return 1, 1, 8
    if device.type == "cuda":
        # Conservative defaults. Increase only if you have ample GPU memory.
        base = 2 if "base" in model_name else 1
        return base, base, 4
    return 1, 1, 1


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------


def normalize_money(value: Any) -> Optional[str]:
    """Normalize money strings to a plain decimal string with two places."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def normalize_date(value: Any) -> Optional[str]:
    """Normalize date-like values to ISO format (YYYY-MM-DD)."""
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
    """Normalize one invoice field for training/evaluation."""
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
    """Remove the leading Donut task token from a decoded sequence."""
    return re.sub(r"^<[^>]+>", "", sequence).strip()


def safe_json_loads(sequence: str) -> Dict[str, Any]:
    """Parse a Donut-like sequence into a dictionary when possible."""
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


def flatten_invoice_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten common Donut invoice payload shapes into a canonical dictionary."""
    if not isinstance(parsed, dict):
        return {}

    if "invoice" in parsed and isinstance(parsed["invoice"], dict):
        invoice = parsed["invoice"]
    elif "header" in parsed and isinstance(parsed["header"], dict):
        invoice = parsed["header"]
    else:
        invoice = parsed

    return {
        "invoice_number": invoice.get("invoice_number") or invoice.get("inv_no") or invoice.get("no"),
        "invoice_date": invoice.get("invoice_date") or invoice.get("date") or invoice.get("issue_date"),
        "seller_name": invoice.get("seller_name") or invoice.get("vendor_name") or invoice.get("supplier_name"),
        "client_name": invoice.get("client_name") or invoice.get("buyer_name") or invoice.get("customer_name"),
        "net_worth": invoice.get("net_worth") or invoice.get("subtotal") or invoice.get("sub_total") or invoice.get("net_amount"),
        "tax": invoice.get("tax") or invoice.get("vat") or invoice.get("vat_amount"),
        "total_amount": invoice.get("total_amount") or invoice.get("total") or invoice.get("grand_total") or invoice.get("gross_worth"),
    }


def normalize_line_items_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalize line-item annotations into a list of canonical dictionaries.

    This is intentionally permissive. If your dataframe does not yet include
    line-item labels, the function returns an empty list and the table task can
    be skipped.
    """
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return []

    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return []
        try:
            payload = json.loads(payload)
        except Exception:
            return []

    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        return []

    normalized = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "item_no": item.get("item_no") or item.get("no") or item.get("line_no"),
            "description": item.get("description") or item.get("desc") or item.get("name"),
            "qty": item.get("qty") or item.get("quantity"),
            "um": item.get("um") or item.get("unit") or item.get("uom"),
            "net_price": item.get("net_price") or item.get("unit_price"),
            "net_worth": item.get("net_worth") or item.get("line_net_worth"),
            "vat_pct": item.get("vat_pct") or item.get("vat") or item.get("tax_rate"),
            "gross_worth": item.get("gross_worth") or item.get("line_gross_worth"),
        })
    return normalized


# -----------------------------------------------------------------------------
# Training-frame builder
# -----------------------------------------------------------------------------


def build_donut_pretraining_frame(
    ground_truth_df: pd.DataFrame,
    image_col: str = "processed_path",
    include_table_task: bool = True,
    table_col_candidates: Sequence[str] = ("line_items_json", "table_json", "line_items"),
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    augment_factor: int = 1,
) -> pd.DataFrame:
    """Convert a labeled dataframe into Donut pretraining-style examples.

    Returns a dataframe with columns:
    - image_path
    - task
    - target_text
    - loss_weight
    - source_idx
    - augment_id

    The target text is JSON-like and begins with a Donut task token, e.g.:
    <s_invoice>{...}
    <s_table>{...}
    """
    df = ground_truth_df.copy()
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        image_path = row.get(image_col)
        if pd.isna(image_path) or not str(image_path).strip():
            continue

        invoice = {
            "invoice_number": normalize_invoice_field(row.get("invoice_number"), "invoice_number"),
            "invoice_date": normalize_invoice_field(row.get("invoice_date"), "invoice_date"),
            "seller_name": normalize_invoice_field(row.get("seller_name"), "seller_name"),
            "client_name": normalize_invoice_field(row.get("client_name"), "client_name"),
            "net_worth": normalize_invoice_field(row.get("net_worth"), "net_worth"),
            "tax": normalize_invoice_field(row.get("tax"), "tax"),
            "total_amount": normalize_invoice_field(row.get("total_amount"), "total_amount"),
        }

        invoice_payload = {"task": "invoice", "invoice": invoice}
        invoice_text = json.dumps(invoice_payload, ensure_ascii=False, sort_keys=True)
        numeric_fields = sum(1 for k in ["invoice_number", "invoice_date", "net_worth", "tax", "total_amount"] if invoice.get(k) not in [None, ""])
        invoice_weight = 1.0 + 0.15 * numeric_fields

        for augment_id in range(max(1, augment_factor)):
            rows.append({
                "image_path": str(image_path),
                "task": "invoice",
                "target_text": f"<s_invoice>{invoice_text}",
                "loss_weight": invoice_weight,
                "source_idx": idx,
                "augment_id": augment_id,
            })

        if include_table_task:
            line_items = None
            for c in table_col_candidates:
                if c in row and pd.notna(row[c]):
                    line_items = normalize_line_items_payload(row[c])
                    break

            if line_items:
                table_payload = {"task": "table", "line_items": line_items}
                table_text = json.dumps(table_payload, ensure_ascii=False, sort_keys=True)
                table_weight = 1.0 + 0.25 * len(line_items)
                for augment_id in range(max(1, augment_factor)):
                    rows.append({
                        "image_path": str(image_path),
                        "task": "table",
                        "target_text": f"<s_table>{table_text}",
                        "loss_weight": table_weight,
                        "source_idx": idx,
                        "augment_id": augment_id,
                    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dataset / collator / trainer
# -----------------------------------------------------------------------------


def augment_document_image(image: Image.Image) -> Image.Image:
    """Apply conservative document-safe augmentation."""
    if random.random() < 0.5:
        angle = random.uniform(-1.5, 1.5)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.95, 1.08))
    return image


class DonutInvoiceDataset(torch.utils.data.Dataset):
    """PyTorch dataset for Donut fine-tuning examples."""

    def __init__(self, df: pd.DataFrame, processor: DonutProcessor, max_length: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.augment and row.get("task") == "invoice":
            image = augment_document_image(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row["target_text"],
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "loss_weight": torch.tensor(float(row.get("loss_weight", 1.0)), dtype=torch.float32),
            "task": row.get("task", "invoice"),
        }


class DonutDataCollator:
    """Stack Donut batches while preserving per-example loss weights."""

    def __init__(self, processor: DonutProcessor):
        self.processor = processor

    def __call__(self, features):
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "loss_weight": torch.stack([f["loss_weight"] for f in features]),
        }


class WeightedDonutTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with per-sample loss weighting for numeric-heavy examples."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weight = inputs.pop("loss_weight", None)
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        valid_tokens = (shift_labels != -100).float()
        sample_loss = (token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1.0)

        if loss_weight is not None:
            sample_loss = sample_loss * loss_weight.to(sample_loss.device)

        loss = sample_loss.mean()
        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Validation metrics
# -----------------------------------------------------------------------------


def build_donut_compute_metrics(processor: DonutProcessor):
    """Create a compute_metrics function for Donut validation.

    Returns exact-match style metrics for invoice fields and table payloads.
    """

    field_keys = ["invoice_number", "invoice_date", "seller_name", "client_name", "net_worth", "tax", "total_amount"]

    def _normalize_payload(payload):
        if not isinstance(payload, dict):
            return {}
        if payload.get("task") == "invoice" and isinstance(payload.get("invoice"), dict):
            inv = payload["invoice"]
            return {k: normalize_invoice_field(v, k) for k, v in inv.items()}
        if payload.get("task") == "table" and isinstance(payload.get("line_items"), list):
            items = []
            for item in payload["line_items"]:
                if not isinstance(item, dict):
                    continue
                items.append({k: normalize_invoice_field(v, k) for k, v in item.items()})
            return {"line_items": items}
        return {k: normalize_invoice_field(v, k) for k, v in payload.items() if k != "task"}

    def _parse(seq: str) -> dict:
        seq = seq.replace(processor.tokenizer.eos_token or "", "").replace(processor.tokenizer.pad_token or "", "")
        seq = _strip_task_token(seq)
        try:
            return json.loads(seq)
        except Exception:
            try:
                return processor.token2json(seq)
            except Exception:
                return {"raw_text": seq}

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        pred_texts = processor.batch_decode(preds, skip_special_tokens=False)
        label_ids = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        label_texts = processor.batch_decode(label_ids, skip_special_tokens=False)

        pred_payloads = [_normalize_payload(_parse(t)) for t in pred_texts]
        label_payloads = [_normalize_payload(_parse(t)) for t in label_texts]

        exact_matches = []
        field_scores = {k: [] for k in field_keys}
        table_exact = []

        for pred, ref in zip(pred_payloads, label_payloads):
            exact_matches.append(int(pred == ref))

            # Invoice payload
            if "line_items" not in ref:
                for k in field_keys:
                    if k in ref or k in pred:
                        field_scores[k].append(int(pred.get(k) == ref.get(k)))

            # Table payload
            if "line_items" in ref:
                def canonical_items(items):
                    return json.dumps(items, sort_keys=True, ensure_ascii=False)
                table_exact.append(int(canonical_items(pred.get("line_items", [])) == canonical_items(ref.get("line_items", []))))

        metrics = {
            "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        }
        for k, vals in field_scores.items():
            metrics[f"{k}_acc"] = float(np.mean(vals)) if vals else np.nan
        if table_exact:
            metrics["table_exact_match"] = float(np.mean(table_exact))
        return metrics

    return compute_metrics


# -----------------------------------------------------------------------------
# Training entry point
# -----------------------------------------------------------------------------


def train_donut_invoice_model(
    ground_truth_df: pd.DataFrame,
    output_dir: str | Path = "./donut_model",
    config: Optional[DonutFineTuningConfig] = None,
    image_col: str = "processed_path",
    include_table_task: bool = True,
    table_col_candidates: Sequence[str] = ("line_items_json", "table_json", "line_items"),
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
    augment_factor: int = 1,
    task_mode: str = "multi",
):
    """Fine-tune a Donut model on invoice data.

    Parameters
    ----------
    ground_truth_df : pandas.DataFrame
        Labeled invoice dataframe containing processed image paths and ground truth fields.
    output_dir : str or Path, optional
        Where to save checkpoints and the trained model.
    config : DonutFineTuningConfig, optional
        Training configuration.
    image_col : str, optional
        Column containing the processed image path.
    include_table_task : bool, optional
        Include table-extraction examples if line-item labels exist.
    table_col_candidates : sequence[str], optional
        Candidate columns for table annotations.
    sample_frac : float or None, optional
        Optionally sample a fraction of the dataset before splitting.
    random_state : int, optional
        Random seed.
    val_size : float, optional
        Validation split fraction.
    test_size : float, optional
        Test split fraction.
    augment_factor : int, optional
        How many training copies to create per row in the pretraining frame.
        Use 1 for no duplication, or >1 to expand the training set.
    task_mode : str, optional
        "invoice", "table", or "multi".

    Returns
    -------
    dict
        Dictionary containing the trained trainer, processor, model, split dataframes,
        test metrics, and the model save path.
    """
    config = config or DonutFineTuningConfig()
    device = resolve_donut_device(config.device)
    train_bs, eval_bs, grad_accum = recommend_donut_batch_sizes(device, config.model_name)

    df = ground_truth_df.copy()
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=random_state)
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test, random_state=random_state)

    processor = DonutProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    model.to(device)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    if task_mode not in {"invoice", "table", "multi"}:
        raise ValueError("task_mode must be one of: 'invoice', 'table', 'multi'.")

    include_table = task_mode in {"table", "multi"} and include_table_task

    train_frame = build_donut_pretraining_frame(
        train_df,
        image_col=image_col,
        include_table_task=include_table,
        table_col_candidates=table_col_candidates,
        augment_factor=augment_factor,
        random_state=random_state,
    )
    val_frame = build_donut_pretraining_frame(
        val_df,
        image_col=image_col,
        include_table_task=include_table,
        table_col_candidates=table_col_candidates,
        augment_factor=1,
        random_state=random_state,
    )
    test_frame = build_donut_pretraining_frame(
        test_df,
        image_col=image_col,
        include_table_task=include_table,
        table_col_candidates=table_col_candidates,
        augment_factor=1,
        random_state=random_state,
    )

    if task_mode == "invoice":
        train_frame = train_frame[train_frame["task"] == "invoice"].reset_index(drop=True)
        val_frame = val_frame[val_frame["task"] == "invoice"].reset_index(drop=True)
        test_frame = test_frame[test_frame["task"] == "invoice"].reset_index(drop=True)
    elif task_mode == "table":
        train_frame = train_frame[train_frame["task"] == "table"].reset_index(drop=True)
        val_frame = val_frame[val_frame["task"] == "table"].reset_index(drop=True)
        test_frame = test_frame[test_frame["task"] == "table"].reset_index(drop=True)

    if include_table and not (train_frame["task"] == "table").any():
        print("WARNING: No table annotations were found. Table fine-tuning will be skipped.")

    max_length = max(config.max_length_invoice, config.max_length_table)
    train_dataset = DonutInvoiceDataset(train_frame, processor, max_length=max_length, augment=True)
    val_dataset = DonutInvoiceDataset(val_frame, processor, max_length=max_length, augment=False)
    test_dataset = DonutInvoiceDataset(test_frame, processor, max_length=max_length, augment=False)

    compute_metrics = build_donut_compute_metrics(processor)

    # Apple silicon / MPS usually benefits from tiny batches and gradient accumulation.
    fp16 = device.type == "cuda"
    bf16 = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=max_length,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=0 if device.type == "mps" else 2,
        eval_accumulation_steps=1 if device.type == "mps" else 4,
    )

    trainer = WeightedDonutTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DonutDataCollator(processor),
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    trainer.train()
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


# -----------------------------------------------------------------------------
# Usage notes
# -----------------------------------------------------------------------------
# 1) Invoice-only fine-tuning:
#    train_donut_invoice_model(ground_truth_df, task_mode="invoice", augment_factor=2)
#
# 2) Table-only fine-tuning (requires line-item labels):
#    train_donut_invoice_model(ground_truth_df, task_mode="table", include_table_task=True)
#
# 3) Multi-task fine-tuning:
#    train_donut_invoice_model(ground_truth_df, task_mode="multi", include_table_task=True)
