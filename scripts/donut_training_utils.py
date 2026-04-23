"""Training utilities for invoice-only Donut fine-tuning.

This module is intentionally separate from the inference pipeline.
It provides:
- canonical invoice payload construction
- mild document-safe augmentation
- dataset + collator
- weighted seq2seq training
- validation metrics
- device / batch-size helpers

The training target format is:
    <s_invoice>{"invoice_number":...,"invoice_date":...,"seller_name":...,"client_name":...,"net_worth":...,"tax":...,"total_amount":...}

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
    label_max_length: int = 128
    generation_max_new_tokens: int = 64
    num_train_epochs: int = 15
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 3
    numeric_loss_weight: float = 1.5
    augment_factor: int = 1
    random_state: int = 42
    device: Optional[str] = None

# Device helpers
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
    """Recommend train/eval batch sizes and gradient accumulation for the device."""
    if device.type == "mps":
        return 1, 1, 8
    if device.type == "cuda":
        base = 2 if "base" in model_name else 1
        return base, base, 4
    return 1, 1, 1



# Normalization and parsing helpers
def normalize_money(value: Any) -> Optional[str]:
    """Normalize money strings to a plain decimal string with two places."""
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
    """Parse a Donut-like sequence into a dictionary when possible.

    This is the single parser used across the module.
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


def flatten_invoice_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a payload into the canonical invoice schema in a fixed order."""
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


def build_canonical_invoice_payload(row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
    """Build the canonical invoice payload in a fixed field order."""
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
    """Convert labeled data into invoice-only Donut training examples.

    Returns a dataframe with columns:
    - image_path
    - target_text
    - loss_weight
    - source_idx
    - augment_id
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
        invoice_text = json.dumps(invoice_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

        numeric_fields = sum(
            1
            for k in ["invoice_number", "invoice_date", "net_worth", "tax", "total_amount"]
            if invoice_payload.get(k) not in [None, ""]
        )
        invoice_weight = 1.0 + 0.15 * numeric_fields

        for augment_id in range(max(1, augment_factor)):
            rows.append(
                {
                    "image_path": str(image_path),
                    "target_text": f"<s_invoice>{invoice_text}",
                    "loss_weight": invoice_weight,
                    "source_idx": idx,
                    "augment_id": augment_id,
                }
            )

    return pd.DataFrame(rows)

def augment_document_image(image: Image.Image) -> Image.Image:
    """No-op augmentation for the first original-image Donut run."""
    return image

class DonutInvoiceDataset(torch.utils.data.Dataset):
    """PyTorch dataset for invoice-only Donut fine-tuning examples."""

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
        #image.thumbnail((960, 960), Image.Resampling.LANCZOS)

        if self.augment:
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
        }


class DonutDataCollator:
    """Stack Donut batches while preserving per-example loss weights."""

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
        inputs = inputs.copy()

        # Remove loss_weight before generate()
        inputs.pop("loss_weight", None)

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# Validation metrics
def build_donut_compute_metrics(processor: DonutProcessor):
    """Create a compute_metrics function for Donut validation."""

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        pred_texts = processor.batch_decode(preds, skip_special_tokens=False)
        label_ids = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        label_texts = processor.batch_decode(label_ids, skip_special_tokens=False)

        pred_payloads = [flatten_invoice_payload(safe_json_loads(t)) for t in pred_texts]
        label_payloads = [flatten_invoice_payload(safe_json_loads(t)) for t in label_texts]

        exact_matches = []
        parse_hits = 0

        stats = {
            field: {"correct": 0, "pred": 0, "gt": 0}
            for field in CANONICAL_INVOICE_FIELDS
        }

        for pred, ref in zip(pred_payloads, label_payloads):
            exact_matches.append(int(pred == ref))

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
                if pred_present and ref_present and pred_val == ref_val:
                    stats[field]["correct"] += 1

        metrics = {
            "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
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
    """Fine-tune a Donut model on invoice data."""
    config = config or DonutFineTuningConfig()
    device = resolve_donut_device(config.device)
    train_bs, eval_bs, grad_accum = recommend_donut_batch_sizes(device, config.model_name)

    processor = DonutProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    model.to(device)

    tokenizer = processor.tokenizer

    tokenizer.model_max_length = config.label_max_length

    # model side 
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # generation side: fresh config, no inherited max_length=20
    model.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        max_new_tokens=config.generation_max_new_tokens,
        num_beams=1,
        do_sample=False,
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

    # generation
    model.generation_config.max_new_tokens = config.generation_max_new_tokens

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
        num_train_epochs=min(config.num_train_epochs, 5),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,

        eval_strategy="no",
        save_strategy="no",

        logging_strategy="steps",
        logging_steps=25,

        predict_with_generate=True,
        generation_max_length=None,
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

