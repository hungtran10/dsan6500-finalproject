from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, set_seed
try:
    # Newer transformers API (commonly used for VLM generation).
    from transformers import AutoModelForVision2Seq as _AutoVisionModel  # type: ignore
except ImportError:  # pragma: no cover
    # Backward-compatible fallback for older transformers releases.
    from transformers import AutoModelForImageTextToText as _AutoVisionModel  # type: ignore

from .eval_utils import evaluate_exact_match


DEFAULT_FIELDS: list[str] = [
    "invoice_number",
    "invoice_date",
    "seller_name",
    "client_name",
    "tax",
    "net_worth",
    "total_amount",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _extract_json_blob(text: str) -> dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class SmolVLMSample:
    processed_file: str
    image_path: str
    target_json: dict[str, str]


class _SmolVLMDataset(Dataset):
    def __init__(
        self,
        *,
        samples: list[SmolVLMSample],
        processor: AutoProcessor,
        prompt_template: str,
        max_label_length: int,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_label_length = max_label_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        prompt = self.prompt_template
        target_text = json.dumps(sample.target_json, ensure_ascii=True)

        # Build a single causal-LM sequence and supervise only the target suffix.
        prompt_enc = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            truncation=True,
        )
        full_text = f"{prompt}\n{target_text}"
        enc = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            truncation=True,
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        prompt_len = int(prompt_enc["input_ids"].shape[-1])
        labels = enc["input_ids"].clone()
        labels[:prompt_len] = -100
        enc["labels"] = labels
        enc["processed_file"] = sample.processed_file
        return enc


def _collate_fn(processor: AutoProcessor, batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    pixel_values = [b["pixel_values"] for b in batch]
    labels = [b["labels"] for b in batch]
    processed_files = [b["processed_file"] for b in batch]

    padded_inputs = processor.tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        return_tensors="pt",
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "pixel_values": torch.stack(pixel_values),
        "labels": padded_labels,
        "processed_file": processed_files,
    }


class SmolVLMInvoiceModel:
    """
    Fine-tuning and inference helper around SmolVLM for invoice field extraction.
    """

    def __init__(
        self,
        *,
        model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        output_dir: str | Path = "outputs/smolvlm",
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = _AutoVisionModel.from_pretrained(model_name).to(self.device)
        self.model.train()

        self.prompt = (
            "<image>\nExtract these invoice fields and return JSON only: "
            "invoice_number, invoice_date, seller_name, client_name, tax, net_worth, total_amount."
        )

    @staticmethod
    def _normalize_target_row(row: pd.Series, fields: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for field in fields:
            out[field] = _safe_str(row.get(field))
        return out

    @staticmethod
    def _ensure_merge_key(
        df: pd.DataFrame,
        *,
        merge_key: str,
        file_name_candidates: tuple[str, ...] = ("File Name", "filename", "original_file"),
    ) -> pd.DataFrame:
        """
        Ensure `merge_key` exists; if missing, derive it from known filename columns.
        """
        out = df.copy()
        if merge_key in out.columns:
            out[merge_key] = out[merge_key].astype(str)
            return out

        source_col = next((c for c in file_name_candidates if c in out.columns), None)
        if source_col is None:
            available = ", ".join(out.columns.tolist())
            raise KeyError(
                f"Missing merge key '{merge_key}'. Could not derive from known filename "
                f"columns {file_name_candidates}. Available columns: [{available}]"
            )

        # Your processed index uses names like "processed_batch1-0494.jpg".
        out[merge_key] = out[source_col].astype(str).apply(
            lambda x: x if x.startswith("processed_") else f"processed_{x}"
        )
        return out

    def build_samples(
        self,
        *,
        ground_truth_df: pd.DataFrame,
        processed_images_df: pd.DataFrame,
        fields: Iterable[str] = DEFAULT_FIELDS,
        merge_key: str = "processed_file",
        image_path_col: str = "processed_path",
        status_col: str = "status",
    ) -> list[SmolVLMSample]:
        gt = self._ensure_merge_key(ground_truth_df, merge_key=merge_key)
        pi = self._ensure_merge_key(processed_images_df, merge_key=merge_key)

        if status_col in pi.columns:
            pi = pi[pi[status_col] == "success"].copy()

        keep_cols = [merge_key, image_path_col]
        pi = pi[keep_cols].dropna()
        merged = gt.merge(pi, on=merge_key, how="inner")

        samples: list[SmolVLMSample] = []
        for _, row in merged.iterrows():
            image_path = str(row[image_path_col])
            if not Path(image_path).exists():
                continue
            samples.append(
                SmolVLMSample(
                    processed_file=str(row[merge_key]),
                    image_path=image_path,
                    target_json=self._normalize_target_row(row, fields),
                )
            )
        return samples

    def train(
        self,
        *,
        train_samples: list[SmolVLMSample],
        val_samples: list[SmolVLMSample] | None = None,
        epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        max_label_length: int = 256,
    ) -> dict[str, list[float]]:
        train_ds = _SmolVLMDataset(
            samples=train_samples,
            processor=self.processor,
            prompt_template=self.prompt,
            max_label_length=max_label_length,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate_fn(self.processor, b),
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        val_loader = None
        if val_samples:
            val_ds = _SmolVLMDataset(
                samples=val_samples,
                processor=self.processor,
                prompt_template=self.prompt,
                max_label_length=max_label_length,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: _collate_fn(self.processor, b),
            )

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"SmolVLM train epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    pixel_values=batch["pixel_values"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_train_loss = running_loss / max(1, len(train_loader))
            history["train_loss"].append(epoch_train_loss)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = self.model(
                            input_ids=batch["input_ids"].to(self.device),
                            attention_mask=batch["attention_mask"].to(self.device),
                            pixel_values=batch["pixel_values"].to(self.device),
                            labels=batch["labels"].to(self.device),
                        )
                        val_loss += float(outputs.loss.item())
                history["val_loss"].append(val_loss / max(1, len(val_loader)))

        return history

    def save(self, save_dir: str | Path | None = None) -> Path:
        save_path = Path(save_dir) if save_dir else self.output_dir / "model"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        return save_path

    def predict_single(
        self,
        *,
        image_path: str | Path,
        max_new_tokens: int = 192,
        num_beams: int = 1,
    ) -> dict[str, str]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        parsed = _extract_json_blob(text)
        return {k: _safe_str(v) for k, v in parsed.items()}

    def predict_dataset(
        self,
        *,
        processed_images_df: pd.DataFrame,
        merge_key: str = "processed_file",
        image_path_col: str = "processed_path",
        status_col: str = "status",
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        work_df = processed_images_df.copy()
        if status_col in work_df.columns:
            work_df = work_df[work_df[status_col] == "success"].copy()

        for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="SmolVLM inference"):
            image_path = row.get(image_path_col)
            if not image_path or not Path(str(image_path)).exists():
                continue
            pred = self.predict_single(image_path=str(image_path))
            out = {merge_key: str(row[merge_key])}
            out.update(pred)
            rows.append(out)

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(self.output_dir / "smolvlm_predictions.csv", index=False)
        return pred_df

    def evaluate_against_ground_truth(
        self,
        *,
        ground_truth_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        fields: Iterable[str] = DEFAULT_FIELDS,
        merge_key: str = "processed_file",
        restrict_to_matched: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        gt = self._ensure_merge_key(ground_truth_df, merge_key=merge_key)
        pr = self._ensure_merge_key(
            pred_df,
            merge_key=merge_key,
            file_name_candidates=("processed_file", "filename", "File Name", "original_file"),
        )
        return evaluate_exact_match(
            ground_truth_df=gt,
            pred_df=pr,
            fields=fields,
            merge_key=merge_key,
            restrict_to_matched=restrict_to_matched,
        )
