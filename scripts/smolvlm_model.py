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
    """
    Normalize arbitrary values to a clean string.

    Notes:
        Returns an empty string for null-like values to simplify JSON targets.

    Inputs:
        value: Any scalar value from model output or dataframe fields.

    Outputs:
        str: Stripped string value, or "" when value is null/NaN.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _extract_json_blob(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object found inside generated text.

    Notes:
        SmolVLM generations can include non-JSON prefixes/suffixes; this parser
        searches for a brace-delimited object and safely parses it.

    Inputs:
        text: Raw generated string from the model.

    Outputs:
        dict[str, Any]: Parsed JSON object, or {} when parsing fails.
    """
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
    """
    Single supervised training/inference record.

    Notes:
        `processed_file` acts as the merge/evaluation key across dataframes.

    Inputs:
        processed_file: Canonical processed filename key.
        image_path: Filesystem path to the processed invoice image.
        target_json: Field-value mapping used as the training target.

    Outputs:
        SmolVLMSample instance with strongly-typed sample metadata.
    """
    processed_file: str
    image_path: str
    target_json: dict[str, str]


class _SmolVLMDataset(Dataset):
    """
    Torch dataset that builds multimodal causal-LM training examples.

    Notes:
        Each item encodes `<image> + prompt + target_json` and masks prompt tokens
        in labels (`-100`) so loss is computed only on the JSON target segment.
    """

    def __init__(
        self,
        *,
        samples: list[SmolVLMSample],
        processor: AutoProcessor,
        prompt_template: str,
        max_label_length: int,
    ) -> None:
        """
        Initialize dataset state.

        Notes:
            `max_label_length` currently controls full-sequence truncation length.

        Inputs:
            samples: Supervised sample list.
            processor: Hugging Face multimodal processor.
            prompt_template: Prompt prefix containing image token and instructions.
            max_label_length: Maximum token length for encoded sequence.

        Outputs:
            None.
        """
        self.samples = samples
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_label_length = max_label_length

    def __len__(self) -> int:
        """
        Return number of available samples.

        Inputs:
            None.

        Outputs:
            int: Dataset size.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Build one encoded training example.

        Notes:
            Returns tokenizer/vision tensors and masked labels aligned with the
            full multimodal sequence length expected by causal-LM loss.

        Inputs:
            idx: Integer sample index.

        Outputs:
            dict[str, Any]: Encoded tensors (`input_ids`, `attention_mask`,
            `pixel_values`, `labels`) plus `processed_file`.
        """
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
            max_length=self.max_label_length,
        )
        full_text = f"{prompt}\n{target_text}"
        enc = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_label_length,
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        prompt_len = int(prompt_enc["input_ids"].shape[-1])
        labels = enc["input_ids"].clone()
        labels[:prompt_len] = -100
        enc["labels"] = labels
        enc["processed_file"] = sample.processed_file
        return enc


def _collate_fn(processor: AutoProcessor, batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate variable-length multimodal items into a batch.

    Notes:
        Pads text inputs with tokenizer rules and pads labels with `-100`
        so ignored positions do not contribute to loss.

    Inputs:
        processor: Processor supplying tokenizer pad behavior.
        batch: List of dataset item dictionaries.

    Outputs:
        dict[str, Any]: Batched tensors and list of `processed_file` keys.
    """
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
    End-to-end SmolVLM workflow for invoice extraction.

    Notes:
        Handles sample building, fine-tuning, batch inference, persistence, and
        exact-match evaluation via the shared project metrics utility.
    """

    def __init__(
        self,
        *,
        model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
        output_dir: str | Path = "outputs/smolvlm",
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize processor/model and runtime configuration.

        Notes:
            Loads pretrained weights immediately and creates output directory.

        Inputs:
            model_name: Hugging Face model id to load.
            output_dir: Directory for predictions/checkpoints.
            device: Optional runtime device override ("cpu"/"cuda").
            seed: Random seed for reproducibility.

        Outputs:
            None.
        """
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
        """
        Convert selected dataframe fields into clean string targets.

        Notes:
            Missing values are normalized to empty strings.

        Inputs:
            row: Source dataframe row.
            fields: Field names to extract from row.

        Outputs:
            dict[str, str]: Normalized field-value mapping.
        """
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
        Ensure a merge key column exists in a dataframe.

        Notes:
            If `merge_key` is missing, derives it from a known filename column and
            prefixes `processed_` when needed to match processed image keys.

        Inputs:
            df: Source dataframe.
            merge_key: Required key column name for joins/evaluation.
            file_name_candidates: Candidate filename columns for fallback derivation.

        Outputs:
            pd.DataFrame: Copy of input dataframe with a valid `merge_key` column.
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
        """
        Build supervised SmolVLM samples from ground truth + processed images.

        Notes:
            Joins on `merge_key`, optionally filters by success status, and skips
            rows whose image files are missing on disk.

        Inputs:
            ground_truth_df: Label dataframe containing invoice fields.
            processed_images_df: Processed image index with paths/status.
            fields: Fields to include in JSON targets.
            merge_key: Join key shared by both inputs.
            image_path_col: Column containing processed image paths.
            status_col: Optional success-status column in processed index.

        Outputs:
            list[SmolVLMSample]: Ready-to-train sample objects.
        """
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
        """
        Fine-tune SmolVLM on invoice extraction samples.

        Notes:
            Uses AdamW and optional validation loss tracking per epoch.

        Inputs:
            train_samples: Training sample list.
            val_samples: Optional validation sample list.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Optimizer learning rate.
            max_label_length: Max sequence length for processor truncation.

        Outputs:
            dict[str, list[float]]: Loss history with `train_loss` and `val_loss`.
        """
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
        """
        Save model and processor to disk.

        Inputs:
            save_dir: Optional explicit save path.

        Outputs:
            Path: Directory containing saved model artifacts.
        """
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
        """
        Run inference for one processed invoice image.

        Notes:
            Decodes model output and extracts the first valid JSON object.

        Inputs:
            image_path: Path to processed invoice image.
            max_new_tokens: Generation length cap.
            num_beams: Beam-search width (1 = greedy decoding).

        Outputs:
            dict[str, str]: Predicted field-value mapping.
        """
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
        """
        Run batched single-image inference over a processed dataframe.

        Notes:
            Filters to successful rows when status is available and persists a CSV
            of predictions to `<output_dir>/smolvlm_predictions.csv`.

        Inputs:
            processed_images_df: Dataframe with keys and image paths.
            merge_key: Output key column name in predictions.
            image_path_col: Column containing image file paths.
            status_col: Optional status column used to filter successful rows.

        Outputs:
            pd.DataFrame: Prediction dataframe keyed by `merge_key`.
        """
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
        """
        Evaluate predictions against ground truth with exact-match metrics.

        Notes:
            Normalizes merge keys in both inputs before delegating to
            `evaluate_exact_match`.

        Inputs:
            ground_truth_df: Ground-truth labels dataframe.
            pred_df: Prediction dataframe from model inference.
            fields: Fields to score.
            merge_key: Join key used for alignment.
            restrict_to_matched: Whether to score only overlapping keys.

        Outputs:
            tuple[pd.DataFrame, dict[str, Any]]:
                - Per-field metrics dataframe
                - Overall aggregated metrics dictionary
        """
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
