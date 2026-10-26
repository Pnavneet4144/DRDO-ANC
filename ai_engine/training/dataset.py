"""Dataset loader for the DRDO-ANC audio dataset."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ai_engine.models.audio_io import load_wav
from common.constants import CLIP_SAMPLES

logger = logging.getLogger(__name__)

# Columns required for every manifest row to be loadable.
_REQUIRED_COLUMNS = {"split", "mixed_path", "clean_path", "reference_path"}


class AncDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for active noise cancellation / speech enhancement.

    Loads noisy mixtures, clean speech, and reference microphone signals
    from a CSV manifest, formatted for training or evaluation.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str = "train",
        target_length: int = CLIP_SAMPLES,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_csv)
        self.split = split
        self.target_length = target_length

        data_root_env = os.environ.get("DATA_ROOT")
        if not data_root_env:
            raise ValueError(
                "DATA_ROOT environment variable is not set. "
                "Set it to the root directory containing the dataset audio files."
            )
        self.data_root = Path(data_root_env)
        if not self.data_root.is_dir():
            raise ValueError(
                f"DATA_ROOT points to a path that is not a directory: {self.data_root}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_path}")

        self.rows: list[dict[str, str]] = []
        skipped = 0
        with self.manifest_path.open(mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # Validate that the CSV header contains all required columns.
            if reader.fieldnames is not None:
                header_set = set(reader.fieldnames)
                missing = _REQUIRED_COLUMNS - header_set
                if missing:
                    raise ValueError(
                        f"Manifest CSV is missing required columns: {sorted(missing)}. "
                        f"Expected at least: {sorted(_REQUIRED_COLUMNS)}"
                    )
            for row_num, row in enumerate(reader, start=2):  # row 1 = header
                if row.get("split") != self.split:
                    continue
                # Skip rows with empty required path fields.
                empty_fields = [
                    col
                    for col in ("mixed_path", "clean_path", "reference_path")
                    if not row.get(col, "").strip()
                ]
                if empty_fields:
                    logger.warning(
                        "Skipping manifest row %d (id=%s): empty fields %s",
                        row_num,
                        row.get("id", "?"),
                        empty_fields,
                    )
                    skipped += 1
                    continue
                self.rows.append(row)

        if skipped:
            logger.warning("Skipped %d invalid rows in split '%s'", skipped, self.split)

        logger.info(
            "Loaded AncDataset split='%s' with %d examples from %s",
            self.split,
            len(self.rows),
            self.manifest_path,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve path relative to DATA_ROOT if not absolute."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.data_root / p

    def _crop_or_pad_aligned(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crop or pad signals to target_length, synchronizing crop when aligned."""
        l_noisy = len(noisy)
        l_clean = len(clean)
        l_ref = len(reference)

        # Synchronize crop offset only when signals are confirmed time-aligned (same length)
        if l_noisy == l_clean == l_ref:
            if l_noisy > self.target_length:
                if self.split == "train":
                    max_start = l_noisy - self.target_length
                    start = int(torch.randint(0, max_start + 1, (1,)).item())
                else:
                    start = 0
                end = start + self.target_length
                noisy = noisy[start:end]
                clean = clean[start:end]
                reference = reference[start:end]
            elif l_noisy < self.target_length:
                pad_len = self.target_length - l_noisy
                noisy = np.pad(noisy, (0, pad_len))
                clean = np.pad(clean, (0, pad_len))
                reference = np.pad(reference, (0, pad_len))
        else:
            # Handle unaligned signals individually
            noisy = self._fix_length_single(noisy)
            clean = self._fix_length_single(clean)
            reference = self._fix_length_single(reference)

        return (
            torch.from_numpy(noisy.astype(np.float32)),
            torch.from_numpy(clean.astype(np.float32)),
            torch.from_numpy(reference.astype(np.float32)),
        )

    def _fix_length_single(self, x: np.ndarray) -> np.ndarray:
        """Crop or pad a single 1-D waveform to target_length."""
        lx = len(x)
        if lx > self.target_length:
            if self.split == "train":
                max_start = lx - self.target_length
                start = int(torch.randint(0, max_start + 1, (1,)).item())
            else:
                start = 0
            return x[start : start + self.target_length]
        if lx < self.target_length:
            return np.pad(x, (0, self.target_length - lx))
        return x

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        row_id = row.get("id", f"index-{index}")
        mixed_path = self._resolve_path(row["mixed_path"])
        clean_path = self._resolve_path(row["clean_path"])
        ref_path = self._resolve_path(row["reference_path"])

        try:
            noisy_arr = load_wav(mixed_path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to load noisy/mixed audio for row '{row_id}': {mixed_path}"
            ) from exc
        try:
            clean_arr = load_wav(clean_path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to load clean audio for row '{row_id}': {clean_path}"
            ) from exc
        try:
            ref_arr = load_wav(ref_path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to load reference audio for row '{row_id}': {ref_path}"
            ) from exc

        noisy, clean, reference = self._crop_or_pad_aligned(noisy_arr, clean_arr, ref_arr)

        return {
            "noisy": noisy,
            "clean": clean,
            "reference": reference,
        }
