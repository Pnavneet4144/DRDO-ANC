"""Unit tests for AncDataset loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from ai_engine.training.dataset import AncDataset
from common.constants import CLIP_SAMPLES, SAMPLE_RATE, SEED


def _create_synthetic_wav(path: Path, num_samples: int) -> None:
    """Write a synthetic float32 sine tone as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, num_samples / SAMPLE_RATE, num_samples, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), tone, SAMPLE_RATE, subtype="FLOAT")


@pytest.fixture()
def synthetic_data_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create synthetic dataset and return path to manifest CSV."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    # Create synthetic WAVs of standard length (CLIP_SAMPLES = 64000)
    manifest_rows = [
        # Train examples
        {
            "id": "tr_01",
            "speaker_id": "spk1",
            "clean_path": "clean/tr_01.wav",
            "noise_path": "noise/tr_01.wav",
            "reference_path": "ref/tr_01.wav",
            "noise_type": "white",
            "snr_db": "5.0",
            "split": "train",
            "mixed_path": "mixed/tr_01.wav",
        },
        {
            "id": "tr_02",
            "speaker_id": "spk2",
            "clean_path": "clean/tr_02.wav",
            "noise_path": "noise/tr_02.wav",
            "reference_path": "ref/tr_02.wav",
            "noise_type": "pink",
            "snr_db": "0.0",
            "split": "train",
            "mixed_path": "mixed/tr_02.wav",
        },
        # Validation example
        {
            "id": "val_01",
            "speaker_id": "spk3",
            "clean_path": "clean/val_01.wav",
            "noise_path": "noise/val_01.wav",
            "reference_path": "ref/val_01.wav",
            "noise_type": "babble",
            "snr_db": "10.0",
            "split": "validation",
            "mixed_path": "mixed/val_01.wav",
        },
        # Test example
        {
            "id": "test_01",
            "speaker_id": "spk4",
            "clean_path": "clean/test_01.wav",
            "noise_path": "noise/test_01.wav",
            "reference_path": "ref/test_01.wav",
            "noise_type": "factory",
            "snr_db": "-5.0",
            "split": "test",
            "mixed_path": "mixed/test_01.wav",
        },
    ]

    for row in manifest_rows:
        for key in ("clean_path", "noise_path", "reference_path", "mixed_path"):
            _create_synthetic_wav(data_root / row[key], CLIP_SAMPLES)

    manifest_csv = tmp_path / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "speaker_id",
            "clean_path",
            "noise_path",
            "reference_path",
            "noise_type",
            "snr_db",
            "split",
            "mixed_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_csv


class TestAncDataset:
    """Test suite for AncDataset."""

    def test_output_keys_shapes_and_dtype(self, synthetic_data_setup: Path) -> None:
        """Verify dataset returns expected keys, shapes [CLIP_SAMPLES], and float32 dtype."""
        dataset = AncDataset(synthetic_data_setup, split="train")
        assert len(dataset) == 2

        item = dataset[0]
        assert set(item.keys()) == {"noisy", "clean", "reference"}

        for k in ("noisy", "clean", "reference"):
            tensor = item[k]
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (CLIP_SAMPLES,)
            assert tensor.dtype == torch.float32

    def test_split_filtering(self, synthetic_data_setup: Path) -> None:
        """Verify dataset only reads rows for requested split and never leaks other splits."""
        train_ds = AncDataset(synthetic_data_setup, split="train")
        assert len(train_ds) == 2
        assert all(row["split"] == "train" for row in train_ds.rows)

        val_ds = AncDataset(synthetic_data_setup, split="validation")
        assert len(val_ds) == 1
        assert val_ds.rows[0]["id"] == "val_01"

        test_ds = AncDataset(synthetic_data_setup, split="test")
        assert len(test_ds) == 1
        assert test_ds.rows[0]["id"] == "test_01"

        empty_ds = AncDataset(synthetic_data_setup, split="nonexistent")
        assert len(empty_ds) == 0

    def test_missing_or_unset_data_root(
        self, synthetic_data_setup: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify initializing AncDataset with unset DATA_ROOT raises a clear ValueError."""
        monkeypatch.delenv("DATA_ROOT", raising=False)
        with pytest.raises(ValueError, match="DATA_ROOT environment variable is not set"):
            AncDataset(synthetic_data_setup, split="train")

    def test_zero_padding_for_shorter_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify files shorter than target length are zero-padded to CLIP_SAMPLES."""
        data_root = tmp_path / "short_data"
        monkeypatch.setenv("DATA_ROOT", str(data_root))

        short_len = 16_000  # 1 second
        row = {
            "id": "short_01",
            "speaker_id": "spk1",
            "clean_path": "clean/s1.wav",
            "noise_path": "noise/s1.wav",
            "reference_path": "ref/s1.wav",
            "noise_type": "white",
            "snr_db": "0.0",
            "split": "train",
            "mixed_path": "mixed/s1.wav",
        }
        for k in ("clean_path", "noise_path", "reference_path", "mixed_path"):
            _create_synthetic_wav(data_root / row[k], short_len)

        csv_path = tmp_path / "short.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        ds = AncDataset(csv_path, split="train")
        item = ds[0]

        for k in ("noisy", "clean", "reference"):
            tensor = item[k]
            assert tensor.shape == (CLIP_SAMPLES,)
            # The padding after short_len must be exactly zeros
            assert (tensor[short_len:] == 0.0).all()
            assert (tensor[:short_len] != 0.0).any()

    def test_deterministic_cropping_for_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify files longer than target length have deterministic crop in validation split."""
        data_root = tmp_path / "long_data"
        monkeypatch.setenv("DATA_ROOT", str(data_root))

        long_len = 80_000  # 5 seconds
        row = {
            "id": "long_01",
            "speaker_id": "spk1",
            "clean_path": "clean/l1.wav",
            "noise_path": "noise/l1.wav",
            "reference_path": "ref/l1.wav",
            "noise_type": "white",
            "snr_db": "0.0",
            "split": "validation",
            "mixed_path": "mixed/l1.wav",
        }
        for k in ("clean_path", "noise_path", "reference_path", "mixed_path"):
            _create_synthetic_wav(data_root / row[k], long_len)

        csv_path = tmp_path / "long.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        ds = AncDataset(csv_path, split="validation")
        item1 = ds[0]
        item2 = ds[0]

        for k in ("noisy", "clean", "reference"):
            assert item1[k].shape == (CLIP_SAMPLES,)
            assert torch.equal(item1[k], item2[k])

    def test_reproducible_random_cropping_for_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify training crop is randomized but reproducible with seeded RNG."""
        data_root = tmp_path / "rand_data"
        monkeypatch.setenv("DATA_ROOT", str(data_root))

        long_len = 80_000
        row = {
            "id": "rand_01",
            "speaker_id": "spk1",
            "clean_path": "clean/r1.wav",
            "noise_path": "noise/r1.wav",
            "reference_path": "ref/r1.wav",
            "noise_type": "white",
            "snr_db": "0.0",
            "split": "train",
            "mixed_path": "mixed/r1.wav",
        }
        for k in ("clean_path", "noise_path", "reference_path", "mixed_path"):
            # Put ramp tone so slices at different offsets are strictly different
            path = data_root / row[k]
            path.parent.mkdir(parents=True, exist_ok=True)
            ramp = np.linspace(0, 1, long_len, dtype=np.float32)
            sf.write(str(path), ramp, SAMPLE_RATE, subtype="FLOAT")

        csv_path = tmp_path / "rand.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        ds = AncDataset(csv_path, split="train")

        torch.manual_seed(SEED)
        item_a = ds[0]

        torch.manual_seed(SEED)
        item_b = ds[0]

        assert torch.equal(item_a["noisy"], item_b["noisy"])
        assert torch.equal(item_a["clean"], item_b["clean"])
        assert torch.equal(item_a["reference"], item_b["reference"])

    def test_synchronized_crop_alignment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify clean, noisy, and reference signals share identical crop offsets when aligned."""
        data_root = tmp_path / "align_data"
        monkeypatch.setenv("DATA_ROOT", str(data_root))

        long_len = 80_000
        # Create unique ramp signal
        ramp = np.linspace(0.1, 0.9, long_len, dtype=np.float32)
        row = {
            "id": "align_01",
            "speaker_id": "spk1",
            "clean_path": "clean/a1.wav",
            "noise_path": "noise/a1.wav",
            "reference_path": "ref/a1.wav",
            "noise_type": "white",
            "snr_db": "0.0",
            "split": "train",
            "mixed_path": "mixed/a1.wav",
        }
        for k in ("clean_path", "reference_path", "mixed_path"):
            path = data_root / row[k]
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), ramp, SAMPLE_RATE, subtype="FLOAT")

        csv_path = tmp_path / "align.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        ds = AncDataset(csv_path, split="train")
        item = ds[0]
        # Because all three had identical initial ramp, synchronized crop means all three are equal
        assert torch.equal(item["noisy"], item["clean"])
        assert torch.equal(item["noisy"], item["reference"])

    def test_missing_manifest_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ValueError names the missing columns when required headers are absent."""
        data_root = tmp_path / "bad_csv_data"
        data_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("DATA_ROOT", str(data_root))

        csv_path = tmp_path / "bad_header.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "split", "clean_path"])
            writer.writeheader()
            writer.writerow({"id": "x", "split": "train", "clean_path": "c.wav"})

        with pytest.raises(ValueError, match="missing required columns"):
            AncDataset(csv_path, split="train")

    def test_audio_load_failure_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify a clear RuntimeError when a referenced audio file does not exist."""
        data_root = tmp_path / "missing_audio"
        data_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("DATA_ROOT", str(data_root))

        row = {
            "id": "missing_01",
            "speaker_id": "spk1",
            "clean_path": "clean/missing.wav",
            "noise_path": "noise/missing.wav",
            "reference_path": "ref/missing.wav",
            "noise_type": "white",
            "snr_db": "0.0",
            "split": "train",
            "mixed_path": "mixed/missing.wav",
        }
        csv_path = tmp_path / "missing.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        ds = AncDataset(csv_path, split="train")
        assert len(ds) == 1
        with pytest.raises(RuntimeError, match="Failed to load.*missing_01"):
            ds[0]

    def test_invalid_data_root_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ValueError when DATA_ROOT points to a non-existent directory."""
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "nonexistent_dir"))
        csv_path = tmp_path / "dummy.csv"
        csv_path.write_text("id,split,clean_path,mixed_path,reference_path\n")

        with pytest.raises(ValueError, match="not a directory"):
            AncDataset(csv_path, split="train")
