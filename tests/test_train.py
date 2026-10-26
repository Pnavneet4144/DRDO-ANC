"""Tests for training loop, CLI, and model overfit runs."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from ai_engine.training.train import HAS_DCCRN, select_device, train, validate_config
from common.constants import CLIP_SAMPLES, SAMPLE_RATE


def _create_synthetic_wav(path: Path, num_samples: int = CLIP_SAMPLES) -> None:
    """Write a synthetic float32 sine tone as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, num_samples / SAMPLE_RATE, num_samples, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), tone, SAMPLE_RATE, subtype="FLOAT")


@pytest.fixture()
def synthetic_training_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Create 8 synthetic training WAVs, manifest CSV, and CKPT_DIR."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("CKPT_DIR", str(ckpt_dir))

    rows = []
    for i in range(8):
        c_rel = f"clean/ex_{i:02d}.wav"
        n_rel = f"noise/ex_{i:02d}.wav"
        r_rel = f"ref/ex_{i:02d}.wav"
        m_rel = f"mixed/ex_{i:02d}.wav"

        _create_synthetic_wav(data_root / c_rel)
        _create_synthetic_wav(data_root / n_rel)
        _create_synthetic_wav(data_root / r_rel)
        _create_synthetic_wav(data_root / m_rel)

        rows.append(
            {
                "id": f"ex_{i:02d}",
                "speaker_id": "spk01",
                "clean_path": c_rel,
                "noise_path": n_rel,
                "reference_path": r_rel,
                "noise_type": "white",
                "snr_db": "5.0",
                "split": "train",
                "mixed_path": m_rel,
            }
        )

    manifest_csv = tmp_path / "train_manifest.csv"
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
        writer.writerows(rows)

    return manifest_csv, ckpt_dir


class TestTraining:
    """Test suite for training pipeline, validation, and overfit checks."""

    def test_select_device(self) -> None:
        """Verify device selection logic for CPU and explicit override."""
        dev = select_device()
        assert isinstance(dev, torch.device)

        dev_cpu = select_device("cpu")
        assert dev_cpu.type == "cpu"

    def test_validate_config_errors(self) -> None:
        """Verify config validation raises informative errors on missing required keys."""
        with pytest.raises(TypeError, match="Configuration must be a dictionary"):
            validate_config("not a dict")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Configuration must contain a valid 'name'"):
            validate_config({})

        with pytest.raises(TypeError, match="Configuration must contain a 'params' dictionary"):
            validate_config({"name": "test"})

        incomplete_params: dict[str, object] = {
            "name": "test",
            "params": {"model": "crn", "lr": 0.001, "batch_size": 4},
        }
        with pytest.raises(ValueError, match="Missing required parameter 'params.epochs'"):
            validate_config(incomplete_params)  # type: ignore[arg-type]

        unsupported_model: dict[str, object] = {
            "name": "test",
            "params": {"model": "invalid_model", "lr": 0.001, "batch_size": 4, "epochs": 1},
        }
        with pytest.raises(ValueError, match="Unsupported model"):
            validate_config(unsupported_model)  # type: ignore[arg-type]

    def test_missing_config_file(self) -> None:
        """Verify FileNotFoundError is raised when config path does not exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            train("nonexistent_config.yaml")

    def test_missing_ckpt_dir(
        self, synthetic_training_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ValueError is raised when CKPT_DIR is unset and missing from config."""
        manifest_csv, _ = synthetic_training_env
        monkeypatch.delenv("CKPT_DIR", raising=False)

        cfg_path = manifest_csv.parent / "no_ckpt.yaml"
        cfg_path.write_text(
            f"""
name: test_no_ckpt
method: crn
seed: 42
dataset_version: v1
manifest_csv: {manifest_csv}
params:
  model: crn
  lr: 0.001
  batch_size: 4
  epochs: 1
  overfit_n: 8
"""
        )

        with pytest.raises(ValueError, match="CKPT_DIR environment variable is not set"):
            train(cfg_path)

    def test_crn_overfit_8_examples_loss_decreases(
        self, synthetic_training_env: tuple[Path, Path]
    ) -> None:
        """Verify CRN overfit on 8 synthetic examples decreases loss and saves checkpoint."""
        manifest_csv, ckpt_dir = synthetic_training_env
        cfg_path = manifest_csv.parent / "crn_overfit.yaml"
        cfg_path.write_text(
            f"""
name: crn_overfit
method: crn
seed: 42
dataset_version: v1
manifest_csv: {manifest_csv}
params:
  model: crn
  lr: 0.005
  batch_size: 4
  epochs: 3
  overfit_n: 8
"""
        )

        res = train(cfg_path, device_override="cpu")

        # Overfit requirement: final loss is lower than initial loss
        assert len(res["train_losses"]) == 3
        initial_loss = res["train_losses"][0]
        final_loss = res["train_losses"][-1]
        assert final_loss < initial_loss, (
            f"CRN loss did not decrease: initial={initial_loss}, final={final_loss}"
        )

        # Checkpoint verification
        ckpt_file = Path(res["checkpoint_path"])
        assert ckpt_file.is_file()
        assert ckpt_file.parent == ckpt_dir
        assert "crn_overfit" in ckpt_file.name

        saved = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        assert "model_state_dict" in saved
        assert "optimizer_state_dict" in saved
        assert "epoch" in saved
        assert "git_hash" in saved

    def test_dccrn_overfit_8_examples_loss_decreases(
        self, synthetic_training_env: tuple[Path, Path]
    ) -> None:
        """Verify DCCRN overfit on 8 synthetic examples decreases loss and saves checkpoint."""
        if not HAS_DCCRN:
            pytest.skip("DCCRN is not available on this branch")

        manifest_csv, ckpt_dir = synthetic_training_env
        cfg_path = manifest_csv.parent / "dccrn_overfit.yaml"
        cfg_path.write_text(
            f"""
name: dccrn_overfit
method: dccrn
seed: 42
dataset_version: v1
manifest_csv: {manifest_csv}
params:
  model: dccrn
  lr: 0.005
  batch_size: 4
  epochs: 3
  overfit_n: 8
"""
        )

        res = train(cfg_path, device_override="cpu")

        assert len(res["train_losses"]) == 3
        initial_loss = res["train_losses"][0]
        final_loss = res["train_losses"][-1]
        assert final_loss < initial_loss, (
            f"DCCRN loss did not decrease: initial={initial_loss}, final={final_loss}"
        )

        ckpt_file = Path(res["checkpoint_path"])
        assert ckpt_file.is_file()
        assert ckpt_file.parent == ckpt_dir
        assert "dccrn_overfit" in ckpt_file.name

    def test_cli_execution(self, synthetic_training_env: tuple[Path, Path]) -> None:
        """Verify python -m ai_engine.training.train --config ... executes successfully."""
        manifest_csv, _ = synthetic_training_env
        cfg_path = manifest_csv.parent / "cli_test.yaml"
        cfg_path.write_text(
            f"""
name: cli_test
method: crn
seed: 42
dataset_version: prototype
manifest_csv: {manifest_csv}
params:
  model: crn
  lr: 0.005
  batch_size: 4
  epochs: 1
  overfit_n: 8
"""
        )

        cmd = [
            sys.executable,
            "-m",
            "ai_engine.training.train",
            "--config",
            str(cfg_path),
        ]
        # Explicitly propagate required env vars so subprocess sees them
        env = {
            **os.environ,
            "DATA_ROOT": os.environ.get("DATA_ROOT", ""),
            "CKPT_DIR": os.environ.get("CKPT_DIR", ""),
        }
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        assert proc.returncode == 0, f"CLI command failed with stderr: {proc.stderr}"
        assert "Training completed successfully" in proc.stderr or "Training completed successfully" in proc.stdout

    def test_reproducible_training_with_seed(
        self, synthetic_training_env: tuple[Path, Path]
    ) -> None:
        """Verify two training runs with the same seed produce identical loss sequences."""
        manifest_csv, _ = synthetic_training_env
        cfg_path = manifest_csv.parent / "repro_test.yaml"
        cfg_path.write_text(
            f"""
name: repro_test
method: crn
seed: 42
dataset_version: prototype
manifest_csv: {manifest_csv}
params:
  model: crn
  lr: 0.005
  batch_size: 4
  epochs: 2
  overfit_n: 8
"""
        )

        res_a = train(cfg_path, device_override="cpu")
        res_b = train(cfg_path, device_override="cpu")

        assert res_a["train_losses"] == res_b["train_losses"], (
            f"Losses differ between seeded runs: {res_a['train_losses']} vs {res_b['train_losses']}"
        )
