"""Training loop and CLI for DRDO-ANC speech enhancement models."""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ai_engine.models.crn import CRN
from ai_engine.models.stft import STFT
from ai_engine.training.dataset import AncDataset
from ai_engine.training.losses import si_snr_loss
from common.constants import SEED

try:
    from ai_engine.models.dccrn import DCCRN

    HAS_DCCRN = True
except ImportError:
    HAS_DCCRN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_git_hash() -> str:
    """Return the current short Git commit hash, or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("Could not retrieve git hash: %s", exc)
    return "unknown"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible execution."""
    logger.info("Setting random seed to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str | None = None) -> torch.device:
    """Select compute device (CPU or CUDA)."""
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_config(config: dict[str, Any]) -> None:
    """Validate that required configuration fields are present."""
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")

    if not config.get("name"):
        raise ValueError("Configuration must contain a valid 'name' field")

    params = config.get("params")
    if not isinstance(params, dict):
        raise TypeError("Configuration must contain a 'params' dictionary")

    required_params = ("model", "lr", "batch_size", "epochs")
    for key in required_params:
        if key not in params or params[key] is None:
            raise ValueError(f"Missing required parameter 'params.{key}' in configuration")

    model_name = str(params["model"]).lower()
    if model_name not in ("crn", "dccrn"):
        raise ValueError(
            f"Unsupported model: '{model_name}'. Only 'crn' and 'dccrn' are supported."
        )

    if model_name == "dccrn" and not HAS_DCCRN:
        raise RuntimeError(
            "DCCRN model requested but ai_engine.models.dccrn is not available on this branch."
        )


def resolve_manifest_path(config: dict[str, Any]) -> Path:
    """Resolve the manifest CSV path from config or environment."""
    manifest_opt = config.get("manifest_csv") or config.get("params", {}).get("manifest_csv")
    if manifest_opt:
        p = Path(manifest_opt)
        if not p.is_file():
            raise FileNotFoundError(f"Configured manifest CSV does not exist: {p}")
        return p

    data_root = os.environ.get("DATA_ROOT")
    candidates: list[Path] = []
    if data_root:
        candidates.append(Path(data_root) / "master_training_split.csv")
        candidates.append(Path(data_root) / "master_training.csv")
    candidates.append(Path("ai_engine/data/master_training_split.csv"))
    candidates.append(Path("ai_engine/data/master_training.csv"))

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "Manifest CSV not specified in config and default manifest files not found."
    )


def resolve_ckpt_dir(config: dict[str, Any]) -> Path:
    """Resolve and create checkpoint directory outside Git."""
    ckpt_dir_val = (
        os.environ.get("CKPT_DIR")
        or config.get("ckpt_dir")
        or config.get("params", {}).get("ckpt_dir")
    )
    if not ckpt_dir_val:
        raise ValueError(
            "CKPT_DIR environment variable is not set and ckpt_dir is not specified in config."
        )

    ckpt_dir = Path(ckpt_dir_val)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def train(config_path: str | Path, device_override: str | None = None) -> dict[str, Any]:
    """Train CRN or DCCRN model according to YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    config_name = config["name"]
    params = config["params"]
    seed = int(config.get("seed", SEED))
    seed_everything(seed)

    device = select_device(device_override or params.get("device"))
    logger.info("Using device: %s", device)

    manifest_path = resolve_manifest_path(config)
    ckpt_dir = resolve_ckpt_dir(config)
    git_hash = get_git_hash()

    batch_size = int(params["batch_size"])
    epochs = int(params["epochs"])
    lr = float(params["lr"])
    overfit_n = params.get("overfit_n")
    if overfit_n is not None:
        overfit_n = int(overfit_n)

    # Initialize datasets
    train_dataset = AncDataset(manifest_path, split="train")
    if overfit_n is not None and overfit_n > 0:
        actual_n = min(overfit_n, len(train_dataset))
        logger.info("Overfit mode: restricting train set to %d examples", actual_n)
        train_dataset = Subset(train_dataset, range(actual_n))  # type: ignore[assignment]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    try:
        val_dataset = AncDataset(manifest_path, split="validation")
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if len(val_dataset) > 0
            else None
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.warning("Could not initialize validation dataset: %s", exc)
        val_loader = None

    # Instantiate model
    model_name = str(params["model"]).lower()
    if model_name == "crn":
        model: torch.nn.Module = CRN().to(device)
    elif model_name == "dccrn":
        model = DCCRN().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    stft = STFT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_loss = float("inf")
    best_ckpt_path = ckpt_dir / f"{config_name}_{git_hash}_best.pt"

    logger.info(
        "Starting training: model=%s, epochs=%d, lr=%.6f, batch_size=%d",
        model_name,
        epochs,
        lr,
        batch_size,
    )

    training_start = time.monotonic()
    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            t_len = noisy.shape[-1]

            optimizer.zero_grad()
            spec = stft(noisy)
            enh_spec = model(spec)
            recon = stft.inverse(enh_spec, length=t_len)
            loss = si_snr_loss(recon, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        train_losses.append(epoch_train_loss)
        epoch_elapsed = time.monotonic() - epoch_start
        logger.info(
            "Epoch %d/%d - Train SI-SNR Loss: %.4f (%.1fs)",
            epoch, epochs, epoch_train_loss, epoch_elapsed,
        )

        # Validation step
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_batch_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    noisy = batch["noisy"].to(device)
                    clean = batch["clean"].to(device)
                    t_len = noisy.shape[-1]
                    spec = stft(noisy)
                    enh_spec = model(spec)
                    recon = stft.inverse(enh_spec, length=t_len)
                    v_loss = si_snr_loss(recon, clean)
                    val_batch_losses.append(v_loss.item())
            epoch_val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else 0.0
            val_losses.append(epoch_val_loss)
            logger.info("Epoch %d/%d - Val SI-SNR Loss: %.4f", epoch, epochs, epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_loss,
                        "config": config,
                        "git_hash": git_hash,
                    },
                    best_ckpt_path,
                )
                logger.info("Saved best validation checkpoint to %s", best_ckpt_path)
        else:
            # When validation set is absent (e.g., overfit test)
            if epoch_train_loss < best_loss:
                best_loss = epoch_train_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": best_loss,
                        "config": config,
                        "git_hash": git_hash,
                    },
                    best_ckpt_path,
                )
                logger.info("Saved best checkpoint to %s", best_ckpt_path)

    # Save final-epoch checkpoint for resumability
    latest_ckpt_path = ckpt_dir / f"{config_name}_{git_hash}_latest.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1] if train_losses else 0.0,
            "val_loss": val_losses[-1] if val_losses else None,
            "config": config,
            "git_hash": git_hash,
        },
        latest_ckpt_path,
    )

    total_time = time.monotonic() - training_start
    logger.info("Training finished in %.1fs. Latest checkpoint: %s", total_time, latest_ckpt_path)

    return {
        "best_loss": best_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "checkpoint_path": str(best_ckpt_path),
        "latest_checkpoint_path": str(latest_ckpt_path),
        "git_hash": git_hash,
    }


def main() -> None:
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(
        description="Train DRDO-ANC speech enhancement model (CRN/DCCRN)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration file",
    )
    args = parser.parse_args()

    try:
        results = train(args.config)
        logger.info("Training completed successfully. Best loss: %.4f", results["best_loss"])
    except (ValueError, TypeError, FileNotFoundError, RuntimeError, OSError) as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
