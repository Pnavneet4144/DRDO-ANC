"""Load, save, and length-fix utilities for WAV audio."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common.constants import SAMPLE_RATE

logger = logging.getLogger(__name__)


def load_wav(path: str | Path) -> np.ndarray:
    """Load a WAV file as mono float32 at SAMPLE_RATE, clipped to [-1, 1]."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Down-mix to mono
    data = data.mean(axis=1)
    # Resample if needed
    if sr != SAMPLE_RATE:
        logger.info("Resampling %s from %d Hz to %d Hz", path, sr, SAMPLE_RATE)
        gcd = np.gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
    data = np.clip(data, -1.0, 1.0)
    return data


def save_wav(path: str | Path, x: np.ndarray) -> None:
    """Save a float32 waveform as a 16 kHz WAV file."""
    sf.write(str(path), x, SAMPLE_RATE, subtype="FLOAT")


def fix_length(x: np.ndarray, n: int) -> np.ndarray:
    """Crop or zero-pad *x* to exactly *n* samples."""
    if x.shape[0] >= n:
        return x[:n]
    pad = np.zeros(n - x.shape[0], dtype=x.dtype)
    return np.concatenate([x, pad])
