"""Project-wide audio and training constants."""

from typing import Final

SAMPLE_RATE: Final[int] = 16_000
N_FFT: Final[int] = 512
HOP_LENGTH: Final[int] = 256
WIN_LENGTH: Final[int] = 512
N_FREQ: Final[int] = N_FFT // 2 + 1  # 257
CLIP_SAMPLES: Final[int] = 64_000  # 4 s at 16 kHz
SEED: Final[int] = 42

# NLMS adaptive-filter defaults
NLMS_TAPS: Final[int] = 4
NLMS_MU: Final[float] = 0.5
NLMS_EPS: Final[float] = 1e-6

# SNR levels for mixing clean + noise
MIX_SNRS_DB: Final[tuple[int, ...]] = (-5, 0, 5, 10, 15)

# Frame / normalisation
FRAME_SAMPLES: Final[int] = 256
PEAK_NORM: Final[float] = 0.99
