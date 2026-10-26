"""Torch STFT / iSTFT wrapper registered as an nn.Module."""

from __future__ import annotations

import torch
from torch import nn

from common.constants import HOP_LENGTH, N_FFT, WIN_LENGTH


class STFT(nn.Module):
    """Windowed STFT and its inverse, with a Hann window stored as a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Compute STFT: [B, T] -> complex [B, N_FREQ, F]."""
        min_len = N_FFT // 2 + 1
        if wave.shape[-1] < min_len:
            # Zero-pad short input on the right so torch.stft reflection padding succeeds
            wave = torch.nn.functional.pad(wave, (0, min_len - wave.shape[-1]))
        return torch.stft(
            wave,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Compute iSTFT: complex [B, N_FREQ, F] -> [B, length]."""
        return torch.istft(
            spec,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            length=length,
        )
