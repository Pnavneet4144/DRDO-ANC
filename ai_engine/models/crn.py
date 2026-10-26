"""Causal Convolutional Recurrent Network (CRN) magnitude mask model."""

from __future__ import annotations

import logging

import torch
from torch import nn

from ai_engine.models.stft import STFT
from common.constants import N_FREQ

logger = logging.getLogger(__name__)


class CausalConv2d(nn.Module):
    """2D convolution with causal padding on the time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 1),
        padding_freq: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_t = kernel_size[1]
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding_freq, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad past time frames and convolve."""
        if self.kernel_t > 1:
            x = nn.functional.pad(x, (self.kernel_t - 1, 0, 0, 0))
        return self.conv(x)


class CausalConvTranspose2d(nn.Module):
    """2D transposed convolution trimmed to preserve causal length on time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 1),
        padding_freq: int = 1,
    ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding_freq, 0),
            output_padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose-convolve and trim time axis to input time length."""
        time_len = x.shape[-1]
        out = self.deconv(x)
        return out[..., :time_len]


class CRN(nn.Module):
    """Causal Convolutional Recurrent Network for magnitude-mask speech enhancement."""

    def __init__(
        self,
        enc_channels: tuple[int, ...] = (16, 32, 64, 64, 64),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 1),
        padding_freq: int = 1,
    ) -> None:
        super().__init__()
        self.enc_channels = enc_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Encoder: 5 causal Conv2d layers
        self.encoder = nn.ModuleList()
        in_c = 1
        curr_freq = N_FREQ
        for out_c in enc_channels:
            self.encoder.append(
                nn.Sequential(
                    CausalConv2d(
                        in_c,
                        out_c,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding_freq=padding_freq,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.ELU(),
                )
            )
            in_c = out_c
            curr_freq = (curr_freq - kernel_size[0] + 2 * padding_freq) // stride[0] + 1

        self.f_bottleneck = curr_freq
        self.lstm_in = enc_channels[-1] * self.f_bottleneck
        self.lstm = nn.LSTM(
            self.lstm_in,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear = nn.Linear(lstm_hidden, self.lstm_in)

        # Decoder: 5 causal ConvTranspose2d layers with skip connections
        self.decoder = nn.ModuleList()
        num_layers = len(enc_channels)
        for i in range(num_layers - 1, -1, -1):
            skip_c = enc_channels[i]
            out_c = enc_channels[i - 1] if i > 0 else 1
            layers: list[nn.Module] = [
                CausalConvTranspose2d(
                    in_c + skip_c,
                    out_c,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding_freq=padding_freq,
                )
            ]
            if i > 0:
                layers.extend([nn.BatchNorm2d(out_c), nn.ELU()])
            else:
                layers.append(nn.Sigmoid())
            self.decoder.append(nn.Sequential(*layers))
            in_c = out_c

    def compute_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute estimated real magnitude mask in [0, 1] from complex spectrum."""
        mag = torch.abs(spec).unsqueeze(1)
        skips: list[torch.Tensor] = []
        x = mag
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        batch_size, channels, freq_bins, time_frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_frames, channels * freq_bins)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x.reshape(batch_size, time_frames, channels, freq_bins).permute(0, 2, 3, 1)

        for i, layer in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        return x.squeeze(1)

    def forward(
        self, spec: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply predicted magnitude mask to noisy complex spectrum."""
        mask = self.compute_mask(spec)
        enhanced = spec * mask
        if return_mask:
            return enhanced, mask
        return enhanced

    def enhance_waveform(
        self, wave: torch.Tensor, stft: STFT | None = None
    ) -> torch.Tensor:
        """Enhance time-domain waveform using STFT -> CRN -> iSTFT."""
        stft_module = stft if stft is not None else STFT()
        orig_len = wave.shape[-1]
        spec = stft_module(wave)
        enh_spec = self.forward(spec)
        return stft_module.inverse(enh_spec, length=orig_len)

    def num_params(self) -> int:
        """Return total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())
