"""Deep Complex Convolution Recurrent Network (DCCRN) model."""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from ai_engine.models.stft import STFT
from common.constants import N_FREQ

logger = logging.getLogger(__name__)


class CausalComplexConv2d(nn.Module):
    """Complex 2D convolution with causal padding on the time axis."""

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
        self.conv_r = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding_freq, 0),
        )
        self.conv_i = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding_freq, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex convolution: (Wr*Xr - Wi*Xi) + j(Wr*Xi + Wi*Xr)."""
        pad_t = self.kernel_t - 1
        if pad_t > 0:
            xr = nn.functional.pad(x.real, (pad_t, 0, 0, 0))
            xi = nn.functional.pad(x.imag, (pad_t, 0, 0, 0))
        else:
            xr, xi = x.real, x.imag
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return torch.complex(yr, yi)


class CausalComplexConvTranspose2d(nn.Module):
    """Complex 2D transposed convolution trimmed to preserve causal length on time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 1),
        padding_freq: int = 1,
    ) -> None:
        super().__init__()
        self.deconv_r = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding_freq, 0),
            output_padding=(0, 0),
        )
        self.deconv_i = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding_freq, 0),
            output_padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex transposed convolution and trim future temporal frames."""
        time_len = x.shape[-1]
        yr = self.deconv_r(x.real) - self.deconv_i(x.imag)
        yi = self.deconv_r(x.imag) + self.deconv_i(x.real)
        return torch.complex(yr[..., :time_len], yi[..., :time_len])


class ComplexBatchNorm2d(nn.Module):
    """Complex batch normalization applying separate normalizations to real and imag parts."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize real and imaginary components independently."""
        return torch.complex(self.bn_r(x.real), self.bn_i(x.imag))


class ComplexELU(nn.Module):
    """Complex ELU activation applied to real and imaginary components."""

    def __init__(self) -> None:
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ELU to real and imaginary components."""
        return torch.complex(self.elu(x.real), self.elu(x.imag))


class ComplexLSTM(nn.Module):
    """Complex LSTM combining two real LSTMs using complex multiplication."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.real_lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.imag_lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear_r = nn.Linear(hidden_size, input_size)
        self.linear_i = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute complex recurrent transitions: (Wr*Xr - Wi*Xi) + j(Wr*Xi + Wi*Xr)."""
        xr, xi = x.real, x.imag
        hrr, _ = self.real_lstm(xr)
        hii, _ = self.imag_lstm(xi)
        hri, _ = self.real_lstm(xi)
        hir, _ = self.imag_lstm(xr)
        hr = hrr - hii
        hi = hri + hir
        out_r = self.linear_r(hr) - self.linear_i(hi)
        out_i = self.linear_r(hi) + self.linear_i(hr)
        return torch.complex(out_r, out_i)


class DCCRN(nn.Module):
    """Deep Complex Convolution Recurrent Network for phase-aware speech enhancement."""

    def __init__(
        self,
        enc_channels: tuple[int, ...] = (16, 32, 64, 64, 64),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 1),
        padding_freq: int = 1,
    ) -> None:
        super().__init__()
        self.enc_channels = enc_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # Encoder: 5 causal complex Conv2d layers
        self.encoder = nn.ModuleList()
        in_c = 1
        curr_freq = N_FREQ
        for out_c in enc_channels:
            self.encoder.append(
                nn.Sequential(
                    CausalComplexConv2d(
                        in_c,
                        out_c,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding_freq=padding_freq,
                    ),
                    ComplexBatchNorm2d(out_c),
                    ComplexELU(),
                )
            )
            in_c = out_c
            curr_freq = (curr_freq - kernel_size[0] + 2 * padding_freq) // stride[0] + 1

        self.f_bottleneck = curr_freq
        self.lstm_in = enc_channels[-1] * self.f_bottleneck
        self.clstm = ComplexLSTM(
            self.lstm_in,
            lstm_hidden,
            num_layers=lstm_layers,
        )

        # Decoder: 5 causal complex ConvTranspose2d layers with skip connections
        self.decoder = nn.ModuleList()
        num_layers = len(enc_channels)
        for i in range(num_layers - 1, -1, -1):
            skip_c = enc_channels[i]
            out_c = enc_channels[i - 1] if i > 0 else 1
            layers: list[nn.Module] = [
                CausalComplexConvTranspose2d(
                    in_c + skip_c,
                    out_c,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding_freq=padding_freq,
                )
            ]
            if i > 0:
                layers.extend([ComplexBatchNorm2d(out_c), ComplexELU()])
            self.decoder.append(nn.Sequential(*layers))
            in_c = out_c

    def compute_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute polar complex ratio mask with bounded magnitude and phase shift."""
        x = spec.unsqueeze(1)
        skips: list[torch.Tensor] = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        batch_size, channels, freq_bins, time_frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_frames, channels * freq_bins)
        x = self.clstm(x)
        x = x.reshape(batch_size, time_frames, channels, freq_bins).permute(0, 2, 3, 1)

        for i, layer in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = layer(x)

        x = x.squeeze(1)
        # Polar complex ratio mask: magnitude bounded in [0, 1], phase in [-pi, pi]
        m_mag = torch.sigmoid(x.real)
        m_phase = math.pi * torch.tanh(x.imag)
        mask_r = m_mag * torch.cos(m_phase)
        mask_i = m_mag * torch.sin(m_phase)
        return torch.complex(mask_r, mask_i)

    def forward(
        self, spec: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply predicted complex ratio mask to noisy complex spectrum."""
        mask = self.compute_mask(spec)
        enhanced = spec * mask
        if return_mask:
            return enhanced, mask
        return enhanced

    def enhance_waveform(
        self, wave: torch.Tensor, stft: STFT | None = None
    ) -> torch.Tensor:
        """Enhance time-domain waveform using STFT -> DCCRN -> iSTFT."""
        stft_module = stft if stft is not None else STFT()
        orig_len = wave.shape[-1]
        spec = stft_module(wave)
        enh_spec = self.forward(spec)
        return stft_module.inverse(enh_spec, length=orig_len)

    def num_params(self) -> int:
        """Return total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())
