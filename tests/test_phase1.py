"""Unit tests for Phase 1 foundation utilities."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from ai_engine.models.audio_io import fix_length, load_wav
from ai_engine.models.stft import STFT
from ai_engine.training.losses import si_snr, si_snr_loss
from common.constants import HOP_LENGTH, N_FREQ, SAMPLE_RATE, SEED

# ---------------------------------------------------------------------------
# STFT round-trip tests
# ---------------------------------------------------------------------------

class TestSTFT:
    """STFT forward → inverse round-trip and shape checks."""

    @pytest.fixture()
    def stft(self) -> STFT:
        """Create a fresh STFT module."""
        return STFT()

    @pytest.mark.parametrize("length", [16000, 15999, 8000, 32001])
    def test_round_trip(self, stft: STFT, length: int) -> None:
        """Round-trip error must be < 1e-4 for various lengths."""
        rng = np.random.default_rng(SEED)
        x = rng.uniform(-1.0, 1.0, size=(2, length)).astype(np.float32)
        wave = torch.from_numpy(x)
        spec = stft(wave)
        recon = stft.inverse(spec, length=length)
        err = (wave - recon).abs().max().item()
        assert err < 1e-4, f"Round-trip error {err:.6e} >= 1e-4 for length={length}"

    def test_output_shape(self, stft: STFT) -> None:
        """STFT output must be [B, 257, F]."""
        B, T = 3, 16000
        wave = torch.randn(B, T)
        spec = stft(wave)
        assert spec.shape[0] == B
        assert spec.shape[1] == N_FREQ  # 257
        expected_frames = T // HOP_LENGTH + 1
        assert spec.shape[2] == expected_frames

    @pytest.mark.parametrize("length", [1, 100, 255, 256, 257, 258])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_boundary_and_short_round_trip(
        self, stft: STFT, length: int, batch_size: int
    ) -> None:
        """Round-trip error < 1e-4 and shape preservation for boundary and short lengths."""
        rng = np.random.default_rng(SEED)
        x = rng.uniform(-1.0, 1.0, size=(batch_size, length)).astype(np.float32)
        wave = torch.from_numpy(x)
        spec = stft(wave)
        assert spec.shape[0] == batch_size
        assert spec.shape[1] == N_FREQ
        recon = stft.inverse(spec, length=length)
        assert recon.shape == (batch_size, length)
        err = (wave - recon).abs().max().item()
        assert err < 1e-4, (
            f"Round-trip error {err:.6e} >= 1e-4 for length={length}, batch={batch_size}"
        )


# ---------------------------------------------------------------------------
# SI-SNR tests
# ---------------------------------------------------------------------------

class TestSISNR:
    """Scale-invariant SNR metric tests."""

    def test_scale_invariance(self) -> None:
        """SI-SNR must be invariant to scaling of the estimate."""
        torch.manual_seed(SEED)
        ref = torch.randn(2, 16000)
        est = ref + 0.01 * torch.randn_like(ref)
        val1 = si_snr(est, ref)
        val2 = si_snr(est * 3.0, ref)
        assert torch.allclose(val1, val2, atol=0.5), "SI-SNR not scale-invariant"

    def test_identity(self) -> None:
        """SI-SNR(x, x) should be very high (> 30 dB)."""
        torch.manual_seed(SEED)
        x = torch.randn(1, 16000)
        val = si_snr(x, x)
        assert val.item() > 30.0, f"SI-SNR(x,x) = {val.item():.1f} dB, expected > 30"

    def test_loss_sign(self) -> None:
        """si_snr_loss should be negative (since SI-SNR is positive for good estimates)."""
        torch.manual_seed(SEED)
        x = torch.randn(2, 16000)
        loss = si_snr_loss(x, x)
        assert loss.item() < 0.0, "si_snr_loss should be negative for identical signals"


# ---------------------------------------------------------------------------
# fix_length tests
# ---------------------------------------------------------------------------

class TestFixLength:
    """Crop and pad tests for fix_length."""

    def test_crop(self) -> None:
        """fix_length must crop longer signals."""
        x = np.ones(100, dtype=np.float32)
        y = fix_length(x, 50)
        assert y.shape == (50,)
        assert np.all(y == 1.0)

    def test_pad(self) -> None:
        """fix_length must zero-pad shorter signals."""
        x = np.ones(50, dtype=np.float32)
        y = fix_length(x, 100)
        assert y.shape == (100,)
        assert np.all(y[:50] == 1.0)
        assert np.all(y[50:] == 0.0)


# ---------------------------------------------------------------------------
# load_wav tests
# ---------------------------------------------------------------------------

class TestLoadWav:
    """Tests for load_wav resampling and dtype."""

    def test_resample_8k_to_16k(self) -> None:
        """Loading an 8 kHz file must produce a 16 kHz float32 mono array."""
        rng = np.random.default_rng(SEED)
        orig_sr = 8000
        duration_s = 1.0
        samples = int(orig_sr * duration_s)
        audio = rng.uniform(-0.5, 0.5, size=samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, orig_sr, subtype="FLOAT")
            loaded = load_wav(f.name)

        expected_len = int(SAMPLE_RATE * duration_s)
        assert loaded.dtype == np.float32, f"Expected float32, got {loaded.dtype}"
        assert loaded.ndim == 1, f"Expected 1-D, got ndim={loaded.ndim}"
        assert abs(loaded.shape[0] - expected_len) <= 1, (
            f"Expected ~{expected_len} samples, got {loaded.shape[0]}"
        )
