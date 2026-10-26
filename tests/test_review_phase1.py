"""Phase 1 review tests — exhaustive verification of foundation utilities.

This file is TEST-ONLY.  No source files are modified.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from ai_engine.models.audio_io import fix_length, load_wav, save_wav
from ai_engine.models.stft import STFT
from ai_engine.training.losses import si_snr, si_snr_loss
from common.constants import (
    CLIP_SAMPLES,
    HOP_LENGTH,
    N_FFT,
    N_FREQ,
    NLMS_EPS,
    NLMS_MU,
    NLMS_TAPS,
    SAMPLE_RATE,
    SEED,
    WIN_LENGTH,
)
from common.interface import check_audio, check_enhanced

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — constants match the contract exactly
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify every constant equals the contract value."""

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 16000

    def test_n_fft(self) -> None:
        assert N_FFT == 512

    def test_hop_length(self) -> None:
        assert HOP_LENGTH == 256

    def test_win_length(self) -> None:
        assert WIN_LENGTH == 512

    def test_n_freq(self) -> None:
        assert N_FREQ == 257

    def test_clip_samples(self) -> None:
        assert CLIP_SAMPLES == 64000

    def test_seed(self) -> None:
        assert SEED == 42

    def test_nlms_taps(self) -> None:
        assert NLMS_TAPS == 4

    def test_nlms_mu(self) -> None:
        assert NLMS_MU == 0.5

    def test_nlms_eps(self) -> None:
        assert NLMS_EPS == 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — STFT shape, dtype, round-trip, and inverse length
# ═══════════════════════════════════════════════════════════════════════════════


class TestSTFTReview:
    """Thorough STFT forward/inverse checks."""

    @pytest.fixture()
    def stft(self) -> STFT:
        return STFT()

    # --- output shape and complex dtype ---
    def test_output_is_complex(self, stft: STFT) -> None:
        """STFT output must be complex dtype."""
        wave = torch.randn(2, 16000)
        spec = stft(wave)
        assert spec.is_complex(), f"Expected complex, got {spec.dtype}"

    def test_output_freq_dim(self, stft: STFT) -> None:
        """Frequency axis must be 257 (= N_FFT//2 + 1)."""
        spec = stft(torch.randn(1, 16000))
        assert spec.shape[1] == 257

    def test_output_batch_preserved(self, stft: STFT) -> None:
        """Batch dimension must pass through unchanged."""
        for B in (1, 4):
            spec = stft(torch.randn(B, 16000))
            assert spec.shape[0] == B, f"Batch {B} → {spec.shape[0]}"

    # --- round-trip for many lengths and batch sizes ---
    @pytest.mark.parametrize("length", [64000, 16000, 16001, 255, 257])
    @pytest.mark.parametrize("batch", [1, 4])
    def test_round_trip(self, stft: STFT, length: int, batch: int) -> None:
        """Forward → inverse max abs error must be < 1e-4."""
        rng = np.random.default_rng(SEED)
        x = rng.uniform(-1.0, 1.0, size=(batch, length)).astype(np.float32)
        wave = torch.from_numpy(x)
        spec = stft(wave)
        recon = stft.inverse(spec, length=length)
        err = (wave - recon).abs().max().item()
        assert err < 1e-4, f"Round-trip error {err:.2e} for len={length}, B={batch}"

    # --- inverse returns exactly T samples ---
    @pytest.mark.parametrize("T", [16000, 16001, 255, 257, 64000])
    def test_inverse_exact_length(self, stft: STFT, T: int) -> None:
        """inverse(spec, length=T) must return tensor with last dim == T."""
        wave = torch.randn(1, T)
        spec = stft(wave)
        recon = stft.inverse(spec, length=T)
        assert recon.shape[-1] == T, f"Expected {T}, got {recon.shape[-1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — SI-SNR: scale invariance, identity, zeros, gradient, loss decrease
# ═══════════════════════════════════════════════════════════════════════════════


class TestSISNRReview:
    """Exhaustive SI-SNR checks."""

    def test_scale_invariance(self) -> None:
        """SI-SNR must be invariant to scaling of the estimate."""
        torch.manual_seed(SEED)
        ref = torch.randn(2, 16000)
        est = ref + 0.01 * torch.randn_like(ref)
        v1 = si_snr(est, ref)
        v2 = si_snr(est * 5.0, ref)
        assert torch.allclose(v1, v2, atol=0.5)

    def test_identity_very_high(self) -> None:
        """SI-SNR(x, x) must be very high (> 30 dB)."""
        torch.manual_seed(SEED)
        x = torch.randn(1, 16000)
        val = si_snr(x, x)
        assert val.item() > 30.0, f"SI-SNR(x,x) = {val.item():.1f}"

    def test_zero_estimate_finite(self) -> None:
        """All-zero estimate must give a finite SI-SNR (no NaN/inf)."""
        ref = torch.randn(1, 16000)
        est = torch.zeros(1, 16000)
        val = si_snr(est, ref)
        assert torch.isfinite(val).all(), f"Got {val}"

    def test_zero_reference_finite(self) -> None:
        """All-zero reference must give a finite SI-SNR (no NaN/inf)."""
        ref = torch.zeros(1, 16000)
        est = torch.randn(1, 16000)
        val = si_snr(est, ref)
        assert torch.isfinite(val).all(), f"Got {val}"

    def test_loss_decreases_with_less_noise(self) -> None:
        """Loss must decrease (improve) when noise is reduced."""
        torch.manual_seed(SEED)
        ref = torch.randn(2, 16000)
        loud = ref + 1.0 * torch.randn_like(ref)
        quiet = ref + 0.01 * torch.randn_like(ref)
        loss_loud = si_snr_loss(loud, ref)
        loss_quiet = si_snr_loss(quiet, ref)
        # Less noise → higher SI-SNR → more negative loss → loss_quiet < loss_loud
        assert loss_quiet < loss_loud, (
            f"loss_quiet={loss_quiet.item():.2f} >= loss_loud={loss_loud.item():.2f}"
        )

    def test_gradient_finite(self) -> None:
        """backward() must produce finite gradients."""
        torch.manual_seed(SEED)
        ref = torch.randn(2, 16000)
        est = torch.randn(2, 16000, requires_grad=True)
        loss = si_snr_loss(est, ref)
        loss.backward()
        assert est.grad is not None, "No gradient computed"
        assert torch.isfinite(est.grad).all(), "Gradient contains NaN/inf"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — audio_io: stereo 44.1 kHz, fix_length, save→load round trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestAudioIOReview:
    """Comprehensive audio_io checks."""

    def test_load_stereo_44100_returns_mono_f32_16k(self) -> None:
        """load_wav on a stereo 44.1 kHz file → mono float32 at 16 kHz in [-1,1]."""
        rng = np.random.default_rng(SEED)
        sr_orig = 44100
        dur = 1.0
        n = int(sr_orig * dur)
        stereo = rng.uniform(-0.8, 0.8, size=(n, 2)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, stereo, sr_orig, subtype="FLOAT")
            out = load_wav(f.name)
        assert out.ndim == 1, f"Expected mono (1-D), got ndim={out.ndim}"
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
        expected_len = int(SAMPLE_RATE * dur)
        assert abs(out.shape[0] - expected_len) <= 1
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_fix_length_shorter(self) -> None:
        """Padding shorter input to target length."""
        x = np.ones(50, dtype=np.float32)
        y = fix_length(x, 100)
        assert y.shape == (100,)
        assert np.all(y[:50] == 1.0) and np.all(y[50:] == 0.0)

    def test_fix_length_equal(self) -> None:
        """Equal-length input passes through unchanged."""
        x = np.ones(100, dtype=np.float32)
        y = fix_length(x, 100)
        assert y.shape == (100,)
        assert np.all(y == 1.0)

    def test_fix_length_longer(self) -> None:
        """Cropping longer input to target length."""
        x = np.arange(200, dtype=np.float32)
        y = fix_length(x, 100)
        assert y.shape == (100,)
        assert np.array_equal(y, x[:100])

    def test_save_load_round_trip(self) -> None:
        """save_wav then load_wav must recover the waveform closely."""
        rng = np.random.default_rng(SEED)
        x = rng.uniform(-0.9, 0.9, size=16000).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_wav(f.name, x)
            y = load_wav(f.name)
        assert y.dtype == np.float32
        assert y.shape == x.shape
        assert np.allclose(x, y, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — check_audio / check_enhanced validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Verify that validation helpers reject bad inputs."""

    # ---- check_audio ----
    def test_check_audio_wrong_dtype(self) -> None:
        with pytest.raises(TypeError):
            check_audio(np.zeros(100, dtype=np.float64))

    def test_check_audio_wrong_ndim(self) -> None:
        with pytest.raises(ValueError):
            check_audio(np.zeros((2, 100), dtype=np.float32))

    def test_check_audio_nan(self) -> None:
        x = np.zeros(100, dtype=np.float32)
        x[50] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            check_audio(x)

    def test_check_audio_out_of_range(self) -> None:
        x = np.zeros(100, dtype=np.float32)
        x[0] = 1.5
        with pytest.raises(ValueError, match="outside"):
            check_audio(x)

    def test_check_audio_valid(self) -> None:
        """Valid input must not raise."""
        check_audio(np.zeros(100, dtype=np.float32))

    # ---- check_enhanced ----
    def test_check_enhanced_wrong_dtype(self) -> None:
        noisy = np.zeros(100, dtype=np.float32)
        enhanced = np.zeros(100, dtype=np.float64)
        with pytest.raises(TypeError):
            check_enhanced(noisy, enhanced)

    def test_check_enhanced_wrong_length(self) -> None:
        noisy = np.zeros(100, dtype=np.float32)
        enhanced = np.zeros(50, dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            check_enhanced(noisy, enhanced)

    def test_check_enhanced_nan(self) -> None:
        noisy = np.zeros(100, dtype=np.float32)
        enhanced = np.zeros(100, dtype=np.float32)
        enhanced[10] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            check_enhanced(noisy, enhanced)

    def test_check_enhanced_valid(self) -> None:
        """Valid matching pair must not raise."""
        noisy = np.zeros(100, dtype=np.float32)
        enhanced = np.zeros(100, dtype=np.float32)
        check_enhanced(noisy, enhanced)
