"""Review test suite for CRN baseline model (Phase 2 review).

Test-only file under tests/. Does not modify any source code.
"""

from __future__ import annotations

import logging
import statistics
import time

import pytest
import torch

from ai_engine.models.crn import CRN
from ai_engine.models.stft import STFT
from ai_engine.training.losses import si_snr_loss
from common.constants import N_FREQ, SEED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check 1: Forward shapes and complex dtype
# ---------------------------------------------------------------------------


class TestForwardShapes:
    """Verify forward pass on various batch sizes and frame lengths."""

    def test_random_complex_batch2(self) -> None:
        """Forward on [2, 257, 251] returns complex [2, 257, 251]."""
        torch.manual_seed(SEED)
        model = CRN().eval()
        spec = torch.randn(2, N_FREQ, 251, dtype=torch.complex64)
        out = model(spec)
        assert out.shape == (2, N_FREQ, 251)
        assert out.dtype == torch.complex64

    @pytest.mark.parametrize("f_frames", [1, 10, 400])
    def test_batch1_variable_frames(self, f_frames: int) -> None:
        """Forward on batch 1 with F in (1, 10, 400) returns correct shape and complex dtype."""
        torch.manual_seed(SEED)
        model = CRN().eval()
        spec = torch.randn(1, N_FREQ, f_frames, dtype=torch.complex64)
        out = model(spec)
        assert out.shape == (1, N_FREQ, f_frames)
        assert out.dtype == torch.complex64


# ---------------------------------------------------------------------------
# Check 2: Mask properties (bounds, magnitude suppression, phase preservation)
# ---------------------------------------------------------------------------


class TestMaskProperties:
    """Verify mask values in [0, 1], magnitude suppression, and phase preservation."""

    def test_mask_and_phase(self) -> None:
        """Mask in [0, 1], out_mag <= in_mag, and out_phase == in_phase."""
        torch.manual_seed(SEED)
        model = CRN().eval()
        spec = torch.randn(2, N_FREQ, 251, dtype=torch.complex64)

        mask = model.compute_mask(spec)
        assert mask.shape == (2, N_FREQ, 251)
        assert mask.min().item() >= 0.0, f"Mask minimum {mask.min().item()} < 0.0"
        assert mask.max().item() <= 1.0, f"Mask maximum {mask.max().item()} > 1.0"

        out = model(spec)
        in_mag = torch.abs(spec)
        out_mag = torch.abs(out)
        mag_diff = (out_mag - in_mag).max().item()
        assert mag_diff <= 1e-6, f"Output magnitude exceeded input magnitude by {mag_diff}"

        in_phase = torch.angle(spec)
        out_phase = torch.angle(out)
        phase_diff = (out_phase - in_phase).abs().max().item()
        assert phase_diff < 1e-5, f"Output phase differed from input phase by {phase_diff}"


# ---------------------------------------------------------------------------
# Check 3: Causality in eval mode and train mode
# ---------------------------------------------------------------------------


class TestCausality:
    """Verify output independence from future frames."""

    def test_causality_eval_mode(self) -> None:
        """In eval() mode, perturbing t >= 100 leaves t < 100 identical (atol=1e-6)."""
        torch.manual_seed(SEED)
        model = CRN().eval()
        spec1 = torch.randn(1, N_FREQ, 251, dtype=torch.complex64)
        spec2 = spec1.clone()
        spec2[:, :, 100:] = torch.randn_like(spec2[:, :, 100:])

        with torch.no_grad():
            out1 = model(spec1)
            out2 = model(spec2)

        earlier_out1 = out1[:, :, :100]
        earlier_out2 = out2[:, :, :100]
        assert torch.allclose(earlier_out1, earlier_out2, atol=1e-6), (
            f"Eval mode: max diff {(earlier_out1 - earlier_out2).abs().max().item()} >= 1e-6"
        )

    def test_causality_train_mode(self) -> None:
        """In train() mode, document whether BatchNorm2d leaks across time."""
        torch.manual_seed(SEED)
        model = CRN().train()
        spec1 = torch.randn(1, N_FREQ, 251, dtype=torch.complex64)
        spec2 = spec1.clone()
        spec2[:, :, 100:] = torch.randn_like(spec2[:, :, 100:])

        out1 = model(spec1)
        out2 = model(spec2)

        earlier_out1 = out1[:, :, :100]
        earlier_out2 = out2[:, :, :100]
        max_diff = (earlier_out1 - earlier_out2).abs().max().item()
        if not torch.allclose(earlier_out1, earlier_out2, atol=1e-6):
            pytest.xfail(
                f"Train mode non-causal due to BatchNorm2d time-dimension statistics (max diff={max_diff:.4f})"
            )


# ---------------------------------------------------------------------------
# Check 4: Backward pass and gradient check
# ---------------------------------------------------------------------------


class TestBackward:
    """Verify end-to-end autograd backward pass."""

    def test_backward_finite_gradients(self) -> None:
        """loss = si_snr_loss(...) gives finite, non-None gradients for all parameters."""
        torch.manual_seed(SEED)
        model = CRN()
        stft = STFT()

        clean = torch.randn(2, 16000)
        noisy = clean + 0.5 * torch.randn(2, 16000)

        spec_noisy = stft(noisy)
        enh_spec = model(spec_noisy)
        recon = stft.inverse(enh_spec, length=16000)
        loss = si_snr_loss(recon, clean)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"


# ---------------------------------------------------------------------------
# Check 5: Determinism with seed 42
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify deterministic weight initialization and outputs."""

    def test_seed_42_determinism(self) -> None:
        """With seed 42, two fresh models give identical outputs."""
        torch.manual_seed(SEED)
        model1 = CRN().eval()

        torch.manual_seed(SEED)
        model2 = CRN().eval()

        spec = torch.randn(2, N_FREQ, 251, dtype=torch.complex64)
        with torch.no_grad():
            out1 = model1(spec)
            out2 = model2(spec)

        diff = (out1 - out2).abs().max().item()
        assert diff == 0.0, f"Models diverged with diff={diff}"


# ---------------------------------------------------------------------------
# Check 6: Silence and loud inputs
# ---------------------------------------------------------------------------


class TestExtremeInputs:
    """Verify model stability on silence and peak-level loud inputs."""

    def test_silence_input(self) -> None:
        """All-zero input gives finite output."""
        model = CRN().eval()
        silence = torch.zeros(1, N_FREQ, 251, dtype=torch.complex64)
        with torch.no_grad():
            out = model(silence)
        assert torch.isfinite(out).all(), "Silence input produced non-finite values"

    def test_loud_input(self) -> None:
        """Peak 1.0 input gives finite output."""
        model = CRN().eval()
        loud = torch.ones(1, N_FREQ, 251, dtype=torch.complex64)
        with torch.no_grad():
            out = model(loud)
        assert torch.isfinite(out).all(), "Loud input produced non-finite values"


# ---------------------------------------------------------------------------
# Check 7: Latency and parameter benchmark (median of 100 runs)
# ---------------------------------------------------------------------------


class TestBenchmark:
    """Measure parameters, 1-second clip CPU latency, and single frame CPU latency."""

    def test_benchmark_measurements(self) -> None:
        """Report num_params(), median CPU time for 1s clip and single frame (100 runs)."""
        model = CRN().eval()
        count = model.num_params()

        # 1-second clip (63 frames)
        spec_1s = torch.randn(1, N_FREQ, 63, dtype=torch.complex64)
        for _ in range(10):
            _ = model(spec_1s)
        times_1s: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(spec_1s)
            t1 = time.perf_counter()
            times_1s.append((t1 - t0) * 1000)
        median_1s = statistics.median(times_1s)

        # Single frame (1 frame)
        spec_1f = torch.randn(1, N_FREQ, 1, dtype=torch.complex64)
        for _ in range(10):
            _ = model(spec_1f)
        times_1f: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(spec_1f)
            t1 = time.perf_counter()
            times_1f.append((t1 - t0) * 1000)
        median_1f = statistics.median(times_1f)

        logger.info("CRN num_params: %d (%.2fM)", count, count / 1e6)
        logger.info("CPU latency 1s clip (63 frames, median of 100): %.3f ms", median_1s)
        logger.info("CPU latency single spectral frame (median of 100): %.3f ms", median_1f)

        assert count < 2_000_000
        assert median_1s > 0.0
        assert median_1f > 0.0


# ---------------------------------------------------------------------------
# Check 8: CUDA smoke test
# ---------------------------------------------------------------------------


class TestCUDA:
    """Smoke test on CUDA device if available."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
    def test_cuda_smoke(self) -> None:
        """Run forward pass on CUDA device if hardware is present."""
        device = torch.device("cuda")
        model = CRN().to(device).eval()
        spec = torch.randn(1, N_FREQ, 251, dtype=torch.complex64, device=device)
        with torch.no_grad():
            out = model(spec)
        assert out.is_cuda
        assert out.shape == (1, N_FREQ, 251)
