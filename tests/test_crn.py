"""Tests for the causal CRN magnitude mask baseline model."""

from __future__ import annotations

import logging

import pytest
import torch

from ai_engine.models.crn import CRN
from ai_engine.models.stft import STFT
from common.constants import N_FREQ, SEED

logger = logging.getLogger(__name__)


@pytest.fixture()
def crn_model() -> CRN:
    """Create a CRN model instance in eval mode with deterministic seed."""
    torch.manual_seed(SEED)
    model = CRN()
    model.eval()
    return model


class TestCRN:
    """Test suite for CRN baseline model."""

    def test_output_shape_and_dtype(self, crn_model: CRN) -> None:
        """Verify output shape and complex dtype on random batch [2, 257, 251]."""
        torch.manual_seed(SEED)
        spec = torch.randn(2, N_FREQ, 251, dtype=torch.complex64)
        out = crn_model(spec)
        assert out.shape == (2, N_FREQ, 251)
        assert out.dtype == torch.complex64

    def test_causality(self, crn_model: CRN) -> None:
        """Verify output at frame t is strictly independent of frames > t."""
        torch.manual_seed(SEED)
        time_len = 251
        t_cut = 100
        spec1 = torch.randn(1, N_FREQ, time_len, dtype=torch.complex64)
        spec2 = spec1.clone()
        spec2[:, :, t_cut:] = torch.randn_like(spec2[:, :, t_cut:])

        with torch.no_grad():
            enh1 = crn_model(spec1)
            enh2 = crn_model(spec2)

        earlier_diff = (enh1[:, :, :t_cut] - enh2[:, :, :t_cut]).abs().max().item()
        assert earlier_diff == 0.0, f"Future frames affected earlier outputs (diff={earlier_diff})"

    def test_mask_range(self, crn_model: CRN) -> None:
        """Verify estimated mask values are strictly within [0, 1]."""
        torch.manual_seed(SEED)
        spec = torch.randn(2, N_FREQ, 251, dtype=torch.complex64)
        mask = crn_model.compute_mask(spec)
        assert mask.shape == (2, N_FREQ, 251)
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0

    def test_param_count_and_logging(self, crn_model: CRN) -> None:
        """Verify param count is under 2M and log the value."""
        count = crn_model.num_params()
        logger.info("CRN total parameter count: %d (%.2fM)", count, count / 1e6)
        assert count < 2_000_000, f"CRN parameter count {count} exceeds 2M limit"

    def test_end_to_end_waveform_with_stft(self, crn_model: CRN) -> None:
        """Verify end-to-end waveform processing via STFT -> CRN -> iSTFT."""
        torch.manual_seed(SEED)
        stft = STFT()
        wave = torch.randn(2, 16000)
        enhanced_wave = crn_model.enhance_waveform(wave, stft=stft)
        assert enhanced_wave.shape == wave.shape
        assert torch.isfinite(enhanced_wave).all()
