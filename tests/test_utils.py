"""Tests for the AdaIN math in src/utils.py. These are plain tensor
operations -- no model weights or GPU needed, runs in well under a second:

    python -m pytest tests/test_utils.py -v
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import adaptive_instance_normalization, calc_mean_std  # noqa: E402


def test_calc_mean_std_matches_manual_computation():
    torch.manual_seed(0)
    feat = torch.randn(2, 3, 8, 8)
    mean, std = calc_mean_std(feat)

    assert mean.shape == (2, 3, 1, 1)
    assert std.shape == (2, 3, 1, 1)

    expected_mean = feat[0, 0].mean()
    assert torch.allclose(mean[0, 0, 0, 0], expected_mean, atol=1e-5)


def test_calc_mean_std_constant_feature_has_zero_std():
    feat = torch.full((1, 4, 5, 5), 3.0)
    mean, std = calc_mean_std(feat)
    assert torch.allclose(mean, torch.full_like(mean, 3.0))
    # not exactly zero because of eps, but should be tiny
    assert (std < 1e-2).all()


def test_adain_output_matches_style_statistics():
    torch.manual_seed(0)
    content = torch.randn(1, 4, 16, 16) * 2.0 + 5.0   # mean ~5, std ~2
    style = torch.randn(1, 4, 16, 16) * 0.3 - 1.0      # mean ~-1, std ~0.3

    result = adaptive_instance_normalization(content, style)
    result_mean, result_std = calc_mean_std(result)
    style_mean, style_std = calc_mean_std(style)

    assert torch.allclose(result_mean, style_mean, atol=1e-4)
    assert torch.allclose(result_std, style_std, atol=1e-4)


def test_adain_is_identity_when_content_equals_style():
    torch.manual_seed(0)
    feat = torch.randn(1, 4, 16, 16)
    result = adaptive_instance_normalization(feat, feat)
    assert torch.allclose(result, feat, atol=1e-4)


def test_calc_mean_std_rejects_wrong_rank():
    try:
        calc_mean_std(torch.randn(4, 4))  # missing batch/channel dims
        assert False, "expected an assertion error for a non-4D tensor"
    except AssertionError:
        pass
