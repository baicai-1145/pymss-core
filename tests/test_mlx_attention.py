"""Attention parity across rolling evaluation stages and SDPA query tails."""
import sys

import numpy as np
import pytest
import torch

from pymss_core.modules.bs_roformer.common import RotaryEmbedding
from pymss_core.modules.bs_roformer.transformer import Attention
from pymss_core.modules.bs_roformer.mlx_attention import _mlx_attention

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal required")


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available(): pytest.skip("MLX Metal required")
    return core


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("length", [1, 31, 513])
@pytest.mark.parametrize("rotary", [False, True])
def test_attention_matches_torch_with_rotary_gates_and_query_tail(mx, dtype, length, rotary):
    torch.manual_seed(17)
    module = Attention(16, heads=2, dim_head=8,
                       rotary_embed=RotaryEmbedding(dim=8) if rotary else None,
                       shared_qkv_bias=torch.nn.Parameter(torch.randn(48)*0.1),
                       shared_out_bias=torch.nn.Parameter(torch.randn(16)*0.1)).eval()
    audio = torch.randn(2, length, 16)
    with torch.inference_mode(): expected = module(audio).numpy()
    x = mx.array(audio.numpy(), dtype=mx.float16 if dtype == torch.float16 else mx.float32)
    actual = _mlx_attention(module, x, dtype)
    tolerance = 3e-3 if dtype == torch.float16 else 2e-5
    np.testing.assert_allclose(np.array(actual), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rotary", [False, True])
def test_attention_normalizes_wide_residual_before_half_projection(mx, dtype, rotary):
    torch.manual_seed(31)
    module = Attention(16, heads=2, dim_head=8,
                       rotary_embed=RotaryEmbedding(dim=8) if rotary else None).eval()
    audio = torch.randn(2, 31, 16) * 1e6
    with torch.inference_mode(): expected = module(audio).numpy()
    actual = _mlx_attention(module, mx.array(audio.numpy()), dtype)
    mx.eval(actual)
    assert actual.dtype == (mx.float16 if dtype == torch.float16 else mx.float32)
    assert bool(mx.all(mx.isfinite(actual)).item())
    np.testing.assert_allclose(np.array(actual), expected, atol=3e-3 if dtype == torch.float16 else 2e-5, rtol=3e-3)
