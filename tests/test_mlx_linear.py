"""Native fused affine operations retain Linear's shape and dtype contracts."""
import sys

import numpy as np
import pytest
import torch

from pymss_core.modules.mlx_backend import linear

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal required")


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available(): pytest.skip("MLX Metal required")
    return core


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(8,), (5, 8), (2, 5, 8), (0, 8)])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_linear_matches_torch_for_vectors_batches_and_empty_inputs(mx, dtype, shape, bias):
    torch.manual_seed(31)
    x = torch.randn(shape)
    weight = torch.randn(11, 8)
    offset = torch.randn(11) if bias else None
    expected = torch.nn.functional.linear(x, weight, offset).numpy()
    mx_type = mx.float16 if dtype == torch.float16 else mx.float32
    actual = linear(mx.array(x.numpy(), dtype=mx_type), mx.array(weight.numpy(), dtype=mx_type),
                    None if offset is None else mx.array(offset.numpy(), dtype=mx_type))
    tolerance = 3e-3 if dtype == torch.float16 else 2e-6
    assert actual.dtype == mx_type
    assert actual.shape == expected.shape
    # MLX 0.31's NumPy bridge dereferences null storage for zero-sized arrays.
    # Empty values are already determined by shape; still evaluate the operation.
    if actual.size:
        np.testing.assert_allclose(np.array(actual), expected, atol=tolerance, rtol=tolerance)
    else:
        mx.eval(actual)


def test_fused_linear_accepts_noncontiguous_inputs_and_weights(mx):
    x = mx.arange(80, dtype=mx.float32).reshape(2, 8, 5).transpose(0, 2, 1) / 80
    weight = mx.arange(88, dtype=mx.float32).reshape(8, 11).T / 88
    offset = mx.arange(11, dtype=mx.float32) / 11
    expected = torch.nn.functional.linear(torch.from_numpy(np.array(x)), torch.from_numpy(np.array(weight)),
                                          torch.from_numpy(np.array(offset))).numpy()
    np.testing.assert_allclose(np.array(linear(x, weight, offset)), expected, atol=2e-6, rtol=2e-6)
