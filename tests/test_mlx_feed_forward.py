"""Compiler parity, bounded cache lifetime and eager fallback on real Metal."""
import sys
import gc
import weakref

import numpy as np
import pytest
import torch

from pymss_core import clear_model_runtime_caches
from pymss_core.modules.bs_roformer import mlx_attention
from pymss_core.modules.bs_roformer.transformer import FeedForward

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal required")


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available(): pytest.skip("MLX Metal required")
    return core


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
def test_compiled_ffn_matches_torch(mx, dtype, bias):
    torch.manual_seed(9)
    module = FeedForward(8, mult=2).eval()
    if not bias:
        module.net[1].bias = module.net[4].bias = None
    audio = torch.randn(2, 7, 8)
    expected = module(audio).detach().numpy()
    actual = mlx_attention._mlx_feed_forward(module, mx.array(audio.numpy(), dtype=mx.float16 if dtype == torch.float16 else mx.float32), dtype)
    assert not getattr(module, "_pymss_mlx_disable_compiled_ffn", False)
    tolerance = 3e-3 if dtype == torch.float16 else 2e-6
    np.testing.assert_allclose(np.array(actual), expected, atol=tolerance, rtol=tolerance)


def test_compiled_ffn_replaces_shapes_weights_and_rebuilds_after_cleanup(mx):
    module = FeedForward(8, mult=2).eval()
    x = mx.ones((1, 4, 8))
    first = np.array(mlx_attention._mlx_feed_forward(module, x, torch.float32))
    cache = module._pymss_mlx_compiled_feed_forward_cache
    original_fn = cache["fn"]
    mlx_attention._mlx_feed_forward(module, x, torch.float32)
    assert cache["fn"] is original_fn
    for shape in ((62, 64, 8), (62, 1151, 8), (2, 16, 8), (1, 4, 8)):
        audio = torch.randn(shape)
        actual = mlx_attention._mlx_feed_forward(module, mx.array(audio.numpy()), torch.float32)
        assert actual.shape == shape
        np.testing.assert_allclose(np.array(actual), module(audio).detach().numpy(), atol=2e-6, rtol=2e-6)
        assert len(cache) == 2 and cache["key"][0] == shape
    with torch.no_grad(): module.net[4].bias.add_(1)
    changed = np.array(mlx_attention._mlx_feed_forward(module, x, torch.float32))
    np.testing.assert_allclose(changed, first+1, atol=2e-6)
    clear_model_runtime_caches(module)
    assert not cache and not module._pymss_mlx_feed_forward_cache
    rebuilt = np.array(mlx_attention._mlx_feed_forward(module, x, torch.float32))
    np.testing.assert_allclose(rebuilt, changed, atol=2e-6)


@pytest.mark.parametrize("failure_stage", ["compile", "execute"])
def test_compile_failure_uses_eager_ffn_and_does_not_retry_same_key(mx, monkeypatch, failure_stage):
    module = FeedForward(8, mult=2).eval()
    audio = torch.randn(1, 5, 8)
    expected = module(audio).detach().numpy()
    calls = []
    def fail_compile(fn, **kwargs):
        calls.append("compile")
        if failure_stage == "compile": raise RuntimeError("compiler unavailable")
        def fail_execute(*args):
            calls.append("execute")
            raise RuntimeError("compiler execution unavailable")
        return fail_execute
    monkeypatch.setattr(mx, "compile", fail_compile)
    for _ in range(2):
        actual = mlx_attention._mlx_feed_forward(module, mx.array(audio.numpy()), torch.float32)
        np.testing.assert_allclose(np.array(actual), expected, atol=2e-6, rtol=2e-6)
    assert calls == (["compile"] if failure_stage == "compile" else ["compile", "execute"])
    assert not getattr(module, "_pymss_mlx_disable_compiled_ffn", False)
    cache = module._pymss_mlx_compiled_feed_forward_cache
    assert "unavailable" in cache["error"]
    assert "fn" not in cache


def test_failed_warmup_shape_does_not_disable_real_shape(mx, monkeypatch):
    module = FeedForward(8, mult=2).eval()
    original_compile = mx.compile
    attempts = []
    def compile_ffn(fn):
        attempts.append(True)
        compiled = original_compile(fn)
        def run(*args):
            if args[0].shape == (1, 4, 8): raise RuntimeError("unsupported warmup shape")
            return compiled(*args)
        return run
    monkeypatch.setattr(mx, "compile", compile_ffn)
    for shape in ((1, 4, 8), (1, 4, 8), (1, 16, 8)):
        audio = torch.randn(shape)
        actual = mlx_attention._mlx_feed_forward(module, mx.array(audio.numpy()), torch.float32)
        np.testing.assert_allclose(np.array(actual), module(audio).detach().numpy(), atol=2e-6, rtol=2e-6)
    assert len(attempts) == 2
    assert "fn" in module._pymss_mlx_compiled_feed_forward_cache


def test_cleanup_retries_a_transient_compiler_failure(mx, monkeypatch):
    module = FeedForward(8, mult=2).eval()
    original_compile = mx.compile
    attempts = []
    def compile_ffn(fn):
        attempts.append(True)
        if len(attempts) == 1: raise RuntimeError("temporary compiler failure")
        return original_compile(fn)
    monkeypatch.setattr(mx, "compile", compile_ffn)
    x = mx.ones((1, 4, 8))
    mlx_attention._mlx_feed_forward(module, x, torch.float32)
    clear_model_runtime_caches(module)
    mlx_attention._mlx_feed_forward(module, x, torch.float32)
    assert len(attempts) == 2
    assert "fn" in module._pymss_mlx_compiled_feed_forward_cache


@pytest.mark.parametrize("change", ["weights", "dtype"])
def test_failed_key_is_retried_after_weight_or_dtype_change(mx, monkeypatch, change):
    module = FeedForward(8, mult=2).eval()
    original_compile = mx.compile
    attempts = []
    def compile_ffn(fn):
        attempts.append(True)
        if len(attempts) == 1: raise RuntimeError("unsupported compilation")
        return original_compile(fn)
    monkeypatch.setattr(mx, "compile", compile_ffn)
    x = mx.ones((1, 4, 8))
    mlx_attention._mlx_feed_forward(module, x, torch.float32)
    dtype = torch.float32
    if change == "weights":
        with torch.no_grad(): module.net[4].bias.add_(1)
    else:
        x, dtype = x.astype(mx.float16), torch.float16
    actual = mlx_attention._mlx_feed_forward(module, x, dtype)
    expected = module(torch.ones((1, 4, 8))).detach().numpy()
    np.testing.assert_allclose(np.array(actual), expected, atol=3e-3, rtol=3e-3)
    assert len(attempts) == 2
    assert "error" not in module._pymss_mlx_compiled_feed_forward_cache


@pytest.mark.parametrize("failure", ["eval", "shape"])
def test_deferred_or_invalid_compiled_output_falls_back(mx, monkeypatch, failure):
    module = FeedForward(8, mult=2).eval()
    audio = torch.randn(1, 5, 8)
    expected = module(audio).detach().numpy()
    attempts = []
    original_compile, original_eval = mx.compile, mx.eval
    def compile_ffn(fn):
        attempts.append(True)
        if failure == "shape": return lambda *args: mx.zeros((1, 1, 8))
        return original_compile(fn)
    def evaluate(*arrays):
        if not getattr(evaluate, "failed", False):
            evaluate.failed = True
            raise RuntimeError("deferred execution failure")
        return original_eval(*arrays)
    monkeypatch.setattr(mx, "compile", compile_ffn)
    if failure == "eval": monkeypatch.setattr(mx, "eval", evaluate)
    for _ in range(2):
        actual = mlx_attention._mlx_feed_forward(module, mx.array(audio.numpy()), torch.float32)
        np.testing.assert_allclose(np.array(actual), expected, atol=2e-6, rtol=2e-6)
    assert len(attempts) == 1
    assert "fn" not in module._pymss_mlx_compiled_feed_forward_cache


def test_shape_changes_and_cleanup_release_compiled_callables(mx, monkeypatch):
    module = FeedForward(8, mult=2).eval()
    original_compile = mx.compile
    references = []
    class TrackedCompiled:
        def __init__(self, fn): self.fn = original_compile(fn)
        def __call__(self, *args): return self.fn(*args)
    def compile_ffn(fn):
        compiled = TrackedCompiled(fn)
        references.append(weakref.ref(compiled))
        return compiled
    monkeypatch.setattr(mx, "compile", compile_ffn)
    for length in (4, 8, 16, 4):
        mlx_attention._mlx_feed_forward(module, mx.ones((1, length, 8)), torch.float32)
        gc.collect()
        assert sum(ref() is not None for ref in references) == 1
    clear_model_runtime_caches(module)
    gc.collect()
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_wide_residual_is_normalized_before_casting_to_compute_dtype(mx, dtype, eager, bias):
    torch.manual_seed(29)
    module = FeedForward(8, mult=2).eval()
    module._pymss_mlx_disable_compiled_ffn = eager
    if not bias:
        module.net[1].bias = module.net[4].bias = None
    audio = torch.randn(2, 7, 8) * 1e6
    expected = module(audio).detach().numpy()
    actual = mlx_attention._mlx_feed_forward(module, mx.array(audio.numpy()), dtype)
    mx.eval(actual)
    assert actual.dtype == (mx.float16 if dtype == torch.float16 else mx.float32)
    assert bool(mx.all(mx.isfinite(actual)).item())
    np.testing.assert_allclose(np.array(actual), expected, atol=3e-3 if dtype == torch.float16 else 2e-6, rtol=3e-3)
    if not eager:
        assert "fn" in module._pymss_mlx_compiled_feed_forward_cache
