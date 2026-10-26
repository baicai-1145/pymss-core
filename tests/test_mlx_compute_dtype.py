"""The configured compute dtype reaches model activations after FP32 FFT work."""
import sys

import numpy as np
import pytest
import torch

from pymss_core.modules.bs_roformer import BSRoformer, MelBandRoformer, BSConformer, MelBandConformer, BSRoformerHyperACE
from pymss_core.modules.bs_roformer import mlx_roformer
from pymss_core.modules.mlx_backend import rms_norm

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal required")


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available(): pytest.skip("MLX Metal required")
    return core


def small_model(kind="bs", *, stereo=False, num_stems=1, depth=1, skip_connection=False):
    options = dict(dim=8, depth=depth, stereo=stereo, num_stems=num_stems, heads=1, dim_head=4,
                   skip_connection=skip_connection,
                   stft_n_fft=16, stft_hop_length=4, stft_win_length=16, mask_estimator_depth=1)
    if "conformer" in kind:
        options.update(time_conformer_depth=1, freq_conformer_depth=1)
    else:
        options.update(time_transformer_depth=1, freq_transformer_depth=1)
    if kind.startswith("mel"):
        options.update(num_bands=4, sample_rate=16000)
    else:
        options["freqs_per_bands"] = (1,)*9
    factory = {"bs":BSRoformer,"mel":MelBandRoformer,"bs_conformer":BSConformer,"mel_conformer":MelBandConformer,"hyperace":BSRoformerHyperACE}[kind]
    return factory(**options).eval()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("kind", ["bs", "mel", "bs_conformer", "mel_conformer"])
def test_compute_dtype_reaches_attention_and_mask_inputs(mx, monkeypatch, dtype, kind):
    torch.manual_seed(12)
    model = small_model(kind)
    expected = mx.float16 if dtype == torch.float16 else mx.float32
    seen = []
    for name in ("_mlx_attention", "_mask_estimator"):
        original = getattr(mlx_roformer, name)
        def record(module, x, compute_dtype, fn=original):
            seen.append(x.dtype)
            return fn(module, x, compute_dtype)
        monkeypatch.setattr(mlx_roformer, name, record)
    y = mlx_roformer.mlx_forward_roformer_mx(model, mx.ones((1, 1, 64)), dtype)
    mx.eval(y)
    assert seen and all(value == expected for value in seen)
    assert y.dtype == expected and y.shape == (1, 1, 64)
    assert bool(mx.all(mx.isfinite(y)).item())


@pytest.mark.parametrize("gamma_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("amplitude", [0., 1e-7, 1e-5, 1., 1e3, 65504.])
@pytest.mark.parametrize("grouped", [False, True])
def test_half_rms_norm_accumulates_safely_for_finite_inputs(mx, amplitude, grouped, gamma_dtype):
    x = mx.full((2, 3, 8), amplitude, dtype=mx.float16)
    expected_dtype = mx.float16 if gamma_dtype == torch.float16 else mx.float32
    gamma = mx.linspace(.5, 1.5, 24 if grouped else 8).reshape((3, 8) if grouped else (8,)).astype(expected_dtype)
    actual = rms_norm(x, gamma)
    ref = torch.from_numpy(np.array(x).astype(np.float32))
    expected = torch.nn.functional.rms_norm(ref, (8,), eps=1e-12).numpy() * np.array(gamma).astype(np.float32)
    assert actual.dtype == expected_dtype
    assert bool(mx.all(mx.isfinite(actual)).item())
    np.testing.assert_allclose(np.array(actual), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("kind", ["bs", "mel", "bs_conformer", "mel_conformer"])
def test_half_model_silent_audio_returns_finite_silence(mx, kind):
    y = mlx_roformer.mlx_forward_roformer_mx(small_model(kind), mx.zeros((1, 1, 64)), torch.float16)
    mx.eval(y)
    assert bool(mx.all(mx.isfinite(y)).item())
    assert float(mx.max(mx.abs(y)).item()) == 0.


@pytest.mark.parametrize("kind", ["bs", "mel", "bs_conformer", "mel_conformer"])
@pytest.mark.parametrize("amplitude", [1e-7, 1e-5, 1.])
def test_half_stereo_multistem_model_matches_fp32_reference(mx, kind, amplitude):
    torch.manual_seed(23)
    model = small_model(kind, stereo=True, num_stems=2)
    raw = mx.array(torch.randn(2, 2, 64).numpy()) * amplitude
    actual = mlx_roformer.mlx_forward_roformer_mx(model, raw, torch.float16)
    reference = mlx_roformer.mlx_forward_roformer_mx(model, raw, torch.float32)
    mx.eval(actual, reference)
    assert actual.shape == reference.shape == (2, 2, 2, 64)
    assert actual.dtype == mx.float16 and reference.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(actual)).item())
    # FP16 output is intentionally not bitwise identical; near silence, use an
    # absolute bound because sample-wise relative error is ill-defined.
    np.testing.assert_allclose(np.array(actual).astype(np.float32), np.array(reference),
                               rtol=2e-2, atol=max(2e-7, 2e-3 * amplitude))


def test_half_mel_masks_accumulate_at_fft_precision(mx, monkeypatch):
    model = small_model("mel")
    original = mlx_roformer._complex_from_ri
    observed = []
    def record(x):
        observed.append(x.dtype)
        return original(x)
    monkeypatch.setattr(mlx_roformer, "_complex_from_ri", record)
    y = mlx_roformer.mlx_forward_roformer_mx(model, mx.ones((1, 1, 64)), torch.float16)
    mx.eval(y)
    assert observed == [mx.float32, mx.float32]


@pytest.mark.parametrize("kind", ["bs", "mel", "bs_conformer", "mel_conformer", "hyperace"])
def test_half_deep_skip_residuals_stay_finite_with_half_transformer_matmuls(mx, monkeypatch, kind):
    from pymss_core.modules.bs_roformer import mlx_attention
    torch.manual_seed(23)
    model = small_model(kind, depth=16, stereo=True, num_stems=2, skip_connection=True)
    raw = mx.array(torch.randn(1, 2, 64).numpy())
    original = mlx_attention._linear
    original_conv = mlx_roformer.conv1d
    seen, conv_inputs = [], []
    def record(x, weight, bias=None):
        seen.append((x.dtype, weight.dtype))
        return original(x, weight, bias)
    def record_conv(module, x, dtype):
        conv_inputs.append(x.dtype)
        return original_conv(module, x, dtype)
    monkeypatch.setattr(mlx_attention, "_linear", record)
    monkeypatch.setattr(mlx_roformer, "conv1d", record_conv)
    actual = mlx_roformer.mlx_forward_roformer_mx(model, raw, torch.float16)
    mx.eval(actual)
    assert actual.shape == (1, 2, 2, 64) and actual.dtype == mx.float16
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert seen and all(pair == (mx.float16, mx.float16) for pair in seen)
    if "conformer" in kind:
        assert conv_inputs and all(value == mx.float16 for value in conv_inputs)
    reference = mlx_roformer.mlx_forward_roformer_mx(model, raw, torch.float32)
    mx.eval(reference)
    np.testing.assert_allclose(np.array(actual).astype(np.float32), np.array(reference), atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("kind", ["bs", "mel"])
def test_skip_band_projection_never_narrows_before_fp32_residual(mx, monkeypatch, kind):
    torch.manual_seed(37)
    model = small_model(kind, stereo=True, skip_connection=True)
    with torch.no_grad():
        for _, projection in model.band_split.to_features:
            projection.weight.fill_(40000.)
            if projection.bias is not None: projection.bias.zero_()
    original = mlx_roformer._band_split
    seen = []
    def record(*args):
        out = original(*args)
        mx.eval(out)
        seen.append((out.dtype, bool(mx.all(mx.isfinite(out)).item()), float(mx.max(mx.abs(out)).item())))
        return out
    monkeypatch.setattr(mlx_roformer, "_band_split", record)
    actual = mlx_roformer.mlx_forward_roformer_mx(model, mx.ones((1, 2, 64)), torch.float16)
    mx.eval(actual)
    assert seen and seen[0][0] == mx.float32 and seen[0][1] and seen[0][2] > 65504.
    assert actual.shape == (1, 2, 64) and bool(mx.all(mx.isfinite(actual)).item())
