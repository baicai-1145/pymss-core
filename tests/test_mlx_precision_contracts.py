"""Mixed-precision contracts, cache transitions, and direct Torch references."""
import sys
import gc
import itertools
import numpy as np
import pytest
import torch
from pymss_core import clear_model_runtime_caches
from pymss_core.modules.mlx_backend import linear
from pymss_core.modules.bs_roformer import BSRoformer,MelBandRoformer,BSConformer,MelBandConformer,BSRoformerHyperACE
from pymss_core.modules.bs_roformer import mlx_roformer,mlx_attention
from pymss_core.modules.bs_roformer.transformer import FeedForward

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal required")
DTYPES=[torch.float16,torch.float32]

@pytest.fixture
def mx():
    core=pytest.importorskip("mlx.core")
    if not core.metal.is_available():pytest.skip("MLX Metal required")
    return core

@pytest.mark.parametrize('shape',[(8,),(3,8),(2,3,8)])
@pytest.mark.parametrize('xd,wd,bd',list(itertools.product(DTYPES,DTYPES,[None,*DTYPES])))
def test_affine_mixed_types_and_broadcast(mx,shape,xd,wd,bd):
    rng=np.random.default_rng(53)
    xr=rng.normal(size=shape).astype(np.float16 if xd==torch.float16 else np.float32)
    wr=rng.normal(size=(11,8)).astype(np.float16 if wd==torch.float16 else np.float32)
    br=None if bd is None else rng.normal(size=(11,)).astype(np.float16 if bd==torch.float16 else np.float32)
    x,w=mx.array(xr),mx.array(wr);b=None if br is None else mx.array(br)
    out=linear(x,w,b)
    old=mx.matmul(x,w.T);old=old if b is None else old+b
    expected=xr.astype(np.float32)@wr.astype(np.float32).T
    if br is not None:expected+=br.astype(np.float32)
    assert out.dtype==old.dtype and out.shape==expected.shape
    tol=4e-3 if out.dtype==mx.float16 else 3e-6
    np.testing.assert_allclose(np.array(out),expected,atol=tol,rtol=tol)

@pytest.mark.parametrize('case',['rows','contract','outputs','batched_weights'])
@pytest.mark.parametrize('dtype',DTYPES)
def test_affine_empty_and_batched_weights(mx,case,dtype):
    xs,ws={'rows':((0,8),(11,8)),'contract':((3,0),(11,0)),
           'outputs':((3,8),(0,8)),'batched_weights':((2,3,8),(2,11,8))}[case]
    md=mx.float16 if dtype==torch.float16 else mx.float32
    x=mx.ones(xs,dtype=md);w=mx.ones(ws,dtype=md);b=mx.ones((ws[-2],),dtype=md)
    out=linear(x,w,b);mx.eval(out)
    expected=np.ones(xs,dtype=np.float32)@np.swapaxes(np.ones(ws,dtype=np.float32),-1,-2)+1
    assert out.shape==expected.shape and out.dtype==md
    if out.size:np.testing.assert_allclose(np.array(out),expected,atol=1e-5)

@pytest.mark.parametrize('xd,td,eager,noncontiguous',list(itertools.product(DTYPES,[torch.float16,torch.float32],[False,True],[False,True])))
def test_ffn_bias_dtype_layout_switches_and_cleanup(mx,xd,td,eager,noncontiguous):
    torch.manual_seed(71)
    model=FeedForward(8,mult=2).eval();model._pymss_mlx_disable_compiled_ffn=eager
    for in_bias,out_bias in [(True,True),(False,True),(True,False),(False,False),(True,True)]:
        for index,enabled in [(1,in_bias),(4,out_bias)]:
            model.net[index].bias=torch.nn.Parameter(torch.randn(model.net[index].out_features)) if enabled else None
        x=mx.arange(56,dtype=mx.float16 if xd==torch.float16 else mx.float32).reshape((1,8,7) if noncontiguous else (1,7,8))/56
        if noncontiguous:x=x.transpose(0,2,1)
        inp=torch.from_numpy(np.array(x).astype(np.float32))
        with torch.inference_mode():ref=model(inp).numpy()
        actual=mlx_attention._mlx_feed_forward(model,x,td);mx.eval(actual)
        assert actual.dtype==(mx.float16 if td==torch.float16 else mx.float32)
        np.testing.assert_allclose(np.array(actual),ref,atol=4e-3 if td==torch.float16 else 3e-6,rtol=4e-3)
        if not eager:
            cache=model._pymss_mlx_compiled_feed_forward_cache
            assert len(cache)==2 and 'fn' in cache and 'error' not in cache
    clear_model_runtime_caches(model)
    assert not model._pymss_mlx_feed_forward_cache
    if not eager:assert not model._pymss_mlx_compiled_feed_forward_cache


def build(kind,skip=False,depth=2):
    options=dict(dim=8,depth=depth,stereo=True,num_stems=1,heads=1,dim_head=4,
                 stft_n_fft=16,stft_hop_length=4,stft_win_length=16,mask_estimator_depth=1,
                 norm_output=True,skip_connection=skip,flash_attn=False)
    if 'conformer' in kind:options.update(time_conformer_depth=1,freq_conformer_depth=1)
    else:options.update(time_transformer_depth=1,freq_transformer_depth=1)
    if kind.startswith('mel'):options.update(num_bands=4,sample_rate=16000)
    else:options.update(freqs_per_bands=(1,)*9)
    return {'bs':BSRoformer,'mel':MelBandRoformer,'bs_conformer':BSConformer,
            'mel_conformer':MelBandConformer,'hyperace':BSRoformerHyperACE}[kind](**options).eval()

@pytest.mark.parametrize('kind',['bs','mel','bs_conformer','mel_conformer','hyperace'])
@pytest.mark.parametrize('skip',[False,True])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('length',[96,97])
def test_end_to_end_direct_torch_reference(mx,kind,skip,dtype,length):
    torch.manual_seed(73)
    model=build(kind,skip)
    raw=torch.randn(1,2,length)
    with torch.inference_mode():reference=model(raw).numpy()
    out=mlx_roformer.mlx_forward_roformer_mx(model,mx.array(raw.numpy()),dtype);mx.eval(out)
    assert out.shape==reference.shape and bool(mx.all(mx.isfinite(out)).item())
    tol=4e-3 if dtype==torch.float16 else 3e-5
    np.testing.assert_allclose(np.array(out),reference,atol=tol,rtol=2e-2 if dtype==torch.float16 else 3e-4)
    clear_model_runtime_caches(model)

@pytest.mark.parametrize('mode',['full','segm_only'])
def test_hyperace_segmentation_fp32_direct_torch_reference(mx,mode):
    dtype=torch.float32
    torch.manual_seed(79)
    model=build('hyperace',depth=1);model.set_mask_mode(mode)
    raw=torch.randn(1,2,256)
    with torch.inference_mode():reference=model(raw).numpy()
    out=mlx_roformer.mlx_forward_roformer_mx(model,mx.array(raw.numpy()),dtype);mx.eval(out)
    assert out.shape==reference.shape and bool(mx.all(mx.isfinite(out)).item())
    np.testing.assert_allclose(np.array(out),reference,atol=5e-4,rtol=2e-3)
    clear_model_runtime_caches(model)
    gc.collect()


@pytest.mark.parametrize('mode',['full','segm_only'])
@pytest.mark.parametrize('silent',[False,True])
def test_hyperace_segmentation_fp16_has_finite_output(mx,mode,silent):
    # FP16 is not assumed numerically equivalent to FP32 for this sensitive
    # segmentation head. The separate FP32 test checks the Torch reference.
    torch.manual_seed(79)
    model=build('hyperace',depth=1);model.set_mask_mode(mode)
    raw=mx.zeros((1,2,256)) if silent else mx.array(torch.randn(1,2,256).numpy())
    out=mlx_roformer.mlx_forward_roformer_mx(model,raw,torch.float16);mx.eval(out)
    assert out.shape==(1,2,256) and out.dtype==mx.float16
    assert bool(mx.all(mx.isfinite(out)).item())
    if silent:assert float(mx.max(mx.abs(out)).item())==0.
    clear_model_runtime_caches(model)
    gc.collect()
