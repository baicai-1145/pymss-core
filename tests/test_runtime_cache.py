import copy
import gc
import weakref

import pytest
import torch

from pymss_core import clear_model_runtime_caches
from pymss_core.modules.bs_roformer.bands import BandSplit, MaskEstimator
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
from pymss_core.modules.bs_roformer.common import RotaryEmbedding
from pymss_core.modules.bs_roformer.transformer import RMSNorm
from pymss_core.modules.look2hear.apollo import Apollo, Roformer as ApolloRoformer


def test_runtime_cache_cleanup_releases_cached_tensors_without_changing_weights_or_user_data():
    model = torch.nn.Sequential(RMSNorm(4), RotaryEmbedding(4))
    model[0].register_buffer("persistent_data", torch.ones(4))
    weights = copy.deepcopy(model.state_dict())
    dtype_cache = model[0]._gamma_dtype_cache = {"weights": torch.ones(4)}
    model[0]._packed_cache = {"user": "keep"}
    model[0]._pymss_cos_sin_cache = {"position": torch.ones(4)}
    model[0]._pymss_group_cache_warm_key = ("cpu", torch.float32)
    model[0].cache = {"user": "keep"}
    model[1].cache["frequency"] = torch.ones(4)
    for _ in range(2):
        clear_model_runtime_caches(model)
        assert model[0]._gamma_dtype_cache is dtype_cache and not dtype_cache
        assert model[0]._packed_cache == {"user": "keep"}
        assert not model[0]._pymss_cos_sin_cache and not model[1].cache
        assert model[0].cache == {"user": "keep"}
        assert model[0]._pymss_group_cache_warm_key is None
        for key, value in weights.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_runtime_cache_cleanup_visits_shared_modules_once_and_preserves_hook_errors():
    class CachedModule(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0; self.failure = None
        def clear_runtime_cache(self):
            self.calls += 1
            if self.failure is not None: raise self.failure
    child = CachedModule()
    model = torch.nn.ModuleList([child, child])
    clear_model_runtime_caches(model)
    assert child.calls == 1
    child.failure = RuntimeError("Runtime cache cleanup failed")
    with pytest.raises(RuntimeError) as caught: clear_model_runtime_caches(model)
    assert caught.value is child.failure
    assert child.calls == 2


def test_cleanup_finishes_other_caches_before_reraising_a_hook_error():
    failure = RuntimeError("Runtime cache cleanup failed")

    class BrokenCache(RMSNorm):
        def clear_runtime_cache(self):
            super().clear_runtime_cache()
            raise failure

    first = BrokenCache(4)
    second = RotaryEmbedding(4)
    first._gamma_dtype_cache = {"weights": torch.ones(4)}
    second.cache["frequency"] = torch.ones(4)
    second._pymss_cos_sin_cache = {"position": torch.ones(4)}
    model = torch.nn.Sequential(first, second)
    weights = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError) as caught:
        clear_model_runtime_caches(model)
    assert caught.value is failure
    assert not first._gamma_dtype_cache and not second.cache and not second._pymss_cos_sin_cache
    for key, value in weights.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("cache_name", [
    "_pymss_mlx_full_param_cache", "_pymss_mlx_cos_sin_cache", "_pymss_mlx_attention_cache",
    "_pymss_mlx_feed_forward_cache", "_pymss_mlx_compiled_feed_forward_cache", "_pymss_mlx_norm_cache", "_pymss_mlx_full_band_split_cache",
    "_pymss_mlx_full_mask_cache", "_pymss_mlx_full_mbr_cache",
])
@pytest.mark.parametrize("hook_failure", [False, True])
def test_cleanup_releases_mlx_cache_references_even_when_a_hook_fails(cache_name, hook_failure):
    failure = RuntimeError("Runtime cache cleanup failed")

    class CachedModule(torch.nn.Linear):
        def clear_runtime_cache(self):
            if hook_failure: raise failure

    model = torch.nn.Sequential(CachedModule(4, 4), torch.nn.Linear(4, 4))
    weights = copy.deepcopy(model.state_dict())
    model[0].cache = {"user": "keep"}
    tensor = torch.ones(4)
    reference = weakref.ref(tensor)
    caches = [{"tensor": tensor}, {"tensor": tensor}]
    for module, cache in zip(model, caches): setattr(module, cache_name, cache)
    del tensor

    if hook_failure:
        with pytest.raises(RuntimeError) as caught: clear_model_runtime_caches(model)
        assert caught.value is failure
    else:
        clear_model_runtime_caches(model)

    gc.collect()
    assert reference() is None
    for module, cache in zip(model, caches):
        assert getattr(module, cache_name) is cache and not cache
    assert model[0].cache == {"user": "keep"}
    for key, value in weights.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("cache_name", [
    "_stft_window_cache", "_gamma_dtype_cache", "_group_cache", "_layer_group_cache", "_index_cache",
    "_packed_layer_group_cache", "_apollo_inference_cache", "_rotary_freq_cache", "_packed_cache",
])
def test_cleanup_preserves_generic_cache_names_on_user_modules(cache_name):
    class UserModule(torch.nn.Module):
        pass

    child = UserModule()
    cache = {"user": ["keep", "original"]}
    setattr(child, cache_name, cache)
    clear_model_runtime_caches(torch.nn.Sequential(child))
    assert getattr(child, cache_name) is cache
    assert cache == {"user": ["keep", "original"]}


def test_apollo_cast_cache_is_namespaced_and_released_without_touching_user_state():
    from pymss_core.modules.look2hear.apollo import _cached_inference_tensor

    module = torch.nn.Conv1d(2, 2, 1)
    user_cache = module._apollo_inference_cache = {"user": "keep"}
    casted = _cached_inference_tensor(module, "weight", module.weight,
                                    torch.zeros(1, 2, 1, dtype=torch.float16), module.weight._version)
    reference = weakref.ref(casted)
    del casted
    assert user_cache == {"user": "keep"}
    cache = module._pymss_apollo_inference_cache
    clear_model_runtime_caches(module)
    gc.collect()
    assert module._pymss_apollo_inference_cache is cache and not cache
    assert reference() is None
    assert user_cache == {"user": "keep"}


def _small_roformer():
    return BSRoformer(dim=8, depth=1, stereo=True, heads=2, dim_head=4,
                      time_transformer_depth=1, freq_transformer_depth=1,
                      freqs_per_bands=(4, 5), stft_n_fft=16, stft_hop_length=4,
                      stft_win_length=16, mask_estimator_depth=1)


@pytest.mark.parametrize("factory,cache_names", [
    (_small_roformer, ("_stft_window_cache",)),
    (lambda: RMSNorm(4), ("_gamma_dtype_cache",)),
    (lambda: BandSplit(4, (4, 4)), ("_group_cache",)),
    (lambda: MaskEstimator(4, (4, 4), depth=1),
     ("_group_cache", "_layer_group_cache", "_index_cache", "_packed_layer_group_cache")),
    (lambda: ApolloRoformer(8, 8, num_head=2, window=4), ("_rotary_freq_cache",)),
    (lambda: Apollo(8000, 20, 8, 0), ("_packed_cache",)),
])
def test_native_owners_clear_their_own_caches_without_changing_state(factory, cache_names):
    model = factory().eval()
    weights = copy.deepcopy(model.state_dict())
    caches, references = [], []
    for name in cache_names:
        cache = getattr(model, name)
        tensor = torch.ones(4)
        references.append(weakref.ref(tensor))
        cache["payload"] = tensor
        caches.append(cache)
        del tensor
    clear_model_runtime_caches(model)
    gc.collect()
    assert all(reference() is None for reference in references)
    for name, cache in zip(cache_names, caches): assert getattr(model, name) is cache and not cache
    for key, value in weights.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_roformer_forward_rebuilds_caches_after_cleanup():
    model = _small_roformer().eval()
    audio = torch.randn(1, 2, 64)
    with torch.inference_mode():
        before = model(audio)
        assert model._stft_window_cache and model.band_split._group_cache
        clear_model_runtime_caches(model)
        assert not model._stft_window_cache and not model.band_split._group_cache
        after = model(audio)
    assert model._stft_window_cache and model.band_split._group_cache
    torch.testing.assert_close(before, after, rtol=0, atol=0)
