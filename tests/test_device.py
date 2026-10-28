from __future__ import annotations

import pytest
import torch

from pymss_core import DEVICE_ALIASES, DeviceProfile, device_profile
from pymss_core.device import (
    ALIAS_TO_VENDOR,
    ASCEND,
    BIREN,
    CPU,
    ENFLAME,
    HYGON,
    ILUVATAR,
    METAX,
    MLU,
    NVIDIA,
    ROCM,
    amp_supported,
    clear_device_cache,
    rnn_op_blocked,
    detect_vendor,
    fft_fallback_required,
    prepare_device_alias,
    profile_for_alias,
    requires_plugin,
    rnn_fallback_required,
)

VENDOR_DEVICE_NAMES = {
    "Hygon DCU K100_AI": HYGON,
    "Iluvatar BI-V150": ILUVATAR,
    "MetaX C500": METAX,
    "Enflame S60": ENFLAME,
    "Biren BR104": BIREN,
    "Cambricon MLU370": MLU,
    "NVIDIA A100-SXM4-40GB": NVIDIA,
}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    clear_device_cache()
    for name in ("PYMSS_DEVICE_BACKEND", "PYMSS_FFT_FALLBACK", "PYMSS_RNN_FALLBACK", "PYMSS_AMP"):
        monkeypatch.delenv(name, raising=False)
    yield
    clear_device_cache()


def _fake_cuda(monkeypatch, name, available=True, count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: name, raising=False)


@pytest.mark.parametrize("device_name,vendor", sorted(VENDOR_DEVICE_NAMES.items()))
def test_detect_vendor_from_cuda_device_name(monkeypatch, device_name, vendor):
    _fake_cuda(monkeypatch, device_name)
    assert detect_vendor("cuda") == vendor


def test_detect_vendor_uses_private_device_types():
    assert detect_vendor("npu") == ASCEND
    assert detect_vendor("gcu") == ENFLAME
    assert detect_vendor("mlu") == MLU
    assert detect_vendor("supa") == BIREN
    assert detect_vendor("cpu") == CPU
    assert detect_vendor("mps") == "mps"


def test_rocm_falls_back_to_hip_version(monkeypatch):
    _fake_cuda(monkeypatch, "AMD Instinct MI300X")
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    assert detect_vendor("cuda") == ROCM


def test_unknown_cuda_device_is_treated_as_nvidia(monkeypatch):
    _fake_cuda(monkeypatch, "Generic Accelerator XYZ")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert detect_vendor("cuda") == NVIDIA


def test_env_backend_forces_detection(monkeypatch):
    _fake_cuda(monkeypatch, "NVIDIA A100-SXM4-40GB")
    monkeypatch.setenv("PYMSS_DEVICE_BACKEND", "metax")
    assert detect_vendor("cuda") == METAX
    assert device_profile("cuda:0").vendor == METAX


def test_ascend_profile_ships_both_fallbacks():
    profile = device_profile("npu")
    assert profile.plugin == "torch_npu"
    assert profile.needs_fft_fallback and profile.needs_rnn_fallback
    assert not profile.supports_data_parallel


def test_domestic_cuda_impersonators_keep_native_operators(monkeypatch):
    _fake_cuda(monkeypatch, "Hygon DCU K100_AI")
    profile = device_profile("cuda:0")
    assert profile.vendor == HYGON
    assert profile.device_type == "cuda"
    assert not profile.needs_fft_fallback
    assert profile.is_accelerator


def test_env_flags_force_fallbacks(monkeypatch):
    _fake_cuda(monkeypatch, "Iluvatar BI-V150")
    assert not fft_fallback_required("cuda:0")
    monkeypatch.setenv("PYMSS_FFT_FALLBACK", "1")
    monkeypatch.setenv("PYMSS_RNN_FALLBACK", "1")
    clear_device_cache()
    assert fft_fallback_required("cuda:0")
    assert rnn_fallback_required("cuda:0")


def test_amp_flag_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PYMSS_AMP", "0")
    clear_device_cache()
    assert not amp_supported("cuda")


def test_cpu_and_mps_are_not_accelerators():
    assert not device_profile("cpu").is_accelerator
    assert not device_profile("mps").is_accelerator


def test_alias_tables_are_consistent():
    assert set(ALIAS_TO_VENDOR) == set(DEVICE_ALIASES)
    for alias in DEVICE_ALIASES:
        profile = profile_for_alias(alias)
        assert isinstance(profile, DeviceProfile)


def test_private_device_aliases_require_their_plugin():
    assert requires_plugin("npu") == "torch_npu"
    assert requires_plugin("gcu") == "torch_gcu"
    assert requires_plugin("mlu") == "torch_mlu"
    assert requires_plugin("biren") == "torch_br"
    assert requires_plugin("hygon") is None


def test_prepare_device_alias_for_plugin_free_vendors():
    assert prepare_device_alias("cpu") == "cpu"
    assert prepare_device_alias("hygon") == "cuda"


def test_prepare_device_alias_reports_missing_plugin():
    with pytest.raises(ImportError):
        prepare_device_alias("gcu")


def test_profile_is_cached_until_cleared(monkeypatch):
    _fake_cuda(monkeypatch, "MetaX C500")
    first = device_profile("cuda")
    assert device_profile("cuda:1") is first
    clear_device_cache()
    _fake_cuda(monkeypatch, "Hygon DCU Z100L")
    assert device_profile("cuda").vendor == HYGON


def test_biren_profile_matches_the_measured_birensupa_stack():
    """BIREN106M + torch-br 1.7.0.20700: no complex stft, no fused GRU cell."""
    profile = device_profile("supa")
    assert profile.vendor == BIREN
    assert profile.device_type == "supa"
    assert profile.complex_fft is False
    assert profile.rnn is True
    assert profile.blocked_rnn_ops == frozenset({"gru"})


def test_rnn_op_blocked_is_per_cell():
    assert rnn_op_blocked("supa", "gru") is True
    assert rnn_op_blocked("supa", "GRU") is True
    assert rnn_op_blocked("supa", "lstm") is False
    assert rnn_op_blocked("supa", "rnn") is False
    # A device level rnn=False blocks every cell.
    assert rnn_op_blocked("npu", "lstm") is True
    assert rnn_op_blocked("npu", "gru") is True
    # Vendors with native recurrent kernels report nothing blocked.
    assert rnn_op_blocked("cuda", "gru") is False
