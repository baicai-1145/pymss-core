"""Tests for the dilated-conv CPU fallback."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import pymss_core.npu_conv as npu_conv


def test_undilated_conv_never_falls_back():
    conv = torch.nn.Conv2d(4, 6, 3, padding=1)
    assert npu_conv.needs_cpu_fallback(conv, torch.randn(1, 4, 8, 8)) is False


def test_dilated_conv_is_native_on_supported_devices():
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2))
    # Plain CPU tensors resolve to the CPU profile, which supports dilation.
    assert npu_conv.needs_cpu_fallback(conv, torch.randn(1, 4, 8, 8)) is False


def test_dilated_conv_falls_back_when_the_profile_blocks_it(monkeypatch):
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2))
    assert npu_conv.needs_cpu_fallback(conv, torch.randn(1, 4, 8, 8)) is True
    conv1d = torch.nn.Conv1d(4, 6, 3, padding=2, dilation=2)
    assert npu_conv.needs_cpu_fallback(conv1d, torch.randn(1, 4, 16)) is True
    flat = torch.nn.Conv1d(4, 6, 3, padding=1)
    assert npu_conv.needs_cpu_fallback(flat, torch.randn(1, 4, 16)) is False


def test_cpu_fallback_matches_the_reference(monkeypatch):
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2)).eval()
    x = torch.randn(2, 4, 21, 12)
    with torch.no_grad():
        expected = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        actual = npu_conv._conv_on_cpu(conv, x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_cpu_fallback_refuses_training(monkeypatch):
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2))
    with pytest.raises(RuntimeError, match="inference only"):
        npu_conv._conv_on_cpu(conv, torch.randn(1, 4, 8, 8, requires_grad=True))


def test_install_is_idempotent_and_keeps_the_original(monkeypatch):
    monkeypatch.setattr(npu_conv, "_installed", False)
    from torch.nn.modules.conv import _ConvNd

    original = _ConvNd.forward
    try:
        npu_conv.install_dilated_conv_fallback()
        patched = _ConvNd.forward
        npu_conv.install_dilated_conv_fallback()
        assert _ConvNd.forward is patched
        assert patched._pymss_conv_original_forward is original
    finally:
        _ConvNd.forward = original


def test_installed_fallback_runs_the_blocked_layer_on_cpu(monkeypatch):
    monkeypatch.setattr(npu_conv, "_installed", False)
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    from torch.nn.modules.conv import _ConvNd

    original = _ConvNd.forward
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2)).eval()
    x = torch.randn(2, 4, 21, 12)
    try:
        with torch.no_grad():
            expected = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        npu_conv.install_dilated_conv_fallback()
        with torch.no_grad():
            actual = conv(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    finally:
        _ConvNd.forward = original
