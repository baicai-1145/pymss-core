import pytest
import torch

from pymss_core.npu_fft import hann_window_npu, istft_npu, stft_npu


@pytest.mark.parametrize("n_fft,hop,length", [(16, 4, 97), (16, 7, 97), (15, 4, 98), (512, 128, 1027), (1024, 320, 2051)])
@pytest.mark.parametrize("normalized", [False, True])
def test_centered_istft_preserves_unaligned_tail(n_fft, hop, length, normalized):
    torch.manual_seed(24)
    wave = torch.randn(2, length)
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(wave, n_fft, hop, window=window, normalized=normalized, return_complex=True)
    expected = torch.istft(spectrum, n_fft, hop, window=window, normalized=normalized, length=length)
    actual = istft_npu(spectrum, n_fft, hop, window=window, normalized=normalized, length=length)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_stft_and_istft_match_torch_without_an_explicit_length(center, normalized):
    torch.manual_seed(19)
    wave = torch.randn(2, 97)
    window = torch.ones(16)
    expected = torch.stft(wave, 16, 4, window=window, center=center, normalized=normalized, return_complex=True)
    actual = stft_npu(wave, 16, 4, window=window, center=center, normalized=normalized)
    torch.testing.assert_close(actual, expected)
    restored = istft_npu(actual, 16, 4, window=window, center=center, normalized=normalized)
    reference = torch.istft(expected, 16, 4, window=window, center=center, normalized=normalized)
    torch.testing.assert_close(restored, reference)


def test_hann_window_is_built_on_cpu_even_with_a_different_default_device():
    with torch.device("meta"):
        window = hann_window_npu(16, device="cpu")
    assert window.device.type == "cpu"
    torch.testing.assert_close(window, torch.hann_window(16))


@pytest.mark.parametrize("win_length", [8, 12])
def test_default_rectangular_window_supports_short_window_lengths(win_length):
    wave = torch.randn(2, 97)
    expected = torch.stft(wave, 16, 4, win_length, window=torch.ones(win_length), return_complex=True)
    actual = stft_npu(wave, 16, 4, win_length)
    torch.testing.assert_close(actual, expected)


def test_stft_rejects_unsupported_real_pair_layout():
    with pytest.raises(ValueError, match="return_complex"):
        stft_npu(torch.randn(97), 16, window=torch.ones(16), return_complex=False)


def test_demucs_npu_hann_is_created_on_cpu(monkeypatch):
    from types import SimpleNamespace
    from pymss_core.modules import demucs_local

    created = []
    result = object()
    device = type("NpuDevice", (), {"type": "npu", "index": 0})()
    ref = SimpleNamespace(device=device, dtype=torch.float32)

    def factory(*args, **kwargs):
        window = torch.hann_window(*args, **kwargs)
        created.append(window.device.type)
        return SimpleNamespace(to=lambda target: result)

    monkeypatch.setattr(demucs_local, "torch", SimpleNamespace(hann_window=factory))
    monkeypatch.setattr(demucs_local, "_HANN_CACHE", {})
    with torch.device("meta"):
        assert demucs_local._hann(16, ref) is result
        assert demucs_local._hann(16, ref) is result
    assert created == ["cpu"]
