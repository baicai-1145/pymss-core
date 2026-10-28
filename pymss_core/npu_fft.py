"""NPU-safe STFT/ISTFT built from validated primitives.

``torch.stft``/``torch.istft`` are unusable on current torch_npu builds for
this device family: the fp32 kernel fails with an aclnn parameter error and
the fp16 kernel faults the vector cores asynchronously. Spectral models route
through these compositions instead. Everything runs on-device in fp32:

- analysis:   pad -> unfold -> window -> rfft        (frames-major)
- synthesis:  irfft -> window -> index_add_ OLA      with window**2 envelope

Implementation notes:

- The unfold view is materialized with ``.contiguous()`` before the broadcast
  window multiply: multiplying a strided unfold view by a vector faults the
  vector cores (async device error 507035) on Ascend 950PR.
- ``hann_window`` must be created with an explicit float dtype before landing
  on the device; the default-dtype device-side path hits an int64/float cast
  bug in this torch_npu build.

Only the code paths exercised by pymss models are implemented: real input,
centered or uncentered, one-sided, complex spectrum, ``reflect``/``constant``
padding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from pymss_core.device import fft_fallback_required

_VALID_PAD_MODES = ("reflect", "constant")
_TINY = 1e-11


def hann_window_npu(window_length: int, *, device, dtype=torch.float32, periodic: bool = True) -> torch.Tensor:
    """``torch.hann_window`` that avoids the device-side default-dtype bug."""
    return torch.hann_window(window_length, periodic=periodic, dtype=dtype, device="cpu").to(device=device)

def _fit_window(window: torch.Tensor, n_fft: int, win_length: int, device, dtype) -> torch.Tensor:
    window = window.to(device=device, dtype=dtype)
    if win_length != n_fft:
        lpad = (n_fft - win_length) // 2
        window = F.pad(window, (lpad, n_fft - win_length - lpad))
    return window

def stft_npu(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = True,
    *,
    return_complex: bool = True,
) -> torch.Tensor:
    """NPU-safe ``torch.stft`` replacement for real inputs.

    Matches ``torch.stft`` output layout ``(…, freq, time)`` for the
    parameter combinations used by pymss models.
    """
    if not torch.is_floating_point(x):
        raise TypeError("stft_npu expects a floating point input")
    if not onesided:
        raise NotImplementedError("two-sided stft_npu is not needed by pymss models")
    if not return_complex:
        raise ValueError("stft_npu requires return_complex=True")
    if pad_mode not in _VALID_PAD_MODES:
        raise ValueError(f"stft_npu supports pad modes {_VALID_PAD_MODES}, got {pad_mode!r}")
    hop = n_fft // 4 if hop_length is None else hop_length
    win_length = n_fft if win_length is None else win_length
    if window is None:
        # torch.stft with no explicit window applies a rectangular (ones) window.
        window = torch.ones(win_length, dtype=x.dtype, device=x.device)
    window = _fit_window(window, n_fft, win_length, x.device, x.dtype)

    *lead, length = x.shape
    flat = x.reshape(-1, 1, length)
    if center:
        if pad_mode == "reflect":
            # aten.reflection_pad1d has no AscendIR converter (torchair graphs);
            # an equivalent flip+cat formulation keeps the whole forward graphable.
            lpad = min(n_fft // 2, flat.shape[-1] - 1)
            rpad = min(n_fft // 2, flat.shape[-1] - 1)
            flat = torch.cat([flat[..., 1:lpad+1].flip(-1), flat, flat[..., -rpad-1:-1].flip(-1)], dim=-1)
        else:
            flat = F.pad(flat, (n_fft // 2, n_fft // 2), mode=pad_mode)
    # aten.unfold has no AscendIR converter (torchair graphs); the identical
    # strided view is expressible through as_strided, which does convert.
    *lead_n, length_n = flat.shape
    n_frames = (length_n - n_fft) // hop + 1
    frames = flat.as_strided((*lead_n, n_frames, n_fft), (*flat.stride()[:-1], hop, 1)).contiguous() * window  # (N, 1, T, n_fft)
    # torch.stft(normalized=True) scales the plain FFT output by sqrt(1/n_fft).
    spec = torch.fft.rfft(frames.squeeze(1), dim=-1)  # (N, T, F)
    if normalized:
        spec = spec * (1.0 / n_fft) ** 0.5
    # torch_npu runs complex64 transposes on the AI_CPU (~20ms at this size);
    # transpose the real/imag planes on the vector cores instead (~0.04ms).
    spec = torch.view_as_complex(torch.view_as_real(spec).permute(0, 2, 1, 3).contiguous())  # (N, F, T)
    return spec.reshape(*lead, spec.shape[-2], spec.shape[-1])


def istft_npu(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = True,
    length: int | None = None,
    *,
    return_complex: bool = False,
) -> torch.Tensor:
    """NPU-safe ``torch.istft`` replacement for one-sided complex inputs."""
    if return_complex:
        raise ValueError("istft_npu reconstructs a real signal; return_complex is not supported")
    if not onesided:
        raise NotImplementedError("two-sided istft_npu is not needed by pymss models")
    hop = n_fft // 4 if hop_length is None else hop_length
    win_length = n_fft if win_length is None else win_length
    if window is None:
        # torch.istft with no explicit window assumes a rectangular one.
        window = torch.ones(win_length, dtype=torch.float32, device=spec.device)
    window = _fit_window(window, n_fft, win_length, spec.device, torch.float32)
    if not spec.is_complex():
        spec = torch.view_as_complex(spec.contiguous())
    if spec.dtype != torch.complex64:
        # torch_npu cannot copy between complex dtypes; rebuild from real parts.
        spec = torch.view_as_complex(torch.view_as_real(spec).float().contiguous())
    if normalized:
        spec = spec * (n_fft ** 0.5)

    *lead, freqs, n_frames = spec.shape
    flat_spec = spec.reshape(-1, freqs, n_frames)
    # torch_npu runs complex64 transposes on the AI_CPU; permute real/imag planes instead.
    flat_spec = torch.view_as_complex(torch.view_as_real(flat_spec).permute(0, 2, 1, 3).contiguous())  # (N, T, F)
    frames = torch.fft.irfft(flat_spec, n=n_fft, dim=-1)  # (N, T, n_fft) contiguous
    frames = frames * window
    batch = frames.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    idx = torch.arange(n_frames, device=spec.device)[:, None] * hop + torch.arange(n_fft, device=spec.device)[None, :]
    idx = idx.reshape(-1)
    out = frames.new_zeros(batch, out_len)
    out.index_add_(1, idx, frames.reshape(batch, -1))
    wsq = (window * window)[None, :].expand(n_frames, n_fft).reshape(-1)
    norm = frames.new_zeros(batch, out_len)
    norm.index_add_(1, idx, wsq[None, :].expand(batch, -1))
    out = out.reshape(*lead, out_len)[..., n_fft // 2 if center else 0 :]
    norm = norm.reshape(*lead, out_len)[..., n_fft // 2 if center else 0 :]
    if center and length is None:
        # An explicit length may include samples reconstructed from right padding.
        trim = min(n_fft // 2, out.shape[-1])
        out = out[..., : out.shape[-1] - trim]
        norm = norm[..., : norm.shape[-1] - trim]
    if length is not None:
        if length > out.shape[-1]:
            out = F.pad(out, (0, length - out.shape[-1]))
            norm = F.pad(norm, (0, length - norm.shape[-1]))
        else:
            out = out[..., :length]
            norm = norm[..., :length]
    return out / norm.clamp_min(_TINY)


def torch_stft_blocked(device) -> bool:
    """Return True when ``torch.stft``/``torch.istft`` must be replaced on this device.

    Delegates to the backend capability profile in :mod:`pymss_core.device`:
    Ascend lacks a usable complex STFT path, and any other accelerator can opt
    into the ``torch.fft`` composition while its stack is being validated
    (``PYMSS_FFT_FALLBACK=1``).
    """
    return fft_fallback_required(device)


__all__ = ["hann_window_npu", "istft_npu", "stft_npu", "torch_stft_blocked"]
