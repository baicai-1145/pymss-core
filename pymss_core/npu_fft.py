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

import math

import numpy as np

import torch
import torch.nn.functional as F

from pymss_core.device import fft_fallback_required

_VALID_PAD_MODES = ("reflect", "constant")
_TINY = 1e-11


def hann_window_npu(window_length: int, *, device, dtype=torch.float32, periodic: bool = True) -> torch.Tensor:
    """``torch.hann_window`` that avoids the device-side default-dtype bug."""
    return torch.hann_window(window_length, periodic=periodic, dtype=dtype, device="cpu").to(device=device)

#: Vendor FFT sizes that are trusted; everything else uses the exact DFT matmul.
_TRUSTED_FFT_MIN = 1024

_DFT_CACHE: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


_VENDOR_FFT_STATUS: dict[tuple[int, str], bool] = {}


def _probe_vendor_fft(n: int, device) -> bool:
    """Ask the vendor kernel to round-trip a known signal and check the answer."""
    try:
        x = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32).reshape(1, n)
        spec = torch.fft.rfft(x, dim=-1)
        planes = torch.view_as_real(spec)
        if not bool(torch.isfinite(planes).all()):
            return False
        if float(planes.abs().max()) == 0.0:
            return False
        back = torch.fft.irfft(spec, n=n, dim=-1)
        ref = x.cpu()
        return bool(torch.allclose(back.cpu(), ref, atol=1e-3, rtol=1e-3))
    except Exception:  # noqa: BLE001 - any vendor failure means "do not trust"
        return False


def vendor_fft_trusted(n: int, device=None) -> bool:
    """Return ``True`` when the vendor FFT can be used for size ``n``.

    BIREN's FFT is not reliable for every shape: size 1411 fails to build a plan
    ("sufft build plan failed") and the 9-bin spectrum of a 16-point STFT comes
    back as all zeros -- silently.  Large power-of-two sizes (2048 for the VR
    models) are proven correct, so those keep the fast path and everything else
    goes through an exact DFT matrix multiply.

    Args:
        n: Transform length.

    Args:
        n: Transform length.
        device: When given, a one-off round-trip probe decides whether this
            device really computes this size correctly (BIREN answered zeros
            for shapes the size alone could not predict).

    Returns:
        ``True`` when the vendor kernel is trusted for this size.
    """
    if n < _TRUSTED_FFT_MIN or (n & (n - 1)) != 0:
        return False
    if device is None:
        return True
    key = (n, str(device))
    cached = _VENDOR_FFT_STATUS.get(key)
    if cached is None:
        cached = _probe_vendor_fft(n, device)
        _VENDOR_FFT_STATUS[key] = cached
    return cached


def _degenerate(out: torch.Tensor, source: torch.Tensor) -> bool:
    """Return ``True`` when the vendor answered with a silent zero tensor.

    BIREN's FFT returns zeros for some shapes without raising (a 513-bin,
    1024-point inverse transform is one), so every vendor result is checked
    before it is trusted.  A zero result from a zero input is legitimate.

    Args:
        out: Vendor FFT result (real or complex).
        source: Its input.

    Returns:
        ``True`` when the answer looks like a silent failure.
    """
    out_planes = torch.view_as_real(out) if out.is_complex() else out
    if float(out_planes.abs().max()) != 0.0:
        return False
    src_planes = torch.view_as_real(source) if source.is_complex() else source
    return float(src_planes.abs().max()) != 0.0


def _dft_matrices(n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cached ``cos``/``sin`` DFT basis for length ``n``."""
    key = (n, str(device), dtype)
    cached = _DFT_CACHE.get(key)
    if cached is None:
        steps = torch.arange(n, device=device, dtype=torch.float64)
        bins = torch.arange(n // 2 + 1, device=device, dtype=torch.float64)
        angle = (2.0 * math.pi / n) * torch.outer(bins, steps)
        cached = (angle.cos().to(dtype), angle.sin().to(dtype))
        _DFT_CACHE[key] = cached
    return cached


def _rfft_matmul(x: torch.Tensor, n: int) -> torch.Tensor:
    """``rfft`` by DFT matrix multiply (real kernels only)."""
    cos, sin = _dft_matrices(n, x.device, x.dtype)
    lead = x.shape[:-1]
    flat = x.reshape(-1, n)
    real = flat @ cos.transpose(0, 1)
    imag = -(flat @ sin.transpose(0, 1))
    bins = cos.shape[0]
    planes = torch.stack((real, imag), dim=-1).reshape(*lead, bins, 2).contiguous()
    return torch.view_as_complex(planes)


def _irfft_matmul(spec: torch.Tensor, n: int) -> torch.Tensor:
    """``irfft`` by DFT matrix multiply (real kernels only)."""
    real_dtype = spec.real.dtype
    cos, sin = _dft_matrices(n, spec.device, real_dtype)
    bins = spec.shape[-1]
    weight = torch.full((bins,), 2.0, device=spec.device, dtype=real_dtype)
    weight[0] = 1.0
    if n % 2 == 0 and bins == n // 2 + 1:
        weight[-1] = 1.0
    lead = spec.shape[:-1]
    planes = torch.view_as_real(spec.reshape(-1, bins).contiguous())
    weighted_real = planes[..., 0] * weight
    weighted_imag = planes[..., 1] * weight
    out = (weighted_real @ cos - weighted_imag @ sin) / n
    return out.reshape(*lead, n)


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
    frames_flat = frames.squeeze(1)
    if vendor_fft_trusted(n_fft, frames_flat.device):
        spec = torch.fft.rfft(frames_flat, dim=-1)  # (N, T, F)
        if _degenerate(spec, frames_flat):
            spec = _rfft_matmul(frames_flat, n_fft)
    else:
        spec = _rfft_matmul(frames_flat, n_fft)
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
    if vendor_fft_trusted(n_fft, flat_spec.device):
        frames = torch.fft.irfft(flat_spec, n=n_fft, dim=-1)  # (N, T, n_fft)
        if _degenerate(frames, flat_spec):
            frames = _irfft_matmul(flat_spec, n_fft)
    else:
        frames = _irfft_matmul(flat_spec, n_fft)
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


def numpy_complex_to_device(spec: "np.ndarray", device) -> torch.Tensor:
    """Upload a complex numpy array through its real planes.

    The mirror image of :func:`complex_to_numpy_cpu`: BIREN rejects a complex
    host->device copy as well, with

        Unsupported BR type for ScalarType: 0

    so the real planes travel and the complex view is rebuilt on the device.

    Args:
        spec: Complex numpy array.
        device: Target torch device.

    Returns:
        A complex tensor on ``device``.
    """
    array = np.ascontiguousarray(spec)
    if not np.iscomplexobj(array):
        return torch.from_numpy(array).to(device)
    planes = np.stack((array.real, array.imag), axis=-1).astype(np.float32, copy=False)
    return torch.view_as_complex(torch.from_numpy(np.ascontiguousarray(planes)).to(device).contiguous())


def complex_to_numpy_cpu(spec: torch.Tensor) -> "np.ndarray":
    """Copy a complex tensor to the host through its real planes.

    BIREN cannot copy complex tensors device->host at all:

        RuntimeError: ERROR (BR TORCH INTERNAL) ... ErrorCode: 101, invalid device

    while copying ``view_as_real(spec)`` works, so the real planes travel and the
    complex view is rebuilt on the host.

    Args:
        spec: Real or complex tensor on any device.

    Returns:
        A numpy array with the same values.
    """
    if not spec.is_complex():
        return spec.detach().cpu().numpy()
    planes = torch.view_as_real(spec.detach()).contiguous().cpu()
    return torch.view_as_complex(planes).numpy()


def masked_spectrum(
    mask: torch.Tensor, spec: torch.Tensor, *, to_cpu: bool = False
) -> torch.Tensor | "np.ndarray":
    """Multiply a real mask into a complex spectrum.

    BIREN has no pointwise kernel for ``complex64 * float32``: it tries to cast
    the mask to complex64 on the device and fails with

        Typecast for BUFFER layout, src_dtype_FP32 and dst_dtype_COMPLEX64 is not implemented
        pointwise No SUPA Function found ... input_dtype = COMPLEX64

    so the product is formed on the real/imaginary planes with plain float
    kernels, which every backend supports.  Non-complex inputs take the direct
    path.

    Args:
        mask: Real mask, broadcastable against the spectrum's leading dims.
        spec: Real or complex spectrum.

    Returns:
        ``mask * spec`` with the spectrum's dtype.
    """
    if not spec.is_complex():
        product = mask * spec
        return complex_to_numpy_cpu(product) if to_cpu else product
    planes = torch.view_as_real(spec)  # (..., 2)
    for _ in range(planes.dim()):
        if mask.dim() >= planes.dim():
            break
        mask = mask.unsqueeze(-1)
    product = (mask * planes).contiguous()
    if to_cpu:
        # Never copy a complex tensor off the device (see complex_to_numpy_cpu).
        return torch.view_as_complex(product.cpu()).numpy()
    return torch.view_as_complex(product)


def torch_stft_blocked(device) -> bool:
    """Return True when ``torch.stft``/``torch.istft`` must be replaced on this device.

    Delegates to the backend capability profile in :mod:`pymss_core.device`:
    Ascend lacks a usable complex STFT path, and any other accelerator can opt
    into the ``torch.fft`` composition while its stack is being validated
    (``PYMSS_FFT_FALLBACK=1``).
    """
    return fft_fallback_required(device)


__all__ = ["hann_window_npu", "istft_npu", "stft_npu", "torch_stft_blocked", "vendor_fft_trusted"]
