"""Device-side shims for vendor kernels that are missing rather than slow.

Several BIREN models fail on the same two primitives, not on their own maths:

* ``complex64 * float32`` -- SUDNN/pointwise has no such kernel and the vendor
  fails while trying to cast the real operand to complex.  VR hit this first
  (``mask * spec``); htdemucs, scnet and bandit hit the identical wall.
* ``torch.hann_window(device=<vendor>)`` -- the window kernel crashes the device
  context; building on the host and copying is fine.

Patching them once in the backend is both smaller and safer than teaching ten
model files about vendor quirks, and both replacements keep the work on the
device: the complex product is formed on the real/imaginary planes, and only the
window is built on the host (a few hundred floats).
"""
from __future__ import annotations

import os
import logging
import warnings

import torch

#: Names of the shims installed for the current device, for logging/debugging.
_installed: set[str] = set()


def _is_number(value) -> bool:
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


def complex_planes_product(a, b) -> torch.Tensor:
    """Multiply with at least one complex operand using only real kernels.

    Args:
        a: Real or complex tensor.
        b: Real or complex tensor, broadcastable against ``a``.

    Returns:
        ``a * b`` with the complex dtype.
    """
    if _is_number(b):
        # Scalar operands hit the same missing kernel (``spec * (1/n)**0.5`` in
        # every normalized STFT), so they are folded into the real planes.
        planes = torch.view_as_real(a)
        if isinstance(b, complex):
            real = planes[..., 0] * b.real - planes[..., 1] * b.imag
            imag = planes[..., 0] * b.imag + planes[..., 1] * b.real
            return torch.view_as_complex(torch.stack((real, imag), dim=-1).contiguous())
        return torch.view_as_complex((planes * b).contiguous())
    if _is_number(a):
        return complex_planes_product(b, a)
    if a.is_complex() and b.is_complex():
        ar, ai = a.real, a.imag
        br, bi = b.real, b.imag
        real = ar * br - ai * bi
        imag = ar * bi + ai * br
        return torch.view_as_complex(torch.stack((real, imag), dim=-1).contiguous())
    complex_operand, real_operand = (a, b) if a.is_complex() else (b, a)
    planes = torch.view_as_real(complex_operand)
    operand = real_operand
    for _ in range(planes.dim()):
        if operand.dim() >= planes.dim():
            break
        operand = operand.unsqueeze(-1)
    return torch.view_as_complex((operand * planes).contiguous())


def _needs_plane_product(a, b) -> bool:
    """Return ``True`` when ``a * b`` needs the real-plane kernel."""
    if _is_number(a) and _is_number(b):
        return False
    if _is_number(a):
        # (scalar, tensor) -- normalise to (tensor, scalar)
        a, b = b, a
    if not isinstance(a, torch.Tensor) or not a.is_complex() or a.dtype != torch.complex64:
        return False
    if _is_number(b):
        return True
    if not isinstance(b, torch.Tensor):
        return False
    if b.is_complex():
        return b.dtype == torch.complex64
    return torch.is_floating_point(b)


def install_complex_mul_shim() -> bool:
    """Route ``complex * real`` through the real planes.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "complex_mul" in _installed:
        return False
    original_mul = torch.Tensor.__mul__
    original_rmul = torch.Tensor.__rmul__
    torch_mul = torch.mul

    def _on_device(operand) -> bool:
        device = getattr(operand, "device", None)
        return device is not None and device.type not in ("cpu", "meta")

    def mul(self, other):
        if _on_device(self) and _needs_plane_product(self, other):
            return complex_planes_product(self, other)
        return original_mul(self, other)

    def rmul(self, other):
        if _on_device(self) and _needs_plane_product(other, self):
            return complex_planes_product(other, self)
        return original_rmul(self, other)

    def mul_fn(a, b, *args, **kwargs):
        if not args and not kwargs and (_on_device(a) or _on_device(b)) and _needs_plane_product(a, b):
            return complex_planes_product(a, b)
        return torch_mul(a, b, *args, **kwargs)

    torch.Tensor.__mul__ = mul
    torch.Tensor.__rmul__ = rmul
    torch.mul = mul_fn
    _installed.add("complex_mul")
    return True


def install_window_shim() -> bool:
    """Build windows on the host and copy them to the device.

    ``torch.hann_window(device="supa")`` crashes the device context
    (``SUPA Kernel Launch Error, invalid parameter``); the vendor has no window
    kernel.  A few hundred floats crossing the bus is free next to that.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "window" in _installed:
        return False
    originals = {name: getattr(torch, name) for name in ("hann_window", "hamming_window", "blackman_window", "bartlett_window")}

    def make(name, original):
        def wrapper(*args, **kwargs):
            device = kwargs.get("device")
            if device is None:
                return original(*args, **kwargs)
            target = torch.device(device)
            if target.type == "cpu":
                return original(*args, **kwargs)
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
            return original(*args, **kwargs).to(target)

        wrapper.__name__ = name
        wrapper._vendor_shim = True
        return wrapper

    for name, original in originals.items():
        setattr(torch, name, make(name, original))
    _installed.add("window")
    return True


def glu_equivalent(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return ``F.glu(x, dim)`` using only elementwise kernels.

    ``a * sigmoid(b)`` is exactly what GLU computes; the fused kernel is missing
    on the SUBLAS path (``no valid backend was found for op: glu``).

    Args:
        x: Input tensor whose ``dim`` splits into two halves.
        dim: Dimension to split on.

    Returns:
        The gated tensor.
    """
    a, b = x.chunk(2, dim=dim)
    return a * torch.sigmoid(b)


def install_glu_shim() -> bool:
    """Replace ``F.glu`` with the elementwise equivalent.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "glu" in _installed:
        return False
    import torch.nn.functional as F

    original = F.glu

    def glu(input, dim=-1):
        if getattr(input, "device", torch.device("cpu")).type in ("cpu", "meta"):
            return original(input, dim)
        return glu_equivalent(input, dim)

    glu._vendor_shim = True
    F.glu = glu
    _installed.add("glu")
    return True


def scatter_add_equivalent(src: torch.Tensor, index: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Accumulate ``src`` into ``size`` slots of a new tensor, without scatter.

    ``scatter_add_`` has no kernel on the SUBLAS path
    (``no valid backend was found for op: ScatterAdd``).  Mel-band models use it
    to fold band outputs back onto the frequency axis, which is a one-hot
    matrix multiply: exact, and built from ops the vendor does implement.

    Args:
        src: Source tensor, shape ``(..., index.numel(), ..., C)`` along ``dim``.
        index: 1-D integer tensor of target slot per source entry.
        dim: Dimension of ``src``/the result that the scatter runs along.
        size: Number of slots in the result.

    Returns:
        The accumulated tensor with ``size`` slots along ``dim``.
    """
    slots = torch.arange(size, device=src.device, dtype=index.dtype)
    one_hot = (index.unsqueeze(0) == slots.unsqueeze(1)).to(src.dtype)  # (size, numel)
    moved = src.movedim(dim, -2) if dim >= 0 else src.movedim(dim, -2)
    leading = moved.shape[:-2]
    flat = moved.reshape(-1, moved.shape[-2], moved.shape[-1])
    out = torch.matmul(one_hot, flat)  # (size, numel) x (N, numel, C) -> broadcast per N
    out = out.reshape(*leading, size, moved.shape[-1])
    if dim >= 0:
        return out.movedim(-2, dim)
    return out.movedim(-2, dim)


def complex_reshape(x: torch.Tensor, *shape) -> torch.Tensor:
    """Reshape a complex tensor through its real planes.

    BIREN fails on ``complex.reshape`` when the reshape needs a copy:

        [SUDNN] Failed to calculate matrix size, since date type 13 is not supported
        SUPASysCtl.cpp:161 dev -1 is not a device

    (dtype 13 is complex64, and the failure also resets the current device).  The
    real planes reshape with the plain kernel and the complex view is rebuilt.

    Args:
        x: Complex tensor.
        *shape: Target shape, without the trailing plane dimension.

    Returns:
        The reshaped complex tensor.
    """
    planes = torch.view_as_real(x)
    return torch.view_as_complex(planes.reshape(*shape, 2).contiguous())


def install_complex_reshape_shim() -> bool:
    """Route ``Tensor.reshape``/``Tensor.view`` for complex operands via the planes.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "complex_reshape" in _installed:
        return False
    original_reshape = torch.Tensor.reshape
    original_view = torch.Tensor.view

    def _on_device(tensor) -> bool:
        return tensor.device.type not in ("cpu", "meta")

    def reshape(self, *shape):
        if self.is_complex() and self.dtype == torch.complex64 and _on_device(self):
            return complex_reshape(self, *shape)
        return original_reshape(self, *shape)

    def view(self, *shape):
        if self.is_complex() and self.dtype == torch.complex64 and _on_device(self):
            return complex_reshape(self, *shape)
        return original_view(self, *shape)

    torch.Tensor.reshape = reshape
    torch.Tensor.view = view
    _installed.add("complex_reshape")
    return True


def group_norm_equivalent(x: torch.Tensor, num_groups: int, weight=None, bias=None, eps: float = 1e-5) -> torch.Tensor:
    """``F.group_norm`` built from reshape/mean/var (exact, all device kernels).

    GroupNorm.cpp on BIREN rejects some channel counts with
    ``SUPA Kernel Launch Error, ErrorCode: 801, operation not supported``.

    Args:
        x: Input tensor ``(N, C, ...)``.
        num_groups: Number of groups.
        weight: Optional per-channel scale.
        bias: Optional per-channel shift.
        eps: Numerical floor for the variance.

    Returns:
        The normalised tensor.
    """
    n = x.shape[0]
    flat = x.reshape(n, num_groups, -1)
    mean = flat.mean(dim=-1, keepdim=True)
    var = flat.var(dim=-1, unbiased=False, keepdim=True)
    out = ((flat - mean) / torch.sqrt(var + eps)).reshape(x.shape)
    if weight is not None:
        shape = (1, -1) + (1,) * (x.dim() - 2)
        out = out * weight.reshape(shape)
        if bias is not None:
            out = out + bias.reshape(shape)
    return out


def install_group_norm_shim() -> bool:
    """Replace ``F.group_norm`` on vendor devices.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "group_norm" in _installed:
        return False
    import torch.nn.functional as F

    original = F.group_norm

    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        device = getattr(input, "device", None)
        if device is None or device.type in ("cpu", "meta"):
            return original(input, num_groups, weight, bias, eps)
        return group_norm_equivalent(input, num_groups, weight, bias, eps)

    group_norm._vendor_shim = True
    F.group_norm = group_norm
    _installed.add("group_norm")
    return True


#: Interpolation modes with no bfloat16 kernel on BIREN.
_BF16_LESS_UPSAMPLE_MODES = frozenset({"bilinear", "bicubic", "linear", "trilinear"})


def install_interpolate_shim() -> bool:
    """Run bf16 interpolation in fp32 on the device.

    ``F.interpolate(..., mode="bilinear")`` on bfloat16 fails with

        ERROR (BR TORCH INTERNAL) ... no valid backend was found for op:
        UpsampleBilinear, ErrorCode: 2

    which used to make us disable autocast for the whole separator model.  The
    operator is linear, so computing it in fp32 and casting back is exact to
    within bf16 rounding, and it keeps everything on the device -- the only cost
    is one cast pair on a memory-bound op at the end of the network.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "interpolate" in _installed:
        return False
    import torch.nn.functional as F

    original = F.interpolate

    def interpolate(input, *args, **kwargs):
        mode = kwargs.get("mode", "nearest")
        device = getattr(input, "device", None)
        needs_fp32 = (
            isinstance(mode, str)
            and mode in _BF16_LESS_UPSAMPLE_MODES
            and input.dtype == torch.bfloat16
            and device is not None
            and device.type not in ("cpu", "meta")
        )
        if not needs_fp32:
            return original(input, *args, **kwargs)
        return original(input.float(), *args, **kwargs).to(torch.bfloat16)

    interpolate._vendor_shim = True
    F.interpolate = interpolate
    _installed.add("interpolate")
    return True


def conv1d_grouped_equivalent(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    """``F.conv1d`` with ``groups`` run as one conv per group (exact).

    BIREN rejects grouped conv1d whenever ``1 < groups < in_channels``:

        Conv1d.cpp:120 SUPA Kernel Launch Error, ErrorCode: 801

    while plain (``groups=1``) and depthwise (``groups == in_channels``) work, so
    the grouped case is split per group and concatenated.

    Args:
        input: Input tensor ``(N, C_in, L)``.
        weight: Kernel ``(C_out, C_in / groups, k)``.
        bias: Optional ``(C_out,)`` bias.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        groups: Number of groups.

    Returns:
        The concatenated per-group results.
    """
    import torch.nn.functional as F

    in_per_group = input.shape[1] // groups
    out_per_group = weight.shape[0] // groups
    outputs = []
    for index in range(groups):
        in_slice = slice(index * in_per_group, (index + 1) * in_per_group)
        out_slice = slice(index * out_per_group, (index + 1) * out_per_group)
        outputs.append(
            F.conv1d(
                input[:, in_slice],
                weight[out_slice],
                None if bias is None else bias[out_slice],
                stride,
                padding,
                dilation,
                1,
            )
        )
    return torch.cat(outputs, dim=1)


def install_conv1d_shim() -> bool:
    """Split grouped ``F.conv1d`` calls on vendor devices.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "conv1d" in _installed:
        return False
    import torch.nn.functional as F

    original = F.conv1d

    def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        in_channels = input.shape[1] if input.dim() > 1 else 0
        device = getattr(input, "device", None)
        grouped = 1 < groups < in_channels
        if not grouped or device is None or device.type in ("cpu", "meta"):
            return original(input, weight, bias, stride, padding, dilation, groups)
        return conv1d_grouped_equivalent(input, weight, bias, stride, padding, dilation, groups)

    conv1d._vendor_shim = True
    F.conv1d = conv1d
    _installed.add("conv1d")
    return True


def install_complex_cat_shim() -> bool:
    """Concatenate complex tensors through their real planes.

    ``torch.cat`` of complex tensors reaches SUDNN on BIREN and dies inside
    Concat.cpp with ``ErrorCode: 11``.  Concatenating the real planes and
    rebuilding the complex view is exact and uses the kernel that works.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "complex_cat" in _installed:
        return False

    original = torch.cat

    def cat(tensors, dim=0, *args, **kwargs):
        seq = list(tensors) if not isinstance(tensors, torch.Tensor) else [tensors]
        if len(seq) > 1 and all(isinstance(t, torch.Tensor) for t in seq):
            device = seq[0].device
            all_complex = all(t.is_complex() for t in seq)
            if all_complex and device.type not in ("cpu", "meta"):
                planes = original([torch.view_as_real(t.contiguous()) for t in seq], dim, *args, **kwargs)
                return torch.view_as_complex(planes.contiguous())
        return original(tensors, dim, *args, **kwargs)

    cat._vendor_shim = True
    torch.cat = cat
    _installed.add("complex_cat")
    return True


def install_fft_shim() -> bool:
    """Route ``torch.fft.rfft``/``irfft`` through the exact DFT implementation.

    The vendor documents ``torch.fft.*`` as unsupported, and the parts that do run
    are not dependable: some shapes answer zeros without raising, 4096-point
    transforms spend minutes in the planner, and a 4-D ``rfft`` from scnet fails
    outright with "Failed to find supa function".  The models call
    ``torch.fft.rfft`` with a ``dim`` and a ``norm``, so the shim reproduces those
    two arguments exactly on top of the matrix implementation.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "fft" in _installed:
        return False
    from pymss_core.npu_fft import irfft_exact, rfft_exact

    original_rfft = torch.fft.rfft
    original_irfft = torch.fft.irfft

    def _vendor(tensor) -> bool:
        device = getattr(tensor, "device", None)
        return device is not None and device.type not in ("cpu", "meta")

    def rfft(input, n=None, dim=-1, norm=None, *, out=None):
        if not _vendor(input) or input.is_complex():
            return original_rfft(input, n, dim, norm, out=out) if out is not None else original_rfft(input, n, dim, norm)
        return rfft_exact(input, n, dim, norm)

    def irfft(input, n=None, dim=-1, norm=None, *, out=None):
        if not _vendor(input) or not input.is_complex():
            return original_irfft(input, n, dim, norm, out=out) if out is not None else original_irfft(input, n, dim, norm)
        return irfft_exact(input, n, dim, norm)

    rfft._vendor_shim = True
    irfft._vendor_shim = True
    torch.fft.rfft = rfft
    torch.fft.irfft = irfft
    _installed.add("fft")
    return True


def std_equivalent(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """``Tensor.std`` rebuilt from ``var``.

    BIREN's ``std`` fails with ``SUPA Kernel Launch Error`` inside std.cpp for
    shapes that legacy_v3_demucs uses:

        std.cpp:38 SUPA Kernel Launch Error

    The variance kernel works, so the standard deviation is its square root.
    ``dim``/``keepdim``/``unbiased``/``correction`` are forwarded unchanged.

    Args:
        x: Input tensor.
        *args: Positional arguments of ``Tensor.std``.
        **kwargs: Keyword arguments of ``Tensor.std``.

    Returns:
        The standard deviation.
    """
    dim = kwargs.pop("dim", args[0] if args else None)
    if len(args) > 1:
        unbiased = args[1]
    else:
        unbiased = kwargs.pop("unbiased", True)
    keepdim = kwargs.pop("keepdim", args[2] if len(args) > 2 else False)
    correction = kwargs.pop("correction", None)
    kwargs.pop("out", None)
    if correction is not None:
        var = torch.var(x, dim=dim, correction=correction, keepdim=keepdim)
    else:
        var = torch.var(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
    return torch.sqrt(var.clamp_min(0))


def sdpa_equivalent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """``scaled_dot_product_attention`` as explicit matmuls.

    This remains the default vendor fallback. Earlier planner failures were
    observed with SDK compiler paths removed from PYTHONPATH; they do not establish
    a hardware or JIT limitation. The explicit suDNN FMHA backend is separate.

    Args:
        query, key, value: ``(..., T, E)`` tensors.
        attn_mask: Optional additive mask.
        dropout_p: Dropout probability (0.0 in inference).
        is_causal: Whether to apply a causal mask.
        scale: Override for ``1/sqrt(E)``.
        enable_gqa: Accepted for signature compatibility; GQA inputs are expected
            to be expanded by the caller.

    Returns:
        The attention output.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    # The model hands over tensors produced by the rotary complex multiply, whose
    # strides make SUDNN fail to compile the GE kernel ("Got GE error", "build graph
    # failed").  The identical bmm succeeds from a plain layout, so normalise first.
    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    # Fold the leading batch/head dimensions into contiguous 3-D operands.
    lead = query.shape[:-2]
    length, dim = query.shape[-2], query.shape[-1]
    flat_q = query.reshape(-1, length, dim).contiguous()
    flat_k = key.reshape(-1, length, dim).contiguous()
    flat_v = value.reshape(-1, length, dim).contiguous()
    # Historical end-to-end measurements favored 1024 over 256. Those runs
    # excluded SDK Python paths, so their shape failures are not backend limits.
    # Frequency attention folds 862 frames * 8 heads = 6896 batches (7 chunks).
    # Keep the chunk configurable; 0 disables it.
    chunk = int(os.environ.get("PYMSS_SDPA_BATCH_CHUNK", "1024"))
    if chunk and flat_q.shape[0] > chunk:
        # Inside a whole model the vendor refuses even batches that a standalone
        # bmm handles, so the batch can be walked in slices.
        # Fold the scale into q: q is 110 MB for the shapes a production roformer uses,
        # while the score tensor is 1.47 GB, so scaling after the bmm costs a full extra
        # read and write of the scores (about 5 ms per attention here, 120 ms per run).
        if scale != 1.0:
            flat_q = flat_q * scale
        # A strided k^T runs the GEMM at 157 GB/s on this device and a contiguous one
        # at 529 GB/s (10.8 ms against 3.2 ms for the score tensor of a production
        # roformer), so it is materialised once, outside the loop.
        kt = flat_k.transpose(-1, -2).contiguous()
        pieces = []
        for start in range(0, flat_q.shape[0], chunk):
            stop = min(start + chunk, flat_q.shape[0])
            piece = torch.bmm(flat_q[start:stop], kt[start:stop])
            pieces.append(piece)
        # A single block is the common case (496 folded rows against a 1024 block) and
        # cat would copy the whole score tensor -- 1.5 GB here -- for nothing.
        scores = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if attn_mask is not None:
            scores = scores + attn_mask.reshape(scores.shape[-2:])
        elif is_causal:
            mask = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        values = []
        for start in range(0, flat_v.shape[0], chunk):
            stop = min(start + chunk, flat_v.shape[0])
            values.append(torch.bmm(probs[start:stop], flat_v[start:stop]))
        out = values[0] if len(values) == 1 else torch.cat(values, dim=0)
        return out.reshape(*lead, length, dim)
    scores = torch.bmm(flat_q * scale, flat_k.transpose(-1, -2).contiguous())
    if attn_mask is not None:
        scores = scores + attn_mask.reshape(scores.shape[-2:])
    elif is_causal:
        mask = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        probs = torch.nn.functional.dropout(probs, dropout_p)
    out = torch.bmm(probs, flat_v)
    return out.reshape(*lead, length, dim)


def sudnn_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               *, scale: float | None = None) -> torch.Tensor:
    """Biren's explicit FMHA entry, validated for unmasked bf16 self-attention.

    Unlike F.sdpa (which selects the math implementation in this SDK), this calls
    the vendor operator directly. Its descriptor accepts 3-D operands, so fold
    batch and heads and restore them on output. SDK PYTHONPATH must be preserved
    for the JIT's br_generator module. Errors propagate rather than silently
    reverting to the math path.
    """
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Biren suDNN backend requires bf16 query, key and value")
    if query.ndim < 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Biren suDNN backend requires matching self-attention shapes")
    length, dim = query.shape[-2:]
    operands = [tensor.reshape(-1, length, dim).contiguous() for tensor in (query, key, value)]
    scale = dim ** -0.5 if scale is None else scale
    output = torch.ops.supa.sudnn_scaled_dot_product_attention_fwd(
        *operands, None, 0.0, False, scale, "FMHA", None, None)[0]
    return output.reshape(query.shape)


def install_sdpa_shim() -> bool:
    """Replace ``F.scaled_dot_product_attention`` on vendor devices.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "sdpa" in _installed:
        return False
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention

    backend = os.environ.get("PYMSS_SDPA_BACKEND", "manual")
    if backend not in ("manual", "sudnn"):
        raise ValueError("PYMSS_SDPA_BACKEND must be manual or sudnn")
    if "PYMSS_SDPA_VENDOR_CHUNK" in os.environ:
        warnings.warn("PYMSS_SDPA_VENDOR_CHUNK is obsolete; select PYMSS_SDPA_BACKEND=manual or sudnn",
                      FutureWarning, stacklevel=2)

    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, *, scale=None, enable_gqa=False):
        if query.device.type in ("cpu", "meta") or query.is_cuda:
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if backend == "sudnn" and query.device.type == "supa":
            if attn_mask is not None or is_causal or dropout_p != 0.0 or enable_gqa:
                raise NotImplementedError("Biren suDNN backend supports unmasked inference self-attention")
            return sudnn_sdpa(query, key, value, scale=scale)
        return sdpa_equivalent(query, key, value, attn_mask, dropout_p, is_causal,
                               scale=scale, enable_gqa=enable_gqa)

    scaled_dot_product_attention._vendor_shim = True
    scaled_dot_product_attention._pymss_sdpa_backend = backend
    logging.getLogger(__name__).info("Installed vendor SDPA backend: %s (sudnn applies to SUPA only)", backend)
    functional.scaled_dot_product_attention = scaled_dot_product_attention
    torch.scaled_dot_product_attention = scaled_dot_product_attention
    _installed.add("sdpa")
    return True


#: supa_rope_apply launch contract, measured with pymss/tools/vendor_rope_shape_sweep2.py
#: and vendor_rope_seq_accuracy.py on BIREN106M / torch-br 1.7.0.20700:
#: * dim0 (batch*heads) must be <= 1024 -- larger launches fail or hang, so the
#:   rotate is chunked along dim0 (dim0 slices of a contiguous tensor are free).
#: * x must be bf16 (batch*heads, seq, head_dim) in the op's SPLIT-HALF pairing:
#:   it rotates component i with component i+head_dim/2, while BS-Roformer q/k
#:   channels are interleaved pairs (2i, 2i+1).  prepare_model_for_vendor_rope()
#:   permutes to_qkv output rows per head into [evens | odds] order so the
#:   projection itself emits the op's pairing at zero cost (measured: folding
#:   the unpack into the weights is what makes the vendor path a win -- a runtime
#:   cat-unpack regressed below the original path).
#: * tables must be fp32 (bf16 is rejected by schema) shaped (1, seq, head_dim)
#:   as concat([half, half], -1); x must be bf16, seq up to 4096 measured fine.
#: * the op output keeps the halves layout, which is exactly the [even | odd]
#:   layout apply_rotary_emb_fast returns, so downstream code is unchanged.
#: End-to-end (10 s stereo, dim=384, depth=12): 1858 -> 1765 ms (-5.0%) with
#: relative-to-fp32 1.5705% vs the original path's 1.5714% (probe logs
#: rope-v3/v5 rounds, .tmp/br-rope-model-ab.json).
_ROPE_MAX_BATCH = 1024
_rope_row_index_cache: dict[tuple[int, int], torch.Tensor] = {}


def _supa_rope_apply(x, cos, sin):
    """Call the vendor RoPE op (standalone so tests can substitute it)."""
    return torch.ops.supa.supa_rope_apply(x, cos, sin)


def _rope_row_index(heads: int, dim_head: int) -> torch.Tensor:
    """Permutation putting each head's q/k rows in [evens | odds] channel order."""
    key = (heads, dim_head)
    if key not in _rope_row_index_cache:
        evens_odds = list(range(0, dim_head, 2)) + list(range(1, dim_head, 2))
        index = (
            [head * dim_head + j for head in range(heads) for j in evens_odds]
            + [heads * dim_head + head * dim_head + j for head in range(heads) for j in evens_odds]
            + list(range(2 * heads * dim_head, 3 * heads * dim_head))
        )
        _rope_row_index_cache[key] = torch.tensor(index, dtype=torch.long)
    return _rope_row_index_cache[key]


def _rope_chunked(x, cos, sin, batch: int):
    """Apply the vendor op, splitting along dim0 to honor the 1024-row launch cap."""
    if batch <= _ROPE_MAX_BATCH:
        return _supa_rope_apply(x, cos, sin)
    chunks = -(-batch // _ROPE_MAX_BATCH)
    step = -(-batch // chunks)
    return torch.cat([_supa_rope_apply(piece, cos, sin) for piece in x.split(step, dim=0)], dim=0)


def _vendor_rope_rotate(rotary_embed, q, k):
    """Vendor RoPE for already-prepared models; mirrors rotate_qk_fast_bnhd."""
    from pymss_core.modules.bs_roformer import transformer as _transformer

    b, s, h, d = q.shape
    batch = b * h
    cos, sin = _transformer.cached_rotary_cos_sin(rotary_embed, s, q.device, torch.float32)
    table_cos = torch.cat((cos, cos), dim=-1).reshape(1, s, d).contiguous()
    table_sin = torch.cat((sin, sin), dim=-1).reshape(1, s, d).contiguous()
    outputs = []
    for tensor in (q, k):
        flat = tensor.permute(0, 2, 1, 3).reshape(batch, s, d)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        outputs.append(_rope_chunked(flat, table_cos, table_sin, batch).view(b, h, s, d).transpose(1, 2))
    return outputs[0], outputs[1]


def prepare_model_for_vendor_rope(model) -> list[str]:
    """Rearrange ``to_qkv`` rows so the vendor RoPE op sees its split-half pairing.

    Must run AFTER loading a raw checkpoint and BEFORE the first forward.  A
    ``load_state_dict`` post-hook re-permutes fresh (raw) checkpoint loads, so
    reloading after preparation stays correct; loading an ALREADY permuted
    state dict is not supported.  Models that never call this keep the original
    rotary path -- the installed shim only accelerates rotary embeddings marked
    here, so skipping the call is always numerically safe.

    Args:
        model: A model containing ``Attention`` modules (``to_qkv`` + rotary).

    Returns:
        The names of the ``to_qkv`` modules that were permuted.
    """
    prepared = []

    def permute_once(name, attention):
        heads = getattr(attention, "heads", None)
        linear = getattr(attention, "to_qkv", None)
        rotary = getattr(attention, "rotary_embed", None)
        if heads is None or linear is None or rotary is None:
            return
        if getattr(linear, "_pymss_rope_permuted", False):
            prepared.append(name + ".to_qkv")
            return
        out_features = linear.out_features
        if out_features % (3 * heads) != 0:
            raise ValueError(f"{name}: to_qkv out_features {out_features} not divisible by 3*heads={3 * heads}")
        dim_head = out_features // (3 * heads)
        if dim_head % 2 != 0:
            raise ValueError(f"{name}: odd head dim {dim_head} cannot be split into pairs")
        index = _rope_row_index(heads, dim_head).to(linear.weight.device)
        with torch.no_grad():
            linear.weight.data = linear.weight.data[index]
            if linear.bias is not None:
                linear.bias.data = linear.bias.data[index]
        linear._pymss_rope_permuted = True
        rotary._pymss_vendor_rope = True
        prepared.append(name + ".to_qkv")

    for name, module in model.named_modules():
        if hasattr(module, "to_qkv") and hasattr(module, "rotary_embed"):
            permute_once(name, module)

    if prepared:
        def _repermute(module, incompatible_keys):
            for name, sub in module.named_modules():
                linear = getattr(sub, "to_qkv", None)
                if linear is None or not getattr(linear, "_pymss_rope_permuted", False):
                    continue
                heads = sub.heads
                dim_head = linear.out_features // (3 * heads)
                index = _rope_row_index(heads, dim_head).to(linear.weight.device)
                with torch.no_grad():
                    linear.weight.data = linear.weight.data[index]
                    if linear.bias is not None:
                        linear.bias.data = linear.bias.data[index]

        model.register_load_state_dict_post_hook(_repermute)
    return prepared


def _vendor_rope_rotate_trio(rotary_embed, trio):
    """Vendor RoPE for the regrouped (3, B, H, S, D) projection tensor.

    q and k are the first two slices of one contiguous allocation, so they can
    be rotated in a single flattened view (2*B*H, S, D) -- one launch (or half
    as many chunks) instead of two separate calls, with no extra copies.
    """
    from pymss_core.modules.bs_roformer import transformer as _transformer

    _, b, h, s, d = trio.shape
    cos, sin = _transformer.cached_rotary_cos_sin(rotary_embed, s, trio.device, torch.float32)
    table_cos = torch.cat((cos, cos), dim=-1).reshape(1, s, d).contiguous()
    table_sin = torch.cat((sin, sin), dim=-1).reshape(1, s, d).contiguous()
    flat = trio[:2].reshape(2 * b * h, s, d)
    out = _rope_chunked(flat, table_cos, table_sin, 2 * b * h)
    out = out.view(2, b, h, s, d)
    return out[0], out[1]


def _vendor_rope_rotate_bhqd(rotary_embed, q, k):
    """Vendor RoPE on already-regrouped (B, H, S, D) contiguous tensors.

    Used by the Attention fast path: the qkv projection is regrouped once into
    (3, B, H, S, D), so q/k reshape to (B*H, S, D) for free and the op output
    views back without any further copies.  Measured bit-identical to the
    original forward and -89.5 ms end-to-end (probe: br-attention-chain.json).
    """
    from pymss_core.modules.bs_roformer import transformer as _transformer

    b, h, s, d = q.shape
    cos, sin = _transformer.cached_rotary_cos_sin(rotary_embed, s, q.device, torch.float32)
    table_cos = torch.cat((cos, cos), dim=-1).reshape(1, s, d).contiguous()
    table_sin = torch.cat((sin, sin), dim=-1).reshape(1, s, d).contiguous()
    outputs = []
    for tensor in (q, k):
        outputs.append(_rope_chunked(tensor.reshape(b * h, s, d), table_cos, table_sin, b * h).view(b, h, s, d))
    return outputs[0], outputs[1]


_supa_fused_ffn_fwd = None
_FFN_BACKEND = os.environ.get('PYMSS_FFN_BACKEND', 'vendor')


def _ffn_fast_path_ok(module) -> bool:
    """Structural gate: Sequential(RMSNorm, Linear, GELU, Dropout, Linear, Dropout)."""
    import torch.nn as nn
    from pymss_core.modules.bs_roformer.transformer import RMSNorm

    net = module.net
    if len(net) != 6:
        return False
    # Sequential(RMSNorm, Linear, GELU, Dropout, Linear, Dropout)
    return (isinstance(net[0], RMSNorm)
            and isinstance(net[1], nn.Linear) and isinstance(net[4], nn.Linear)
            and isinstance(net[2], nn.GELU)
            and isinstance(net[3], nn.Dropout) and isinstance(net[5], nn.Dropout)
            and net[1].bias is not None and net[4].bias is not None
            and float(net[3].p) == 0.0 and float(net[5].p) == 0.0)


def _vendor_fused_ffn(module, normalized):
    """Fused Linear->GELU->Linear via supa_fused_mlp_linear_act_linear_fwd.

    act_mode='gelu' (lowercase only -- other spellings silently disable the
    activation and corrupt the output; validated rel 0.41% vs eager bf16).
    Returns output only (op returns [output, ...]).
    """
    first, second = module.net[1], module.net[4]
    outputs = _supa_fused_ffn_fwd(normalized, first.weight, second.weight,
                                  first.bias, second.bias, act_mode='gelu')
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def install_rope_shim() -> bool:
    """Route prepared rotary embeddings through the vendor RoPE kernel.

    Non-SUPA devices and rotary embeddings that were never passed through
    :func:`prepare_model_for_vendor_rope` keep ``rotate_qk_fast_bnhd`` untouched,
    so installing the shim alone never changes numerics.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was or
        the vendor op is unavailable in this build.
    """
    if "rope" in _installed:
        return False
    if getattr(torch.ops.supa, "supa_rope_apply", None) is None:
        return False
    from pymss_core.modules.bs_roformer import transformer as _transformer

    original = _transformer.rotate_qk_fast_bnhd
    original_attention_forward = _transformer.Attention.forward
    original_feedforward_forward = _transformer.FeedForward.forward

    global _supa_fused_ffn_fwd
    if _supa_fused_ffn_fwd is None:
        import torch_br
        _supa_fused_ffn_fwd = torch_br.supa_fused_mlp_linear_act_linear_fwd

    def feedforward_forward(self, x):
        """Fused FFN core (one kernel) for SUPA bf16 inference; else original."""
        if (x.device.type == "supa" and x.dtype == torch.bfloat16
                and not self.training
                and _FFN_BACKEND == 'vendor'
                and _supa_fused_ffn_fwd is not None
                and _ffn_fast_path_ok(self)):
            return _vendor_fused_ffn(self, self.net[0](x))
        return original_feedforward_forward(self, x)

    def rotate_qk_fast_bnhd(rotary_embed, q, k):
        if (q.device.type == "supa"
                and getattr(rotary_embed, "_pymss_vendor_rope", False)
                and getattr(torch.ops.supa, "supa_rope_apply", None) is not None):
            return _vendor_rope_rotate(rotary_embed, q, k)
        return original(rotary_embed, q, k)

    def attention_forward(self, x):
        """Regrouped qkv chain for prepared SUPA bf16 rotary attention.

        One (3, B, H, S, D) copy replaces the three scattered copies (q and k
        permutes inside the rotate plus the SDPA shim's v ``.contiguous()``);
        rotary and the FMHA adapter then run on free views.  Everything else --
        other devices, dtypes, PoPE, training mode, unprepared models -- keeps
        the original forward.
        """
        if (x.device.type == "supa" and x.dtype == torch.bfloat16
                and not self.training
                and self.rotary_embed is not None and self.pope_embed is None
                and getattr(self.rotary_embed, "_pymss_vendor_rope", False)
                and getattr(torch.ops.supa, "supa_rope_apply", None) is not None):
            heads = self.heads
            dim_head = self.to_qkv.out_features // (3 * heads)
            normalized = self.norm(x)
            qkv = self.to_qkv(normalized)
            b, n, _ = qkv.shape
            trio = qkv.view(b, n, 3, heads, dim_head).permute(2, 0, 3, 1, 4).contiguous()
            q, k = _vendor_rope_rotate_trio(self.rotary_embed, trio)
            out = self._attention(q, k, trio[2])
            return self.to_out((out.transpose(1, 2) * self.to_gates(normalized).sigmoid().unsqueeze(-1)).flatten(start_dim=-2))
        return original_attention_forward(self, x)

    rotate_qk_fast_bnhd._pymss_original = original
    attention_forward._pymss_original = original_attention_forward
    feedforward_forward._pymss_original = original_feedforward_forward
    logging.getLogger(__name__).info(
        "Installed vendor RoPE + FFN shim (SUPA only; accelerate via prepare_model_for_vendor_rope)")
    _transformer.rotate_qk_fast_bnhd = rotate_qk_fast_bnhd
    _transformer.Attention.forward = attention_forward
    _transformer.FeedForward.forward = feedforward_forward
    _installed.add("rope")
    return True


def complex_to_host(tensor: torch.Tensor, copy) -> torch.Tensor:
    """Copy a complex tensor with ``copy`` applied to its real/imaginary planes.

    Args:
        tensor: A complex tensor.
        copy: Callable that moves a real tensor to the host (usually ``.cpu``).

    Returns:
        The complex tensor on the host.
    """
    planes = copy(torch.view_as_real(tensor).contiguous())
    return torch.view_as_complex(planes)


def install_cpu_shim() -> bool:
    """Copy non-contiguous device tensors to the host through a contiguous copy.

    ``.cpu()`` on a non-contiguous device tensor fails with
    ``ErrorCode: 11, Sudnn Error`` -- scnet returns a ``(1, 1, 2, 16384)`` view with
    stride ``(34816, 34816, 17408, 1)`` and its very first host copy raised it,
    while ``contiguous().cpu()``, ``reshape(-1).cpu()`` and ``clone().cpu()`` all
    work.  Users copy audio back to the host, so this has to be transparent.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "cpu" in _installed:
        return False
    original = torch.Tensor.cpu

    def cpu(self, *args, **kwargs):
        if self.device.type in ("cpu", "meta"):
            return original(self, *args, **kwargs)
        if self.is_complex():
            # A complex host copy is also refused ("ErrorCode: 101, invalid
            # device"); the real/imaginary planes copy fine and rebuild here.
            return complex_to_host(self, lambda plane: original(plane, *args, **kwargs))
        if not self.is_contiguous():
            return original(self.contiguous(), *args, **kwargs)
        return original(self, *args, **kwargs)

    cpu._vendor_shim = True
    torch.Tensor.cpu = cpu
    _installed.add("cpu")
    return True


def install_std_shim() -> bool:
    """Replace ``Tensor.std`` on vendor devices.

    Returns:
        ``True`` when the shim was installed, ``False`` when it already was.
    """
    if "std" in _installed:
        return False
    original = torch.Tensor.std

    def std(self, *args, **kwargs):
        if self.device.type in ("cpu", "meta"):
            return original(self, *args, **kwargs)
        return std_equivalent(self, *args, **kwargs)

    std._vendor_shim = True
    torch.Tensor.std = std
    _installed.add("std")
    return True


#: Shim name -> installer.
SHIMS = {
    "complex_mul": install_complex_mul_shim,
    "complex_reshape": install_complex_reshape_shim,
    "window": install_window_shim,
    "glu": install_glu_shim,
    "group_norm": install_group_norm_shim,
    "interpolate": install_interpolate_shim,
    "conv1d": install_conv1d_shim,
    "complex_cat": install_complex_cat_shim,
    "fft": install_fft_shim,
    "std": install_std_shim,
    "cpu": install_cpu_shim,
    "sdpa": install_sdpa_shim,
"rope": install_rope_shim,
}


def install_vendor_shims(
    device_type: str,
    *,
    complex_mul: bool = True,
    window_on_device: bool = True,
    glu: bool = True,
    complex_reshape: bool = True,
    group_norm: bool = True,
    interpolate: bool = True,
    conv1d: bool = False,
    complex_cat: bool = False,
    fft: bool = False,
    std: bool = False,
    cpu: bool = False,
    sdpa: bool = False,
    rope: bool = False,
) -> list[str]:
    """Install the shims a device profile asks for.

    Args:
        device_type: Device type string, e.g. ``"supa"``.
        complex_mul: ``False`` when the backend has no ``complex * real`` kernel.
        window_on_device: ``False`` when windows cannot be built on the device.
        glu: ``True`` when ``F.glu`` should be replaced by its elementwise form.
        complex_reshape: ``False`` when complex tensors cannot be reshaped.
        group_norm: ``True`` when ``F.group_norm`` should be replaced.
        interpolate: ``True`` when bf16 interpolation must run in fp32.
        conv1d: ``False`` when grouped conv1d must be split per group.
        complex_cat: ``True`` when complex ``torch.cat`` must go via the planes.
        fft: ``True`` when ``torch.fft.rfft``/``irfft`` must be replaced.
        std: ``True`` when ``Tensor.std`` must be rebuilt from ``var``.
        cpu: ``True`` when a non-contiguous host copy must go via ``contiguous``.
        sdpa: ``True`` when fused attention must be expressed as matmuls.
        rope: ``True`` when the vendor RoPE kernel should replace the rotary
            path for models prepared with ``prepare_model_for_vendor_rope``.

    Returns:
        The shim names that were installed by this call.
    """
    if device_type in ("cpu", "meta"):
        return []
    wanted = []
    if not complex_mul:
        wanted.append("complex_mul")
    if not complex_reshape:
        wanted.append("complex_reshape")
    if not window_on_device:
        wanted.append("window")
    if glu:
        wanted.append("glu")
    if group_norm:
        wanted.append("group_norm")
    if interpolate:
        wanted.append("interpolate")
    if conv1d:
        wanted.append("conv1d")
    if complex_cat:
        wanted.append("complex_cat")
    if fft:
        wanted.append("fft")
    if std:
        wanted.append("std")
    if cpu:
        wanted.append("cpu")
    if sdpa:
        wanted.append("sdpa")
    if rope:
        wanted.append("rope")
    installed = []
    for name in wanted:
        if SHIMS[name]():
            installed.append(name)
    return installed


def installed_shims() -> frozenset[str]:
    """Return the shims installed in this process."""
    return frozenset(_installed)


def reset_shims() -> None:
    """Forget which shims were installed (tests only; patches stay in place)."""
    _installed.clear()
