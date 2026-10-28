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


#: Shim name -> installer.
SHIMS = {
    "complex_mul": install_complex_mul_shim,
    "complex_reshape": install_complex_reshape_shim,
    "window": install_window_shim,
    "glu": install_glu_shim,
}


def install_vendor_shims(
    device_type: str,
    *,
    complex_mul: bool = True,
    window_on_device: bool = True,
    glu: bool = True,
    complex_reshape: bool = True,
) -> list[str]:
    """Install the shims a device profile asks for.

    Args:
        device_type: Device type string, e.g. ``"supa"``.
        complex_mul: ``False`` when the backend has no ``complex * real`` kernel.
        window_on_device: ``False`` when windows cannot be built on the device.
        glu: ``True`` when ``F.glu`` should be replaced by its elementwise form.
        complex_reshape: ``False`` when complex tensors cannot be reshaped.

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
