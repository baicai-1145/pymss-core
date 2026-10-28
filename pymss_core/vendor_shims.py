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


def complex_planes_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply with at least one complex operand using only real kernels.

    Args:
        a: Real or complex tensor.
        b: Real or complex tensor, broadcastable against ``a``.

    Returns:
        ``a * b`` with the complex dtype.
    """
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
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and (a.is_complex() or b.is_complex())
        and (a.is_complex() or torch.is_floating_point(a))
        and (b.is_complex() or torch.is_floating_point(b))
        and not (a.is_complex() and a.dtype != torch.complex64)
        and not (b.is_complex() and b.dtype != torch.complex64)
    )


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

    def mul(self, other):
        if _needs_plane_product(self, other):
            return complex_planes_product(self, other)
        return original_mul(self, other)

    def rmul(self, other):
        if _needs_plane_product(other, self):
            return complex_planes_product(other, self)
        return original_rmul(self, other)

    def mul_fn(a, b, *args, **kwargs):
        if not args and not kwargs and _needs_plane_product(a, b):
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


#: Shim name -> installer.
SHIMS = {
    "complex_mul": install_complex_mul_shim,
    "window": install_window_shim,
    "glu": install_glu_shim,
}


def install_vendor_shims(
    device_type: str,
    *,
    complex_mul: bool = True,
    window_on_device: bool = True,
    glu: bool = True,
) -> list[str]:
    """Install the shims a device profile asks for.

    Args:
        device_type: Device type string, e.g. ``"supa"``.
        complex_mul: ``False`` when the backend has no ``complex * real`` kernel.
        window_on_device: ``False`` when windows cannot be built on the device.
        glu: ``True`` when ``F.glu`` should be replaced by its elementwise form.

    Returns:
        The shim names that were installed by this call.
    """
    if device_type in ("cpu", "meta"):
        return []
    wanted = []
    if not complex_mul:
        wanted.append("complex_mul")
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
