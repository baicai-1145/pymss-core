"""CPU fallback for convolutions the vendor kernel cannot plan.

BIREN's SUDNN rejects most dilated convolutions on the current SUPA stack
(1.9.0.0.rc1 / torch-br 1.7.0.20700).  The failure is not a Python error: the
plan builder throws a C++ exception that reaches ``terminate()``, so a model
with a dilated resblock kills the whole process instead of raising.  Measured
on a BIREN106M (2x192x21x32 input, 3x3 kernel, fp32):

===================  =========
dilation / padding   result
===================  =========
(1, 1)               native OK
(2, 2) / (2, 2)      native OK
(4, 4) / (4, 4)      becomes SUDNN plan failure
(4, 2), (1, 2), (2, 1) any padding -> process abort
BRTB_ENABLE_SUPA_FALLBACK=1        no effect on conv
bfloat16             no effect
===================  =========

Dilated layers are small in the models that use them, so bouncing those calls
through CPU keeps inference usable.  Inference only: a call with autograd
enabled raises instead of silently detaching.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pymss_core.device import dilated_conv_blocked

_installed = False


def _dilation_of(conv) -> tuple[int, int]:
    dilation = conv.dilation
    if isinstance(dilation, int):
        return (dilation, dilation)
    values = tuple(int(value) for value in dilation)
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[-1])


def needs_cpu_fallback(conv, input) -> bool:
    """Return ``True`` when this conv must run on CPU for the current device."""
    if not isinstance(input, torch.Tensor) or not hasattr(conv, "weight"):
        return False
    if input.dim() < 3:
        return False
    dilation = _dilation_of(conv)
    if dilation == (1, 1):
        return False
    return dilated_conv_blocked(input.device, dilation)


def _conv_on_cpu(conv, input):
    """Evaluate ``conv`` on CPU in fp32 and move the result back."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "The dilated-conv CPU fallback supports inference only; disable autograd before calling it."
        )
    device, dtype = input.device, input.dtype
    weight = conv.weight.detach().to(device="cpu", dtype=torch.float32)
    bias = conv.bias.detach().to(device="cpu", dtype=torch.float32) if conv.bias is not None else None
    args = (input.detach().to(device="cpu", dtype=torch.float32), weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    if input.dim() == 4:
        out = F.conv2d(*args)
    elif input.dim() == 5:
        out = F.conv3d(*args)
    else:
        out = F.conv1d(*args)
    return out.to(device=device, dtype=dtype)


def install_dilated_conv_fallback():
    """Patch ``_ConvNd.forward`` so unsupported dilated convs run on CPU."""
    global _installed
    if _installed:
        return
    from torch.nn.modules.conv import _ConvNd

    original = getattr(_ConvNd.forward, "_pymss_conv_original_forward", _ConvNd.forward)

    def forward(self, input):
        if needs_cpu_fallback(self, input):
            return _conv_on_cpu(self, input)
        return original(self, input)

    forward._pymss_conv_original_forward = original
    _ConvNd.forward = forward
    _installed = True


__all__ = ["install_dilated_conv_fallback", "needs_cpu_fallback"]
