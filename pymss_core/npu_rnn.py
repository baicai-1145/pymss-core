"""NPU RNN fallback: run LSTM/GRU cells on CPU with device-transparent results.

Ascend torch_npu builds used by pymss ship ``npu_lstm``/``npu_gru`` without an
aclnn implementation ("Current device only support aclnn operator"), so any
model with recurrent cells (bandit's BSRNN, SCNet's dual-path RNN, VR 5.x
LSTM modules) faults on device. This shim patches ``nn.LSTM``/``nn.GRU``
forward passes on NPU tensors to bounce through CPU: inputs and module
weights round-trip once per call and the output lands back on the NPU. RNN
weights are small relative to the surrounding convolutional/transformer
blocks, so the transfer cost keeps inference usable on affected builds.
Autograd on NPU recurrent modules is not supported by this fallback.

Enabled via ``pymss_core.npu_rnn.install_npu_rnn_fallback()``.
"""
from __future__ import annotations

from functools import wraps

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

_installed = False


class _NpuRnnCpuFallback(nn.Module):
    def __init__(self, rnn: nn.RNNBase, original_forward):
        super().__init__()
        self._rnn = rnn
        self._original_forward = original_forward

    def _cpu_module(self, dtype):
        # The adapter is per-call; keep the original module and parameters untouched.
        import copy

        return copy.deepcopy(self._rnn).to(device="cpu", dtype=dtype)

    def forward(self, input, hx=None):
        if torch.is_grad_enabled():
            raise RuntimeError("The NPU recurrent fallback supports inference only; disable autograd before calling it.")
        tensor = input.data if isinstance(input, PackedSequence) else input
        input_dtype = tensor.dtype
        weight_dtype = self._rnn.weight_ih_l0.dtype
        low_precision = (torch.float16, torch.bfloat16)
        compute_dtype = torch.float32 if input_dtype in low_precision or weight_dtype in low_precision else weight_dtype
        input_cpu = input.to(device="cpu", dtype=compute_dtype) if isinstance(input, PackedSequence) else input.detach().to(device="cpu", dtype=compute_dtype)
        hx_cpu = None
        if hx is not None:
            hx_cpu = tuple(h.detach().to(device="cpu", dtype=compute_dtype) for h in hx) if isinstance(hx, (tuple, list)) else hx.detach().to(device="cpu", dtype=compute_dtype)
        with torch.autocast("cpu", enabled=False):
            cpu_rnn = self._cpu_module(compute_dtype)
            output, hidden = self._original_forward(cpu_rnn, input_cpu) if hx_cpu is None else self._original_forward(cpu_rnn, input_cpu, hx_cpu)
        out_device = tensor.device
        hidden = tuple(value.to(device=out_device, dtype=input_dtype) for value in hidden) if isinstance(hidden, tuple) else hidden.to(device=out_device, dtype=input_dtype)
        return output.to(device=out_device, dtype=input_dtype), hidden


def _is_npu_input(input):
    tensor = input.data if isinstance(input, PackedSequence) else input
    return isinstance(tensor, torch.Tensor) and tensor.device.type == "npu"


def _patched_rnn_forward(original_forward):
    @wraps(original_forward)
    def forward(self, input, hx=None):
        if not _is_npu_input(input):
            return original_forward(self, input) if hx is None else original_forward(self, input, hx)
        return _NpuRnnCpuFallback(self, original_forward)(input, hx)

    forward._pymss_npu_original_forward = original_forward
    return forward


def install_npu_rnn_fallback():
    """Patch recurrent forwards for NPU inference; native CPU/CUDA paths stay intact."""
    global _installed
    if _installed:
        return
    for cls in (nn.LSTM, nn.GRU, nn.RNN):
        original = getattr(cls.forward, "_pymss_npu_original_forward", cls.forward)
        cls.forward = _patched_rnn_forward(original)
    _installed = True
