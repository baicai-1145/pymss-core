"""NPU RNN fallback: run LSTM/GRU cells on CPU with device-transparent results.

Ascend torch_npu builds used by pymss ship ``npu_lstm``/``npu_gru`` without an
aclnn implementation ("Current device only support aclnn operator"), so any
model with recurrent cells (bandit's BSRNN, SCNet's dual-path RNN, VR 5.x
LSTM modules) faults on device. This shim patches ``nn.LSTM``/``nn.GRU``
forward passes on NPU tensors to bounce through CPU: inputs and module
weights round-trip once per call and the output lands back on the NPU. RNN
weights are small relative to the surrounding convolutional/transformer
blocks, so the transfer cost keeps these models usable while torch_npu
catches up; it is automatically bypassed once the native kernel works.

Enabled via ``pymss_core.npu_rnn.install_npu_rnn_fallback()``.
"""
from __future__ import annotations

import torch
from torch import nn

_installed = False


class _NpuRnnCpuFallback(nn.Module):
    def __init__(self, rnn: nn.RNNBase):
        super().__init__()
        self._rnn = rnn
        self._cpu_rnn = None
        self._cpu_weight_version = None

    def _cpu_module(self):
        weight_version = self._rnn.weight_ih_l0._version
        if self._cpu_rnn is None or self._cpu_weight_version != weight_version:
            # deepcopy keeps the original module untouched on NPU
            import copy

            self._cpu_rnn = copy.deepcopy(self._rnn).cpu()
            self._cpu_weight_version = weight_version
        return self._cpu_rnn

    def forward(self, input, hx=None):
        input_cpu = input.detach().cpu()
        hx_cpu = None
        if hx is not None:
            hx_cpu = tuple(h.detach().cpu() for h in hx) if isinstance(hx, (tuple, list)) else hx.detach().cpu()
        out = self._cpu_module()(input_cpu, hx_cpu)
        out_device = input.device
        if isinstance(out, tuple):
            return (out[0].to(out_device),) + tuple(
                (o.to(out_device) if isinstance(o, torch.Tensor) else tuple(t.to(out_device) for t in o)) for o in out[1:]
            )
        return out.to(out_device)


def _patched_rnn_forward(self, input, hx=None):
    if not isinstance(input, torch.Tensor) or input.device.type != "npu":
        return _NpuRnnCpuFallback._ORIG_FORWARDS[type(self)](self, input) if hx is None else _NpuRnnCpuFallback._ORIG_FORWARDS[type(self)](self, input, hx)
    return _NpuRnnCpuFallback(self)(input, hx)


def install_npu_rnn_fallback():
    """Patch nn.LSTM/nn.GRU/nn.RNN forwards to bounce NPU inputs through CPU."""
    global _installed
    if _installed:
        return
    orig = {}
    for cls in (nn.LSTM, nn.GRU, nn.RNN):
        orig[cls] = cls.forward
        cls.forward = _patched_rnn_forward
    _NpuRnnCpuFallback._ORIG_FORWARDS = orig
    _installed = True
