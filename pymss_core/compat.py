"""Small torch-version compatibility shims shared by model modules."""
from __future__ import annotations

import torch
import torch.nn.functional as F

if not hasattr(F, "rms_norm"):  # torch<2.4
    def _rms_norm_compat(input, normalized_shape, weight=None, eps=None):
        eps = 1e-6 if eps is None else eps
        dims = tuple(range(input.dim() - len(normalized_shape), input.dim()))
        variance = input.pow(2).mean(dim=dims, keepdim=True)
        out = input * torch.rsqrt(variance + eps)
        if weight is not None:
            out = out * weight
        return out

    F.rms_norm = _rms_norm_compat
