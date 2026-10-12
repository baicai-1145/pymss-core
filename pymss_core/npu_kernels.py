"""Optional triton-ascend fused kernels for NPU inference (branch: ascend-kernels).

Status after the 2026-10 PoC on Ascend 950PR + triton-ascend 3.2.1:

- Correctness: all kernels below are numerically sound (the fused FF matmul
  was bit-exact vs eager fp16).
- Performance: **not yet enabled** in the model path. triton-ascend 3.2.1
  generates unvectorized memory kernels on 950PR — a plain copy runs 0.239 ms
  where ``aten::clone`` needs 0.081 ms (3x). Since the eager aclnn elementwise
  ops already saturate ~75-90% of HBM bandwidth, a triton fusion that saves 2
  round trips but pays 3x per trip is a net loss (measured: 62x vs 78x RTFx
  end-to-end on bs_roformer hyperacev2).
- Revisit when triton-ascend ships vectorized loads (tl.load of 8xfp16 →
  128-bit LDG equivalent). The 2.88x 5-op-chain PoC (bench_fused.py) shows the
  ceiling once per-trip speed reaches parity.

Activated when ALL of:
- running on an ``npu`` device,
- triton-ascend is importable and exposes a working NPU driver,
- ``PYMSS_CORE_NPU_TRITON`` != "0".

The eager torch_npu path remains the default fallback; every kernel here must
be numerically equivalent (fp32 intermediates are allowed — they are *more*
accurate than the fp16 eager chain).
"""
from __future__ import annotations

import os

import torch

_available: bool | None = None


def triton_ascend_available() -> bool:
    """True when triton-ascend is importable with a live NPU driver."""
    global _available
    if _available is not None:
        return _available
    if os.environ.get("PYMSS_CORE_NPU_TRITON", "1") == "0":
        _available = False
        return _available
    try:
        import torch_npu  # noqa: F401  (driver availability depends on torch_npu init)

        import torch as _torch

        if not _torch.npu.is_available():
            _available = False
            return _available
        import triton
    except Exception:
        _available = False
        return _available
    try:
        target = triton.runtime.driver.active.get_current_target()
        _available = "npu" in str(getattr(target, "backend", "")).lower()
    except Exception:
        _available = False
    return _available


if triton_ascend_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_gated_out(gates_ptr, gstride, attn_ptr, out_ptr, n, D: tl.constexpr, BLOCK: tl.constexpr):
        """out[i] = attn[i] * sigmoid(gates[i // D]) — gates stride-read, no materialization."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(attn_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(gates_ptr + (offs // D) * gstride, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, (a * tl.sigmoid(g)).to(tl.float16), mask=mask)

    @triton.jit
    def _fused_residual_scale_chain(x_ptr, res_ptr, out_ptr, scale, n, BLOCK: tl.constexpr):
        """out = ((x * res) + x) * scale  — reads x/res once, writes out once."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        r = tl.load(res_ptr + offs, mask=mask).to(tl.float32)
        y = (x * r + x) * scale
        tl.store(out_ptr + offs, y.to(tl.float16), mask=mask)

    @triton.jit
    def _fused_rmsnorm_glu(x_ptr, w_ptr, g_ptr, out_ptr, n_rows, n_cols, eps, ROWS_PER_PROG: tl.constexpr, BLOCK_N: tl.constexpr):
        """Row-wise RMSNorm(x*w) followed by a per-element sigmoid gate.

        Each program handles ROWS_PER_PROG consecutive rows: reads each row
        once, computes the mean-square in registers, applies weight and gate,
        writes once — replaces norm+mul+sigmoid+mul (4 HBM round trips).
        Grouping keeps the launch grid under the 65535 coreDim limit for tall
        activations (e.g. 124x690 rows in the roformer time transformer).
        """
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(ROWS_PER_PROG):
            row = pid * ROWS_PER_PROG + i
            if row < n_rows:
                base = row.to(tl.int64) * n_cols
                x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
                ms = tl.sum(x * x, axis=0) / n_cols
                inv = 1.0 / tl.sqrt(ms + eps)
                y = x * inv * w
                g = tl.load(g_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
                out = y * tl.sigmoid(g)
                tl.store(out_ptr + base + cols, out.to(tl.float16), mask=mask)


def fused_residual_scale_chain(x: torch.Tensor, residual: torch.Tensor, scale: float) -> torch.Tensor:
    """``((x * residual) + x) * scale`` — one HBM pass when triton is up.

    Microbench (Ascend 950PR, 43.7M-elem fp16 hot shape): 0.468 ms eager vs
    0.185 ms fused at BLOCK=16384 (**2.5x**). BLOCK matters a lot: the per-program
    prologue dominates at small blocks (4096 -> 0.24 ms even for a pure copy),
    16384 reaches copy-bandwidth parity with aten.
    """
    if not _kernel_ok(x, residual):
        return (x * residual + x) * scale
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 16384),)
    _fused_residual_scale_chain[grid](x, residual, out, scale, n, BLOCK=16384)
    return out


def fused_gelu_linear_residual(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, residual: torch.Tensor | None = None) -> torch.Tensor:
    """Eager reference for ``gelu(x @ W.T + b) (+ residual)``.

    The triton tl.dot variant benchmarked at parity with MatMulV3 (both are
    cube-core bound), so the fused path is not enabled — kept for tilelang-style
    experiments on branch ascend-kernels.
    """
    y = torch.nn.functional.linear(x, weight, bias)
    y = torch.nn.functional.gelu(y)
    return y if residual is None else y + residual


def fused_gated_out(attn: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """``attn * sigmoid(gates)`` where gates broadcast over the head dim.

    attn: (B, H, T, D) contiguous, gates: (B, H, T) — replaces
    sigmoid+unsqueeze+mul (one HBM pass for attn instead of three; the gates
    row is stride-read once per D elements and stays in cache).
    """
    B, H, T, D = attn.shape
    if not _kernel_ok(attn) or gates.shape != (B, H, T) or not gates.is_contiguous():
        return attn * gates.sigmoid().unsqueeze(-1)
    out = torch.empty_like(attn)
    n = attn.numel()
    grid = (triton.cdiv(n, 4096),)
    _fused_gated_out[grid](gates, 1, attn, out, n, D=D, BLOCK=4096)
    return out


def _kernel_ok(*tensors: torch.Tensor) -> bool:
    """Common eligibility gate for the fused kernels."""
    if not triton_ascend_available():
        return False
    for t in tensors:
        if t.device.type != "npu" or t.dtype != torch.float16 or not t.is_contiguous():
            return False
    return True


def fused_rmsnorm_glu(x: torch.Tensor, weight: torch.Tensor, gate_proj_weight: torch.Tensor) -> torch.Tensor:
    """``RMSNorm(x) * sigmoid(gate_proj(x))`` fused into one kernel.

    ``gate_proj_weight`` is the (n_cols, n_cols) linear weight applied to x for
    the gate (no bias); falls back to eager otherwise.
    """
    if not _kernel_ok(x) or weight.dim() != 1 or x.shape[-1] != weight.shape[0]:
        # eager reference
        normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight)
        gate = torch.nn.functional.linear(x, gate_proj_weight)
        return normed * gate.sigmoid()
    lead, n_cols = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, n_cols)
    gate = torch.nn.functional.linear(flat, gate_proj_weight)  # matmul stays on cube cores
    out = torch.empty_like(flat)
    n_rows = flat.shape[0]
    BLOCK_N = triton.next_power_of_2(n_cols)
    if BLOCK_N > 32768:
        normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight)
        return normed * gate.reshape(*lead, n_cols).sigmoid() * 1.0
    ROWS = 4 if n_rows > 65535 // 4 else 1  # stay under the 65535 coreDim launch limit
    if n_rows % ROWS:
        ROWS = 1
    _fused_rmsnorm_glu[(triton.cdiv(n_rows, ROWS),)](flat, weight, gate, out, n_rows, n_cols, 1e-12, ROWS_PER_PROG=ROWS, BLOCK_N=BLOCK_N)
    return out.reshape(*lead, n_cols)
