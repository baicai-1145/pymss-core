"""Accelerator backend profiles.

pymss runs on several PyTorch accelerators, and they do not agree on how they
present themselves to ``torch``:

* **NVIDIA / AMD ROCm**  -- ``device.type == "cuda"``.
* **Iluvatar CoreX / MetaX MACA / Hygon DCU** -- vendor PyTorch builds that
  *impersonate* CUDA: ``device.type`` is ``"cuda"`` and ``torch.cuda.is_available()``
  is ``True``.  ``torch.cuda.get_device_name()`` still reports the real vendor.
* **Huawei Ascend** (``torch_npu``) -- private device type ``"npu"``.
* **Enflame GCU** (``torch_gcu``) -- private device type ``"gcu"``.
* **Cambricon MLU** (``torch_mlu``) -- private device type ``"mlu"``.
* **Biren** (``torch_br``) -- private device type ``"supa"`` on newer stacks,
  CUDA-impersonating on older ones.

Hardcoding ``device.type == "npu"``-style checks ties the code to a single
vendor, so this module centralises detection and exposes capability flags
instead.  Every flag can be forced from the environment while a new vendor is
being brought up::

    PYMSS_DEVICE_BACKEND=biren   # force vendor detection
    PYMSS_FFT_FALLBACK=1         # force the complex-safe STFT/ISTFT path
    PYMSS_RNN_FALLBACK=1         # force the CPU fallback for LSTM/GRU/RNN
    PYMSS_AMP=0                  # disable autocast

Capability defaults are deliberately conservative only where a vendor is known
to be incomplete (Ascend).  Vendors that are still being validated default to
the native path and can be switched to the fallback path per-run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

NVIDIA = "nvidia"
ROCM = "rocm"
HYGON = "hygon"
ILUVATAR = "iluvatar"
METAX = "metax"
ENFLAME = "enflame"
BIREN = "biren"
ASCEND = "ascend"
MLU = "mlu"
MPS = "mps"
CPU = "cpu"

#: Vendors a user may pass as ``--device``.  ``auto`` is handled by callers.
DEVICE_ALIASES: tuple[str, ...] = (
    "cpu",
    "cuda",
    "rocm",
    "mps",
    "mlx",
    "npu",
    "gcu",
    "mlu",
    "supa",
    "biren",
    "iluvatar",
    "metax",
    "enflame",
    "hygon",
)

#: ``--device`` alias -> vendor key.
ALIAS_TO_VENDOR: dict[str, str] = {
    "cuda": NVIDIA,
    "rocm": ROCM,
    "mps": MPS,
    "mlx": MPS,
    "npu": ASCEND,
    "gcu": ENFLAME,
    "mlu": MLU,
    "supa": BIREN,
    "biren": BIREN,
    "iluvatar": ILUVATAR,
    "metax": METAX,
    "enflame": ENFLAME,
    "hygon": HYGON,
    "cpu": CPU,
}


@dataclass(frozen=True)
class DeviceProfile:
    """Capabilities of the accelerator pymss is currently running on.

    Attributes:
        vendor: Canonical vendor key (see the module level constants).
        device_type: torch device type string to build devices with.
        plugin: Import name of the vendor PyTorch extension, if one is required
            before the backend can be used (``torch_npu``, ``torch_gcu``,
            ``torch_mlu``, ``torch_br``).
        complex_fft: ``torch.stft``/``torch.istft`` handle complex tensors on the
            device.  When ``False`` the ``torch.fft`` based fallback is used.
        rnn: ``nn.LSTM``/``nn.GRU``/``nn.RNN`` run natively on the device.  When
            ``False`` recurrent layers bounce through CPU.
        rnn_blocked: individual cells (``"lstm"``/``"gru"``/``"rnn"``) that must
            bounce through CPU although the rest of the family is native.
        amp: ``torch.amp.autocast`` is usable for the device type.
        channels_last: ``channels_last`` memory format is beneficial/supported.
        multi_device: more than one device index is usable.
    """

    vendor: str
    device_type: str
    plugin: str | None = None
    complex_fft: bool = True
    rnn: bool = True
    rnn_blocked: frozenset[str] = frozenset()
    amp: bool = True
    channels_last: bool = True
    multi_device: bool = True

    @property
    def is_accelerator(self) -> bool:
        return self.device_type not in ("cpu", "mps")

    @property
    def needs_fft_fallback(self) -> bool:
        return not self.complex_fft

    @property
    def needs_rnn_fallback(self) -> bool:
        return not self.rnn

    @property
    def blocked_rnn_ops(self) -> frozenset[str]:
        """Recurrent cells that must bounce through CPU even when ``rnn`` is set."""
        return frozenset() if not self.rnn else self.rnn_blocked

    @property
    def supports_amp(self) -> bool:
        return self.amp

    @property
    def supports_data_parallel(self) -> bool:
        return self.multi_device and self.device_type == "cuda"


#: Baseline profiles.  ``complex_fft``/``rnn`` are ``False`` only where the
#: vendor stack is known to be missing the operator.
_PROFILES: dict[str, DeviceProfile] = {
    NVIDIA: DeviceProfile(NVIDIA, "cuda"),
    ROCM: DeviceProfile(ROCM, "cuda"),
    HYGON: DeviceProfile(HYGON, "cuda"),
    ILUVATAR: DeviceProfile(ILUVATAR, "cuda"),
    METAX: DeviceProfile(METAX, "cuda"),
    ENFLAME: DeviceProfile(ENFLAME, "gcu", plugin="torch_gcu"),
    # Measured on BIREN106M + torch 2.7.0+cpu / torch-br 1.7.0.20700 (SUPA 1.9.0):
    # ``stft(return_complex=True)`` raises "does not support return_complex=True
    # now", while the rfft-based composition matches torch to 9.5e-07; GRU hits
    # an unimplemented ``aten::_thnn_fused_gru_cell`` although LSTM is native.
    BIREN: DeviceProfile(BIREN, "supa", plugin="torch_br", complex_fft=False, rnn_blocked=frozenset({"gru"})),
    MLU: DeviceProfile(MLU, "mlu", plugin="torch_mlu"),
    # Ascend's torch_npu builds ship npu_lstm/npu_gru without inference support
    # and lack a complex STFT path, so both fallbacks are part of the profile.
    ASCEND: DeviceProfile(ASCEND, "npu", plugin="torch_npu", complex_fft=False, rnn=False, multi_device=False),
    MPS: DeviceProfile(MPS, "mps", complex_fft=False, rnn=True, amp=True, channels_last=False, multi_device=False),
    CPU: DeviceProfile(CPU, "cpu", amp=False, channels_last=False, multi_device=False),
}

#: Substrings looked for in ``torch.cuda.get_device_name()`` (lowercased).
_VENDOR_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (HYGON, ("hygon", "dtk", "dcu", "k100", "z100", "bw1000", "shuxian")),
    (ILUVATAR, ("iluvatar", "corex", "bi-v", "mr-v", "bi150", "bi-v150")),
    (METAX, ("metax", "maca", "mxc", "c500", "n260", "xiyun")),
    (ENFLAME, ("enflame", "gcu", "s60", "t20", "cloudblazer")),
    (BIREN, ("biren", "supa", "br100", "br104", "bili")),
    (MLU, ("cambricon", "mlu")),
    (ASCEND, ("ascend", "npu")),
)

#: Private device types that imply a vendor without any probing.
_DEVICE_TYPE_VENDOR: dict[str, str] = {
    "npu": ASCEND,
    "gcu": ENFLAME,
    "mlu": MLU,
    "supa": BIREN,
    "mps": MPS,
    "cpu": CPU,
}

_cache: dict[str, DeviceProfile] = {}


def _env_flag(name: str) -> bool | None:
    """Return a tri-state boolean from ``name`` (``1/0``, ``true/false``, ...)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _device_type_of(device) -> str:
    """Normalise ``device`` (str, ``torch.device``, tensor, ``None``) to a type."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.Tensor):
        return device.device.type
    try:
        return torch.device(device).type
    except (TypeError, RuntimeError):
        # Backends that are not registered in this build (``"npu:0"`` on a CUDA
        # host) cannot be parsed by torch.device; fall back to the type prefix so
        # the profile is still resolvable in tests and on mixed hosts.
        prefix = str(device).split(":", 1)[0].strip().lower()
        return prefix or "cpu"


def _cuda_device_name() -> str:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return str(torch.cuda.get_device_name(0)).lower()
    except Exception:  # a half-initialised vendor backend must not break detection
        return ""
    return ""


def detect_vendor(device_type: str) -> str:
    """Best-effort vendor detection for a torch device type.

    ``PYMSS_DEVICE_BACKEND`` wins over every probe.

    Args:
        device_type: ``torch.device(...).type`` of the target device.

    Returns:
        Vendor key; defaults to :data:`NVIDIA` for ``"cuda"`` and :data:`CPU`
        for anything unknown.
    """
    forced = os.environ.get("PYMSS_DEVICE_BACKEND", "").strip().lower()
    if forced in _PROFILES:
        return forced

    if device_type in _DEVICE_TYPE_VENDOR:
        return _DEVICE_TYPE_VENDOR[device_type]
    if device_type != "cuda":
        return CPU

    name = _cuda_device_name()
    for vendor, keywords in _VENDOR_KEYWORDS:
        if any(keyword in name for keyword in keywords):
            return vendor
    if getattr(torch.version, "hip", None):
        return ROCM
    return NVIDIA


def _with_overrides(profile: DeviceProfile) -> DeviceProfile:
    """Apply ``PYMSS_FFT_FALLBACK`` / ``PYMSS_RNN_FALLBACK`` / ``PYMSS_AMP``."""
    updates: dict[str, bool] = {}
    fft_fallback = _env_flag("PYMSS_FFT_FALLBACK")
    if fft_fallback is not None:
        updates["complex_fft"] = not fft_fallback
    rnn_fallback = _env_flag("PYMSS_RNN_FALLBACK")
    if rnn_fallback is not None:
        updates["rnn"] = not rnn_fallback
    amp = _env_flag("PYMSS_AMP")
    if amp is not None:
        updates["amp"] = amp
    return replace(profile, **updates) if updates else profile


def device_profile(device=None) -> DeviceProfile:
    """Return the (cached) capability profile for ``device``.

    Args:
        device: Optional device spec -- ``None``, a device string, a
            ``torch.device`` or a tensor.

    Returns:
        The matching :class:`DeviceProfile`.
    """
    device_type = _device_type_of(device)
    cached = _cache.get(device_type)
    if cached is not None:
        return cached
    profile = _with_overrides(_PROFILES[detect_vendor(device_type)])
    _cache[device_type] = profile
    return profile


def clear_device_cache() -> None:
    """Drop the detection cache (tests, or after loading a vendor plugin)."""
    _cache.clear()


def profile_for_alias(alias: str | None) -> DeviceProfile | None:
    """Return the profile implied by a ``--device`` alias, or ``None``."""
    if not alias:
        return None
    vendor = ALIAS_TO_VENDOR.get(str(alias).strip().lower())
    return _PROFILES[vendor] if vendor else None


def requires_plugin(alias: str) -> str | None:
    """Return the vendor extension that must be imported for ``alias``."""
    profile = profile_for_alias(alias)
    return profile.plugin if profile else None


def prepare_device_alias(alias: str) -> str:
    """Import the vendor extension for ``alias`` and return the device type.

    Every domestic vendor needs its extension imported before the backend is
    registered on ``torch`` (``torch_npu``, ``torch_gcu``, ``torch_mlu``,
    ``torch_br``).  Imports are best-effort: an ImportError is reported to the
    caller so it can raise a helpful error message.

    Args:
        alias: A ``--device`` value such as ``"biren"`` or ``"npu"``.

    Returns:
        The torch device type string to build devices with.
    """
    profile = profile_for_alias(alias)
    if profile is None:
        return str(alias)
    if profile.plugin:
        __import__(profile.plugin)
    clear_device_cache()
    device_type = profile.device_type
    if device_type == "supa" and not hasattr(torch, "supa"):
        # Older BIRENSUPA releases expose SUPA devices as CUDA.
        device_type = "cuda"
    return device_type


def fft_fallback_required(device=None) -> bool:
    """Return ``True`` when ``torch.stft``/``torch.istft`` must be replaced."""
    return device_profile(device).needs_fft_fallback


def rnn_fallback_required(device=None) -> bool:
    """Return ``True`` when recurrent layers must bounce through CPU."""
    return device_profile(device).needs_rnn_fallback


def rnn_op_blocked(device, op: str) -> bool:
    """Return ``True`` when the recurrent cell ``op`` must bounce through CPU.

    ``op`` is the lowercase cell name (``"lstm"``, ``"gru"``, ``"rnn"``).  A
    device level ``rnn=False`` blocks every cell; otherwise only the cells listed
    in :attr:`DeviceProfile.rnn_blocked` are affected.
    """
    profile = device_profile(device)
    if not profile.rnn:
        return True
    return str(op).lower() in profile.rnn_blocked


def amp_supported(device=None) -> bool:
    """Return ``True`` when autocast is usable on ``device``."""
    return device_profile(device).supports_amp


__all__ = (
    "ALIAS_TO_VENDOR",
    "ASCEND",
    "BIREN",
    "CPU",
    "DEVICE_ALIASES",
    "DeviceProfile",
    "ENFLAME",
    "HYGON",
    "ILUVATAR",
    "METAX",
    "MLU",
    "MPS",
    "NVIDIA",
    "ROCM",
    "amp_supported",
    "clear_device_cache",
    "detect_vendor",
    "device_profile",
    "fft_fallback_required",
    "prepare_device_alias",
    "profile_for_alias",
    "requires_plugin",
    "rnn_fallback_required",
    "rnn_op_blocked",
)
