# pymss-core

[中文文档](README_CN.md)

Core model, configuration, and checkpoint package for music source separation.

`pymss-core` is the shared low-level package for higher-level projects such as
`pymss` inference and `pymsst` training. It contains model definitions,
configuration loading, and checkpoint compatibility helpers. It intentionally
does not include inference DSP pipelines, chunked demixing, audio file I/O,
model downloads, catalog management, CLI, HTTP server, WebUI, datasets, losses,
or training loops.

## Install

```bash
pip install pymss-core
```

For local development:

```bash
uv sync --dev
```

Optional MLX backend on Apple Silicon:

```bash
pip install "pymss-core[mlx]"
```

## Ascend NPU

Install matching PyTorch, `torch_npu`, CANN, firmware, and driver versions from the
[Ascend compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).
`torch_npu` is optional and is not installed by this package. Spectral models use
NPU-specific STFT/ISTFT helpers; CPU, CUDA, and MPS retain their native paths.

For recurrent models, enable the CPU fallback after importing `torch_npu`:

```python
import torch
from pymss_core.npu_rnn import install_npu_rnn_fallback

install_npu_rnn_fallback()
model.eval()
with torch.inference_mode():
    output = model(audio)
```

The fallback handles NPU RNN/LSTM/GRU inputs only, including packed sequences.
It copies recurrent weights and inputs to CPU for each call, computes low-precision
inputs in float32, and returns results to the input device and dtype. It supports
inference only; NPU training with this fallback raises an error. CPU/CUDA recurrent
execution and autograd remain native. `pymss` enables the fallback automatically
when selecting NPU.

## Public API

```python
from pymss_core import (
    clear_model_runtime_caches,
    get_model_from_config,
    load_config,
    load_model_weights,
)

model, config = get_model_from_config("bs_roformer", "config.yaml")
load_model_weights(model, "model.ckpt", model_type="bs_roformer", strict=True)

model.eval()
```

- Use `get_model_from_config("auto", "config.yaml")` to detect the architecture from YAML. `detect_model_type(config)` exposes the same detection for an already loaded configuration. Unknown or conflicting architectures raise `ModelTypeDetectionError` (a `RuntimeError` subclass); an explicit model type overrides detection.
- `bs_roformer` also loads BS PolarFormer checkpoints when the matching YAML sets `model.use_pope: true`.

After inference has finished, clear model-owned derived caches before releasing the model:

```python
clear_model_runtime_caches(model)
```

Cleanup preserves parameters and registered buffers. It calls module-owned `clear_runtime_cache()` hooks and releases known inference-cache references. If a hook fails, cleanup continues through the remaining modules before re-raising the first error.

Process-wide shared MLX caches, including overlap-add indices and Hann windows, and backend allocator caches are managed separately. They are outside the scope of this model-level cleanup.

## Package Boundary

Included:

- YAML config loading with `AttrDict`
- PyTorch model definitions under `pymss_core.modules`
- Optional MLX backend implementations for supported model forward paths
- Model factory: `get_model_from_config(model_type, config_path)`
- Checkpoint helpers for common MSS checkpoint containers
- Cleanup for model-owned inference caches
- Small model-internal DSP math needed to construct model structures
- VR network structures and VR model parameter JSON files

Excluded:

- Audio file decoding/encoding
- Resampling, preprocessing, and full inference DSP pipelines
- Tensor-level chunked demixing runtime
- Model catalog, aliases, downloads, and model-file cache management
- CLI, server, WebUI, and endpoint schemas
- Dataset, augmentation, loss, metrics, and trainer code
- Any default dependency on MLX, Librosa, tqdm, Lightning, FastAPI, Uvicorn,
  PyAV, WandB, or training extras

## Repository Roles

```text
pymss-core
  shared model/config/checkpoint layer

pymss
  user-facing inference package built on pymss-core, with audio I/O and demix

pymsst
  training package built on pymss-core, with training data/loss/runtime code
```
