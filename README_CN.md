# pymss-core

[English](README.md)

`pymss-core` 是面向音乐源分离的底层模型、配置与 checkpoint 包。它会作为上层项目的共享基础层，例如：

- `pymss`：面向用户的推理包
- `pymsst`：面向训练的训练包

这个包只保留模型推理和训练都需要复用的核心能力：模型定义、配置加载、
checkpoint 兼容处理。它不包含推理 DSP pipeline、分块 demix、音频文件读写、
模型下载、模型 catalog、CLI、HTTP server、WebUI、dataset、loss 或训练循环。

## 安装

```bash
pip install pymss-core
```

本地开发：

```bash
uv sync --dev
```

Apple Silicon 上可选安装 MLX backend：

```bash
pip install "pymss-core[mlx]"
```

## 昇腾 NPU

请按 [Ascend 兼容性矩阵](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
安装相互匹配的 PyTorch、`torch_npu`、CANN、固件及驱动。`torch_npu` 为可选组件，
本包不会自动安装。频谱模型在 NPU 上使用专用 STFT/ISTFT 实现；CPU、CUDA 和 MPS
继续使用原有路径。

循环网络模型需要在导入 `torch_npu` 后启用 CPU 回退：

```python
import torch
from pymss_core.npu_rnn import install_npu_rnn_fallback

install_npu_rnn_fallback()
model.eval()
with torch.inference_mode():
    output = model(audio)
```

回退仅处理 NPU 上的 RNN/LSTM/GRU 输入，支持打包序列。每次调用会将循环层权重及输入
复制到 CPU，低精度输入使用 float32 计算，结果返回输入的设备及数据类型。
此回退仅支持推理；启用梯度的 NPU 调用会明确报错。CPU/CUDA 的循环层执行及自动微分
保持原生行为。`pymss` 选择 NPU 时会自动启用此回退。

## 公共 API

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

- 使用 `get_model_from_config("auto", "config.yaml")` 可根据 YAML 识别架构；`detect_model_type(config)` 为已加载的配置提供同一识别能力。未知或冲突的架构会抛出 `ModelTypeDetectionError`，手动指定模型类型可覆盖自动识别。
- `bs_roformer` 也可加载 BS PolarFormer checkpoint，匹配的 YAML 配置需保留 `model.use_pope: true`

推理结束后，在释放模型前清理模型持有的派生缓存：

```python
clear_model_runtime_caches(model)
```

清理会保留模型参数和注册的 buffer，调用模块自身的 `clear_runtime_cache()` hook，并释放已知推理缓存的引用。如果某个 hook 失败，会继续清理其余模块，最后重新抛出首个错误。

MLX 的 overlap-add 索引、Hann 窗口等进程级共享缓存，以及后端分配器缓存，需要单独管理，不属于此模型级清理函数的范围。

## 包边界

包含：

- YAML 配置加载与 `AttrDict`
- `pymss_core.modules` 下的 PyTorch 模型定义
- 支持部分模型 forward 路径的可选 MLX backend 实现
- 模型工厂：`get_model_from_config(model_type, config_path)`
- 常见 MSS checkpoint 容器的加载与 state dict 解析
- 模型持有的推理缓存清理
- 构造模型结构所需的少量内部 DSP 数学函数
- VR 网络结构和 `resources/vr_modelparams/*.json`

不包含：

- 音频文件解码/编码
- 重采样、预处理和完整推理 DSP pipeline
- tensor 级别的分块 demix runtime
- 模型 catalog、别名、下载及模型文件缓存管理
- CLI、server、WebUI 和接口 schema
- dataset、augmentation、loss、metric 和 trainer 代码
- 默认安装不会引入 MLX、Librosa、tqdm、Lightning、FastAPI、Uvicorn、
  PyAV、WandB 等训练或产品层依赖

## 关于 VR JSON

`pymss_core/resources/vr_modelparams/*.json` 不是新增的下载数据，而是从旧路径
`pymss/resources/vr_modelparams/*.json` 移动过来的 VR/UVR 模型结构参数。它们是
VR 网络结构的一部分，因此保留在 core 中。

已删除的 `model_catalog.json` 属于模型下载和 catalog 管理层，不属于 core。
完整的 `VRSeparator`、重采样、STFT/ISTFT pipeline 属于上层推理包。

## 仓库分工

```text
pymss-core
  共享的模型、配置、checkpoint 层

pymss
  基于 pymss-core 的用户推理包，包含音频 IO 和 demix

pymsst
  基于 pymss-core 的训练包，包含训练数据、loss 和 runtime
```
