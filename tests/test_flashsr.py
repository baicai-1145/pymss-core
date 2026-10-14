import torch
import yaml

from pymss_core import detect_model_type, get_model_from_config
from pymss_core.modules.flashsr import FlashSR


CONFIG = {
    "model": {
        "spec_channels": 128, "segment_size": 30, "resblock": "0",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [3], "upsample_initial_channel": 32, "upsample_kernel_sizes": [3],
    },
}


def test_flashsr_is_detected_and_instantiated(tmp_path):
    config_path = tmp_path / "flashsr.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    assert detect_model_type(CONFIG) == "flashsr"
    model, _ = get_model_from_config("flashsr", config_path)
    assert isinstance(model, FlashSR)


def test_flashsr_upsamples_three_times():
    model = FlashSR().eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 1, 1600)).shape == (1, 1, 4800)


def test_flashsr_loads_the_published_checkpoint_layout(tmp_path):
    model = FlashSR()
    torch.save({"model": model.state_dict()}, tmp_path / "upsampler.pth")
    checkpoint = torch.load(tmp_path / "upsampler.pth", map_location="cpu", weights_only=True)
    assert model.load_state_dict(checkpoint["model"]).missing_keys == []
