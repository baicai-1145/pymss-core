import torch
import pytest
import yaml

from pymss_core import detect_model_type, get_model_from_config, load_model_weights, load_state_dict
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


@pytest.mark.parametrize("from_path", [False, True])
def test_flashsr_loads_the_published_checkpoint_layout(tmp_path, from_path):
    model = FlashSR()
    checkpoint = {"model": model.state_dict(), "iteration": 1, "optimizer": None, "learning_rate": 0.0001}
    path = tmp_path / "upsampler.pth"
    torch.save(checkpoint, path)
    loaded = FlashSR()
    result = load_model_weights(loaded, path if from_path else checkpoint, model_type="flashsr")
    assert result.missing_keys == result.unexpected_keys == []
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_flashsr_load_state_dict_unwraps_the_published_layout(tmp_path):
    state_dict = FlashSR().state_dict()
    path = tmp_path / "upsampler.pth"
    torch.save({"model": state_dict}, path)
    loaded = load_state_dict(path, model_type="flashsr", weights_only=True)
    assert loaded.keys() == state_dict.keys()
