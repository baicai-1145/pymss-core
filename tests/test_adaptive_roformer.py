"""Contracts for unknown-count separation, including zero speakers and stop tokens."""
import pytest
import torch
from torch import nn

from pymss_core.modules.bs_roformer import AdaptiveBSRoformer, AdaptiveMelBandRoformer
from pymss_core.modules.bs_roformer.adaptive import AdaptiveSeparationOutput
from pymss_core.utils import get_model_from_config


def tiny_model(mel=False, stereo=False):
    kwargs = {'dim': 16, 'depth': 1, 'max_speakers': 3, 'decoder_heads': 2, 'memory_frames': 8,
                  'stereo': stereo, 'heads': 2, 'dim_head': 8, 'time_transformer_depth': 1,
                  'freq_transformer_depth': 1, 'stft_n_fft': 32, 'stft_win_length': 32,
                  'stft_hop_length': 8, 'attn_dropout': 0., 'ff_dropout': 0.}
    kwargs.update({'num_bands': 4, 'sample_rate': 8000} if mel else {'freqs_per_bands': (4, 4, 4, 5)})
    return (AdaptiveMelBandRoformer if mel else AdaptiveBSRoformer)(**kwargs)


@pytest.mark.parametrize('mel', [False, True])
@pytest.mark.parametrize('stereo', [False, True])
def test_shapes_backward_and_checkpoint(mel, stereo):
    torch.manual_seed(3)
    model = tiny_model(mel, stereo)
    channels = 2 if stereo else 1
    x = torch.randn(2, channels, 129)
    out = model(x)
    assert out.audio.shape == (2, 3, channels, 129)
    assert out.existence_logits.shape == (2, 4)
    assert torch.isfinite(out.audio).all()
    (out.audio.square().mean() + out.existence_logits.square().mean()).backward()
    for component in [model.backbone.band_split, model.attractor_cell, model.film, model.existence_head]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    restored = tiny_model(mel, stereo).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x).audio, restored(x).audio)


class ScriptedExistence(nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.counts = counts
        self.step = 0

    def forward(self, x):
        logits = x.new_tensor([10. if self.step < n else -10. for n in self.counts])[:, None]
        self.step += 1
        return logits


@pytest.mark.parametrize('counts', [[0, 0], [1, 3], [2, 1]])
def test_ragged_inference_stops_per_sample(counts):
    model = tiny_model().eval()
    model.existence_head = ScriptedExistence(counts)
    tracks = model.separate(torch.randn(2, 1, 129))
    assert [track.shape for track in tracks] == [(n, 1, 129) for n in counts]
    assert model.existence_head.step == max(counts) + 1


def test_stop_is_prefix_not_thresholded_slot_filter():
    out = AdaptiveSeparationOutput(torch.zeros(1, 3, 1, 64), torch.tensor([[8., -8., 8., -8.]]))
    assert out.counts().tolist() == [1]


def test_explicit_decode_limit_and_validation():
    model = tiny_model().eval()
    model.existence_head = ScriptedExistence([5])
    assert model.separate(torch.randn(1, 1, 129), speaker_limit=5)[0].shape == (5, 1, 129)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 129), speaker_limit=0)
    with pytest.raises(ValueError):
        model.separate(torch.randn(1, 1, 129), threshold=1.1)


@pytest.mark.parametrize('mel', [False, True])
def test_inference_then_training_discards_inference_only_caches(mel):
    model = tiny_model(mel).eval()
    audio = torch.randn(1, 1, 129)
    with torch.inference_mode():
        model(audio)
    model.train()
    output = model(audio)
    (output.audio.square().mean() + output.existence_logits.square().mean()).backward()
    assert model.backbone.band_split.to_features[0][1].weight.grad is not None


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_silent_audio_has_finite_training_gradients(mixed_precision):
    model = tiny_model().train()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        output = model(torch.zeros(1, 1, 129))
        loss = output.audio.square().mean() + output.existence_logits.float().square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_auto_config_load(tmp_path):
    import yaml
    model = tiny_model()
    config = {'model_type': 'adaptive_bs_roformer', 'model': {
        'dim': 16, 'depth': 1, 'max_speakers': 3, 'decoder_heads': 2, 'memory_frames': 8,
        'heads': 2, 'dim_head': 8, 'freqs_per_bands': [4, 4, 4, 5],
        'stft_n_fft': 32, 'stft_win_length': 32, 'stft_hop_length': 8}}
    path = tmp_path / 'model.yaml'
    path.write_text(yaml.safe_dump(config))
    loaded, _ = get_model_from_config('auto', path)
    assert isinstance(loaded, type(model))
