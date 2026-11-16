"""Unknown-count separation with recurrent speaker attractors and one shared mask head."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .bs_roformer import BSRoformer
from .common import forward_roformer_features, istft_roformer, mask_to_complex_shape, stft_roformer
from .mel_band_roformer import MelBandRoformer


@dataclass
class AdaptiveSeparationOutput:
    audio: torch.Tensor                 # [batch, candidate_speakers, channels, samples]
    existence_logits: torch.Tensor      # [batch, candidates + 1], including end-of-sequence

    def counts(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        valid = self.existence_logits[:, :self.audio.shape[1]].sigmoid() >= threshold
        return valid.long().cumprod(dim=1).sum(dim=1)

    def at_limit(self, threshold=0.5):
        """True when the extra candidate still exists after exhausting the decode budget."""
        n = self.audio.shape[1]
        return (self.counts(threshold) == n) & (self.existence_logits[:, n].sigmoid() >= threshold)


class AdaptiveRoformer(nn.Module):
    backbone_class = BSRoformer

    def __init__(self, dim, *, max_speakers=5, decoder_heads=4, memory_frames=256, **kwargs):
        super().__init__()
        if max_speakers < 1 or memory_frames < 1 or decoder_heads < 1 or dim % decoder_heads:
            raise ValueError("positive speaker/memory/head counts and dim divisible by decoder_heads are required")
        if kwargs.pop("num_stems", 1) != 1:
            raise ValueError("adaptive models use one shared mask head; use max_speakers instead of num_stems")
        self.max_speakers, self.memory_frames = int(max_speakers), int(memory_frames)
        self.backbone = self.backbone_class(dim=dim, num_stems=1, **kwargs)
        self.audio_channels = self.backbone.audio_channels
        self.frequency_pool = nn.Linear(dim, 1)
        self.start_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.history_attention = nn.MultiheadAttention(dim, decoder_heads, batch_first=True)
        self.mixture_attention = nn.MultiheadAttention(dim, decoder_heads, batch_first=True)
        self.attractor_cell = nn.GRUCell(dim, dim)
        self.attractor_norm = nn.LayerNorm(dim)
        self.existence_head = nn.Linear(dim, 1)
        self.film = nn.Linear(dim, dim * 2)
        self.refine = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def train(self, mode=True):
        if mode and not self.training:
            from pymss_core.runtime_cache import clear_model_runtime_caches
            # Validation can populate positional/window caches with inference-only tensors.
            clear_model_runtime_caches(self)
        return super().train(mode)

    def _memory(self, features):
        weights = self.frequency_pool(features).softmax(dim=2)
        memory = (features * weights).sum(dim=2)
        if memory.shape[1] > self.memory_frames:
            memory = F.adaptive_avg_pool1d(memory.transpose(1, 2), self.memory_frames).transpose(1, 2)
        return memory

    def _decode_mask(self, features, attractor):
        scale, bias = self.film(attractor).chunk(2, dim=-1)
        conditioned = features * (1 + scale[:, None, None]) + bias[:, None, None]
        conditioned = conditioned + F.gelu(self.refine(conditioned.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        mask = self.backbone.mask_estimators[0](conditioned).unsqueeze(1)
        return mask_to_complex_shape(mask)

    def forward(self, raw_audio, *, speaker_limit=None, stop_early=False, threshold=0.5):
        """Training returns padded candidates; inference may stop before the safety limit.

        speaker_limit may exceed the training count, but that extrapolation needs evaluation.
        Audio is encoded once. No identity state is retained between separate calls.
        """
        limit = self.max_speakers if speaker_limit is None else speaker_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("speaker_limit must be a positive integer")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if raw_audio.ndim not in (2, 3) or raw_audio.shape[-1] < 1:
            raise ValueError("audio must have shape [batch, samples] or [batch, channels, samples]")
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)
        if raw_audio.shape[1] != self.audio_channels:
            raise ValueError(f"expected {self.audio_channels} audio channels")
        # Reflect-padding in torch.stft requires more than n_fft/2 input samples.
        length = raw_audio.shape[-1]
        raw_audio = F.pad(raw_audio.float(), (0, max(0, self.backbone.stft_kwargs['n_fft'] // 2 + 1 - length)))
        spectrum, context = stft_roformer(self.backbone, raw_audio)
        selected = spectrum[:, self.backbone.freq_indices] if isinstance(self.backbone, MelBandRoformer) else spectrum
        features = forward_roformer_features(self.backbone, selected)
        memory = self._memory(features)
        state = self.start_query[None].expand(features.shape[0], -1).to(features.dtype)
        alive = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        history, masks, logits = [], [], []
        for step in range(limit + 1):
            query = state[:, None]
            previous = torch.stack(history, dim=1) if history else query
            query = query + self.history_attention(query, previous, previous, need_weights=False)[0]
            context_vector = self.mixture_attention(query, memory, memory, need_weights=False)[0][:, 0]
            state = self.attractor_cell(context_vector, state)
            attractor = self.attractor_norm(state + context_vector)
            logits.append(self.existence_head(attractor).squeeze(-1))
            if step == limit:
                break
            alive = alive & (logits[-1].sigmoid() >= threshold)
            if stop_early and not alive.any():
                break
            history.append(attractor)
            masks.append(self._decode_mask(features, attractor))
        if masks:
            masks = torch.cat(masks, dim=1).float().contiguous()
            if isinstance(self.backbone, MelBandRoformer):
                estimates = self.backbone.apply_masks(spectrum, masks)
            else:
                estimates = torch.view_as_complex(spectrum.contiguous())[:, None] * torch.view_as_complex(masks)
            audio = istft_roformer(self.backbone, estimates, context, context.audio_length)
            if audio.ndim == 3:
                audio = audio.unsqueeze(1)
            audio = audio[..., :length]
        else:
            audio = raw_audio.new_empty(raw_audio.shape[0], 0, self.audio_channels, length)
        return AdaptiveSeparationOutput(audio, torch.stack(logits, dim=1))

    @torch.inference_mode()
    def separate(self, raw_audio, *, speaker_limit=None, threshold=0.5):
        """Return one [speakers, channels, samples] tensor per input; call eval() first."""
        output = self(raw_audio, speaker_limit=speaker_limit, stop_early=True, threshold=threshold)
        return [audio[:int(count)] for audio, count in zip(output.audio, output.counts(threshold))]


class AdaptiveBSRoformer(AdaptiveRoformer):
    backbone_class = BSRoformer


class AdaptiveMelBandRoformer(AdaptiveRoformer):
    backbone_class = MelBandRoformer
