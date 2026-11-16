# Adaptive speaker RoFormer

`adaptive_bs_roformer` and `adaptive_mel_band_roformer` separate an unknown number of speaker identities without enrollment audio. They are experimental architectures, not pretrained quality guarantees.

The existing spectral frontend, axial time/frequency RoFormer trunk and inverse STFT remain. A recurrent decoder reads frequency-pooled, temporally compressed trunk features and its previous attractors. Each attractor modulates the full-resolution features through FiLM and a depthwise time/frequency refinement layer. One shared band mask estimator predicts a complex mask per speaker. A sigmoid existence head supplies an end-of-sequence decision. Mel-band masks retain the original overlap averaging.

The shared trunk is encoded once per call. There is no persistent speaker state or temporary forward mutation; calls do not share identities. Original model state dictionaries keep their layout. Adaptive checkpoints have a `backbone.` prefix and additional decoder parameters, so original checkpoints are not directly interchangeable.

```python
import torch
from pymss_core import get_model_from_config

model, config = get_model_from_config("auto", "adaptive_bs_roformer_l4.yaml")
model.eval()
audio = torch.randn(2, 1, 64000)

# Differentiable padded training output, including one extra stop candidate.
output = model(audio)
assert output.audio.shape == (2, 5, 1, 64000)
assert output.existence_logits.shape == (2, 6)

# Ragged inference: one [detected_speakers, channels, samples] tensor per item.
tracks = model.separate(audio, threshold=0.5)
```

`max_speakers` sets the default training/unroll budget. `speaker_limit` can override the inference budget without adding parameters; quality beyond the training range is unverified. The extra existence prediction reports whether the budget was exhausted while another speaker was still predicted. `output.at_limit()` exposes that condition. Existence decisions form a prefix: once a speaker is absent, later candidates are not returned. Each batch item has its own stop position. Zero speakers returns a zero-length speaker axis.

Training always unfolds the padded candidates regardless of predicted existence. For N identities, the first N candidates are matched to the N references by a permutation-invariant objective; their existence labels are one and subsequent labels are zero. Silent targets use an energy loss rather than SI-SDR. See the companion pymsst branch for dynamic mixtures, matching, validation and L4 recipes.

Limitations: training only covers explicitly sampled counts; no unlimited-count claim is made. The full-clip trunk still has temporal attention costs. Streaming, automatic long-recording chunk stitching and cross-call identity tracking are not implemented. Speech training does not establish singing-voice performance. The initial frontend recipe is 16 kHz mono; the core architecture also supports stereo.

Related work: [encoder-decoder attractors](https://www.audiolabs-erlangen.de/resources/2022-Interspeech-BSS-EDA) and [one-and-rest recursive separation](https://www.isca-archive.org/interspeech_2019/takahashi19_interspeech.pdf). Combining these established ideas with this backbone does not itself establish research novelty.
