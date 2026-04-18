# VoxCPM-0.5B-distill

LoRA-distilled weights for `VoxCPM-0.5B`, extracted from the local `chonggou` desktop runtime.

This repository publishes the distilled adapter and reference voice assets only. The upstream base model is **not** bundled here.

## Included files

- `lora_weights.safetensors`
- `lora_config.json`
- `voice_refs/sweet_female_prompt.wav`
- `voice_refs/sweet_female_prompt_15.wav`
- `voice_refs/sweet_female_prompt_20.wav`
- `voice_refs/sweet_female_prompt_30.wav`
- `voice_refs/sweet_female_prompt_40.wav`

## Base model

Download the official base model separately:

- Upstream model: `VoxCPM-0.5B`
- Suggested local path: `E:\model_cache\VoxCPM-0.5B`

The upstream VoxCPM weights and code are released under Apache-2.0.

## LoRA config

Current adapter config:

- rank: `32`
- alpha: `32`
- dropout: `0.0`
- LM target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- DiT target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- proj modules: `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj`

## Quick start

1. Download the official `VoxCPM-0.5B` base model.
2. Clone this repository.
3. Point your local runtime to:

```text
VOXCPM_DISTILL_BASE=<path-to-VoxCPM-0.5B>
VOXCPM_DISTILL_LORA=<path-to-this-repo>\lora_weights.safetensors
VOXCPM_DISTILL_LORA_CONFIG=<path-to-this-repo>\lora_config.json
VOXCPM_SWEET_PROMPT_WAV=<path-to-this-repo>\voice_refs\sweet_female_prompt.wav
```

## Example

```python
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
import json
from pathlib import Path

base_model = Path(r"E:\model_cache\VoxCPM-0.5B")
repo_dir = Path(r".")

raw = json.loads((repo_dir / "lora_config.json").read_text(encoding="utf-8"))
lora_config = LoRAConfig(**raw["lora_config"])

model = VoxCPM.from_pretrained(
    str(base_model),
    load_denoiser=False,
    optimize=False,
    lora_config=lora_config,
    lora_weights_path=str(repo_dir / "lora_weights.safetensors"),
)
```

## Notes

- This repository is intended for local/runtime integration and reproducible deployment.
- It does not include the upstream full base weights.
- The reference voice files are included because the desktop runtime depends on them for the shipped `sweet/gentle` prompt presets.
