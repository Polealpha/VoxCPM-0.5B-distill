# VoxCPM-0.5B-distill

Complete local deployment bundle for the `VoxCPM-0.5B` base model plus the distilled LoRA adapter used by the `chonggou` desktop runtime.

## Included files

- `base_model/`
  - upstream `VoxCPM-0.5B` base model files
- `lora_weights.safetensors`
- `lora_config.json`
- `voice_refs/`
  - reference voice prompt wav files used by the shipped `sweet/gentle` presets
- `integration/chonggou/`
  - stable local TTS/STT/OpenClaw wiring files from the desktop app

## Base model

This repository now includes the upstream `VoxCPM-0.5B` base model under `base_model/`.

Large model artifacts are stored through **Git LFS**. Before pulling, run:

```bash
git lfs install
git lfs pull
```

The upstream VoxCPM weights and code are released under Apache-2.0, and this repository keeps that license.

## LoRA config

Current adapter config:

- rank: `32`
- alpha: `32`
- dropout: `0.0`
- LM target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- DiT target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- proj modules: `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj`

## Quick start

1. Clone this repository.
2. Run `git lfs pull`.
3. Point your local runtime to:

```text
VOXCPM_DISTILL_BASE=<path-to-this-repo>\base_model
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

base_model = Path(r".\base_model")
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

## Stable integration files

`integration/chonggou/` contains the current desktop wiring that was used locally:

- Electron startup/runtime wiring
- local VoxCPM TTS service
- local SenseVoice STT service
- backend assistant/OpenClaw bridge files

## Notes

- This repository is intended for local/runtime integration and reproducible deployment.
- The reference voice files are included because the desktop runtime depends on them for the shipped `sweet/gentle` prompt presets.
