# chonggou integration

This directory contains the local desktop wiring files that were used to connect:

- `VoxCPM-0.5B + distill LoRA` for TTS
- `SenseVoiceSmall` for STT
- the local backend bridge
- the Electron startup/runtime bridge

Included files:

- `scripts/serve_voxcpm_tts_api.py`
- `scripts/serve_sensevoice_stt_api.py`
- `backend/assistant_service.py`
- `backend/desktop_speech.py`
- `backend/main.py`
- `backend/openclaw_gateway.py`
- `backend/settings.py`
- `app_windows/electron/main.cjs`

These are published for teammate sync and local reproduction.
