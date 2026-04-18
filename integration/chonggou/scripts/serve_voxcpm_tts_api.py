#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, HTTPException
import torch
import torchaudio
import uvicorn

from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig as VoxCPMLoRAConfig
from voxcpm.model.voxcpm2 import LoRAConfig as VoxCPM2LoRAConfig


def _soundfile_load(uri: str, *args, **kwargs):
    audio, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(np.ascontiguousarray(audio.T))
    return tensor, int(sample_rate)


torchaudio.load = _soundfile_load


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "voxcpm_distill": {
        "base_model": os.environ.get("VOXCPM_DISTILL_BASE", "/root/voxcpm_distill/models/VoxCPM-0.5B"),
        "lora_weights": os.environ.get(
            "VOXCPM_DISTILL_LORA",
            "/root/voxcpm_distill/runs/student_en_240/checkpoints/latest/lora_weights.safetensors",
        ),
        "lora_config": os.environ.get(
            "VOXCPM_DISTILL_LORA_CONFIG",
            "/root/voxcpm_distill/runs/student_en_240/checkpoints/latest/lora_config.json",
        ),
        "model_name": "VoxCPM-0.5B-distill",
    },
    "voxcpm_base": {
        "base_model": os.environ.get("VOXCPM_BASE_MODEL", "/root/voxcpm_distill/models/VoxCPM-0.5B"),
        "lora_weights": None,
        "lora_config": None,
        "model_name": "VoxCPM-0.5B-base",
    },
}

VOICE_STYLE_PRESETS: dict[str, dict[str, str]] = {
    "sweet": {
        "prompt_wav_path": os.environ.get(
            "VOXCPM_SWEET_PROMPT_WAV",
            os.environ.get(
                "LOCAL_VOXCPM_SWEET_PROMPT_WAV",
                str(Path("app/assistant_data/voice_refs/sweet_female_prompt.wav").resolve()),
            ),
        ),
        "prompt_text": os.environ.get(
            "VOXCPM_SWEET_PROMPT_TEXT",
            "\u4f60\u597d\u5440\uff0c\u6211\u4f1a\u4e00\u76f4\u6e29\u67d4\u5730\u966a\u7740\u4f60\uff0c\u6162\u6162\u804a\u5c31\u597d\u3002",
        ),
    },
    "gentle": {
        "prompt_wav_path": os.environ.get(
            "VOXCPM_GENTLE_PROMPT_WAV",
            os.environ.get(
                "VOXCPM_SWEET_PROMPT_WAV",
                os.environ.get(
                    "LOCAL_VOXCPM_SWEET_PROMPT_WAV",
                    str(Path("app/assistant_data/voice_refs/sweet_female_prompt.wav").resolve()),
                ),
            ),
        ),
        "prompt_text": os.environ.get(
            "VOXCPM_GENTLE_PROMPT_TEXT",
            "\u4f60\u597d\u5440\uff0c\u6211\u4f1a\u7528\u6e29\u67d4\u3001\u81ea\u7136\u3001\u6e05\u6670\u7684\u58f0\u97f3\u966a\u4f60\u804a\u5929\u3002",
        ),
    }
}


app = FastAPI(title="VoxCPM Local TTS Service")
_MODELS: dict[str, VoxCPM] = {}
_MODELS_LOCK = Lock()
_AUDIO_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_AUDIO_CACHE_LOCK = Lock()
_PROMPT_CACHE: dict[str, Any] = {}
_PROMPT_CACHE_LOCK = Lock()
_TTS_HEALTH: dict[str, Any] = {
    "ok": False,
    "ready": False,
    "provider": "voxcpm_distill",
    "model_name": MODEL_REGISTRY["voxcpm_distill"]["model_name"],
    "detail": "local_voxcpm_tts_service_loading",
    "probe": {},
}
_TTS_HEALTH_LOCK = Lock()
HOST = os.environ.get("VOXCPM_TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOXCPM_TTS_PORT", "18884"))


def _load_lora_config(config_path: str | None, base_model_path: str):
    if not config_path or not Path(config_path).exists():
        return None
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = raw.get("lora_config", raw)
    config = json.loads(Path(base_model_path, "config.json").read_text(encoding="utf-8"))
    architecture = str(config.get("architecture", "voxcpm")).lower()
    if architecture == "voxcpm2":
        return VoxCPM2LoRAConfig(**cfg)
    return VoxCPMLoRAConfig(**cfg)


def get_model(provider: str) -> VoxCPM:
    normalized = provider if provider in MODEL_REGISTRY else "voxcpm_distill"
    with _MODELS_LOCK:
        if normalized in _MODELS:
            return _MODELS[normalized]
        spec = MODEL_REGISTRY[normalized]
        lora_config = _load_lora_config(spec.get("lora_config"), spec["base_model"])
        model = VoxCPM.from_pretrained(
            spec["base_model"],
            load_denoiser=False,
            optimize=False,
            lora_config=lora_config,
            lora_weights_path=spec.get("lora_weights"),
        )
        _MODELS[normalized] = model
        return model


def _prepare_prompt_wav_path(style_key: str, preset: dict[str, str]) -> str | None:
    source = str(preset.get("prompt_wav_path") or "").strip()
    if not source:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    if style_key != "gentle":
        return str(source_path)
    try:
        for suffix in ("_30", "_40", "_20", "_15"):
            candidate = source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}")
            if candidate.exists():
                return str(candidate)
        info = sf.info(str(source_path))
        max_duration = float(os.environ.get("VOXCPM_GENTLE_PROMPT_MAX_SEC", "3.0"))
        if info.duration <= max_duration + 1e-3:
            return str(source_path)
        trimmed_path = source_path.with_name(f"{source_path.stem}.gentle_{int(max_duration * 10):02d}.wav")
        if trimmed_path.exists():
            return str(trimmed_path)
        audio, sample_rate = sf.read(str(source_path), dtype="float32")
        frames = max(1, min(len(audio), int(sample_rate * max_duration)))
        sf.write(str(trimmed_path), audio[:frames], sample_rate)
        return str(trimmed_path)
    except Exception:
        return str(source_path)


def _prompt_cache_key(provider: str, style_key: str, preset: dict[str, str] | None) -> str:
    payload = {
        "provider": provider,
        "style": style_key,
        "prompt_wav_path": str((preset or {}).get("prompt_wav_path") or ""),
        "prompt_text": str((preset or {}).get("prompt_text") or ""),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _get_prompt_cache(provider: str, style_key: str) -> Any:
    preset = VOICE_STYLE_PRESETS.get(style_key)
    cache_key = _prompt_cache_key(provider, style_key, preset)
    with _PROMPT_CACHE_LOCK:
        cached = _PROMPT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        model = get_model(provider)
        prompt_cache = None
        if preset:
            prompt_wav_path = _prepare_prompt_wav_path(style_key, preset)
            prompt_text = str(preset.get("prompt_text") or "").strip()
            if prompt_wav_path and Path(prompt_wav_path).exists() and prompt_text:
                prompt_cache = model.tts_model.build_prompt_cache(
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav_path,
                )
        _PROMPT_CACHE[cache_key] = prompt_cache
        return prompt_cache


def _cache_key(provider: str, voice_style: str, text: str, cfg_value: float, inference_timesteps: int) -> str:
    return json.dumps(
        {
            "provider": provider,
            "voice_style": voice_style,
            "text": text,
            "cfg_value": round(float(cfg_value), 4),
            "inference_timesteps": int(inference_timesteps),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _cache_audio_result(key: str, result: dict[str, Any]) -> None:
    with _AUDIO_CACHE_LOCK:
        _AUDIO_CACHE[key] = result
        _AUDIO_CACHE.move_to_end(key)
        while len(_AUDIO_CACHE) > 32:
            _AUDIO_CACHE.popitem(last=False)


def _get_cached_audio_result(key: str) -> dict[str, Any] | None:
    with _AUDIO_CACHE_LOCK:
        cached = _AUDIO_CACHE.get(key)
        if cached is None:
            return None
        _AUDIO_CACHE.move_to_end(key)
        return dict(cached)


def _warmup_generation(provider: str) -> None:
    normalized = provider if provider in MODEL_REGISTRY else "voxcpm_distill"
    try:
        get_model(normalized)
        _get_prompt_cache(normalized, "gentle")
    except Exception as error:
        print(f"voxcpm warmup skipped: {error}")


def _set_tts_health(ready: bool, detail: str, provider: str, probe: dict[str, Any] | None = None) -> None:
    normalized = provider if provider in MODEL_REGISTRY else "voxcpm_distill"
    with _TTS_HEALTH_LOCK:
        _TTS_HEALTH.update(
            {
                "ok": bool(ready),
                "ready": bool(ready),
                "provider": normalized,
                "model_name": MODEL_REGISTRY[normalized]["model_name"],
                "detail": str(detail or "").strip(),
                "probe": dict(probe or {}),
            }
        )


def _get_tts_health() -> dict[str, Any]:
    with _TTS_HEALTH_LOCK:
        return dict(_TTS_HEALTH)


def _collect_probe_stats(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    duration_sec = float(len(wav) / max(1, sample_rate))
    rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float32))))) if len(wav) else 0.0
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    silence_ratio = float(np.mean(np.abs(wav) < 0.003)) if len(wav) else 1.0
    clipped_ratio = float(np.mean(np.abs(wav) >= 0.995)) if len(wav) else 0.0
    return {
        "duration_sec": round(duration_sec, 4),
        "rms": round(rms, 6),
        "silence_ratio": round(silence_ratio, 6),
        "peak": round(peak, 6),
        "clipped_ratio": round(clipped_ratio, 6),
        "sample_rate": int(sample_rate),
    }


def _resolve_generation_limits(text: str) -> tuple[int, int]:
    compact_len = len("".join(str(text or "").split()))
    if compact_len <= 2:
        return 1, 192
    if compact_len <= 8:
        return 2, 320
    if compact_len <= 16:
        return 2, 480
    if compact_len <= 32:
        return 2, 768
    scaled = max(768, min(2048, compact_len * 32))
    return 2, int(scaled)


def _probe_is_healthy(stats: dict[str, Any]) -> bool:
    duration_sec = float(stats.get("duration_sec") or 0.0)
    rms = float(stats.get("rms") or 0.0)
    silence_ratio = float(stats.get("silence_ratio") or 1.0)
    peak = float(stats.get("peak") or 0.0)
    clipped_ratio = float(stats.get("clipped_ratio") or 0.0)
    if duration_sec < 0.45 or duration_sec > 20:
        return False
    if rms < 0.005:
        return False
    if silence_ratio > 0.94:
        return False
    if peak <= 0.02 or peak >= 1.0:
        return False
    if clipped_ratio > 0.02:
        return False
    return True


def _run_tts_probe(provider: str) -> None:
    normalized = provider if provider in MODEL_REGISTRY else "voxcpm_distill"
    try:
        model = get_model(normalized)
        prompt_cache = _get_prompt_cache(normalized, "gentle")
        generate_result = model.tts_model._generate_with_prompt_cache(
            target_text="你好，我已经准备好了。",
            prompt_cache=prompt_cache,
            min_len=2,
            max_len=768,
            inference_timesteps=5,
            cfg_value=1.35,
            retry_badcase=False,
            streaming=False,
        )
        wav = next(generate_result)[0].squeeze(0).cpu().numpy().astype(np.float32)
        stats = _collect_probe_stats(wav, int(model.tts_model.sample_rate))
        if not _probe_is_healthy(stats):
            _set_tts_health(False, "local_voxcpm_tts_probe_failed", normalized, stats)
            return
        _set_tts_health(True, "local_voxcpm_tts_probe_ready", normalized, stats)
    except Exception as error:
        _set_tts_health(False, f"local_voxcpm_tts_probe_error: {error}", normalized, {})


@app.on_event("startup")
def warmup_default_model():
    provider = os.environ.get("VOXCPM_WARMUP_PROVIDER", "voxcpm_distill").strip() or "voxcpm_distill"
    # Load the model on the main server thread. Loading CUDA models from a
    # background worker was causing the Windows sidecar to crash shortly after
    # startup, which made the packaged app fall back to system TTS.
    _warmup_generation(provider)
    _run_tts_probe(provider)


@app.get("/api/tts/health")
async def health():
    base_model = str(MODEL_REGISTRY["voxcpm_distill"].get("base_model", "") or "")
    lora_weights = str(MODEL_REGISTRY["voxcpm_distill"].get("lora_weights", "") or "")
    normalized = "voxcpm_distill"
    health = _get_tts_health()
    return {
        "ok": bool(health.get("ok")),
        "ready": bool(health.get("ready")) and normalized in _MODELS,
        "provider": str(health.get("provider") or normalized),
        "model_name": str(health.get("model_name") or MODEL_REGISTRY["voxcpm_distill"]["model_name"]),
        "detail": str(health.get("detail") or "local_voxcpm_tts_service_loading"),
        "assets_present": Path(base_model).exists() and Path(lora_weights).exists(),
        "probe": health.get("probe") or {},
    }


@app.post("/api/tts/synthesize")
async def synthesize(payload: dict = Body(...)):
    text = str(payload.get("text", "") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")
    provider = str(payload.get("provider", "voxcpm_distill") or "voxcpm_distill")
    normalized = provider if provider in MODEL_REGISTRY else "voxcpm_distill"
    cfg_value = float(payload.get("cfg_value", 2.0))
    inference_timesteps = int(payload.get("inference_timesteps", 10))
    style_key = str(payload.get("voice_style", "sweet") or "").strip().lower()
    health = _get_tts_health()
    if normalized == str(health.get("provider") or normalized) and not bool(health.get("ready")):
        raise HTTPException(status_code=503, detail=str(health.get("detail") or "local_voxcpm_tts_not_ready"))
    if style_key == "gentle":
        inference_timesteps = max(inference_timesteps, 4)
        cfg_value = max(cfg_value, 1.3)
    min_len, max_len = _resolve_generation_limits(text)
    cache_key = _cache_key(normalized, style_key, text, cfg_value, inference_timesteps)
    cached = _get_cached_audio_result(cache_key)
    if cached is not None:
        return cached

    model = get_model(normalized)
    prompt_cache = _get_prompt_cache(normalized, style_key)
    generate_result = model.tts_model._generate_with_prompt_cache(
        target_text=text,
        prompt_cache=prompt_cache,
        min_len=min_len,
        max_len=max_len,
        inference_timesteps=inference_timesteps,
        cfg_value=cfg_value,
        retry_badcase=False,
        streaming=False,
    )
    wav = next(generate_result)[0].squeeze(0).cpu().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, wav, model.tts_model.sample_rate, format="WAV")
    audio_bytes = buffer.getvalue()
    spec = MODEL_REGISTRY[normalized]
    result = {
        "ok": True,
        "provider": normalized,
        "model_name": spec["model_name"],
        "sample_rate": int(model.tts_model.sample_rate),
        "mime_type": "audio/wav",
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }
    _cache_audio_result(cache_key, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
