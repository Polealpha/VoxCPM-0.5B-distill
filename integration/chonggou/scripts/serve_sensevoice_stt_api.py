#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from pathlib import Path
from threading import Lock

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


MODEL_DIR = os.environ.get(
    "SENSEVOICE_MODEL_DIR",
    os.environ.get("LOCAL_STT_MODEL_DIR", "/root/sensevoice_stt/SenseVoiceSmall"),
)
DEVICE = os.environ.get("SENSEVOICE_DEVICE", os.environ.get("LOCAL_STT_DEVICE", "cuda:0"))
HOST = os.environ.get("SENSEVOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SENSEVOICE_PORT", "18885"))
_MODEL = None
_LOCK = Lock()


def get_model():
    global _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        _MODEL = AutoModel(
            model=MODEL_DIR,
            trust_remote_code=False,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=DEVICE,
        )
        return _MODEL

app = FastAPI(title="SenseVoice Local STT Service")


@app.on_event("startup")
async def warmup_model():
    try:
        await asyncio.to_thread(get_model)
    except Exception as error:
        print(f"sensevoice warmup skipped: {error}")


@app.get("/api/stt/health")
async def health():
    return {
        "ok": True,
        "ready": Path(MODEL_DIR).exists(),
        "provider": "sensevoice_small",
        "model_name": "SenseVoiceSmall",
        "detail": "local_sensevoice_stt_service_ready",
    }


@app.post("/api/stt/transcribe")
async def transcribe(payload: dict = Body(...)):
    audio_base64 = str(payload.get("audio_base64", "") or "")
    if not audio_base64:
        raise HTTPException(status_code=400, detail="empty_audio_base64")

    audio_bytes = base64.b64decode(audio_base64)
    filename = str(payload.get("filename", "audio.wav") or "audio.wav")
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    t0 = time.time()
    try:
        model = get_model()
        result = model.generate(
            input=temp_path,
            cache={},
            language=str(payload.get("language", "auto") or "auto"),
            use_itn=bool(payload.get("use_itn", True)),
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        raw_text = ""
        if isinstance(result, list) and result:
            raw_text = str(result[0].get("text", "") or "")
        transcript = rich_transcription_postprocess(raw_text)
        return {
            "ok": True,
            "ready": True,
            "provider": "sensevoice_small",
            "model_name": "SenseVoiceSmall",
            "context": str(payload.get("context", "chat") or "chat"),
            "transcript": transcript,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
