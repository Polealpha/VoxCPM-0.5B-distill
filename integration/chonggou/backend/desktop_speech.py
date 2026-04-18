from __future__ import annotations

from array import array
import io
import re
import time
import threading
import wave
from typing import Any, Dict, Optional

try:
    import audioop
except ModuleNotFoundError:
    audioop = None

from engine.core.config import AsrConfig
from engine.nlp.asr_module import AsrModule

from .settings import (
    DESKTOP_STT_BEAM_SIZE,
    DESKTOP_STT_BEST_OF,
    DESKTOP_STT_CHUNK_LENGTH,
    DESKTOP_STT_COMPRESSION_RATIO_THRESHOLD,
    DESKTOP_STT_COMPUTE_TYPE,
    DESKTOP_STT_DEVICE,
    DESKTOP_STT_FALLBACK_PROVIDER,
    DESKTOP_STT_HOTWORDS,
    DESKTOP_STT_INITIAL_PROMPT,
    DESKTOP_STT_LANGUAGE,
    DESKTOP_STT_LOG_PROB_THRESHOLD,
    DESKTOP_STT_MAX_SEC,
    DESKTOP_STT_MODEL_NAME,
    DESKTOP_STT_NO_SPEECH_THRESHOLD,
    DESKTOP_STT_NUM_THREADS,
    DESKTOP_STT_PATIENCE,
    DESKTOP_STT_PREPROCESS,
    DESKTOP_STT_PROVIDER,
    DESKTOP_STT_REPETITION_PENALTY,
    DESKTOP_STT_SILENCE_THRESHOLD,
    DESKTOP_STT_TARGET_PEAK,
    DESKTOP_STT_TRIM_SILENCE,
    DESKTOP_STT_VAD_FILTER,
)


class DesktopSpeechService:
    def __init__(self) -> None:
        self._primary_asr: Optional[AsrModule] = None
        self._fallback_asr: Optional[AsrModule] = None
        self._primary_provider = str(DESKTOP_STT_PROVIDER or "auto").strip().lower()
        self._fallback_provider = str(DESKTOP_STT_FALLBACK_PROVIDER or "sherpa_onnx").strip().lower()
        self._initialized = False
        self._init_lock = threading.Lock()

    def _init_modules(self) -> None:
        primary = self._build_module(self._primary_provider)
        fallback = None
        if self._fallback_provider and self._fallback_provider != self._primary_provider:
            fallback = self._build_module(self._fallback_provider)
        self._primary_asr = primary
        self._fallback_asr = fallback
        self._initialized = True

    def _ensure_modules(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._init_modules()

    def _build_module(self, provider: str) -> Optional[AsrModule]:
        if provider in {"", "none", "disabled"}:
            return None
        if provider == "auto":
            provider = "faster_whisper"
        if provider in {"whisper", "faster_whisper"}:
            return AsrModule(
                AsrConfig(
                    enabled=True,
                    engine="faster_whisper",
                    language=DESKTOP_STT_LANGUAGE,
                    max_sec=DESKTOP_STT_MAX_SEC,
                    model_name=DESKTOP_STT_MODEL_NAME,
                    device=DESKTOP_STT_DEVICE,
                    compute_type=DESKTOP_STT_COMPUTE_TYPE,
                    beam_size=DESKTOP_STT_BEAM_SIZE,
                    best_of=DESKTOP_STT_BEST_OF,
                    patience=DESKTOP_STT_PATIENCE,
                    repetition_penalty=DESKTOP_STT_REPETITION_PENALTY,
                    no_speech_threshold=DESKTOP_STT_NO_SPEECH_THRESHOLD,
                    log_prob_threshold=DESKTOP_STT_LOG_PROB_THRESHOLD,
                    compression_ratio_threshold=DESKTOP_STT_COMPRESSION_RATIO_THRESHOLD,
                    chunk_length=DESKTOP_STT_CHUNK_LENGTH,
                    vad_filter=DESKTOP_STT_VAD_FILTER,
                    initial_prompt=DESKTOP_STT_INITIAL_PROMPT,
                    hotwords=DESKTOP_STT_HOTWORDS,
                )
            )
        if provider in {"sherpa", "sherpa_onnx"}:
            return AsrModule(
                AsrConfig(
                    enabled=True,
                    engine="sherpa_onnx",
                    language="zh",
                    max_sec=DESKTOP_STT_MAX_SEC,
                    num_threads=DESKTOP_STT_NUM_THREADS,
                )
            )
        if provider == "vosk":
            return AsrModule(
                AsrConfig(
                    enabled=True,
                    engine="vosk",
                    language="zh",
                    max_sec=DESKTOP_STT_MAX_SEC,
                )
            )
        return None

    def status(self) -> Dict[str, Any]:
        if not self._initialized:
            try:
                self._ensure_modules()
            except Exception:
                pass
        if not self._initialized:
            return {
                "ok": True,
                "ready": False,
                "provider_preference": self._primary_provider,
                "fallback_provider": self._fallback_provider,
                "active_provider": self._primary_provider or "disabled",
                "primary_ready": False,
                "primary_engine": self._primary_provider or "disabled",
                "primary_error": None,
                "fallback_ready": False,
                "fallback_engine": self._fallback_provider or "disabled",
                "fallback_error": None,
                "language": DESKTOP_STT_LANGUAGE,
                "max_sec": DESKTOP_STT_MAX_SEC,
                "model_name": DESKTOP_STT_MODEL_NAME,
                "beam_size": DESKTOP_STT_BEAM_SIZE,
                "best_of": DESKTOP_STT_BEST_OF,
                "preprocess_enabled": DESKTOP_STT_PREPROCESS,
                "trim_silence_enabled": DESKTOP_STT_TRIM_SILENCE,
                "initial_prompt_enabled": bool(str(DESKTOP_STT_INITIAL_PROMPT or "").strip()),
                "hotwords_enabled": bool(str(DESKTOP_STT_HOTWORDS or "").strip()),
            }
        primary = self._primary_asr
        fallback = self._fallback_asr
        ready = bool(primary and primary.ready) or bool(fallback and fallback.ready)
        return {
            "ok": True,
            "ready": ready,
            "provider_preference": self._primary_provider,
            "fallback_provider": self._fallback_provider,
            "active_provider": self._resolve_active_provider(),
            "primary_ready": bool(primary and primary.ready),
            "primary_engine": primary.active_engine if primary else "disabled",
            "primary_error": primary.error if primary else None,
            "fallback_ready": bool(fallback and fallback.ready),
            "fallback_engine": fallback.active_engine if fallback else "disabled",
            "fallback_error": fallback.error if fallback else None,
            "language": DESKTOP_STT_LANGUAGE,
            "max_sec": DESKTOP_STT_MAX_SEC,
            "model_name": DESKTOP_STT_MODEL_NAME,
            "beam_size": DESKTOP_STT_BEAM_SIZE,
            "best_of": DESKTOP_STT_BEST_OF,
            "preprocess_enabled": DESKTOP_STT_PREPROCESS,
            "trim_silence_enabled": DESKTOP_STT_TRIM_SILENCE,
            "initial_prompt_enabled": bool(str(DESKTOP_STT_INITIAL_PROMPT or "").strip()),
            "hotwords_enabled": bool(str(DESKTOP_STT_HOTWORDS or "").strip()),
        }

    def transcribe_upload(
        self,
        *,
        audio_bytes: bytes,
        filename: str = "",
        content_type: str = "",
        context: str = "chat",
    ) -> Dict[str, Any]:
        self._ensure_modules()
        started = time.perf_counter()
        pcm, sample_rate, duration_ms = self._decode_audio(audio_bytes, filename=filename, content_type=content_type)
        pcm = self._preprocess_pcm(pcm, sample_rate)
        duration_ms = int((len(pcm) / 2) / max(1, sample_rate) * 1000) if pcm else 0
        if not pcm:
            raise ValueError("empty_audio")
        provider = self._resolve_active_provider()
        transcript = ""
        used_fallback = False
        primary = self._primary_asr
        fallback = self._fallback_asr

        initial_prompt = self._build_context_prompt(context)
        hotwords = self._build_context_hotwords(context)

        if primary and primary.ready:
            transcript = self._normalize_text(
                primary.transcribe(
                    pcm,
                    sample_rate,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                )
            )
            provider = primary.active_engine
        if not transcript and fallback and fallback.ready:
            transcript = self._normalize_text(
                fallback.transcribe(
                    pcm,
                    sample_rate,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                )
            )
            provider = fallback.active_engine
            used_fallback = True
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": bool(transcript),
            "transcript": transcript,
            "provider": provider,
            "used_fallback": used_fallback,
            "duration_ms": duration_ms,
            "latency_ms": latency_ms,
            "context": str(context or "chat"),
            "ready": bool(transcript) or self.status()["ready"],
        }

    def _resolve_active_provider(self) -> str:
        if self._primary_asr and self._primary_asr.ready:
            return self._primary_asr.active_engine
        if self._fallback_asr and self._fallback_asr.ready:
            return self._fallback_asr.active_engine
        if self._primary_asr:
            return self._primary_asr.active_engine
        if self._fallback_asr:
            return self._fallback_asr.active_engine
        return "disabled"

    def _decode_audio(self, audio_bytes: bytes, *, filename: str, content_type: str) -> tuple[bytes, int, int]:
        if not audio_bytes:
            return b"", 16000, 0
        lower_name = str(filename or "").lower()
        lower_type = str(content_type or "").lower()
        if lower_name.endswith(".wav") or "wav" in lower_type or audio_bytes[:4] == b"RIFF":
            return self._decode_wav(audio_bytes)
        raise ValueError("unsupported_audio_format")

    def _decode_wav(self, audio_bytes: bytes) -> tuple[bytes, int, int]:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = int(wav_file.getnchannels() or 1)
            sample_width = int(wav_file.getsampwidth() or 2)
            sample_rate = int(wav_file.getframerate() or 16000)
            frame_count = int(wav_file.getnframes() or 0)
            pcm = wav_file.readframes(frame_count)

        pcm = self._convert_sample_width(pcm, sample_width)
        pcm = self._mix_to_mono(pcm, channels)
        if sample_rate != 16000:
            pcm = self._resample_pcm(pcm, sample_rate, 16000)
            sample_rate = 16000

        max_frames = max(1, int(DESKTOP_STT_MAX_SEC) * sample_rate)
        max_bytes = max_frames * 2
        if len(pcm) > max_bytes:
            pcm = pcm[:max_bytes]

        duration_ms = int((len(pcm) / 2) / max(1, sample_rate) * 1000)
        return pcm, sample_rate, duration_ms

    def _normalize_text(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        normalized = " ".join(raw.split())
        normalized = re.sub(r"([一-龥])\s+([一-龥])", r"\1\2", normalized)
        normalized = re.sub(r"\s+([，。！？：；、,.!?])", r"\1", normalized)
        return normalized.strip()

    def _build_context_prompt(self, context: str) -> str:
        base = str(DESKTOP_STT_INITIAL_PROMPT or "").strip()
        key = str(context or "chat").strip().lower()
        context_prompts = {
            "chat": "这是中文陪伴机器人和主人的自然口语对话，请优先识别控制意图、人物称呼和情绪表达。",
            "activation_assessment": "这是机器人首次激活和人格测评时的中文回答，请准确识别人名、关系、性格描述和生活习惯。",
            "activation_identity": "这是机器人首次登录后的身份确认，请准确识别人名、称呼、与机器人的关系和自我介绍。",
            "command": "这是中文控制指令，请优先准确识别动作、设置项、应用名和设备名。",
        }
        parts = [part for part in [base, context_prompts.get(key)] if str(part or "").strip()]
        return " ".join(parts).strip()

    def _build_context_hotwords(self, context: str) -> str:
        base_terms = [item.strip() for item in str(DESKTOP_STT_HOTWORDS or "").split(",") if item.strip()]
        key = str(context or "chat").strip().lower()
        extra = {
            "chat": ["关怀", "陪伴", "主动关怀"],
            "activation_assessment": ["主人", "人格", "绑定", "测评", "性格"],
            "activation_identity": ["主人", "绑定", "机器人", "小念"],
            "command": ["云台", "跟踪", "设置页", "网易云音乐", "打开", "关闭"],
        }.get(key, [])
        merged: list[str] = []
        for item in [*base_terms, *extra]:
            if item and item not in merged:
                merged.append(item)
        return ",".join(merged)

    def _preprocess_pcm(self, pcm: bytes, sample_rate: int) -> bytes:
        if not pcm or sample_rate <= 0 or not DESKTOP_STT_PREPROCESS:
            return pcm
        processed = pcm
        if DESKTOP_STT_TRIM_SILENCE:
            processed = self._trim_edge_silence(processed, sample_rate)
        processed = self._normalize_peak(processed)
        return processed

    def _trim_edge_silence(self, pcm: bytes, sample_rate: int) -> bytes:
        if not pcm:
            return pcm
        frame_size = 320
        threshold = max(32, int(DESKTOP_STT_SILENCE_THRESHOLD))
        total_samples = len(pcm) // 2
        if total_samples <= frame_size:
            return pcm
        ints = memoryview(pcm).cast("h")
        start = 0
        end = total_samples
        while start + frame_size < total_samples:
            window = ints[start:start + frame_size]
            if max(abs(int(v)) for v in window) >= threshold:
                break
            start += frame_size
        while end - frame_size > start:
            window = ints[end - frame_size:end]
            if max(abs(int(v)) for v in window) >= threshold:
                break
            end -= frame_size
        # Keep a small pad so clipped syllables are less likely.
        pad = int(sample_rate * 0.12)
        start = max(0, start - pad)
        end = min(total_samples, end + pad)
        if end <= start:
            return pcm
        return pcm[start * 2:end * 2]

    def _normalize_peak(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        target_peak = max(2048, int(DESKTOP_STT_TARGET_PEAK))
        if audioop is None:
            samples = array("h")
            samples.frombytes(pcm)
            peak = max((abs(int(v)) for v in samples), default=0)
            if peak <= 0 or peak >= target_peak:
                return pcm
            gain = min(6.0, target_peak / max(1.0, float(peak)))
            if gain <= 1.05:
                return pcm
            scaled = array("h", [max(-32768, min(32767, int(round(v * gain)))) for v in samples])
            return scaled.tobytes()
        try:
            peak = audioop.max(pcm, 2)
        except Exception:
            return pcm
        if peak <= 0:
            return pcm
        if peak >= target_peak:
            return pcm
        gain = min(6.0, target_peak / max(1.0, float(peak)))
        if gain <= 1.05:
            return pcm
        try:
            return audioop.mul(pcm, 2, gain)
        except Exception:
            return pcm

    def _convert_sample_width(self, pcm: bytes, sample_width: int) -> bytes:
        if sample_width == 2:
            return pcm
        if audioop is not None:
            if sample_width == 1:
                return audioop.lin2lin(audioop.bias(pcm, 1, -128), 1, 2)
            if sample_width == 4:
                return audioop.lin2lin(pcm, 4, 2)
        if sample_width == 1:
            samples = array("B", pcm)
            converted = array("h", [((int(v) - 128) << 8) for v in samples])
            return converted.tobytes()
        if sample_width == 4:
            ints = array("i")
            ints.frombytes(pcm)
            converted = array("h", [max(-32768, min(32767, int(v / 65536))) for v in ints])
            return converted.tobytes()
        raise ValueError("unsupported_sample_width")

    def _mix_to_mono(self, pcm: bytes, channels: int) -> bytes:
        if channels <= 1:
            return pcm
        if audioop is not None:
            return audioop.tomono(pcm, 2, 0.5, 0.5)
        samples = array("h")
        samples.frombytes(pcm)
        mixed = array("h")
        for start in range(0, len(samples), channels):
            frame = samples[start:start + channels]
            if not frame:
                continue
            mixed.append(int(sum(int(v) for v in frame) / len(frame)))
        return mixed.tobytes()

    def _resample_pcm(self, pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
        if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
            return pcm
        if audioop is not None:
            converted, _state = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
            return converted
        samples = array("h")
        samples.frombytes(pcm)
        if not samples:
            return pcm
        target_len = max(1, int(round(len(samples) * dst_rate / src_rate)))
        resampled = array("h")
        for i in range(target_len):
            src_index = min(len(samples) - 1, int(i * src_rate / dst_rate))
            resampled.append(int(samples[src_index]))
        return resampled.tobytes()
