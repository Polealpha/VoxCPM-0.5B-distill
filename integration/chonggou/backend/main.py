from __future__ import annotations

import json
import asyncio
import time
import threading
import shutil
import os
import re
import subprocess
import base64
import hashlib
import secrets
import binascii
from difflib import SequenceMatcher
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from sqlite3 import Connection
import httpx
import uvicorn
from cryptography.fernet import Fernet, InvalidToken
from engine.core.config import EngineConfig, load_engine_config
from engine.core.types import VideoFrame
from engine.vision.face_detector import FaceDetector
from engine.vision.frame_decode import decode_rgb
from engine.vision.vision_risk import VisionRiskScorer

from . import auth
from .activation_prompts import ACTIVATION_SYSTEM_PROMPT, IDENTITY_EXTRACTION_PROMPT
from .assessment_engine import (
    PAIR_KEYS,
    QUESTION_MAP,
    build_final_profile,
    build_initial_session,
    build_memory_summary,
    compute_dimension_confidence,
    derive_type_code,
    empty_pair_confidence,
    empty_score_map,
    extract_next_question_from_model,
    extract_turn_analysis_from_model,
    fallback_next_question,
    fallback_turn_analysis,
    merge_scoring,
    normalize_confidence,
    normalize_scores,
    parse_json_dict,
    score_answer_heuristic,
    select_next_question,
)
from .assessment_prompts import (
    ASSESSMENT_CONDUCTOR_PROMPT,
    ASSESSMENT_MEMORY_WRITER_PROMPT,
    ASSESSMENT_TURN_PROMPT,
)
from .personality_prompts import PERSONALITY_EXTRACTION_PROMPT, PERSONALITY_SYSTEM_PROMPT
from .assistant_service import AssistantService, build_session_key, normalize_surface
from .care_prompts import CARE_RESPONSE_RULES, CARE_SYSTEM_PROMPT
from .desktop_speech import DesktopSpeechService
from .db import get_db, init_db
from .openclaw_gateway import OpenClawGatewayError
from .schemas import (
    AssistantBridgeSendRequest,
    ActivationCompleteRequest,
    ActivationAssessmentFinishResponse,
    ActivationAssessmentStartRequest,
    ActivationAssessmentStateResponse,
    ActivationAssessmentTurnRequest,
    ActivationAssessmentTurnResponse,
    ActivationAssessmentVoiceRequest,
    ActivationAssessmentVoicePollRequest,
    ActivationAssessmentVoicePollResponse,
    ActivationIdentityInferRequest,
    ActivationIdentityInferResponse,
    ActivationPersonalityCompleteRequest,
    ActivationPersonalityInferRequest,
    ActivationPersonalityInferResponse,
    ActivationPersonalityStateResponse,
    ActivationProfileResponse,
    ActivationRuntimeStatusResponse,
    ActivationPromptPackResponse,
    AssistantMemorySearchResponse,
    AssistantRuntimeStatusResponse,
    AssistantSendRequest,
    AssistantSendResponse,
    AssistantSessionResetRequest,
    AssistantSessionStatusResponse,
    AssistantTodoCreateRequest,
    AssistantTodoItem,
    AssistantTodoListResponse,
    AssistantTodoUpdateRequest,
    CareRequest,
    CareResponse,
    CameraEmotionAnalyzeRequest,
    CameraEmotionAnalyzeResponse,
    CameraEmotionFaceBox,
    DailySummaryRequest,
    DailySummaryResponse,
    DesktopRuntimeStatusResponse,
    DesktopVoiceStatusResponse,
    DesktopVoiceTranscribeResponse,
    DeviceInfoResponse,
    DeviceSettingsPageRequest,
    DeviceSettingsResponse,
    DeviceSettingsUpdateRequest,
    DeviceHeartbeatRequest,
    DeviceHeartbeatResponse,
    DeviceClaimRequest,
    DeviceClaimResponse,
    DeviceClaimStatusResponse,
    ClientSessionHeartbeatRequest,
    ClientSessionHeartbeatResponse,
    DeviceStatusResponse,
    EngineEventRequest,
    EngineSignalPullRequest,
    EngineSignalPullResponse,
    EngineSignalRequest,
    OwnerEnrollmentRequest,
    OwnerEnrollmentStartRequest,
    OwnerEnrollmentStartResponse,
    OwnerEnrollmentResponse,
    OwnerStatusResponse,
    EmotionEventRequest,
    EmotionEventResponse,
    LoginEmailRequest,
    LoginRequest,
    LoginResponse,
    LogoutRequest,
    ProfileResponse,
    ProfileUpdateRequest,
    RealtimeScoresResponse,
    RealtimeRiskDetailResponse,
    RefreshRequest,
    RegisterRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    TokenResponse,
    UserResponse,
)
from .settings import (
    ACCESS_TOKEN_EXPIRE_SEC,
    ALLOWED_ORIGINS,
    ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS,
    ASSISTANT_BRIDGE_TOKEN,
    ASSISTANT_BRIDGE_USER_ID,
    AUTH_SECRET_KEY,
    OPENCLAW_PREFERRED_CODE_MODEL,
    OPENCLAW_PREFERRED_MODE,
)

for _proxy_key in (
    "OPENCLAW_PROXY_URL",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
):
    if str(os.environ.get("OPENCLAW_GATEWAY_URL", "")).startswith("ws://127.0.0.1:"):
        os.environ.pop(_proxy_key, None)

if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    # Uvicorn on Python 3.14 + Windows can intermittently drop the listener under
    # the default Proactor loop. Pinning the selector policy keeps the local
    # desktop backend stable.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Auth Backend", version="0.1.0")
UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
CHAT_UPLOAD_ROOT = UPLOAD_ROOT / "chat"
CHAT_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")

cors_origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["*"]
allow_all_origins = len(cors_origins) == 1 and cors_origins[0] == "*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[] if allow_all_origins else cors_origins,
    allow_origin_regex=".*" if allow_all_origins else None,
    # With wildcard origins, credentials must be disabled to keep CORS valid in browsers.
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)
_llm = None
_llm_config = None
_llm_init_attempted = False
_llm_init_lock = threading.Lock()
_camera_vision_lock = threading.Lock()
_camera_analysis_lock = threading.Lock()
_camera_vision_runtime: Optional[Dict[str, Any]] = None
HEARTBEAT_STALE_MS = 30000
CLIENT_SESSION_STALE_MS = 45000
_RUNTIME_BOOT_TS = int(time.time() * 1000)
_TOOL_LEAK_PATTERNS = [
    r'"function_call"\s*:',
    r'"tool_call"\s*:',
    r"\bweb_search\b",
    r"^\s*\{[\s\S]*\}\s*$",
]


def _safe_git(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(Path(__file__).resolve().parents[1]), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _runtime_version_payload() -> Dict[str, object]:
    return {
        "git_sha": _safe_git(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
        "git_branch": _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "build_ts": _RUNTIME_BOOT_TS,
        "bridge_git_sha": _safe_git(["git", "rev-parse", "--short", "HEAD"]) or "unknown",
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    return str((_repo_root() / candidate).resolve())


def _expression_label_to_zh(label: str) -> str:
    mapping = {
        "neutral": "平静",
        "happiness": "开心",
        "surprise": "惊讶",
        "sadness": "低落",
        "anger": "愤怒",
        "disgust": "厌恶",
        "fear": "紧张",
        "contempt": "轻蔑",
        "unknown": "未识别",
    }
    return mapping.get(str(label or "").strip().lower(), "未识别")


def _decode_camera_image_payload(image_data_url: str) -> bytes:
    raw = str(image_data_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="image_data_url is required")
    if "," in raw:
        _, raw = raw.split(",", 1)
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid image_data_url") from exc


def _get_camera_vision_runtime() -> Dict[str, Any]:
    global _camera_vision_runtime
    if _camera_vision_runtime is not None:
        return _camera_vision_runtime
    with _camera_vision_lock:
        if _camera_vision_runtime is not None:
            return _camera_vision_runtime
        video_cfg = EngineConfig().video
        video_cfg.expression_enabled = True
        video_cfg.expression_model_path = _resolve_repo_path(video_cfg.expression_model_path)
        video_cfg.expression_mp_model_path = _resolve_repo_path(video_cfg.expression_mp_model_path)
        runtime: Dict[str, Any] = {
            "scorer": VisionRiskScorer(video_cfg),
            "detector": FaceDetector(
                {
                    "detector": "mediapipe",
                    "min_detection_confidence": 0.35,
                    "min_face_area_ratio": 0.015,
                    "max_face_area_ratio": 0.92,
                    "multi_face_policy": "largest",
                }
            ),
            "counter": None,
            "counter_source": "fallback",
        }
        try:
            from mediapipe.tasks import python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore

            runtime["counter"] = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=video_cfg.expression_mp_model_path),
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=2,
                    min_face_detection_confidence=0.25,
                    min_face_presence_confidence=0.25,
                    min_tracking_confidence=0.25,
                )
            )
            runtime["counter_source"] = "mediapipe_tasks"
        except Exception:
            runtime["counter"] = None
            runtime["counter_source"] = "fallback"
        _camera_vision_runtime = runtime
        return runtime


def _count_faces_for_camera(frame: VideoFrame, runtime: Dict[str, Any]) -> tuple[int, Optional[tuple[int, int, int, int]], str]:
    rgb = decode_rgb(frame)
    counter = runtime.get("counter")
    if rgb is not None and counter is not None:
        try:
            import mediapipe as mp  # type: ignore

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = counter.detect(mp_image)
            detections = getattr(results, "face_landmarks", None) or []
            candidates: List[tuple[int, int, int, int]] = []
            for face_landmarks in detections:
                xs = [float(point.x) for point in face_landmarks]
                ys = [float(point.y) for point in face_landmarks]
                if not xs or not ys:
                    continue
                expanded = _expand_face_bbox_from_landmarks(xs, ys, frame.width, frame.height)
                if not expanded:
                    continue
                candidates.append(expanded)
            if candidates:
                candidates.sort(key=lambda item: int(item[2]) * int(item[3]), reverse=True)
                return len(candidates), candidates[0], "mediapipe_tasks"
            return 0, None, "mediapipe_tasks"
        except Exception:
            pass
    detector = runtime.get("detector")
    if detector is not None:
        face_det = detector.detect(frame)
        if face_det.found and face_det.bbox:
            return 1, face_det.bbox, "fallback"
    return 0, None, "fallback"


def _bbox_to_percent(bbox: Optional[tuple[int, int, int, int]], width: int, height: int) -> Optional[Dict[str, float]]:
    if not bbox or width <= 0 or height <= 0:
        return None
    x, y, w, h = bbox
    return {
        "left": max(0.0, min(100.0, float(x / width * 100.0))),
        "top": max(0.0, min(100.0, float(y / height * 100.0))),
        "width": max(0.0, min(100.0, float(w / width * 100.0))),
        "height": max(0.0, min(100.0, float(h / height * 100.0))),
    }


def _expand_face_bbox_from_landmarks(
    xs: List[float], ys: List[float], frame_w: int, frame_h: int
) -> Optional[tuple[int, int, int, int]]:
    if not xs or not ys or frame_w <= 0 or frame_h <= 0:
        return None
    x1 = max(0.0, min(xs)) * frame_w
    y1 = max(0.0, min(ys)) * frame_h
    x2 = min(1.0, max(xs)) * frame_w
    y2 = min(1.0, max(ys)) * frame_h
    raw_w = max(0.0, x2 - x1)
    raw_h = max(0.0, y2 - y1)
    if raw_w <= 0.0 or raw_h <= 0.0:
        return None

    # Landmarker points tend to hug the inner face contour and cut off hair/chin.
    # Expand and bias slightly downward so the visible face is fully enclosed.
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5 + raw_h * 0.05
    box_w = raw_w * 1.38
    box_h = raw_h * 1.58

    left = int(max(0.0, cx - box_w * 0.5))
    top = int(max(0.0, cy - box_h * 0.5))
    right = int(min(float(frame_w), cx + box_w * 0.5))
    bottom = int(min(float(frame_h), cy + box_h * 0.5))
    width = max(0, right - left)
    height = max(0, bottom - top)
    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def _looks_tool_call_leak_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return any(re.search(p, raw, flags=re.IGNORECASE) for p in _TOOL_LEAK_PATTERNS)


def _sanitize_outbound_bot_text(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    if not raw:
        return "", False
    if not _looks_tool_call_leak_text(raw):
        return raw, False
    if re.search(r'"function_call"\s*:', raw, flags=re.IGNORECASE) or re.search(
        r'"tool_call"\s*:', raw, flags=re.IGNORECASE
    ):
        return "我已经拿到工具结果，但中间格式异常。你再问一次，我直接给你结论。", True
    cleaned = re.sub(r'(?is)\{\s*"function_call"\s*:\s*\{.*?\}\s*\}', " ", raw)
    cleaned = re.sub(r'(?is)\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}', " ", cleaned)
    cleaned = re.sub(r"(?im)^\s*(function_call|tool_call)\s*[:：].*$", " ", cleaned)
    cleaned = re.sub(r"^```(?:json)?|```$", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if cleaned:
        return cleaned, True
    return "我已经拿到工具结果，但中间格式异常。你再问一次，我直接给你结论。", True


class EventManager:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: Dict[str, object]) -> None:
        if not self._connections:
            return
        dead: List[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


event_manager = EventManager()
assistant_service = AssistantService()
desktop_speech_service = DesktopSpeechService()
DESKTOP_TTS_BASE_URL = str(os.environ.get("DESKTOP_TTS_BASE_URL", "http://127.0.0.1:18884")).rstrip("/")
DESKTOP_STT_BASE_URL = str(os.environ.get("DESKTOP_STT_BASE_URL", "http://127.0.0.1:18885")).rstrip("/")
DESKTOP_TTS_URL = str(os.environ.get("DESKTOP_TTS_URL", f"{DESKTOP_TTS_BASE_URL}/api/tts/synthesize")).rstrip("/")
DESKTOP_TTS_HEALTH = str(os.environ.get("DESKTOP_TTS_HEALTH", f"{DESKTOP_TTS_BASE_URL}/api/tts/health")).rstrip("/")
DESKTOP_STT_URL = str(os.environ.get("DESKTOP_STT_URL", f"{DESKTOP_STT_BASE_URL}/api/stt/transcribe")).rstrip("/")
DESKTOP_STT_HEALTH = str(os.environ.get("DESKTOP_STT_HEALTH", f"{DESKTOP_STT_BASE_URL}/api/stt/health")).rstrip("/")


async def _desktop_voice_fetch_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 20.0,
) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=3.0)) as client:
        if method.upper() == "POST":
            response = await client.post(url, json=payload or {})
        else:
            response = await client.get(url)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {"ok": False, "detail": "invalid_voice_sidecar_response"}

ACTIVATION_ASSESSMENT_MODEL_TIMEOUT_MS = 35_000
ACTIVATION_ASSESSMENT_MEMORY_TIMEOUT_MS = 30_000
ACTIVATION_ASSESSMENT_RETRY_DELAYS_MS = (1200, 2600, 5200)
_ENGINE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "engine_config.json"


@app.on_event("startup")
def _startup() -> None:
    init_db()
    _ensure_signal_state()


@app.get("/api/runtime/version")
def runtime_version() -> Dict[str, object]:
    payload = _runtime_version_payload()
    payload["ok"] = True
    return payload


def _desktop_runtime_status_payload() -> Dict[str, object]:
    engine_cfg = load_engine_config(str(_ENGINE_CONFIG_PATH))
    engine_root = Path(__file__).resolve().parent.parent / "engine"
    care_templates = engine_root / "policy" / "templates_zh.json"
    text_lexicon = engine_root / "nlp" / "lexicon_zh.txt"
    assistant_chain = assistant_service.runtime_status()
    voice_chain = desktop_speech_service.status()
    return {
        "ok": True,
        "source": "desktop_backend",
        "emotion_chain_ready": True,
        "proactive_care_ready": bool(assistant_chain.get("gateway_ready")) and bool(assistant_chain.get("provider_network_ok")),
        "active_care_strategy": str(engine_cfg.policy.care_delivery_strategy or "policy"),
        "voice_chain": voice_chain,
        "assistant_chain": assistant_chain,
        "components": {
            "fusion_enabled": True,
            "trigger_enabled": True,
            "text_risk_lexicon": str(text_lexicon),
            "text_risk_ready": text_lexicon.exists(),
            "care_policy_templates": str(care_templates),
            "care_policy_ready": care_templates.exists(),
            "care_openclaw_entrypoint": "llm_care",
            "summary_enabled": True,
            "desktop_backend_local": True,
        },
    }


@app.get("/api/desktop/runtime/status", response_model=DesktopRuntimeStatusResponse)
def desktop_runtime_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DesktopRuntimeStatusResponse:
    _parse_access_token_for_local_desktop(credentials, conn, request)
    return DesktopRuntimeStatusResponse(**_desktop_runtime_status_payload())


@app.get("/api/desktop/voice/status", response_model=DesktopVoiceStatusResponse)
async def desktop_voice_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DesktopVoiceStatusResponse:
    _parse_access_token_for_local_desktop(credentials, conn, request)
    try:
        payload = await _desktop_voice_fetch_json(DESKTOP_STT_HEALTH, timeout_s=20.0)
        return DesktopVoiceStatusResponse(
            ok=bool(payload.get("ok", True)),
            ready=bool(payload.get("ready", False)),
            provider_preference="sensevoice_small",
            fallback_provider="none",
            active_provider=str(payload.get("provider", "sensevoice_small")),
            primary_ready=bool(payload.get("ready", False)),
            primary_engine=str(payload.get("provider", "sensevoice_small")),
            primary_error=None,
            fallback_ready=False,
            fallback_engine="none",
            fallback_error=None,
            language="auto",
            max_sec=120,
            model_name=str(payload.get("model_name", "SenseVoiceSmall")),
            beam_size=1,
            best_of=1,
            preprocess_enabled=True,
            trim_silence_enabled=True,
            initial_prompt_enabled=False,
            hotwords_enabled=False,
        )
    except Exception:
        return DesktopVoiceStatusResponse(**desktop_speech_service.status())


@app.get("/api/desktop/voice/synthesize/status")
async def desktop_voice_synthesize_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, Any]:
    _parse_access_token_for_local_desktop(credentials, conn, request)
    try:
        payload = await _desktop_voice_fetch_json(DESKTOP_TTS_HEALTH, timeout_s=20.0)
        payload["bridge_mode"] = "local-http"
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "ready": False,
            "provider": "voxcpm_distill",
            "model_name": "VoxCPM-0.5B-distill",
            "detail": str(exc),
            "bridge_mode": "local-http",
        }


def _wifi_cipher() -> Fernet:
    secret = (AUTH_SECRET_KEY or "change-this-secret").encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
    return Fernet(key)


def _encrypt_wifi_password(password: str) -> str:
    raw = str(password or "")
    if not raw:
        return ""
    return _wifi_cipher().encrypt(raw.encode("utf-8")).decode("utf-8")


def _decrypt_wifi_password(token: str) -> str:
    raw = str(token or "").strip()
    if not raw:
        return ""
    try:
        return _wifi_cipher().decrypt(raw.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError):
        return ""


def _ensure_signal_state() -> None:
    if not hasattr(app.state, "signal_queue"):
        app.state.signal_queue = deque()
    if not hasattr(app.state, "signal_lock"):
        app.state.signal_lock = threading.Lock()


def _enqueue_signal(signal: Dict[str, object]) -> None:
    _ensure_signal_state()
    lock = app.state.signal_lock
    with lock:
        app.state.signal_queue.append(signal)


def _drain_signals(limit: int) -> List[Dict[str, object]]:
    _ensure_signal_state()
    lock = app.state.signal_lock
    signals: List[Dict[str, object]] = []
    with lock:
        while app.state.signal_queue and len(signals) < limit:
            signals.append(app.state.signal_queue.popleft())
    return signals


def _is_local_request(request: Request) -> bool:
    if not request.client:
        return False
    host = str(request.client.host or "")
    return host in {"127.0.0.1", "::1", "localhost"}


def _ensure_llm_loaded():
    global _llm, _llm_config, _llm_init_attempted
    if _llm_init_attempted:
        return _llm
    with _llm_init_lock:
        if _llm_init_attempted:
            return _llm
        try:
            from engine.core.config import load_engine_config
            from engine.llm.llm_responder import LLMResponder

            config = load_engine_config("config/engine_config.json")
            _llm = LLMResponder(config.llm)
            _llm_config = config.llm
        except Exception:
            _llm = None
            _llm_config = None
        _llm_init_attempted = True
        return _llm


def _get_user_by_username(conn: Connection, username: str) -> Dict:
    cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return dict(row) if row else {}


def _get_user_by_id(conn: Connection, user_id: int) -> Dict:
    cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    return dict(row) if row else {}


def _profile_from_user(user: Dict) -> Dict:
    username = str(user.get("username") or "").strip()
    display_name = str(user.get("display_name") or "").strip()
    if not display_name:
        display_name = username or "鍏遍福鏃呬汉"
    avatar_url = user.get("avatar_url")
    if avatar_url is not None:
        avatar_url = str(avatar_url).strip() or None
    bio = user.get("bio")
    if bio is not None:
        bio = str(bio).strip() or None
    location = user.get("location")
    if location is not None:
        location = str(location).strip() or None
    return {
        "id": int(user.get("id", 0)),
        "username": username,
        "display_name": display_name,
        "avatar_url": avatar_url,
        "bio": bio,
        "location": location,
        "created_at": int(user.get("created_at") or 0),
        "updated_at": int(user.get("updated_at") or 0) or None,
    }


def _get_default_user_id(conn: Connection) -> Optional[int]:
    cur = conn.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    return int(row["id"]) if row else None


def _set_user_configured(conn: Connection, user_id: int, value: bool) -> None:
    conn.execute("UPDATE users SET is_configured = ? WHERE id = ?", (1 if value else 0, user_id))
    conn.commit()


def _get_activation_profile(conn: Connection, user_id: int) -> Dict[str, object]:
    row = conn.execute(
        """
        SELECT *
        FROM user_activation_profiles
        WHERE user_id = ?
        """,
        (int(user_id),),
    ).fetchone()
    if not row:
        return {}
    payload = dict(row)
    try:
        profile_json = json.loads(str(payload.get("profile_json") or "{}"))
    except Exception:
        profile_json = {}
    if not isinstance(profile_json, dict):
        profile_json = {}
    merged = dict(profile_json)
    merged.update(
        {
            "preferred_name": str(payload.get("preferred_name") or "").strip() or None,
            "role_label": str(payload.get("role_label") or "").strip() or None,
            "relation_to_robot": str(payload.get("relation_to_robot") or "").strip() or None,
            "pronouns": str(payload.get("pronouns") or "").strip() or None,
            "identity_summary": str(payload.get("identity_summary") or "").strip() or None,
            "onboarding_notes": str(payload.get("onboarding_notes") or "").strip() or None,
            "voice_intro_summary": str(payload.get("voice_intro_summary") or "").strip() or None,
            "activation_version": str(payload.get("activation_version") or "v1"),
            "completed_at_ms": int(payload.get("completed_at_ms") or 0) or None,
        }
    )
    return merged


def _upsert_activation_profile(conn: Connection, user_id: int, payload: Dict[str, object]) -> Dict[str, object]:
    now_ms = int(time.time() * 1000)
    profile_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    preferred_name = str(payload.get("preferred_name") or "").strip()
    role_label = str(payload.get("role_label") or "owner").strip() or "owner"
    relation_to_robot = str(payload.get("relation_to_robot") or "primary_user").strip() or "primary_user"
    pronouns = str(payload.get("pronouns") or "").strip()
    identity_summary = str(payload.get("identity_summary") or "").strip()
    onboarding_notes = str(payload.get("onboarding_notes") or "").strip()
    voice_intro_summary = str(payload.get("voice_intro_summary") or "").strip()
    activation_version = str(payload.get("activation_version") or "v1").strip() or "v1"
    completed_at_ms = int(payload.get("completed_at_ms") or 0) or None
    conn.execute(
        """
        INSERT INTO user_activation_profiles (
            user_id,
            preferred_name,
            role_label,
            relation_to_robot,
            pronouns,
            identity_summary,
            onboarding_notes,
            voice_intro_summary,
            profile_json,
            activation_version,
            completed_at_ms,
            updated_at,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            preferred_name = excluded.preferred_name,
            role_label = excluded.role_label,
            relation_to_robot = excluded.relation_to_robot,
            pronouns = excluded.pronouns,
            identity_summary = excluded.identity_summary,
            onboarding_notes = excluded.onboarding_notes,
            voice_intro_summary = excluded.voice_intro_summary,
            profile_json = excluded.profile_json,
            activation_version = excluded.activation_version,
            completed_at_ms = excluded.completed_at_ms,
            updated_at = excluded.updated_at
        """,
        (
            int(user_id),
            preferred_name or None,
            role_label,
            relation_to_robot,
            pronouns or None,
            identity_summary or None,
            onboarding_notes or None,
            voice_intro_summary or None,
            profile_json,
            activation_version,
            completed_at_ms,
            now_ms,
            now_ms,
        ),
    )
    conn.commit()
    return _get_activation_profile(conn, user_id)


def _activation_response(conn: Connection, user: Dict[str, object], profile: Dict[str, object]) -> ActivationProfileResponse:
    is_configured = bool(user.get("is_configured", 0))
    psychometric_completed = bool(_get_psychometric_profile(conn, int(user["id"])))
    owner_binding = _get_owner_binding_state(conn, int(user["id"]), is_configured, psychometric_completed)
    preferred_name = str(profile.get("preferred_name") or "").strip() or None
    role_label = str(profile.get("role_label") or "").strip() or None
    relation_to_robot = str(profile.get("relation_to_robot") or "").strip() or None
    pronouns = str(profile.get("pronouns") or "").strip() or None
    identity_summary = str(profile.get("identity_summary") or "").strip() or None
    onboarding_notes = str(profile.get("onboarding_notes") or "").strip() or None
    voice_intro_summary = str(profile.get("voice_intro_summary") or "").strip() or None
    return ActivationProfileResponse(
        ok=True,
        is_configured=is_configured,
        activation_required=not is_configured,
        assessment_required=is_configured and not psychometric_completed,
        psychometric_completed=psychometric_completed,
        owner_binding_required=bool(owner_binding["required"]),
        owner_binding_completed=bool(owner_binding["completed"]),
        preferred_device_id=owner_binding["device_id"],
        preferred_name=preferred_name,
        role_label=role_label,
        relation_to_robot=relation_to_robot,
        pronouns=pronouns,
        identity_summary=identity_summary,
        onboarding_notes=onboarding_notes,
        voice_intro_summary=voice_intro_summary,
        activation_version=str(profile.get("activation_version") or "v1"),
        completed_at_ms=int(profile.get("completed_at_ms") or 0) or None,
        preferred_mode=OPENCLAW_PREFERRED_MODE,
        preferred_code_model=OPENCLAW_PREFERRED_CODE_MODEL,
    )


def _get_owner_binding_state(
    conn: Connection,
    user_id: int,
    is_configured: bool,
    psychometric_completed: bool,
) -> Dict[str, object]:
    # Face enrollment is no longer part of the activation gate.
    devices = _list_devices(conn, int(user_id))
    if not devices:
        return {"required": False, "completed": False, "device_id": None}
    preferred = devices[0]
    device_id = str(preferred.get("device_id") or "").strip() or None
    if not device_id:
        return {"required": False, "completed": False, "device_id": None}
    profile = _get_owner_profile(conn, int(user_id), device_id)
    completed = bool(profile)
    del is_configured, psychometric_completed
    return {"required": False, "completed": completed, "device_id": device_id}


def _json_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _get_personality_profile(conn: Connection, user_id: int) -> Dict[str, object]:
    row = conn.execute(
        """
        SELECT *
        FROM user_personality_profiles
        WHERE user_id = ?
        """,
        (int(user_id),),
    ).fetchone()
    if not row:
        return {}
    payload = dict(row)
    try:
        profile_json = json.loads(str(payload.get("profile_json") or "{}"))
    except Exception:
        profile_json = {}
    if not isinstance(profile_json, dict):
        profile_json = {}
    merged = dict(profile_json)
    merged.update(
        {
            "summary": str(payload.get("summary") or "").strip(),
            "response_style": str(payload.get("response_style") or "").strip(),
            "care_style": str(payload.get("care_style") or "").strip(),
            "traits": _json_list(payload.get("traits_json")),
            "topics": _json_list(payload.get("topics_json")),
            "boundaries": _json_list(payload.get("boundaries_json")),
            "signals": _json_list(payload.get("signals_json")),
            "confidence": float(payload.get("confidence") or 0.0),
            "sample_count": int(payload.get("sample_count") or 0),
            "inference_version": str(payload.get("inference_version") or "v1").strip() or "v1",
            "updated_at_ms": int(payload.get("updated_at") or 0) or None,
        }
    )
    return merged


def _personality_response(profile: Dict[str, object]) -> ActivationPersonalityStateResponse:
    return ActivationPersonalityStateResponse(
        ok=True,
        exists=bool(profile),
        summary=str(profile.get("summary") or "").strip(),
        response_style=str(profile.get("response_style") or "").strip(),
        care_style=str(profile.get("care_style") or "").strip(),
        traits=[str(item) for item in profile.get("traits") or [] if str(item).strip()],
        topics=[str(item) for item in profile.get("topics") or [] if str(item).strip()],
        boundaries=[str(item) for item in profile.get("boundaries") or [] if str(item).strip()],
        signals=[str(item) for item in profile.get("signals") or [] if str(item).strip()],
        confidence=max(0.0, min(1.0, float(profile.get("confidence") or 0.0))),
        sample_count=max(0, int(profile.get("sample_count") or 0)),
        inference_version=str(profile.get("inference_version") or "v1"),
        updated_at_ms=int(profile.get("updated_at_ms") or 0) or None,
    )


def _upsert_personality_profile(conn: Connection, user_id: int, payload: Dict[str, object]) -> Dict[str, object]:
    now_ms = int(time.time() * 1000)
    traits = [str(item).strip() for item in payload.get("traits") or [] if str(item).strip()]
    topics = [str(item).strip() for item in payload.get("topics") or [] if str(item).strip()]
    boundaries = [str(item).strip() for item in payload.get("boundaries") or [] if str(item).strip()]
    signals = [str(item).strip() for item in payload.get("signals") or [] if str(item).strip()]
    profile_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    conn.execute(
        """
        INSERT INTO user_personality_profiles (
            user_id,
            summary,
            response_style,
            care_style,
            traits_json,
            topics_json,
            boundaries_json,
            signals_json,
            profile_json,
            confidence,
            sample_count,
            inference_version,
            updated_at,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            summary = excluded.summary,
            response_style = excluded.response_style,
            care_style = excluded.care_style,
            traits_json = excluded.traits_json,
            topics_json = excluded.topics_json,
            boundaries_json = excluded.boundaries_json,
            signals_json = excluded.signals_json,
            profile_json = excluded.profile_json,
            confidence = excluded.confidence,
            sample_count = excluded.sample_count,
            inference_version = excluded.inference_version,
            updated_at = excluded.updated_at
        """,
        (
            int(user_id),
            str(payload.get("summary") or "").strip() or None,
            str(payload.get("response_style") or "").strip() or None,
            str(payload.get("care_style") or "").strip() or None,
            json.dumps(traits, ensure_ascii=False),
            json.dumps(topics, ensure_ascii=False),
            json.dumps(boundaries, ensure_ascii=False),
            json.dumps(signals, ensure_ascii=False),
            profile_json,
            max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
            max(0, int(payload.get("sample_count") or 0)),
            str(payload.get("inference_version") or "v1").strip() or "v1",
            now_ms,
            now_ms,
        ),
    )
    conn.commit()
    return _get_personality_profile(conn, user_id)


def _get_active_assessment_session_row(conn: Connection, user_id: int) -> Optional[Dict[str, object]]:
    row = conn.execute(
        """
        SELECT *
        FROM user_assessment_sessions
        WHERE user_id = ? AND status = 'active'
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (int(user_id),),
    ).fetchone()
    return dict(row) if row else None


def _get_latest_assessment_session_row(conn: Connection, user_id: int) -> Optional[Dict[str, object]]:
    row = conn.execute(
        """
        SELECT *
        FROM user_assessment_sessions
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (int(user_id),),
    ).fetchone()
    return dict(row) if row else None


def _load_assessment_session(conn: Connection, user_id: int, active_only: bool = False) -> tuple[Optional[int], Dict[str, object]]:
    row = _get_active_assessment_session_row(conn, user_id) if active_only else _get_latest_assessment_session_row(conn, user_id)
    if not row:
        return None, {}
    payload = parse_json_dict(str(row.get("session_json") or "{}"))
    if not payload:
        payload = {}
    payload.setdefault("status", str(row.get("status") or "idle"))
    payload.setdefault("started_at_ms", int(row.get("started_at_ms") or 0) or None)
    payload.setdefault("completed_at_ms", int(row.get("completed_at_ms") or 0) or None)
    payload.setdefault("updated_at_ms", int(row.get("updated_at") or 0) or None)
    payload["scores"] = normalize_scores(payload.get("scores"))
    payload["cognitive_scores"] = normalize_scores(payload.get("cognitive_scores") or payload.get("scores"))
    payload["dimension_confidence"] = normalize_confidence(payload.get("dimension_confidence"))
    payload["function_confidence"] = normalize_confidence(payload.get("function_confidence") or payload.get("dimension_confidence"))
    return int(row["id"]), payload


def _save_assessment_session(conn: Connection, user_id: int, session_payload: Dict[str, object], session_id: Optional[int] = None) -> int:
    now_ms = int(time.time() * 1000)
    payload = dict(session_payload)
    payload["scores"] = normalize_scores(payload.get("scores"))
    payload["cognitive_scores"] = normalize_scores(payload.get("cognitive_scores") or payload.get("scores"))
    payload["dimension_confidence"] = normalize_confidence(payload.get("dimension_confidence"))
    payload["function_confidence"] = normalize_confidence(payload.get("function_confidence") or payload.get("dimension_confidence"))
    session_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    status_value = str(payload.get("status") or "active").strip() or "active"
    started_at_ms = int(payload.get("started_at_ms") or now_ms)
    completed_at_ms = int(payload.get("completed_at_ms") or 0) or None
    if session_id is None:
        conn.execute("UPDATE user_assessment_sessions SET status = 'superseded', updated_at = ? WHERE user_id = ? AND status = 'active'", (now_ms, int(user_id)))
        cursor = conn.execute(
            """
            INSERT INTO user_assessment_sessions (
                user_id, status, session_json, started_at_ms, completed_at_ms, updated_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (int(user_id), status_value, session_json, started_at_ms, completed_at_ms, now_ms, now_ms),
        )
        conn.commit()
        return int(cursor.lastrowid)
    conn.execute(
        """
        UPDATE user_assessment_sessions
        SET status = ?, session_json = ?, started_at_ms = ?, completed_at_ms = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
        """,
        (status_value, session_json, started_at_ms, completed_at_ms, now_ms, int(session_id), int(user_id)),
    )
    conn.commit()
    return int(session_id)


def _append_assessment_turn_event(
    conn: Connection,
    user_id: int,
    session_id: int,
    turn_index: int,
    question_id: str,
    question_text: str,
    answer_text: str,
    transcript_text: str,
    scoring: Dict[str, object],
) -> None:
    conn.execute(
        """
        INSERT INTO assessment_turn_events (
            user_id, session_id, turn_index, question_id, question_text,
            answer_text, transcript_text, scoring_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            int(session_id),
            int(turn_index),
            str(question_id or "").strip() or None,
            str(question_text or "").strip() or None,
            str(answer_text or "").strip() or None,
            str(transcript_text or "").strip() or None,
            json.dumps(scoring or {}, ensure_ascii=False, separators=(",", ":")),
            int(time.time() * 1000),
        ),
    )
    conn.commit()


def _get_psychometric_profile(conn: Connection, user_id: int) -> Dict[str, object]:
    row = conn.execute(
        """
        SELECT *
        FROM user_psychometric_profiles
        WHERE user_id = ?
        """,
        (int(user_id),),
    ).fetchone()
    if not row:
        return {}
    payload = dict(row)
    merged = parse_json_dict(str(payload.get("profile_json") or "{}"))
    raw_scores = parse_json_dict(str(payload.get("scores_json") or "{}"))
    raw_confidence = parse_json_dict(str(payload.get("dimension_confidence_json") or "{}"))
    merged.update(
        {
            "type_code": str(payload.get("type_code") or "").strip(),
            "mapped_type_code": str(merged.get("mapped_type_code") or payload.get("type_code") or "").strip(),
            "scores": raw_scores,
            "cognitive_scores": parse_json_dict(str((merged.get("cognitive_scores") or raw_scores) or "{}")),
            "dimension_confidence": raw_confidence,
            "function_confidence": parse_json_dict(str((merged.get("function_confidence") or raw_confidence) or "{}")),
            "evidence_summary": parse_json_dict(str(payload.get("evidence_summary_json") or "{}")),
            "summary": str(payload.get("summary") or "").strip(),
            "response_style": str(payload.get("response_style") or "").strip(),
            "care_style": str(payload.get("care_style") or "").strip(),
            "interaction_preferences": [str(item).strip() for item in merged.get("interaction_preferences") or [] if str(item).strip()],
            "decision_style": str(merged.get("decision_style") or payload.get("response_style") or "").strip(),
            "stress_response": str(merged.get("stress_response") or "").strip(),
            "comfort_preferences": [str(item).strip() for item in merged.get("comfort_preferences") or [] if str(item).strip()],
            "avoid_patterns": [str(item).strip() for item in merged.get("avoid_patterns") or [] if str(item).strip()],
            "care_guidance": str(merged.get("care_guidance") or payload.get("care_style") or "").strip(),
            "confidence": float(merged.get("confidence") or 0.0),
            "conversation_count": int(payload.get("conversation_count") or 0),
            "completed_at_ms": int(payload.get("completed_at_ms") or 0) or None,
            "updated_at_ms": int(payload.get("updated_at") or 0) or None,
            "inference_version": str(payload.get("inference_version") or "activation-dialogue-v3").strip() or "activation-dialogue-v3",
            "dominant_stack": [str(item).strip() for item in merged.get("dominant_stack") or [] if str(item).strip()],
            "assessment_ready": bool(merged.get("assessment_ready", True)),
            "ai_required": bool(merged.get("ai_required", True)),
            "completion_reason": str(merged.get("completion_reason") or "").strip(),
        }
    )
    merged["scores"] = normalize_scores(merged.get("scores"))
    merged["cognitive_scores"] = normalize_scores(merged.get("cognitive_scores") or merged.get("scores"))
    merged["dimension_confidence"] = normalize_confidence(merged.get("dimension_confidence"))
    merged["function_confidence"] = normalize_confidence(merged.get("function_confidence") or merged.get("dimension_confidence"))
    evidence = merged.get("evidence_summary")
    if not isinstance(evidence, dict):
        evidence = {}
    merged["evidence_summary"] = {
        "highlights": [str(item).strip() for item in evidence.get("highlights") or [] if str(item).strip()],
        "notes": str(evidence.get("notes") or "").strip(),
    }
    return merged


def _upsert_psychometric_profile(conn: Connection, user_id: int, profile: Dict[str, object]) -> Dict[str, object]:
    now_ms = int(time.time() * 1000)
    scores = normalize_scores(profile.get("cognitive_scores") or profile.get("scores"))
    confidence = normalize_confidence(profile.get("function_confidence") or profile.get("dimension_confidence"))
    evidence_summary = profile.get("evidence_summary") if isinstance(profile.get("evidence_summary"), dict) else {}
    payload = dict(profile)
    payload["scores"] = scores
    payload["cognitive_scores"] = scores
    payload["dimension_confidence"] = confidence
    payload["function_confidence"] = confidence
    payload["evidence_summary"] = evidence_summary
    conn.execute(
        """
        INSERT INTO user_psychometric_profiles (
            user_id, type_code, scores_json, dimension_confidence_json, evidence_summary_json,
            summary, response_style, care_style, conversation_count, completed_at_ms,
            inference_version, profile_json, updated_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            type_code = excluded.type_code,
            scores_json = excluded.scores_json,
            dimension_confidence_json = excluded.dimension_confidence_json,
            evidence_summary_json = excluded.evidence_summary_json,
            summary = excluded.summary,
            response_style = excluded.response_style,
            care_style = excluded.care_style,
            conversation_count = excluded.conversation_count,
            completed_at_ms = excluded.completed_at_ms,
            inference_version = excluded.inference_version,
            profile_json = excluded.profile_json,
            updated_at = excluded.updated_at
        """,
        (
            int(user_id),
            str(profile.get("mapped_type_code") or profile.get("type_code") or "").strip() or None,
            json.dumps(scores, ensure_ascii=False, separators=(",", ":")),
            json.dumps(confidence, ensure_ascii=False, separators=(",", ":")),
            json.dumps(evidence_summary, ensure_ascii=False, separators=(",", ":")),
            str(profile.get("summary") or "").strip() or None,
            str(profile.get("decision_style") or profile.get("response_style") or "").strip() or None,
            str(profile.get("care_guidance") or profile.get("care_style") or "").strip() or None,
            int(profile.get("conversation_count") or 0),
            int(profile.get("completed_at_ms") or 0) or None,
            str(profile.get("inference_version") or "activation-dialogue-v3").strip() or "activation-dialogue-v3",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            now_ms,
            now_ms,
        ),
    )
    conn.commit()
    return _get_psychometric_profile(conn, user_id)


def _assessment_response(session: Dict[str, object], device_online: bool = False, exists: bool = True) -> ActivationAssessmentStateResponse:
    final = session.get("final_result") if isinstance(session.get("final_result"), dict) else {}
    evidence = (final or session).get("evidence_summary")
    if not isinstance(evidence, dict):
        evidence = {}
    voice_mode = str(session.get("voice_mode") or "idle")
    voice_session_active = bool(session.get("voice_session_active"))
    blocking_reason = str(session.get("blocking_reason") or "").strip()
    if blocking_reason:
        mode_hint = "ai_blocked"
    elif not bool(device_online):
        mode_hint = "device_offline_text_available"
    elif voice_session_active and voice_mode == "robot":
        mode_hint = "robot_voice_active"
    elif voice_mode == "robot":
        mode_hint = "robot_voice_ready"
    else:
        mode_hint = "text_mode_ready"
    return ActivationAssessmentStateResponse(
        ok=True,
        exists=bool(exists and session),
        status=str(session.get("status") or "idle"),
        started_at_ms=int(session.get("started_at_ms") or 0) or None,
        updated_at_ms=int(session.get("updated_at_ms") or 0) or None,
        completed_at_ms=int((final or session).get("completed_at_ms") or 0) or None,
        turn_count=max(0, int(session.get("turn_count") or 0)),
        effective_turn_count=max(0, int(session.get("effective_turn_count") or 0)),
        conversation_count=max(0, int((final or session).get("conversation_count") or session.get("effective_turn_count") or 0)),
        latest_question=str(session.get("latest_question") or ""),
        latest_transcript=str(session.get("latest_transcript") or ""),
        last_question_id=str(session.get("last_question_id") or ""),
        evidence_summary={
            "highlights": [str(item).strip() for item in evidence.get("highlights") or [] if str(item).strip()],
            "notes": str(evidence.get("notes") or "").strip(),
        },
        finish_reason=str(session.get("finish_reason") or ""),
        voice_mode=voice_mode,
        voice_session_active=voice_session_active,
        device_online=bool(device_online),
        summary=str((final or session).get("summary") or (session.get("profile_preview") or {}).get("summary") or ""),
        inference_version=str((final or session).get("inference_version") or "activation-dialogue-v4"),
        question_source=str(session.get("question_source") or "ai_required"),
        scoring_source=str(session.get("scoring_source") or "pending"),
        current_focus=str(session.get("current_focus") or session.get("question_pair") or ""),
        mode_hint=mode_hint,
        can_submit_text=not bool(blocking_reason),
        assessment_ready=bool((final or session).get("assessment_ready")),
        ai_required=bool((final or session).get("ai_required", True)),
        blocking_reason=blocking_reason,
        dialogue_turns=[
            {
                "role": str(item.get("role") or "").strip(),
                "text": str(item.get("text") or "").strip(),
                "timestamp_ms": int(item.get("timestamp_ms") or 0) or None,
            }
            for item in ((final or session).get("dialogue_turns") or session.get("dialogue_turns") or [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ],
        interaction_preferences=[str(item).strip() for item in ((final or session).get("interaction_preferences") or []) if str(item).strip()],
        decision_style=str((final or session).get("decision_style") or (final or session).get("response_style") or ""),
        stress_response=str((final or session).get("stress_response") or ""),
        comfort_preferences=[str(item).strip() for item in ((final or session).get("comfort_preferences") or []) if str(item).strip()],
        avoid_patterns=[str(item).strip() for item in ((final or session).get("avoid_patterns") or []) if str(item).strip()],
        care_guidance=str((final or session).get("care_guidance") or (final or session).get("care_style") or ""),
        confidence=float((final or session).get("confidence") or 0.0),
    )


_ASSESSMENT_FALLBACK_QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "fallback-support-style",
        "pair": "interaction_preferences",
        "focus": "interaction_preferences",
        "prompt": "如果你状态不太好时，有人来关心你，你更希望对方怎么开口，才不会让你觉得有压力？",
    },
    {
        "id": "fallback-decision-style",
        "pair": "decision_style",
        "focus": "decision_style",
        "prompt": "遇到要做选择的事情时，你通常是很快拍板，还是会先反复想一会儿？",
    },
    {
        "id": "fallback-stress-response",
        "pair": "stress_response",
        "focus": "stress_response",
        "prompt": "当你很累、很烦或者压力上来时，你更容易先沉默自己扛着，还是会想找人说一说？",
    },
    {
        "id": "fallback-comfort-preferences",
        "pair": "comfort_preferences",
        "focus": "comfort_preferences",
        "prompt": "如果我要安抚你，你更能接受哪种方式：先安静陪着、先给明确建议，还是先陪你把情绪说出来？",
    },
    {
        "id": "fallback-avoid-patterns",
        "pair": "avoid_patterns",
        "focus": "avoid_patterns",
        "prompt": "有没有哪种说话方式会让你立刻不想继续聊，比如催、说教、追问或者太多鸡汤？",
    },
]


def _assessment_fallback_question(session_payload: Dict[str, object]) -> Dict[str, str]:
    asked_ids = {
        str(item.get("question_id") or "").strip()
        for item in (session_payload.get("question_history") or [])
        if isinstance(item, dict) and str(item.get("question_id") or "").strip()
    }
    for item in _ASSESSMENT_FALLBACK_QUESTIONS:
        if item["id"] not in asked_ids:
            return dict(item)
    turn_index = int(session_payload.get("turn_count") or 0)
    return dict(_ASSESSMENT_FALLBACK_QUESTIONS[turn_index % len(_ASSESSMENT_FALLBACK_QUESTIONS)])


async def _assessment_pick_question_with_fallback(
    user_id: int,
    activation_profile: Dict[str, object],
    session_payload: Dict[str, object],
) -> tuple[Dict[str, object], str]:
    try:
        model_question = await _assessment_pick_model_question(int(user_id), activation_profile, session_payload)
    except Exception:
        model_question = {}
    if model_question:
        return model_question, "ai"
    return _assessment_fallback_question(session_payload), "fallback"


def _activation_ai_runtime_snapshot() -> Dict[str, object]:
    runtime = assistant_service.runtime_status()
    gateway_ready = bool(runtime.get("gateway_ready"))
    provider_network_ok = bool(runtime.get("provider_network_ok"))
    ai_detail = str(runtime.get("gateway_error") or runtime.get("provider_network_detail") or "").strip()
    blocking_reason = ""
    if not gateway_ready:
        blocking_reason = str(runtime.get("gateway_error") or "OpenClaw gateway unavailable").strip()
    elif not provider_network_ok:
        blocking_reason = str(runtime.get("provider_network_detail") or "OpenClaw provider network unavailable").strip()
    return {
        "runtime": runtime,
        "gateway_ready": gateway_ready,
        "provider_network_ok": provider_network_ok,
        "ai_ready": bool(gateway_ready and provider_network_ok),
        "ai_detail": ai_detail,
        "blocking_reason": blocking_reason,
    }


def _compact_text(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?：:；;\"'“”‘’（）()【】\[\]<>《》-]+", "", str(text or "").strip().lower())


def _get_device_json(device_ip: str, path: str, timeout_sec: float = 4.0) -> Dict[str, object]:
    url = f"http://{device_ip}{path}"
    response = httpx.get(url, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    raise ValueError("Invalid device payload")


def _assessment_speak_text(device_ip: str, text: str, timeout_sec: float = 8.0) -> Dict[str, object]:
    clean = str(text or "").strip()
    if not clean:
        return {"ok": False, "detail": "empty_text"}
    return _post_device_json(device_ip, "/speak", {"text": clean}, timeout_sec=timeout_sec)


def _apply_assessment_question(payload: Dict[str, object], question: Dict[str, object], *, default_source: str = "ai") -> None:
    next_question = dict(question or {})
    if not next_question:
        return
    payload["latest_question"] = str(next_question.get("prompt") or payload.get("latest_question") or "")
    payload["last_question_id"] = str(next_question.get("id") or payload.get("last_question_id") or "")
    payload["question_pair"] = str(next_question.get("pair") or payload.get("question_pair") or "")
    payload["current_focus"] = str(
        next_question.get("focus") or next_question.get("pair") or payload.get("current_focus") or ""
    )
    rationale = str(next_question.get("rationale") or "").strip().lower()
    payload["question_source"] = "fallback" if rationale.startswith("fallback") else str(default_source or "ai")


def _assessment_should_ignore_transcript(session_payload: Dict[str, object], transcript: str) -> bool:
    clean = _compact_text(transcript)
    if not clean:
        return True
    last_consumed = _compact_text(str(session_payload.get("voice_last_consumed_transcript") or ""))
    if last_consumed and clean == last_consumed:
        return True

    latest_question = _compact_text(str(session_payload.get("latest_question") or ""))
    if latest_question:
        similarity = SequenceMatcher(None, clean, latest_question).ratio()
        if clean == latest_question or similarity >= 0.82:
            return True

    latest_prompt = _compact_text(str(session_payload.get("voice_last_prompt") or ""))
    if latest_prompt:
        similarity = SequenceMatcher(None, clean, latest_prompt).ratio()
        if clean == latest_prompt or similarity >= 0.82:
            return True

    if clean in {_compact_text("我在"), _compact_text("你好"), _compact_text("收到")}:
        return True
    return False


async def _assessment_apply_turn(
    conn: Connection,
    user_id: int,
    session_id: int,
    session_payload: Dict[str, object],
    *,
    answer_text: str,
    transcript_text: str,
    device_id: Optional[str],
    voice_mode: str,
    surface: str,
    client_turn_id: str = "",
) -> tuple[Dict[str, object], ActivationAssessmentTurnResponse]:
    ai_snapshot = _activation_ai_runtime_snapshot()
    if not bool(ai_snapshot["ai_ready"]):
        session_payload["status"] = "blocked"
        session_payload["blocking_reason"] = str(ai_snapshot["blocking_reason"] or "AI 未就绪，正式建档已暂停。")
        session_payload["assessment_ready"] = False
        _save_assessment_session(conn, int(user_id), session_payload, session_id=int(session_id))
        device_online, _selected = _assessment_device_online(conn, int(user_id), device_id)
        response = _assessment_response(session_payload, device_online=device_online, exists=True)
        return session_payload, ActivationAssessmentTurnResponse(
            **response.model_dump(),
            question_changed=False,
            just_completed=False,
        )

    question = {
        "id": str(session_payload.get("last_question_id") or "").strip() or f"dialogue-{int(time.time() * 1000)}",
        "pair": str(session_payload.get("current_focus") or session_payload.get("question_pair") or "").strip(),
        "prompt": str(session_payload.get("latest_question") or "").strip(),
    }
    try:
        analysis = await _assessment_analyze_turn_model(int(user_id), question, session_payload, answer_text)
        if not analysis:
            raise RuntimeError("assessment_turn_analysis_empty")
        scoring_source = "ai"
    except Exception as exc:
        analysis = fallback_turn_analysis(question, session_payload, answer_text, error=str(exc))
        scoring_source = "fallback"

    scoring = {
        "effective": bool(analysis.get("effective", True)),
        "profile_updates": dict(analysis.get("profile_updates") or {}),
        "evidence_summary": list(analysis.get("evidence_summary") or []),
        "reasoning": str(analysis.get("reasoning") or "").strip(),
        "next_focus": str(analysis.get("missing_area") or "").strip(),
        "stable_enough": bool(analysis.get("should_finish", False)),
        "confidence": float(analysis.get("confidence") or 0.0),
        "summary_hint": str((analysis.get("profile_updates") or {}).get("summary") or "").strip(),
    }
    merged = merge_scoring(session_payload, question, answer_text, scoring, int(time.time() * 1000))
    merged["voice_mode"] = str(voice_mode or "text").strip() or "text"
    merged["scoring_source"] = scoring_source
    merged["question_pair"] = str(question.get("pair") or merged.get("question_pair") or "")
    merged["status"] = "active"
    merged["blocking_reason"] = ""
    should_finish = bool(analysis.get("should_finish"))
    finish_reason = str(analysis.get("finish_reason") or "need_more_signal").strip() or "need_more_signal"
    missing_area = str(analysis.get("missing_area") or "").strip()
    next_question = dict(analysis.get("next_question") or {})

    if should_finish and merged.get("status") != "completed":
        merged["completed_at_ms"] = int(time.time() * 1000)
        merged["finish_reason"] = finish_reason or "model_finish"
        merged["confidence"] = max(float(merged.get("confidence") or 0.0), float(analysis.get("confidence") or 0.0))
        merged["final_result"] = build_final_profile(merged)
        if bool((merged.get("final_result") or {}).get("assessment_ready")):
            merged["status"] = "completed"
            merged["latest_question"] = ""
            merged["last_question_id"] = ""
            merged["current_focus"] = ""
        else:
            merged["status"] = "blocked"
            merged["blocking_reason"] = "AI 认为当前信息还不够稳定，正式建档继续等待更多回答。"
    elif merged.get("status") != "completed":
        if not next_question:
            fallback_question = fallback_next_question(merged, missing_area)
            if fallback_question:
                next_question = fallback_question
            else:
                merged["status"] = "blocked"
                merged["blocking_reason"] = "AI 下一题生成未完成：缺少下一题内容"
                merged["assessment_ready"] = False
        if next_question and merged.get("status") != "blocked":
            if missing_area and not str(next_question.get("focus") or "").strip():
                next_question["focus"] = missing_area
            _apply_assessment_question(merged, next_question)

    _append_assessment_turn_event(
        conn,
        int(user_id),
        int(session_id),
        int(merged.get("turn_count") or 0),
        str(question.get("id") or ""),
        str(question.get("prompt") or ""),
        answer_text,
        transcript_text,
        scoring,
    )
    if merged.get("status") == "completed":
        try:
            merged["final_result"] = await _persist_assessment_completion(conn, int(user_id), merged)
        except Exception as exc:
            merged["status"] = "blocked"
            merged["assessment_ready"] = False
            merged["blocking_reason"] = f"AI 记忆压缩写入失败：{exc}"
            merged["final_result"] = {}

    _save_assessment_session(conn, int(user_id), merged, session_id=int(session_id))
    device_online, _selected = _assessment_device_online(conn, int(user_id), device_id)
    response = _assessment_response(merged, device_online=device_online, exists=True)
    turn_response = ActivationAssessmentTurnResponse(
        **response.model_dump(),
        question_changed=bool(response.latest_question),
        just_completed=bool(merged.get("status") == "completed"),
    )
    if str(client_turn_id or "").strip():
        merged["last_client_turn_id"] = str(client_turn_id).strip()
        merged["last_turn_response"] = turn_response.model_dump()
        _save_assessment_session(conn, int(user_id), merged, session_id=int(session_id))
    return merged, turn_response


def _heuristic_personality_profile(text: str) -> Dict[str, object]:
    raw = str(text or "").strip()
    lowered = raw.lower()
    traits: List[str] = []
    topics: List[str] = []
    boundaries: List[str] = []
    signals: List[str] = []
    response_style = "先给结论，再补解释。"
    care_style = "以低打扰、可执行、短句陪伴为主。"
    if any(token in raw for token in ["直接", "别绕", "简洁", "效率"]):
        traits.append("偏直接")
        response_style = "先给结论，避免绕圈，必要时再补步骤。"
    if any(token in raw for token in ["理性", "逻辑", "分析"]):
        traits.append("偏理性")
    if any(token in raw for token in ["敏感", "容易想多", "内耗", "焦虑"]):
        traits.append("容易内耗")
        topics.append("压力与焦虑管理")
        care_style = "先安抚情绪，再给一个很小的下一步。"
    if any(token in raw for token in ["睡眠", "熬夜", "作息"]):
        topics.append("睡眠节律")
    if any(token in raw for token in ["工作", "任务", "效率", "项目"]):
        topics.append("工作压力")
    if any(token in raw for token in ["不要催", "别催", "不喜欢被催"]):
        boundaries.append("不要频繁催促")
    if any(token in raw for token in ["不要说教", "别说教"]):
        boundaries.append("不要说教")
    if any(token in raw for token in ["先自己扛", "先不说", "先沉默", "不想马上说"]):
        signals.append("压力大时可能先沉默")
    if "不是生气" in raw or "只是着急" in raw or "只是烦" in raw:
        signals.append("语气直接不等于敌意")
    if "幽默" in raw or "轻松" in raw:
        care_style = "允许轻松一点，但避免过度玩笑。"
    if not traits:
        traits = ["需要稳定感", "偏长期陪伴"]
    if not topics:
        topics = ["日常压力", "情绪节律"]
    if not boundaries:
        boundaries = ["避免一次问太多问题"]
    if not signals:
        signals = ["需要先被理解，再接受建议"]
    summary = "这个用户更适合稳定、低打扰、持续记忆式陪伴。"
    return {
        "summary": summary,
        "response_style": response_style,
        "care_style": care_style,
        "traits": traits[:5],
        "topics": topics[:5],
        "boundaries": boundaries[:5],
        "signals": signals[:5],
        "confidence": 0.42 if raw else 0.0,
        "sample_count": max(1, len([line for line in raw.splitlines() if line.strip()])) if raw else 0,
        "inference_version": "v1",
        "heuristic": True,
    }


def _build_assistant_identity_context(conn: Connection, user_id: int) -> Dict[str, object]:
    activation = _get_activation_profile(conn, user_id)
    personality = _get_personality_profile(conn, user_id)
    psychometric = _get_psychometric_profile(conn, user_id)
    return {
        "identity": {
            "preferred_name": str(activation.get("preferred_name") or "").strip(),
            "role_label": str(activation.get("role_label") or "").strip(),
            "relation_to_robot": str(activation.get("relation_to_robot") or "").strip(),
            "identity_summary": str(activation.get("identity_summary") or "").strip(),
            "voice_intro_summary": str(activation.get("voice_intro_summary") or "").strip(),
        },
        "personality": {
            "summary": str(personality.get("summary") or "").strip(),
            "response_style": str(personality.get("response_style") or "").strip(),
            "care_style": str(personality.get("care_style") or "").strip(),
            "traits": [str(item) for item in personality.get("traits") or [] if str(item).strip()],
            "topics": [str(item) for item in personality.get("topics") or [] if str(item).strip()],
            "boundaries": [str(item) for item in personality.get("boundaries") or [] if str(item).strip()],
            "signals": [str(item) for item in personality.get("signals") or [] if str(item).strip()],
        },
        "psychometric": {
            "summary": str(psychometric.get("summary") or "").strip(),
            "interaction_preferences": [
                str(item) for item in psychometric.get("interaction_preferences") or [] if str(item).strip()
            ],
            "decision_style": str(psychometric.get("decision_style") or psychometric.get("response_style") or "").strip(),
            "stress_response": str(psychometric.get("stress_response") or "").strip(),
            "comfort_preferences": [
                str(item) for item in psychometric.get("comfort_preferences") or [] if str(item).strip()
            ],
            "avoid_patterns": [
                str(item) for item in psychometric.get("avoid_patterns") or [] if str(item).strip()
            ],
            "care_guidance": str(psychometric.get("care_guidance") or psychometric.get("care_style") or "").strip(),
            "confidence": float(psychometric.get("confidence") or 0.0),
            "conversation_count": int(psychometric.get("conversation_count") or 0),
            "completion_reason": str(psychometric.get("completion_reason") or "").strip(),
        },
        "runtime_preferences": {
            "preferred_mode": OPENCLAW_PREFERRED_MODE,
            "preferred_code_model": OPENCLAW_PREFERRED_CODE_MODEL,
        },
    }


def _extract_json_block(text: str) -> Dict[str, object]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _assessment_device_online(conn: Connection, user_id: int, device_id: Optional[str] = None) -> tuple[bool, Optional[Dict[str, object]]]:
    selected = _get_device(conn, int(user_id), device_id) if device_id else {}
    if not selected:
        devices = _list_devices(conn, int(user_id))
        selected = devices[0] if devices else {}
    resolved_ip = str(selected.get("device_ip") or "").strip() if selected else ""
    return bool(selected and resolved_ip), (selected or None)


def _assessment_sync_voice_prompt(
    session_payload: Dict[str, object],
    *,
    device_ip: str,
    speak_question: bool = True,
) -> tuple[Dict[str, object], bool, str]:
    prompt_text = str(session_payload.get("latest_question") or "").strip()
    if not speak_question:
        return session_payload, False, "speak_disabled"
    if not prompt_text:
        return session_payload, False, "no_prompt"
    if not bool(session_payload.get("voice_session_active")):
        return session_payload, False, "voice_inactive"
    if _compact_text(prompt_text) == _compact_text(str(session_payload.get("voice_last_prompt") or "")):
        return session_payload, False, "prompt_already_spoken"
    _assessment_speak_text(device_ip, prompt_text)
    session_payload["voice_last_prompt"] = prompt_text
    session_payload["voice_last_prompt_ms"] = int(time.time() * 1000)
    return session_payload, True, "prompt_spoken"


async def _gateway_send_message_with_timeout(session_key: str, prompt: str, timeout_ms: int) -> str:
    try:
        return await assistant_service.gateway.send_message(session_key, prompt, timeout_ms=timeout_ms)
    except TypeError:
        return await assistant_service.gateway.send_message(session_key, prompt)


def _is_retryable_assessment_gateway_error(exc: Exception) -> bool:
    detail = str(exc or "").strip().lower()
    if not detail:
        return False
    retry_markers = (
        "429",
        "rate limit",
        "速率限制",
        "network_error",
        "timeout",
        "timed out",
        "fetch failed",
        "provider finish_reason: network_error",
    )
    return any(marker in detail for marker in retry_markers)


async def _gateway_send_message_with_backoff(
    session_key: str,
    prompt: str,
    timeout_ms: int,
    *,
    retry_delays_ms: tuple[int, ...] = ACTIVATION_ASSESSMENT_RETRY_DELAYS_MS,
) -> str:
    last_exc: Optional[Exception] = None
    attempts = 1 + len(retry_delays_ms)
    for attempt in range(attempts):
        try:
            return await _gateway_send_message_with_timeout(session_key, prompt, timeout_ms=timeout_ms)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_retryable_assessment_gateway_error(exc):
                raise
            await asyncio.sleep(max(0.0, retry_delays_ms[attempt] / 1000.0))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("assessment_gateway_retry_failed")


def _assessment_gateway_session_key(user_id: int, stage: str) -> str:
    clean_stage = re.sub(r"[^a-z0-9_-]+", "-", str(stage or "turn").strip().lower()).strip("-") or "turn"
    return f"activation:{int(user_id)}:assessment:{clean_stage}"


async def _assessment_pick_model_question(
    user_id: int,
    activation_profile: Dict[str, object],
    session_payload: Dict[str, object],
) -> Dict[str, object]:
    def _trim_text(value: object, limit: int = 140) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _compact_profile_preview(raw: Dict[str, object]) -> Dict[str, object]:
        return {
            "summary": _trim_text(raw.get("summary"), 120),
            "interaction_preferences": [str(item).strip() for item in (raw.get("interaction_preferences") or []) if str(item).strip()][:3],
            "decision_style": _trim_text(raw.get("decision_style"), 96),
            "stress_response": _trim_text(raw.get("stress_response"), 96),
            "comfort_preferences": [str(item).strip() for item in (raw.get("comfort_preferences") or []) if str(item).strip()][:3],
            "avoid_patterns": [str(item).strip() for item in (raw.get("avoid_patterns") or []) if str(item).strip()][:3],
            "care_guidance": _trim_text(raw.get("care_guidance"), 120),
        }

    compact_session = {
        "conversation_count": int(session_payload.get("conversation_count") or 0),
        "effective_turn_count": int(session_payload.get("effective_turn_count") or 0),
        "current_focus": str(session_payload.get("current_focus") or "").strip(),
        "profile_preview": _compact_profile_preview(dict(session_payload.get("profile_preview") or {})),
        "dialogue_turns": [
            {
                "role": str(item.get("role") or "").strip(),
                "text": _trim_text(item.get("text"), 180),
            }
            for item in (session_payload.get("dialogue_turns") or [])[-3:]
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ],
    }
    compact_identity = {
        "preferred_name": str(activation_profile.get("preferred_name") or "").strip(),
        "identity_summary": _trim_text(activation_profile.get("identity_summary"), 120),
        "voice_intro_summary": _trim_text(activation_profile.get("voice_intro_summary"), 120),
    }
    prompt = (
        f"{ASSESSMENT_CONDUCTOR_PROMPT}\n\n"
        f"输入数据：{json.dumps({'user_id': user_id, 'identity': compact_identity, 'session': compact_session}, ensure_ascii=False)}"
    )
    session_key = _assessment_gateway_session_key(int(user_id), "conductor")
    try:
        raw = await _gateway_send_message_with_backoff(
            session_key,
            prompt,
            timeout_ms=ACTIVATION_ASSESSMENT_MODEL_TIMEOUT_MS,
        )
        parsed = extract_next_question_from_model(raw)
        if parsed:
            return parsed
    except Exception:
        pass
    return fallback_next_question(session_payload, str(session_payload.get("current_focus") or "").strip())


async def _assessment_analyze_turn_model(
    user_id: int,
    question: Dict[str, object],
    session_payload: Dict[str, object],
    answer: str,
) -> Dict[str, object]:
    def _trim_text(value: object, limit: int = 140) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _compact_profile_preview(raw: Dict[str, object]) -> Dict[str, object]:
        return {
            "summary": _trim_text(raw.get("summary"), 120),
            "interaction_preferences": [str(item).strip() for item in (raw.get("interaction_preferences") or []) if str(item).strip()][:3],
            "decision_style": _trim_text(raw.get("decision_style"), 96),
            "stress_response": _trim_text(raw.get("stress_response"), 96),
            "comfort_preferences": [str(item).strip() for item in (raw.get("comfort_preferences") or []) if str(item).strip()][:3],
            "avoid_patterns": [str(item).strip() for item in (raw.get("avoid_patterns") or []) if str(item).strip()][:3],
            "care_guidance": _trim_text(raw.get("care_guidance"), 120),
        }

    compact_session = {
        "conversation_count": int(session_payload.get("conversation_count") or 0),
        "effective_turn_count": int(session_payload.get("effective_turn_count") or 0),
        "required_min_turns": int(session_payload.get("required_min_turns") or 4),
        "current_focus": str(session_payload.get("current_focus") or session_payload.get("question_pair") or "").strip(),
        "profile_preview": _compact_profile_preview(dict(session_payload.get("profile_preview") or {})),
        "dialogue_turns": [
            {
                "role": str(item.get("role") or "").strip(),
                "text": _trim_text(item.get("text"), 120),
            }
            for item in (session_payload.get("dialogue_turns") or [])[-2:]
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ],
    }
    prompt = (
        f"{ASSESSMENT_TURN_PROMPT}\n\n"
        f"输入数据：{json.dumps({'question': {'id': question.get('id'), 'prompt': _trim_text(question.get('prompt'), 120), 'focus': question.get('focus') or question.get('pair')}, 'session': compact_session, 'answer': _trim_text(answer, 160)}, ensure_ascii=False)}"
    )
    session_key = _assessment_gateway_session_key(int(user_id), "turn")
    raw = await _gateway_send_message_with_backoff(
        session_key,
        prompt,
        timeout_ms=ACTIVATION_ASSESSMENT_MODEL_TIMEOUT_MS,
    )
    return extract_turn_analysis_from_model(raw)


async def _assessment_memory_writer_model(user_id: int, final_profile: Dict[str, object], session_payload: Dict[str, object]) -> Dict[str, object]:
    prompt = (
        f"{ASSESSMENT_MEMORY_WRITER_PROMPT}\n\n"
        f"输入数据：{json.dumps({'final_profile': final_profile, 'session': session_payload}, ensure_ascii=False)}"
    )
    session_key = _assessment_gateway_session_key(int(user_id), "memory")
    raw = await _gateway_send_message_with_backoff(
        session_key,
        prompt,
        timeout_ms=ACTIVATION_ASSESSMENT_MEMORY_TIMEOUT_MS,
    )
    parsed = parse_json_dict(raw)
    machine_readable = str(parsed.get("machine_readable") or "").strip()
    ai_readable = str(parsed.get("ai_readable") or "").strip()
    if not machine_readable or not ai_readable:
        raise RuntimeError("AI memory compression returned incomplete payload")
    return {
        "memory_title": str(parsed.get("memory_title") or "activation_dialogue_profile").strip() or "activation_dialogue_profile",
        "machine_readable": machine_readable,
        "ai_readable": ai_readable,
    }


def _assessment_sync_personality_profile(conn: Connection, user_id: int, psychometric: Dict[str, object]) -> None:
    summary = str(psychometric.get("summary") or "").strip()
    _upsert_personality_profile(
        conn,
        int(user_id),
        {
            "summary": summary,
            "response_style": str(psychometric.get("decision_style") or psychometric.get("response_style") or "").strip(),
            "care_style": str(psychometric.get("care_guidance") or psychometric.get("care_style") or "").strip(),
            "traits": [str(item).strip() for item in psychometric.get("interaction_preferences") or [] if str(item).strip()][:6],
            "topics": [],
            "boundaries": [str(item).strip() for item in psychometric.get("avoid_patterns") or [] if str(item).strip()][:6],
            "signals": [
                *([str(psychometric.get("stress_response") or "").strip()] if str(psychometric.get("stress_response") or "").strip() else []),
                *[str(item) for item in ((psychometric.get("evidence_summary") or {}).get("highlights") or [])[:3]],
            ],
            "confidence": float(psychometric.get("confidence") or 0.0),
            "sample_count": int(psychometric.get("conversation_count") or 0),
            "inference_version": str(psychometric.get("inference_version") or "activation-dialogue-v3"),
            "profile": {"source": "activation_dialogue", "psychometric_profile": psychometric},
        },
    )


async def _persist_assessment_completion(conn: Connection, user_id: int, session_payload: Dict[str, object]) -> Dict[str, object]:
    final_profile = build_final_profile(session_payload)
    if not bool(final_profile.get("assessment_ready")):
        raise RuntimeError("assessment_result_not_ready")
    memory_payload = await _assessment_memory_writer_model(int(user_id), final_profile, session_payload)
    final_profile["memory_summary"] = {
        "machine_readable": memory_payload["machine_readable"],
        "ai_readable": memory_payload["ai_readable"],
    }
    psychometric = _upsert_psychometric_profile(conn, int(user_id), final_profile)
    _assessment_sync_personality_profile(conn, int(user_id), psychometric)
    activation = _get_activation_profile(conn, int(user_id))
    preferred_name = str(activation.get("preferred_name") or "").strip()
    assistant_service.store.append_memory(
        int(user_id),
        title=str(memory_payload["memory_title"] or "activation_dialogue_profile"),
        content=f"{memory_payload['machine_readable']}\n{memory_payload['ai_readable']}",
        tags=["activation", "assessment", "dialogue_profile"],
    )
    assistant_service.store.append_memory(
        int(user_id),
        title="activation_dialogue_profile_backup",
        content=build_memory_summary(psychometric, preferred_name=preferred_name),
        tags=["activation", "assessment", "dialogue_profile", "backup"],
    )
    return psychometric


def _heuristic_activation_identity(transcript: str, observed_name: str = "") -> Dict[str, object]:
    raw = str(transcript or "").strip()
    lowered = raw.lower()
    preferred_name = str(observed_name or "").strip()
    patterns = [
        r"我叫([^\s，。,.!！?？]{1,12})",
        r"我是([^\s，。,.!！?？]{1,12})",
        r"你可以叫我([^\s，。,.!！?？]{1,12})",
        r"叫我([^\s，。,.!！?？]{1,12})就行",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            preferred_name = str(match.group(1) or "").strip("，。,.!！?？ ")
            if preferred_name:
                break
    role_label = "unknown"
    relation = "unknown"
    if any(token in raw for token in ["主人", "owner"]):
        role_label = "owner"
        relation = "primary_user"
    elif any(token in raw for token in ["家人", "妈妈", "爸爸", "姐姐", "哥哥", "妹妹", "弟弟", "family"]):
        role_label = "family"
        relation = "family_member"
    elif any(token in raw for token in ["护工", "照护", "护理", "caregiver"]):
        role_label = "caregiver"
        relation = "caregiver"
    elif any(token in lowered for token in ["admin", "管理员", "调试", "维护", "operator"]):
        role_label = "operator" if "operator" in lowered or "调试" in raw else "admin"
        relation = "maintainer"
    elif any(token in raw for token in ["病人", "患者"]):
        role_label = "patient"
        relation = "primary_user"
    summary = ""
    if preferred_name:
        summary = f"{preferred_name} 是当前与机器人首次确认身份的用户。"
    if role_label == "owner":
        summary = f"{preferred_name or '该用户'} 是机器人的主人，后续应优先按主人身份服务。"
    elif role_label == "family":
        summary = f"{preferred_name or '该用户'} 是与机器人相关的家庭成员，后续应按家庭成员身份理解。"
    elif role_label == "caregiver":
        summary = f"{preferred_name or '该用户'} 是照护相关人员，后续应按照护者身份支持。"
    elif role_label in {"operator", "admin"}:
        summary = f"{preferred_name or '该用户'} 更像是设备操作或维护人员，不应默认视作主人。"
    notes = "待确认：本结果由本地规则保守推断，建议在激活页人工确认。"
    voice_intro = raw[:80]
    confidence = 0.28
    if preferred_name:
        confidence += 0.18
    if role_label != "unknown":
        confidence += 0.24
    return {
        "preferred_name": preferred_name,
        "role_label": role_label,
        "relation_to_robot": relation,
        "pronouns": "",
        "identity_summary": summary[:80],
        "onboarding_notes": notes[:120],
        "voice_intro_summary": voice_intro,
        "confidence": max(0.0, min(0.75, confidence)),
        "heuristic": True,
    }


def _activation_page_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>首次激活</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe7;
      --card: #fffaf1;
      --ink: #1c2a1f;
      --muted: #5c6b60;
      --line: #d7cdbb;
      --accent: #2e7d5b;
      --accent-2: #b85c38;
    }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(46,125,91,0.12), transparent 28%),
        radial-gradient(circle at bottom left, rgba(184,92,56,0.14), transparent 26%),
        var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 980px;
      margin: 32px auto;
      padding: 0 20px 40px;
    }}
    h1 {{
      font-size: 32px;
      margin: 0 0 8px;
    }}
    p {{
      color: var(--muted);
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(20, 32, 24, 0.06);
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin: 10px 0 6px;
    }}
    input, textarea, select {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: white;
      color: var(--ink);
    }}
    textarea {{
      min-height: 112px;
      resize: vertical;
    }}
    .row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font: inherit;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }}
    button.secondary {{
      background: var(--accent-2);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #172018;
      color: #eaf5ee;
      padding: 14px;
      border-radius: 14px;
      min-height: 140px;
    }}
    .hint {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>首次激活与身份确认</h1>
    <p>第一次登录后，先确认“这个人是谁、和机器人是什么关系、机器人应该如何理解与服务他/她”。这一步会沉淀成长期身份卡，不直接混进普通聊天上下文。</p>
    <div class="grid">
      <section class="card">
        <label>Bearer Token</label>
        <input id="token" placeholder="粘贴登录返回的 token，或使用 ?token=..." />
        <div class="hint" id="modeHint">默认模式：{OPENCLAW_PREFERRED_MODE}；默认模型偏好：{OPENCLAW_PREFERRED_CODE_MODEL}</div>

        <label>首段语音 / 首次对话记录</label>
        <textarea id="transcript" placeholder="例如：你好，我叫小北，是这个机器人的主人，你可以叫我小北。"></textarea>
        <label>观察到的名字（可选）</label>
        <input id="observed_name" placeholder="如 ASR 或联系人资料里已有名字" />
        <div class="row">
          <button id="loadBtn" type="button">读取当前状态</button>
          <button id="inferBtn" class="secondary" type="button">从首段语音推断身份</button>
        </div>
      </section>

      <section class="card">
        <label>称呼</label>
        <input id="preferred_name" />
        <label>角色</label>
        <select id="role_label">
          <option value="owner">owner</option>
          <option value="family">family</option>
          <option value="caregiver">caregiver</option>
          <option value="guest">guest</option>
          <option value="operator">operator</option>
          <option value="admin">admin</option>
          <option value="patient">patient</option>
          <option value="unknown">unknown</option>
        </select>
        <label>与机器人关系</label>
        <select id="relation_to_robot">
          <option value="primary_user">primary_user</option>
          <option value="family_member">family_member</option>
          <option value="caregiver">caregiver</option>
          <option value="visitor">visitor</option>
          <option value="maintainer">maintainer</option>
          <option value="observer">observer</option>
          <option value="unknown">unknown</option>
        </select>
        <label>代词 / 称谓偏好</label>
        <input id="pronouns" />
        <label>身份摘要</label>
        <textarea id="identity_summary"></textarea>
        <label>激活备注</label>
        <textarea id="onboarding_notes"></textarea>
        <label>首次语音摘要</label>
        <textarea id="voice_intro_summary"></textarea>
        <div class="row">
          <button id="saveBtn" type="button">完成激活</button>
        </div>
      </section>
    </div>

    <section class="card" style="margin-top:18px">
      <label>调试输出</label>
      <pre id="output">等待操作...</pre>
    </section>
  </main>
  <script>
    const q = (id) => document.getElementById(id);
    const output = q("output");
    const tokenInput = q("token");
    const qsToken = new URLSearchParams(window.location.search).get("token") || "";
    tokenInput.value = qsToken || localStorage.getItem("activationToken") || "";

    function headers() {{
      const token = (tokenInput.value || "").trim();
      if (!token) throw new Error("请先提供 Bearer token");
      localStorage.setItem("activationToken", token);
      return {{
        "Authorization": `Bearer ${{token}}`,
        "Content-Type": "application/json"
      }};
    }}

    function fillForm(data) {{
      q("preferred_name").value = data.preferred_name || "";
      q("role_label").value = data.role_label || "owner";
      q("relation_to_robot").value = data.relation_to_robot || "primary_user";
      q("pronouns").value = data.pronouns || "";
      q("identity_summary").value = data.identity_summary || "";
      q("onboarding_notes").value = data.onboarding_notes || "";
      q("voice_intro_summary").value = data.voice_intro_summary || "";
    }}

    async function fetchJson(url, options) {{
      const res = await fetch(url, options);
      const data = await res.json().catch(() => ({{ ok: false, detail: "invalid json" }}));
      if (!res.ok) {{
        throw new Error(data.detail || JSON.stringify(data));
      }}
      return data;
    }}

    async function loadState() {{
      const [state, prompts] = await Promise.all([
        fetchJson("/api/activation/state", {{ headers: headers() }}),
        fetchJson("/api/activation/prompt-pack", {{ headers: headers() }})
      ]);
      fillForm(state);
      q("modeHint").textContent = `默认模式：${{prompts.preferred_mode}}；默认模型偏好：${{prompts.preferred_code_model}}`;
      output.textContent = JSON.stringify(state, null, 2);
    }}

    async function inferIdentity() {{
      const payload = {{
        transcript: q("transcript").value,
        observed_name: q("observed_name").value,
        surface: "robot",
        context: {{
          entrypoint: "activation_page"
        }}
      }};
      const data = await fetchJson("/api/activation/identity/infer", {{
        method: "POST",
        headers: headers(),
        body: JSON.stringify(payload)
      }});
      fillForm(data);
      output.textContent = JSON.stringify(data, null, 2);
    }}

    async function completeActivation() {{
      const payload = {{
        preferred_name: q("preferred_name").value,
        role_label: q("role_label").value,
        relation_to_robot: q("relation_to_robot").value,
        pronouns: q("pronouns").value,
        identity_summary: q("identity_summary").value,
        onboarding_notes: q("onboarding_notes").value,
        voice_intro_summary: q("voice_intro_summary").value,
        profile: {{
          source: "activation_page"
        }},
        activation_version: "v1"
      }};
      const data = await fetchJson("/api/activation/complete", {{
        method: "POST",
        headers: headers(),
        body: JSON.stringify(payload)
      }});
      output.textContent = JSON.stringify(data, null, 2);
    }}

    q("loadBtn").addEventListener("click", () => loadState().catch((err) => output.textContent = String(err)));
    q("inferBtn").addEventListener("click", () => inferIdentity().catch((err) => output.textContent = String(err)));
    q("saveBtn").addEventListener("click", () => completeActivation().catch((err) => output.textContent = String(err)));

    loadState().catch((err) => output.textContent = `等待 token 或接口可用：${{err}}`);
  </script>
</body>
</html>"""


def _list_emotion_events(
    conn: Connection,
    user_id: int,
    limit: int,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> List[Dict]:
    query = (
        "SELECT * FROM emotion_events "
        "WHERE user_id = ? AND type IN ('HAPPY','SAD','ANGRY','CALM','TIRED','ANXIOUS')"
    )
    params: List[Any] = [user_id]
    if start_ms is not None:
        query += " AND timestamp_ms >= ?"
        params.append(int(start_ms))
    if end_ms is not None:
        query += " AND timestamp_ms <= ?"
        params.append(int(end_ms))
    query += " ORDER BY timestamp_ms DESC LIMIT ?"
    params.append(limit)
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def _insert_emotion_event(conn: Connection, user_id: int, payload: EmotionEventRequest) -> int:
    cur = conn.execute(
        """
        INSERT INTO emotion_events (
            user_id, timestamp_ms, type, description, v, a, t, s, intensity, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            payload.timestamp_ms,
            payload.type,
            payload.description,
            payload.V,
            payload.A,
            payload.T,
            payload.S,
            payload.intensity,
            payload.source,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _list_chat_messages(
    conn: Connection,
    user_id: int,
    limit: int,
    session_key: Optional[str] = None,
    surface: Optional[str] = None,
) -> List[Dict]:
    query = [
        "SELECT * FROM chat_messages",
        "WHERE user_id = ?",
    ]
    params: List[Any] = [user_id]
    if session_key:
        query.append("AND session_key = ?")
        params.append(str(session_key))
    elif surface:
        query.append("AND surface = ?")
        params.append(str(surface))
    query.append("ORDER BY timestamp_ms ASC LIMIT ?")
    params.append(limit)
    cur = conn.execute("\n".join(query), params)
    rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        attachments_raw = row.get("attachments_json", "[]")
        try:
            attachments = json.loads(str(attachments_raw or "[]"))
            if not isinstance(attachments, list):
                attachments = []
        except Exception:
            attachments = []
        row["attachments"] = attachments
        row["content_type"] = str(row.get("content_type") or "text")
        row["surface"] = str(row.get("surface") or "desktop")
        row["session_key"] = str(row.get("session_key") or "").strip() or None
    return rows


def _insert_chat_message(conn: Connection, user_id: int, payload: ChatMessageRequest) -> int:
    attachments_json = "[]"
    try:
        attachments_json = json.dumps(payload.attachments or [], ensure_ascii=False, separators=(",", ":"))
    except Exception:
        attachments_json = "[]"
    cur = conn.execute(
        """
        INSERT INTO chat_messages (user_id, sender, text, content_type, attachments_json, timestamp_ms, surface, session_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            payload.sender,
            payload.text,
            str(payload.content_type or "text"),
            attachments_json,
            payload.timestamp_ms,
            str(payload.surface or "desktop"),
            str(payload.session_key or "").strip() or None,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _resolve_wechat_mirror_target() -> Optional[Dict[str, str]]:
    state_dir = Path(__file__).resolve().parents[1] / "assistant_data" / "openclaw_state" / "openclaw-weixin"
    accounts_index = state_dir / "accounts.json"
    try:
        account_ids = json.loads(accounts_index.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(account_ids, list) or not account_ids:
        return None
    account_id = str(account_ids[0] or "").strip()
    if not account_id:
        return None
    account_path = state_dir / "accounts" / f"{account_id}.json"
    try:
        account_payload = json.loads(account_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(account_payload, dict):
        return None
    token = str(account_payload.get("token") or "").strip()
    base_url = str(account_payload.get("baseUrl") or "https://ilinkai.weixin.qq.com").strip()
    preferred_user_id = str(account_payload.get("userId") or "").strip()
    context_path = state_dir / "accounts" / f"{account_id}.context-tokens.json"
    try:
        context_payload = json.loads(context_path.read_text(encoding="utf-8"))
    except Exception:
        context_payload = {}
    if not isinstance(context_payload, dict):
        context_payload = {}
    user_id = preferred_user_id
    context_token = str(context_payload.get(user_id) or "").strip() if user_id else ""
    if (not user_id or not context_token) and context_payload:
        for raw_user_id, raw_context in context_payload.items():
            candidate_user_id = str(raw_user_id or "").strip()
            candidate_context = str(raw_context or "").strip()
            if candidate_user_id and candidate_context:
                user_id = candidate_user_id
                context_token = candidate_context
                break
    if not token or not base_url or not user_id:
        return None
    return {
        "account_id": account_id,
        "base_url": base_url,
        "token": token,
        "user_id": user_id,
        "context_token": context_token,
        "channel_version": "2.1.1",
        "app_id": "bot",
    }


def _build_wechat_client_version(version: str) -> int:
    parts = [int(part) if str(part).isdigit() else 0 for part in str(version or "0.0.0").split(".")]
    major = parts[0] if len(parts) > 0 else 0
    minor = parts[1] if len(parts) > 1 else 0
    patch = parts[2] if len(parts) > 2 else 0
    return ((major & 0xFF) << 16) | ((minor & 0xFF) << 8) | (patch & 0xFF)


async def _mirror_bot_reply_to_wechat(text: str) -> bool:
    clean_text = str(text or "").strip()
    if not clean_text:
        return False
    target = _resolve_wechat_mirror_target()
    if not target:
        return False
    client_id = f"desktop-sync-{int(time.time() * 1000)}-{secrets.token_hex(4)}"
    x_wechat_uin = base64.b64encode(str(secrets.randbits(32)).encode("utf-8")).decode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Authorization": f"Bearer {target['token']}",
        "iLink-App-Id": target["app_id"],
        "iLink-App-ClientVersion": str(_build_wechat_client_version(target["channel_version"])),
        "X-WECHAT-UIN": x_wechat_uin,
    }
    body = {
        "msg": {
            "from_user_id": "",
            "to_user_id": target["user_id"],
            "client_id": client_id,
            "message_type": 2,
            "message_state": 2,
            "item_list": [
                {
                    "type": 1,
                    "text_item": {
                        "text": clean_text,
                    },
                }
            ],
            "context_token": target.get("context_token") or None,
        },
        "base_info": {
            "channel_version": target["channel_version"],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            response = await client.post(
                f"{target['base_url'].rstrip('/')}/ilink/bot/sendmessage",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[wechat-mirror] outbound mirror failed: {exc}")
        return False


def _sync_wechat_transcript_history(
    conn: Connection,
    user_id: int,
    session_key: str,
    surface: str = "desktop",
) -> List[ChatMessageResponse]:
    transcript_items = assistant_service.list_wechat_mirror_messages(session_key, limit=200)
    if not transcript_items:
        return []
    existing_rows = _list_chat_messages(conn, user_id, 400, session_key=session_key)
    inserted: List[ChatMessageResponse] = []
    for item in transcript_items:
        sender = str(item.get("sender") or "").strip()
        text = str(item.get("text") or "").strip()
        timestamp_ms = int(item.get("timestamp_ms") or 0)
        if not sender or not text or timestamp_ms <= 0:
            continue
        duplicate = next(
            (
                row for row in existing_rows
                if str(row.get("sender") or "").strip() == sender
                and str(row.get("text") or "").strip() == text
                and abs(int(row.get("timestamp_ms") or 0) - timestamp_ms) <= 4000
            ),
            None,
        )
        if duplicate is not None:
            continue
        msg_id = _insert_chat_message(
            conn,
            user_id,
            ChatMessageRequest(
                sender=sender,
                text=text,
                content_type="text",
                attachments=[],
                timestamp_ms=timestamp_ms,
                surface=surface,
                session_key=session_key,
            ),
        )
        existing_rows.append(
            {
                "id": msg_id,
                "sender": sender,
                "text": text,
                "timestamp_ms": timestamp_ms,
                "surface": surface,
                "session_key": session_key,
            }
        )
        inserted.append(
            ChatMessageResponse(
                id=msg_id,
                sender=sender,
                text=text,
                content_type="text",
                attachments=[],
                timestamp_ms=timestamp_ms,
                surface=surface,
                session_key=session_key,
            )
        )
    if inserted:
        conn.commit()
    return inserted


def _chat_response_from_row(row: Dict[str, object]) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=int(row["id"]),
        sender=str(row.get("sender") or ""),
        text=str(row.get("text") or ""),
        content_type=str(row.get("content_type") or "text"),
        attachments=row.get("attachments") if isinstance(row.get("attachments"), list) else [],
        timestamp_ms=int(row.get("timestamp_ms") or 0),
        surface=str(row.get("surface") or "desktop"),
        session_key=str(row.get("session_key") or "").strip() or None,
    )


def _usage_date_key(ts_ms: Optional[int] = None) -> str:
    now_ms = int(ts_ms if ts_ms is not None else time.time() * 1000)
    return time.strftime("%Y-%m-%d", time.localtime(now_ms / 1000.0))


def _get_tool_usage_daily(conn: Connection, user_id: int, date_key: str) -> Dict[str, int]:
    cur = conn.execute(
        """
        SELECT web_search_count, emotion_auto_search_count
        FROM tool_usage_daily
        WHERE user_id = ? AND date_key = ?
        """,
        (int(user_id), str(date_key)),
    )
    row = cur.fetchone()
    if not row:
        return {"web_search_count": 0, "emotion_auto_search_count": 0}
    return {
        "web_search_count": int(row["web_search_count"] or 0),
        "emotion_auto_search_count": int(row["emotion_auto_search_count"] or 0),
    }


def _bump_tool_usage_daily(
    conn: Connection,
    user_id: int,
    date_key: str,
    web_search_delta: int = 0,
    emotion_auto_delta: int = 0,
) -> None:
    if web_search_delta == 0 and emotion_auto_delta == 0:
        return
    now_ms = int(time.time() * 1000)
    conn.execute(
        """
        INSERT INTO tool_usage_daily (
            user_id, date_key, web_search_count, emotion_auto_search_count, updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, date_key) DO UPDATE SET
            web_search_count = web_search_count + excluded.web_search_count,
            emotion_auto_search_count = emotion_auto_search_count + excluded.emotion_auto_search_count,
            updated_at = excluded.updated_at
        """,
        (int(user_id), str(date_key), int(web_search_delta), int(emotion_auto_delta), now_ms),
    )
    conn.commit()


def _list_devices(conn: Connection, user_id: int) -> List[Dict]:
    cur = conn.execute(
        """
        SELECT * FROM devices
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    )
    return [dict(row) for row in cur.fetchall()]


def _get_device(conn: Connection, user_id: int, device_id: str) -> Dict:
    cur = conn.execute(
        """
        SELECT * FROM devices
        WHERE user_id = ? AND device_id = ?
        """,
        (user_id, device_id),
    )
    row = cur.fetchone()
    return dict(row) if row else {}


def _select_device_for_user(conn: Connection, user_id: int, explicit_device_id: Optional[str] = None) -> Dict:
    if explicit_device_id:
        return _get_device(conn, user_id, explicit_device_id)
    devices = _list_devices(conn, user_id)
    return devices[0] if devices else {}


def _default_device_settings() -> Dict[str, object]:
    return {
        "mode": "normal",
        "care_delivery_strategy": "policy",
        "assistant": {"mode": "product", "native_control_enabled": True},
        "media": {"camera_enabled": True, "audio_enabled": True},
        "wake": {"enabled": True, "wake_phrase": "小念", "ack_text": "我在"},
        "behavior": {"cooldown_min": 30, "daily_trigger_limit": 5, "settings_auto_return_sec": 0},
        "tracking": {"pan_enabled": True, "tilt_enabled": True},
        "voice": {
            "desktop_stt_provider": "sensevoice_small",
            "desktop_stt_model": "SenseVoiceSmall",
            "robot_tts_provider": "voxcpm_distill",
            "robot_voice_style": "gentle",
        },
    }


def _default_device_ui_state() -> Dict[str, object]:
    return {
        "page": "expression",
        "screen_awake": True,
        "source": "backend",
        "opened_at_ms": None,
        "last_closed_at_ms": None,
    }


def _merge_settings(base: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base)
    for key, value in (incoming or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_settings(dict(merged.get(key) or {}), value)
        else:
            merged[key] = value
    return merged


def _get_device_settings(conn: Connection, user_id: int, device_id: str) -> Dict[str, object]:
    row = conn.execute(
        """
        SELECT settings_json, updated_at
        FROM device_settings_profiles
        WHERE user_id = ? AND device_id = ?
        """,
        (int(user_id), str(device_id)),
    ).fetchone()
    settings = _default_device_settings()
    updated_at_ms = None
    if row:
        try:
            stored = json.loads(str(row["settings_json"] or "{}"))
        except Exception:
            stored = {}
        if isinstance(stored, dict):
            settings = _merge_settings(settings, stored)
        updated_at_ms = int(row["updated_at"] or 0) or None
    return {"settings": settings, "updated_at_ms": updated_at_ms}


def _get_user_assistant_settings(
    conn: Connection,
    user_id: int,
    explicit_device_id: Optional[str] = None,
) -> Dict[str, object]:
    selected = _select_device_for_user(conn, int(user_id), explicit_device_id)
    if not selected:
        return dict(_default_device_settings().get("assistant") or {})
    payload = _get_device_settings(conn, int(user_id), str(selected.get("device_id") or ""))
    settings = payload.get("settings") if isinstance(payload, dict) else {}
    assistant = settings.get("assistant") if isinstance(settings, dict) else {}
    if isinstance(assistant, dict):
        return _merge_settings(dict(_default_device_settings().get("assistant") or {}), assistant)
    return dict(_default_device_settings().get("assistant") or {})


def _upsert_device_settings(conn: Connection, user_id: int, device_id: str, patch: Dict[str, object]) -> Dict[str, object]:
    current = _get_device_settings(conn, user_id, device_id)
    merged = _merge_settings(dict(current.get("settings") or {}), dict(patch or {}))
    now_ms = int(time.time() * 1000)
    existing = conn.execute(
        """
        SELECT id
        FROM device_settings_profiles
        WHERE user_id = ? AND device_id = ?
        """,
        (int(user_id), str(device_id)),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE device_settings_profiles
            SET settings_json = ?, updated_at = ?
            WHERE user_id = ? AND device_id = ?
            """,
            (json.dumps(merged, ensure_ascii=False), now_ms, int(user_id), str(device_id)),
        )
    else:
        conn.execute(
            """
            INSERT INTO device_settings_profiles (user_id, device_id, settings_json, updated_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(user_id), str(device_id), json.dumps(merged, ensure_ascii=False), now_ms, now_ms),
        )
    conn.commit()
    return {"settings": merged, "updated_at_ms": now_ms}


def _cached_ui_state(device_row: Dict[str, object]) -> Dict[str, object]:
    ui_state = _default_device_ui_state()
    if not device_row:
        return ui_state
    try:
        status_payload = json.loads(str(device_row.get("status_json") or "{}"))
    except Exception:
        status_payload = {}
    if isinstance(status_payload, dict):
        candidate = status_payload.get("ui_state")
        if isinstance(candidate, dict):
            ui_state = _merge_settings(ui_state, candidate)
    return ui_state


def _get_device_owner(conn: Connection, device_id: str) -> Dict:
    cur = conn.execute(
        """
        SELECT * FROM devices
        WHERE device_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (device_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else {}


def _list_wifi_profiles(conn: Connection, user_id: int, device_id: str) -> List[Dict]:
    rows = conn.execute(
        """
        SELECT * FROM wifi_profiles
        WHERE user_id = ? AND device_id = ?
        ORDER BY COALESCE(last_success_at, 0) DESC, updated_at DESC
        """,
        (user_id, device_id),
    ).fetchall()
    return [dict(row) for row in rows]


def _upsert_wifi_profile(
    conn: Connection,
    user_id: int,
    device_id: str,
    ssid: str,
    password: str,
    client_type: str,
) -> Dict:
    now_ms = int(time.time() * 1000)
    existing = conn.execute(
        """
        SELECT * FROM wifi_profiles
        WHERE user_id = ? AND device_id = ? AND ssid = ?
        """,
        (user_id, device_id, ssid),
    ).fetchone()
    encrypted = _encrypt_wifi_password(password)
    if existing:
        conn.execute(
            """
            UPDATE wifi_profiles
            SET encrypted_password = ?,
                last_seen_client_type = ?,
                updated_at = ?
            WHERE user_id = ? AND device_id = ? AND ssid = ?
            """,
            (encrypted, client_type, now_ms, user_id, device_id, ssid),
        )
    else:
        conn.execute(
            """
            INSERT INTO wifi_profiles (
                user_id, device_id, ssid, encrypted_password,
                last_success_at, last_seen_client_type, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (user_id, device_id, ssid, encrypted, client_type, now_ms, now_ms),
        )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM wifi_profiles
        WHERE user_id = ? AND device_id = ? AND ssid = ?
        """,
        (user_id, device_id, ssid),
    ).fetchone()
    return dict(row) if row else {}


def _mark_wifi_profile_success(conn: Connection, user_id: int, device_id: str, ssid: str) -> None:
    now_ms = int(time.time() * 1000)
    conn.execute(
        """
        UPDATE wifi_profiles
        SET last_success_at = ?, updated_at = ?
        WHERE user_id = ? AND device_id = ? AND ssid = ?
        """,
        (now_ms, now_ms, user_id, device_id, ssid),
    )
    conn.commit()


def _upsert_client_session(
    conn: Connection,
    user_id: int,
    client_type: str,
    client_id: str,
    current_ssid: Optional[str],
    client_ip: Optional[str],
    is_active: bool,
) -> Dict:
    now_ms = int(time.time() * 1000)
    row = conn.execute(
        """
        SELECT * FROM client_sessions
        WHERE user_id = ? AND client_type = ? AND client_id = ?
        """,
        (user_id, client_type, client_id),
    ).fetchone()
    if row:
        conn.execute(
            """
            UPDATE client_sessions
            SET current_ssid = ?,
                client_ip = ?,
                last_seen_ms = ?,
                is_active = ?,
                updated_at = ?
            WHERE user_id = ? AND client_type = ? AND client_id = ?
            """,
            (
                current_ssid,
                client_ip,
                now_ms,
                1 if is_active else 0,
                now_ms,
                user_id,
                client_type,
                client_id,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO client_sessions (
                user_id, client_type, client_id, current_ssid, client_ip, last_seen_ms, is_active, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, client_type, client_id, current_ssid, client_ip, now_ms, 1 if is_active else 0, now_ms),
        )
    conn.commit()
    fresh = conn.execute(
        """
        SELECT * FROM client_sessions
        WHERE user_id = ? AND client_type = ? AND client_id = ?
        """,
        (user_id, client_type, client_id),
    ).fetchone()
    return dict(fresh) if fresh else {}


def _list_active_client_sessions(conn: Connection, user_id: int) -> List[Dict]:
    cutoff = int(time.time() * 1000) - CLIENT_SESSION_STALE_MS
    rows = conn.execute(
        """
        SELECT * FROM client_sessions
        WHERE user_id = ? AND is_active = 1 AND last_seen_ms >= ?
        ORDER BY
            CASE client_type WHEN 'desktop' THEN 0 ELSE 1 END ASC,
            last_seen_ms DESC
        """,
        (user_id, cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def _apply_device_network_state(
    conn: Connection,
    user_id: int,
    device_id: str,
    desired_ssid: Optional[str],
    network_mismatch: bool,
    missing_profile: bool,
    last_switch_reason: Optional[str],
) -> None:
    conn.execute(
        """
        UPDATE devices
        SET desired_ssid = ?,
            network_mismatch = ?,
            missing_profile = ?,
            last_switch_reason = ?,
            updated_at = ?
        WHERE user_id = ? AND device_id = ?
        """,
        (
            desired_ssid,
            1 if network_mismatch else 0,
            1 if missing_profile else 0,
            last_switch_reason,
            int(time.time() * 1000),
            user_id,
            device_id,
        ),
    )
    conn.commit()


def _compute_device_network_strategy(conn: Connection, user_id: int, device: Dict) -> Dict[str, Any]:
    device_id = str(device.get("device_id") or "").strip()
    current_ssid = str(device.get("ssid") or "").strip()
    sessions = _list_active_client_sessions(conn, user_id)
    preferred = next(
        (row for row in sessions if str(row.get("current_ssid") or "").strip()),
        None,
    )
    desired_ssid = str((preferred or {}).get("current_ssid") or "").strip() or None
    profiles = _list_wifi_profiles(conn, user_id, device_id)
    known_ssids = {str(item.get("ssid") or "").strip() for item in profiles if item.get("ssid")}
    network_mismatch = bool(desired_ssid and current_ssid and desired_ssid != current_ssid)
    missing_profile = bool(desired_ssid and desired_ssid not in known_ssids)
    last_switch_reason = None
    if desired_ssid:
        source = str((preferred or {}).get("client_type") or "mobile")
        if missing_profile:
            last_switch_reason = f"{source}_priority_missing_profile"
        elif network_mismatch:
            last_switch_reason = f"{source}_priority_network_sync"
        else:
            last_switch_reason = f"{source}_priority_aligned"
    return {
        "desired_ssid": desired_ssid,
        "network_mismatch": network_mismatch,
        "missing_profile": missing_profile,
        "last_switch_reason": last_switch_reason,
        "profiles": profiles,
    }


def _upsert_device(
    conn: Connection,
    user_id: int,
    device_id: str,
    device_ip: Optional[str] = None,
    ssid: Optional[str] = None,
    device_mac: Optional[str] = None,
) -> Dict:
    now_ms = int(time.time() * 1000)
    owner = _get_device_owner(conn, device_id)
    if owner and int(owner.get("user_id") or 0) != int(user_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Device already bound to another account")
    existing = _get_device(conn, user_id, device_id)
    if existing:
        conn.execute(
            """
            UPDATE devices
            SET device_ip = COALESCE(?, device_ip),
                device_mac = COALESCE(?, device_mac),
                ssid = COALESCE(?, ssid),
                updated_at = ?
            WHERE user_id = ? AND device_id = ?
            """,
            (device_ip, device_mac, ssid, now_ms, user_id, device_id),
        )
        conn.commit()
        return _get_device(conn, user_id, device_id)
    conn.execute(
        """
        INSERT INTO devices (user_id, device_id, device_ip, device_mac, ssid, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, device_id, device_ip, device_mac, ssid, now_ms),
    )
    conn.commit()
    return _get_device(conn, user_id, device_id)


def _update_device_status(
    conn: Connection,
    user_id: int,
    device_id: str,
    last_seen_ms: Optional[int],
    status: Optional[dict],
) -> None:
    status_json = json.dumps(status, ensure_ascii=False) if status else None
    now_ms = int(time.time() * 1000)
    conn.execute(
        """
        UPDATE devices
        SET last_seen_ms = COALESCE(?, last_seen_ms),
            status_json = COALESCE(?, status_json),
            onboarding_state = COALESCE(?, onboarding_state),
            identity_state = COALESCE(?, identity_state),
            identity_version = COALESCE(?, identity_version),
            owner_last_seen_ms = COALESCE(?, owner_last_seen_ms),
            updated_at = ?
        WHERE user_id = ? AND device_id = ?
        """,
        (
            last_seen_ms,
            status_json,
            (status or {}).get("onboarding_state") if isinstance(status, dict) else None,
            (status or {}).get("identity_state") if isinstance(status, dict) else None,
            (status or {}).get("embedding_version") if isinstance(status, dict) else None,
            last_seen_ms if isinstance(status, dict) and bool(status.get("owner_recognized")) else None,
            now_ms,
            user_id,
            device_id,
        ),
    )
    conn.commit()


def _update_devices_by_device_id(
    conn: Connection,
    device_id: str,
    device_ip: Optional[str],
    device_mac: Optional[str],
    ssid: Optional[str],
    last_seen_ms: Optional[int],
    status: Optional[dict],
) -> int:
    status_json = json.dumps(status, ensure_ascii=False) if status else None
    now_ms = int(time.time() * 1000)
    cur = conn.execute(
        """
        UPDATE devices
        SET device_ip = COALESCE(?, device_ip),
            device_mac = COALESCE(?, device_mac),
            ssid = COALESCE(?, ssid),
            last_seen_ms = COALESCE(?, last_seen_ms),
            status_json = COALESCE(?, status_json),
            onboarding_state = COALESCE(?, onboarding_state),
            identity_state = COALESCE(?, identity_state),
            identity_version = COALESCE(?, identity_version),
            owner_last_seen_ms = COALESCE(?, owner_last_seen_ms),
            updated_at = ?
        WHERE device_id = ?
        """,
        (
            device_ip,
            device_mac,
            ssid,
            last_seen_ms,
            status_json,
            (status or {}).get("onboarding_state") if isinstance(status, dict) else None,
            (status or {}).get("identity_state") if isinstance(status, dict) else None,
            (status or {}).get("embedding_version") if isinstance(status, dict) else None,
            last_seen_ms if isinstance(status, dict) and bool(status.get("owner_recognized")) else None,
            now_ms,
            device_id,
        ),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def _create_claim_session(conn: Connection, user_id: int, device_id: str) -> Dict:
    now_ms = int(time.time() * 1000)
    expires_at_ms = now_ms + (10 * 60 * 1000)
    claim_token = secrets.token_urlsafe(24)
    conn.execute(
        """
        UPDATE device_claim_sessions
        SET is_active = 0, updated_at = ?
        WHERE device_id = ?
        """,
        (now_ms, device_id),
    )
    conn.execute(
        """
        INSERT INTO device_claim_sessions (
            user_id, device_id, claim_token, expires_at_ms,
            claimed_at_ms, claimed_user_id, is_active, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (user_id, device_id, claim_token, expires_at_ms, now_ms, user_id, now_ms, now_ms),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM device_claim_sessions
        WHERE claim_token = ?
        """,
        (claim_token,),
    ).fetchone()
    return dict(row) if row else {}


def _get_active_claim_session(conn: Connection, device_id: str) -> Dict:
    now_ms = int(time.time() * 1000)
    row = conn.execute(
        """
        SELECT * FROM device_claim_sessions
        WHERE device_id = ? AND is_active = 1 AND expires_at_ms > ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (device_id, now_ms),
    ).fetchone()
    return dict(row) if row else {}


def _get_claim_session_by_token(conn: Connection, claim_token: str) -> Dict:
    now_ms = int(time.time() * 1000)
    row = conn.execute(
        """
        SELECT * FROM device_claim_sessions
        WHERE claim_token = ? AND is_active = 1 AND expires_at_ms > ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (claim_token, now_ms),
    ).fetchone()
    return dict(row) if row else {}


def _upsert_owner_profile(conn: Connection, user_id: int, payload: OwnerEnrollmentRequest) -> Dict:
    now_ms = int(time.time() * 1000)
    existing = conn.execute(
        """
        SELECT * FROM device_owner_profiles
        WHERE user_id = ? AND device_id = ?
        """,
        (user_id, payload.device_id),
    ).fetchone()
    params = (
        payload.owner_label,
        payload.embedding_version,
        int(payload.enrolled_at_ms),
        now_ms,
        1,
        int(payload.sample_count),
        float(payload.similarity_threshold),
        payload.embedding_backend,
        now_ms,
        user_id,
        payload.device_id,
    )
    if existing:
        conn.execute(
            """
            UPDATE device_owner_profiles
            SET owner_label = ?, embedding_version = ?, enrolled_at_ms = ?,
                last_sync_ms = ?, recognition_enabled = ?, sample_count = ?,
                similarity_threshold = ?, embedding_backend = ?, updated_at = ?
            WHERE user_id = ? AND device_id = ?
            """,
            params,
        )
    else:
        conn.execute(
            """
            INSERT INTO device_owner_profiles (
                owner_label, embedding_version, enrolled_at_ms, last_sync_ms,
                recognition_enabled, sample_count, similarity_threshold, embedding_backend,
                created_at, user_id, device_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.owner_label,
                payload.embedding_version,
                int(payload.enrolled_at_ms),
                now_ms,
                1,
                int(payload.sample_count),
                float(payload.similarity_threshold),
                payload.embedding_backend,
                now_ms,
                user_id,
                payload.device_id,
                now_ms,
            ),
        )
    conn.execute(
        """
        UPDATE devices
        SET identity_state = ?, identity_version = ?, owner_last_seen_ms = ?, updated_at = ?
        WHERE user_id = ? AND device_id = ?
        """,
        ("ready", payload.embedding_version, now_ms, now_ms, user_id, payload.device_id),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM device_owner_profiles
        WHERE user_id = ? AND device_id = ?
        """,
        (user_id, payload.device_id),
    ).fetchone()
    return dict(row) if row else {}


def _get_owner_profile(conn: Connection, user_id: int, device_id: str) -> Dict:
    row = conn.execute(
        """
        SELECT * FROM device_owner_profiles
        WHERE user_id = ? AND device_id = ?
        """,
        (user_id, device_id),
    ).fetchone()
    return dict(row) if row else {}


def _fetch_device_status(device_ip: str, timeout_sec: float = 2.0) -> Dict:
    url = f"http://{device_ip}/status"
    response = httpx.get(url, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    raise ValueError("Invalid status payload")


def _post_device_json(device_ip: str, path: str, payload: Dict[str, object], timeout_sec: float = 4.0) -> Dict:
    url = f"http://{device_ip}{path}"
    response = httpx.post(url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data
    raise ValueError("Invalid device payload")


def _emotion_type_from_tags(tags: List[str], s_value: float) -> str:
    tags_lower = [str(tag).lower() for tag in tags]
    if any(tag in tags_lower for tag in ("anger", "angry")):
        return "ANGRY"
    if any(tag in tags_lower for tag in ("fatigue", "tired")):
        return "TIRED"
    if any(tag in tags_lower for tag in ("lonely", "sad")):
        return "SAD"
    if s_value >= 0.7:
        return "ANXIOUS"
    return "CALM"


def _event_to_emotion(event: EngineEventRequest) -> Optional[EmotionEventRequest]:
    # Runtime transport events should be broadcast only and not persisted
    # into emotion history.
    if event.type in {
        "FaceTrackUpdate",
        "FaceTrackState",
        "MediaState",
        "RiskUpdate",
        "TriggerCandidate",
        "WakeState",
        "WakeWordDetected",
        "WakeAudioState",
        "WakeDiag",
        "VoiceChatUser",
        "VoiceChatBot",
        "ChatMessage",
    }:
        return None

    payload = event.payload or {}
    reason = {}
    tags: List[str] = []
    if isinstance(payload, dict):
        reason = payload.get("reason") or {}
        if not reason and isinstance(payload.get("care_plan"), dict):
            reason = payload.get("care_plan", {}).get("reason", {}) or {}
        if isinstance(reason, dict):
            tags = reason.get("tags", []) if isinstance(reason.get("tags"), list) else []

    try:
        v = float(reason.get("V", 0.0))
        a = float(reason.get("A", 0.0))
        t = float(reason.get("T", 0.0)) if reason.get("T") is not None else 0.0
        s = float(reason.get("S", max(v, a, t)))
    except Exception:
        v = a = t = 0.0
        s = 0.0

    description = ""
    if isinstance(payload, dict):
        care_plan = payload.get("care_plan")
        if isinstance(care_plan, dict):
            description = str(care_plan.get("text", "")).strip()
        if not description:
            description = str(payload.get("summary", "") or payload.get("transcript", "")).strip()
    if not description:
        description = f"event:{event.type}"

    return EmotionEventRequest(
        timestamp_ms=event.timestamp_ms,
        type=_emotion_type_from_tags(tags, s),
        description=description,
        V=v,
        A=a,
        T=t,
        S=s,
        intensity=int(min(100, max(0, s * 100))),
        source="engine",
    )


def _insert_refresh_token(conn: Connection, user_id: int, refresh_token: str, expires_at: int) -> None:
    token_hash = auth.hash_token(refresh_token)
    conn.execute(
        """
        INSERT INTO refresh_tokens (user_id, token_hash, expires_at, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, token_hash, expires_at, int(time.time())),
    )
    conn.commit()


def _revoke_refresh_token(conn: Connection, refresh_token: str) -> None:
    token_hash = auth.hash_token(refresh_token)
    conn.execute(
        "UPDATE refresh_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
        (int(time.time()), token_hash),
    )
    conn.commit()


def _refresh_token_valid(conn: Connection, refresh_token: str) -> bool:
    token_hash = auth.hash_token(refresh_token)
    cur = conn.execute(
        """
        SELECT id FROM refresh_tokens
        WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?
        """,
        (token_hash, int(time.time())),
    )
    return cur.fetchone() is not None


def _parse_access_token(
    credentials: HTTPAuthorizationCredentials,
    conn: Connection,
) -> Dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    payload = auth.decode_token(credentials.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id = int(payload.get("sub", 0))
    user = _get_user_by_id(conn, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def _is_loopback_request(request: Optional[Request]) -> bool:
    if request is None or request.client is None:
        return False
    host = str(request.client.host or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _ensure_shadow_user(
    conn: Connection,
    user_id: int,
    username: str,
    *,
    is_configured: bool = False,
) -> Dict[str, Any]:
    now = int(time.time())
    existing = _get_user_by_id(conn, user_id)
    if existing:
        if username and str(existing["username"]) != username:
            conn.execute("UPDATE users SET username = ?, updated_at = ? WHERE id = ?", (username, now, user_id))
            conn.commit()
            refreshed = _get_user_by_id(conn, user_id)
            if refreshed:
                return refreshed
        return existing
    shadow_username = username or f"desktop-shadow-{user_id}"
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, created_at, updated_at, is_configured)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            shadow_username,
            auth.hash_password(f"shadow-user-{user_id}"),
            now,
            now,
            1 if is_configured else 0,
        ),
    )
    conn.commit()
    created = _get_user_by_id(conn, user_id)
    if not created:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Shadow user bootstrap failed")
    return created


def _parse_access_token_for_local_desktop(
    credentials: HTTPAuthorizationCredentials,
    conn: Connection,
    request: Optional[Request] = None,
) -> Dict:
    if ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS and _is_loopback_request(request):
        if credentials is None or str(getattr(credentials, "scheme", "")).lower() != "bearer":
            return _ensure_shadow_user(conn, 1, "desktop-local")
    try:
        return _parse_access_token(credentials, conn)
    except HTTPException:
        if not ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS or not _is_loopback_request(request):
            raise
        if credentials is None or credentials.scheme.lower() != "bearer":
            return _ensure_shadow_user(conn, 1, "desktop-local")
        payload = auth.decode_token_unverified(credentials.credentials)
        if not payload or payload.get("type") != "access":
            return _ensure_shadow_user(conn, 1, "desktop-local")
        try:
            user_id = int(payload.get("sub", 0))
        except Exception as exc:
            return _ensure_shadow_user(conn, 1, "desktop-local")
        if user_id <= 0:
            return _ensure_shadow_user(conn, 1, "desktop-local")
        username = str(payload.get("username") or f"desktop-shadow-{user_id}").strip()
        return _ensure_shadow_user(conn, user_id, username)


def _parse_access_token_for_local_activation(
    credentials: HTTPAuthorizationCredentials,
    conn: Connection,
    request: Optional[Request] = None,
) -> Dict:
    try:
        return _parse_access_token(credentials, conn)
    except HTTPException:
        if not ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS or not _is_loopback_request(request):
            raise
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise
        payload = auth.decode_token_unverified(credentials.credentials)
        if not payload or payload.get("type") != "access":
            raise
        try:
            user_id = int(payload.get("sub", 0))
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject") from exc
        if user_id <= 0:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
        username = str(payload.get("username") or f"desktop-shadow-{user_id}").strip()
        return _ensure_shadow_user(conn, user_id, username, is_configured=False)


def _bridge_user_from_request(request: Request, conn: Connection) -> Dict:
    token = str(request.headers.get("x-assistant-bridge-token") or "").strip()
    if not token or token != ASSISTANT_BRIDGE_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bridge token")
    user = _get_user_by_id(conn, int(ASSISTANT_BRIDGE_USER_ID))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bridge user not found")
    return user


@app.post("/auth/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest, conn: Connection = Depends(get_db)) -> UserResponse:
    existing = _get_user_by_username(conn, payload.username)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")
    password_hash = auth.hash_password(payload.password)
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (payload.username, password_hash, now),
    )
    conn.commit()
    user_id = cur.lastrowid
    return UserResponse(id=user_id, username=payload.username, created_at=now)


@app.post("/api/auth/register", response_model=UserResponse, status_code=201)
def register_api(payload: RegisterRequest, conn: Connection = Depends(get_db)) -> UserResponse:
    return register(payload, conn)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, conn: Connection = Depends(get_db)) -> TokenResponse:
    username = str(payload.username or "").strip()
    if not username:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Username is required")
    user = _get_user_by_username(conn, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access = auth.create_access_token(user["id"], user["username"])
    refresh = auth.create_refresh_token(user["id"], user["username"])
    expires_at = int(time.time()) + refresh["expires_in"]
    _insert_refresh_token(conn, user["id"], refresh["token"], expires_at)
    return TokenResponse(
        access_token=access["token"],
        refresh_token=refresh["token"],
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_SEC,
    )


@app.post("/api/auth/login", response_model=LoginResponse)
def login_api(payload: LoginEmailRequest, conn: Connection = Depends(get_db)) -> LoginResponse:
    email = str(payload.email or "").strip()
    if not email:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Email is required")
    user = _get_user_by_username(conn, email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access = auth.create_access_token(user["id"], user["username"])
    refresh = auth.create_refresh_token(user["id"], user["username"])
    expires_at = int(time.time()) + refresh["expires_in"]
    _insert_refresh_token(conn, user["id"], refresh["token"], expires_at)
    psychometric_completed = bool(_get_psychometric_profile(conn, int(user["id"])))
    owner_binding = _get_owner_binding_state(
        conn,
        int(user["id"]),
        bool(user.get("is_configured", 0)),
        psychometric_completed,
    )

    return LoginResponse(
        token=access["token"],
        refresh_token=refresh["token"],
        user_id=int(user["id"]),
        is_configured=bool(user.get("is_configured", 0)),
        activation_required=not bool(user.get("is_configured", 0)),
        assessment_required=bool(user.get("is_configured", 0)) and not psychometric_completed,
        owner_binding_required=bool(owner_binding["required"]),
        owner_binding_completed=bool(owner_binding["completed"]),
        preferred_device_id=owner_binding["device_id"],
        activation_path="/activate",
    )


@app.get("/activate", response_class=HTMLResponse)
def activation_page() -> HTMLResponse:
    return HTMLResponse(_activation_page_html())


@app.get("/api/auth/me", response_model=UserResponse)
def me_api(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> UserResponse:
    user = _parse_access_token(credentials, conn)
    return UserResponse(id=user["id"], username=user["username"], created_at=user["created_at"])


@app.get("/api/activation/state", response_model=ActivationProfileResponse)
def activation_state(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationProfileResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    profile = _get_activation_profile(conn, int(user["id"]))
    return _activation_response(conn, user, profile)


@app.get("/api/activation/runtime/status", response_model=ActivationRuntimeStatusResponse)
def activation_runtime_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationRuntimeStatusResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    snapshot = _activation_ai_runtime_snapshot()
    device_online, selected = _assessment_device_online(conn, int(user["id"]))
    preferred_device_id = str((selected or {}).get("device_id") or "").strip()
    return ActivationRuntimeStatusResponse(
        ok=True,
        ai_ready=bool(snapshot["ai_ready"]),
        ai_detail=str(snapshot["ai_detail"] or ""),
        gateway_ready=bool(snapshot["gateway_ready"]),
        provider_network_ok=bool(snapshot["provider_network_ok"]),
        blocking_reason=str(snapshot["blocking_reason"] or ""),
        text_assessment_ready=bool(snapshot["ai_ready"]),
        desktop_voice_ready=False,
        desktop_voice_detail="桌面麦克风转写由本地桌面后端提供，需本机运行时可用。",
        device_online=device_online,
        robot_voice_ready=bool(device_online and preferred_device_id),
        preferred_device_id=preferred_device_id,
    )


async def _assessment_bootstrap_first_question(
    conn: Connection,
    user_id: int,
    *,
    voice_mode: str = "text",
) -> None:
    session_id, existing = _load_assessment_session(conn, int(user_id), active_only=True)
    if not existing or session_id is None:
        latest_session_id, latest_payload = _load_assessment_session(conn, int(user_id), active_only=False)
        latest_status = str(latest_payload.get("status") or "").strip()
        if latest_payload and latest_session_id is not None and latest_status != "completed":
            session_id, existing = latest_session_id, latest_payload
    if existing and session_id is not None and str(existing.get("latest_question") or "").strip():
        return
    ai_snapshot = _activation_ai_runtime_snapshot()
    if not bool(ai_snapshot["ai_ready"]):
        blocked_payload = dict(existing or build_initial_session(int(time.time() * 1000)))
        blocked_payload["status"] = "blocked"
        blocked_payload["blocking_reason"] = str(ai_snapshot["blocking_reason"] or "AI 未就绪，正式建档已暂停。")
        blocked_payload["assessment_ready"] = False
        _save_assessment_session(conn, int(user_id), blocked_payload, session_id=int(session_id) if session_id is not None else None)
        return
    activation = _get_activation_profile(conn, int(user_id))
    if existing and session_id is not None:
        session_payload = dict(existing)
        session_payload["status"] = "active"
        session_payload["blocking_reason"] = ""
        session_payload["voice_mode"] = str(voice_mode or session_payload.get("voice_mode") or "text").strip() or "text"
    else:
        now_ms = int(time.time() * 1000)
        session_payload = build_initial_session(now_ms)
        session_payload["voice_mode"] = str(voice_mode or "text").strip() or "text"
        session_id = None
    model_question = await _assessment_pick_model_question(int(user_id), activation, session_payload)
    if not model_question:
        session_payload["status"] = "blocked"
        session_payload["blocking_reason"] = "AI 没有生成首个正式建档问题。"
    else:
        session_payload["last_question_id"] = str(model_question.get("id") or "")
        session_payload["latest_question"] = str(model_question.get("prompt") or "")
        session_payload["question_pair"] = str(model_question.get("pair") or "")
        session_payload["current_focus"] = str(model_question.get("focus") or model_question.get("pair") or "")
        session_payload["question_source"] = "ai"
    _save_assessment_session(conn, int(user_id), session_payload, session_id=int(session_id) if session_id is not None else None)


@app.post("/api/activation/complete", response_model=ActivationProfileResponse)
async def activation_complete(
    payload: ActivationCompleteRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationProfileResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    preferred_name = str(payload.preferred_name or "").strip()
    if not preferred_name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="preferred_name is required")
    now_ms = int(time.time() * 1000)
    summary = str(payload.identity_summary or "").strip()
    if not summary:
        summary = f"{preferred_name} 是当前机器人的主要使用者，后续应按已确认身份持续服务。"
    profile_payload = {
        "preferred_name": preferred_name,
        "role_label": str(payload.role_label or "owner").strip() or "owner",
        "relation_to_robot": str(payload.relation_to_robot or "primary_user").strip() or "primary_user",
        "pronouns": str(payload.pronouns or "").strip(),
        "identity_summary": summary,
        "onboarding_notes": str(payload.onboarding_notes or "").strip(),
        "voice_intro_summary": str(payload.voice_intro_summary or "").strip(),
        "activation_version": str(payload.activation_version or "v1").strip() or "v1",
        "completed_at_ms": now_ms,
        "profile": payload.profile or {},
    }
    profile = _upsert_activation_profile(conn, int(user["id"]), profile_payload)
    _set_user_configured(conn, int(user["id"]), True)
    assistant_service.store.append_memory(
        int(user["id"]),
        title="activation_profile",
        content=(
            f"首次激活完成。称呼：{preferred_name}；角色：{profile_payload['role_label']}；"
            f"关系：{profile_payload['relation_to_robot']}；摘要：{summary}"
        ),
        tags=["activation", "identity", "profile"],
    )
    updated_user = _get_user_by_id(conn, int(user["id"]))
    return _activation_response(conn, updated_user, profile)


@app.post("/api/activation/identity/infer", response_model=ActivationIdentityInferResponse)
async def activation_identity_infer(
    payload: ActivationIdentityInferRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationIdentityInferResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    transcript = str(payload.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="transcript is required")
    ai_snapshot = _activation_ai_runtime_snapshot()
    if not bool(ai_snapshot["ai_ready"]):
        return ActivationIdentityInferResponse(
            ok=False,
            preferred_name="",
            role_label="owner",
            relation_to_robot="primary_user",
            pronouns="",
            identity_summary="",
            onboarding_notes="AI 未就绪，身份草稿暂停生成。你可以先手动填写基础身份信息。",
            voice_intro_summary="",
            confidence=0.0,
            inference_source="blocked",
            inference_detail=str(ai_snapshot["blocking_reason"] or "AI 未就绪，身份草稿暂停。"),
            raw_json={"blocking_reason": str(ai_snapshot["blocking_reason"] or "")},
        )
    inference_input = {
        "transcript": transcript,
        "surface": str(payload.surface or "robot").strip() or "robot",
        "observed_name": str(payload.observed_name or "").strip(),
        "context": payload.context or {},
    }
    session_key = f"activation:{int(user['id'])}:infer:{int(time.time() * 1000)}"
    prompt = (
        f"{ACTIVATION_SYSTEM_PROMPT}\n\n"
        f"{IDENTITY_EXTRACTION_PROMPT}\n\n"
        f"输入数据：{json.dumps(inference_input, ensure_ascii=False)}"
    )
    raw = ""
    inference_detail = ""
    raw = await assistant_service.gateway.send_message(session_key, prompt)
    parsed = _extract_json_block(raw)
    if not parsed or not any(
        str(parsed.get(key) or "").strip()
        for key in ("preferred_name", "role_label", "relation_to_robot", "identity_summary")
    ):
        return ActivationIdentityInferResponse(
            ok=False,
            preferred_name="",
            role_label="owner",
            relation_to_robot="primary_user",
            pronouns="",
            identity_summary="",
            onboarding_notes="AI 返回为空，身份草稿暂停生成。你可以先手动填写基础身份信息。",
            voice_intro_summary="",
            confidence=0.0,
            inference_source="blocked",
            inference_detail="AI 身份草稿返回为空，请稍后重试。",
            raw_json={"raw_text": raw},
        )
    preferred_name = str(parsed.get("preferred_name") or "").strip()
    if not preferred_name and inference_input["observed_name"]:
        preferred_name = str(inference_input["observed_name"]).strip()
    role_label = str(parsed.get("role_label") or "unknown").strip() or "unknown"
    relation_to_robot = str(parsed.get("relation_to_robot") or "unknown").strip() or "unknown"
    pronouns = str(parsed.get("pronouns") or "").strip()
    identity_summary = str(parsed.get("identity_summary") or "").strip()
    onboarding_notes = str(parsed.get("onboarding_notes") or "").strip()
    voice_intro_summary = str(parsed.get("voice_intro_summary") or "").strip()
    confidence = float(parsed.get("confidence") or 0.0)
    if not onboarding_notes and not confidence:
        onboarding_notes = "待确认：请在首次激活页再次确认身份信息。"
    return ActivationIdentityInferResponse(
        ok=True,
        preferred_name=preferred_name,
        role_label=role_label,
        relation_to_robot=relation_to_robot,
        pronouns=pronouns,
        identity_summary=identity_summary,
        onboarding_notes=onboarding_notes,
        voice_intro_summary=voice_intro_summary,
        confidence=max(0.0, min(1.0, confidence)),
        inference_source="ai",
        inference_detail=inference_detail,
        raw_json=parsed or {"raw_text": raw},
    )


@app.get("/api/activation/prompt-pack", response_model=ActivationPromptPackResponse)
def activation_prompt_pack(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationPromptPackResponse:
    _parse_access_token_for_local_activation(credentials, conn, request)
    return ActivationPromptPackResponse(
        ok=True,
        system_prompt=ACTIVATION_SYSTEM_PROMPT,
        extraction_prompt=IDENTITY_EXTRACTION_PROMPT,
        preferred_mode=OPENCLAW_PREFERRED_MODE,
        preferred_code_model=OPENCLAW_PREFERRED_CODE_MODEL,
    )


@app.post("/api/activation/assessment/start", response_model=ActivationAssessmentStateResponse)
async def activation_assessment_start(
    payload: ActivationAssessmentStartRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationAssessmentStateResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    if not bool(user.get("is_configured", 0)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Identity activation must complete first")
    ai_snapshot = _activation_ai_runtime_snapshot()
    session_id, existing = _load_assessment_session(conn, int(user["id"]), active_only=True)
    if (not existing or session_id is None) and not payload.reset:
        latest_session_id, latest_payload = _load_assessment_session(conn, int(user["id"]), active_only=False)
        latest_status = str(latest_payload.get("status") or "").strip()
        if latest_payload and latest_session_id is not None and latest_status != "completed":
            session_id, existing = latest_session_id, latest_payload
    if not bool(ai_snapshot["ai_ready"]):
        blocked_payload = dict(existing or build_initial_session(int(time.time() * 1000)))
        blocked_payload["status"] = "blocked"
        blocked_payload["blocking_reason"] = str(ai_snapshot["blocking_reason"] or "AI 未就绪，正式建档已暂停。")
        blocked_payload["assessment_ready"] = False
        if existing and session_id is not None:
            _save_assessment_session(conn, int(user["id"]), blocked_payload, session_id=int(session_id))
        device_online, _selected = _assessment_device_online(conn, int(user["id"]), payload.device_id)
        return _assessment_response(blocked_payload, device_online=device_online, exists=bool(existing))
    if payload.reset or not existing:
        now_ms = int(time.time() * 1000)
        session_payload = build_initial_session(now_ms)
        session_payload["voice_mode"] = str(payload.voice_mode or "text").strip() or "text"
        activation = _get_activation_profile(conn, int(user["id"]))
        model_question, question_source = await _assessment_pick_question_with_fallback(int(user["id"]), activation, session_payload)
        if not model_question:
            session_payload["status"] = "blocked"
            session_payload["blocking_reason"] = "AI 没有生成首个正式建档问题。"
        else:
            session_payload["last_question_id"] = str(model_question.get("id") or "")
            session_payload["latest_question"] = str(model_question.get("prompt") or "")
            session_payload["question_pair"] = str(model_question.get("pair") or "")
            session_payload["current_focus"] = str(model_question.get("focus") or model_question.get("pair") or "")
            session_payload["question_source"] = question_source
        session_id = _save_assessment_session(conn, int(user["id"]), session_payload, session_id=None)
        existing = session_payload
    elif not str(existing.get("latest_question") or "").strip() and str(existing.get("status") or "") != "completed":
        activation = _get_activation_profile(conn, int(user["id"]))
        model_question, question_source = await _assessment_pick_question_with_fallback(int(user["id"]), activation, existing)
        if model_question:
            existing["status"] = "active"
            existing["blocking_reason"] = ""
            existing["last_question_id"] = str(model_question.get("id") or "")
            existing["latest_question"] = str(model_question.get("prompt") or "")
            existing["question_pair"] = str(model_question.get("pair") or "")
            existing["current_focus"] = str(model_question.get("focus") or model_question.get("pair") or "")
            existing["question_source"] = question_source
            _save_assessment_session(conn, int(user["id"]), existing, session_id=int(session_id))
    elif existing and session_id is not None and str(existing.get("status") or "").strip() == "blocked" and bool(ai_snapshot["ai_ready"]):
        existing["status"] = "active"
        existing["blocking_reason"] = ""
        existing["voice_mode"] = str(payload.voice_mode or existing.get("voice_mode") or "text").strip() or "text"
        _save_assessment_session(conn, int(user["id"]), existing, session_id=int(session_id))
    device_online, _selected = _assessment_device_online(conn, int(user["id"]), payload.device_id)
    return _assessment_response(existing, device_online=device_online, exists=True)


@app.get("/api/activation/assessment/state", response_model=ActivationAssessmentStateResponse)
def activation_assessment_state(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationAssessmentStateResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    _session_id, session_payload = _load_assessment_session(conn, int(user["id"]), active_only=False)
    device_online, _selected = _assessment_device_online(conn, int(user["id"]))
    if not session_payload:
        completed = _get_psychometric_profile(conn, int(user["id"]))
        if completed:
            session_payload = {
                "status": "completed",
                "completed_at_ms": completed.get("completed_at_ms"),
                "updated_at_ms": completed.get("updated_at_ms"),
                "effective_turn_count": completed.get("conversation_count"),
                "turn_count": completed.get("conversation_count"),
                "voice_mode": "idle",
                "voice_session_active": False,
                "dialogue_turns": completed.get("dialogue_turns") or [],
                "final_result": completed,
            }
            return _assessment_response(session_payload, device_online=device_online, exists=True)
        return _assessment_response({}, device_online=device_online, exists=False)
    return _assessment_response(session_payload, device_online=device_online, exists=True)


@app.post("/api/activation/assessment/turn", response_model=ActivationAssessmentTurnResponse)
async def activation_assessment_turn(
    payload: ActivationAssessmentTurnRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationAssessmentTurnResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    client_turn_id = str(payload.client_turn_id or "").strip()
    session_id, session_payload = _load_assessment_session(conn, int(user["id"]), active_only=True)
    if not session_payload or session_id is None:
        latest_session_id, latest_payload = _load_assessment_session(conn, int(user["id"]), active_only=False)
        latest_status = str(latest_payload.get("status") or "").strip()
        if latest_payload and latest_status == "completed":
            cached = latest_payload.get("last_turn_response")
            if client_turn_id and str(latest_payload.get("last_client_turn_id") or "").strip() == client_turn_id and isinstance(cached, dict):
                return ActivationAssessmentTurnResponse(**cached)
            response = _assessment_response(latest_payload, device_online=_assessment_device_online(conn, int(user["id"]), payload.device_id)[0], exists=True)
            return ActivationAssessmentTurnResponse(**response.model_dump(), question_changed=False, just_completed=False)
        if latest_payload and latest_session_id is not None and latest_status != "completed":
            ai_snapshot = _activation_ai_runtime_snapshot()
            if not bool(ai_snapshot["ai_ready"]):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=str(ai_snapshot["blocking_reason"] or "Assessment session not started"),
                )
            latest_payload["status"] = "active"
            latest_payload["blocking_reason"] = ""
            latest_payload["voice_mode"] = (
                str(payload.voice_mode or latest_payload.get("voice_mode") or "text").strip() or "text"
            )
            if not str(latest_payload.get("latest_question") or "").strip():
                activation = _get_activation_profile(conn, int(user["id"]))
                model_question = await _assessment_pick_model_question(int(user["id"]), activation, latest_payload)
                if not model_question:
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assessment session not started")
                latest_payload["last_question_id"] = str(model_question.get("id") or "")
                latest_payload["latest_question"] = str(model_question.get("prompt") or "")
                latest_payload["question_pair"] = str(model_question.get("pair") or "")
                latest_payload["current_focus"] = str(model_question.get("focus") or model_question.get("pair") or "")
                latest_payload["question_source"] = "ai"
            _save_assessment_session(conn, int(user["id"]), latest_payload, session_id=int(latest_session_id))
            session_id, session_payload = latest_session_id, latest_payload
    if not session_payload or session_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assessment session not started")
    if str(session_payload.get("status") or "") == "completed":
        response = _assessment_response(session_payload, device_online=_assessment_device_online(conn, int(user["id"]), payload.device_id)[0], exists=True)
        return ActivationAssessmentTurnResponse(**response.model_dump(), question_changed=False, just_completed=False)
    if client_turn_id and str(session_payload.get("last_client_turn_id") or "").strip() == client_turn_id:
        cached = session_payload.get("last_turn_response")
        if isinstance(cached, dict):
            return ActivationAssessmentTurnResponse(**cached)
        response = _assessment_response(session_payload, device_online=_assessment_device_online(conn, int(user["id"]), payload.device_id)[0], exists=True)
        return ActivationAssessmentTurnResponse(
            **response.model_dump(),
            question_changed=bool(response.latest_question),
            just_completed=bool(str(session_payload.get("status") or "") == "completed"),
        )
    answer_text = str(payload.answer or payload.transcript or "").strip()
    if not answer_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="answer is required")
    _merged, response = await _assessment_apply_turn(
        conn,
        int(user["id"]),
        int(session_id),
        session_payload,
        answer_text=answer_text,
        transcript_text=str(payload.transcript or answer_text),
        device_id=payload.device_id,
        voice_mode=payload.voice_mode,
        surface=payload.surface,
        client_turn_id=client_turn_id,
    )
    return response


@app.post("/api/activation/assessment/finish", response_model=ActivationAssessmentFinishResponse)
async def activation_assessment_finish(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationAssessmentFinishResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    session_id, session_payload = _load_assessment_session(conn, int(user["id"]), active_only=True)
    if not session_payload or session_id is None:
        profile = _get_psychometric_profile(conn, int(user["id"]))
        if profile:
            return ActivationAssessmentFinishResponse(
                **_assessment_response(
                    {"status": "completed", "final_result": profile, "completed_at_ms": profile.get("completed_at_ms")},
                    device_online=_assessment_device_online(conn, int(user["id"]))[0],
                    exists=True,
                ).model_dump(),
                persisted=True,
            )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assessment session not started")
    if not bool(_activation_ai_runtime_snapshot()["ai_ready"]):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="AI not ready; formal assessment cannot finish")
    final_profile = session_payload.get("final_result") if isinstance(session_payload.get("final_result"), dict) else build_final_profile(session_payload)
    if not bool(final_profile.get("assessment_ready")):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assessment signal is not stable enough to finish")
    session_payload["status"] = "completed"
    session_payload["completed_at_ms"] = int(time.time() * 1000)
    session_payload["finish_reason"] = str(final_profile.get("completion_reason") or session_payload.get("finish_reason") or "formal_finish")
    session_payload["final_result"] = await _persist_assessment_completion(conn, int(user["id"]), session_payload)
    _save_assessment_session(conn, int(user["id"]), session_payload, session_id=int(session_id))
    device_online, _selected = _assessment_device_online(conn, int(user["id"]))
    return ActivationAssessmentFinishResponse(
        **_assessment_response(session_payload, device_online=device_online, exists=True).model_dump(),
        persisted=True,
    )


@app.post("/api/activation/assessment/voice/start")
def activation_assessment_voice_start(
    payload: ActivationAssessmentVoiceRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, object]:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    device_online, selected = _assessment_device_online(conn, int(user["id"]), payload.device_id)
    if not selected or not device_online:
        return {"ok": True, "device_online": False, "state": "offline", "detail": "Device offline; voice mode unavailable"}
    resolved_ip = str(selected.get("device_ip") or "").strip()
    try:
        response = _post_device_json(resolved_ip, "/voice/session/start", {"mode": str(payload.session_mode or "assessment")})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    session_id, session_payload = _load_assessment_session(conn, int(user["id"]), active_only=True)
    prompt_spoken = False
    if session_payload and session_id is not None:
        session_payload["voice_mode"] = "robot"
        session_payload["voice_session_active"] = True
        session_payload, prompt_spoken, _detail = _assessment_sync_voice_prompt(
            session_payload,
            device_ip=resolved_ip,
            speak_question=True,
        )
        _save_assessment_session(conn, int(user["id"]), session_payload, session_id=int(session_id))
    return {
        "ok": True,
        "device_online": True,
        "prompt_spoken": prompt_spoken,
        "state": response,
        "assessment": _assessment_response(session_payload or {}, device_online=True, exists=bool(session_payload)),
    }


@app.post("/api/activation/assessment/voice/stop")
def activation_assessment_voice_stop(
    payload: ActivationAssessmentVoiceRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, object]:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    device_online, selected = _assessment_device_online(conn, int(user["id"]), payload.device_id)
    if not selected or not device_online:
        return {"ok": True, "device_online": False, "state": "offline"}
    resolved_ip = str(selected.get("device_ip") or "").strip()
    try:
        response = _post_device_json(resolved_ip, "/voice/session/stop", {"mode": str(payload.session_mode or "assessment")})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    session_id, session_payload = _load_assessment_session(conn, int(user["id"]), active_only=True)
    if session_payload and session_id is not None:
        session_payload["voice_session_active"] = False
        session_payload["voice_mode"] = "text"
        _save_assessment_session(conn, int(user["id"]), session_payload, session_id=int(session_id))
    return {"ok": True, "device_online": True, "state": response}


@app.post("/api/activation/assessment/voice/poll", response_model=ActivationAssessmentVoicePollResponse)
async def activation_assessment_voice_poll(
    payload: ActivationAssessmentVoicePollRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationAssessmentVoicePollResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    user_id = int(user["id"])
    session_id, session_payload = _load_assessment_session(conn, user_id, active_only=True)
    device_online, selected = _assessment_device_online(conn, user_id, payload.device_id)
    base_state = _assessment_response(session_payload or {}, device_online=device_online, exists=bool(session_payload))
    if not session_payload or session_id is None:
        return ActivationAssessmentVoicePollResponse(
            ok=True,
            device_online=device_online,
            detail="assessment_not_started",
            state=base_state,
        )
    if not selected or not device_online:
        return ActivationAssessmentVoicePollResponse(
            ok=True,
            device_online=False,
            detail="device_offline",
            state=base_state,
        )

    resolved_ip = str(selected.get("device_ip") or "").strip()
    prompt_spoken = False
    detail = "waiting_for_transcript"
    transcript_text = ""
    try:
        session_payload, prompt_spoken, prompt_detail = _assessment_sync_voice_prompt(
            session_payload,
            device_ip=resolved_ip,
            speak_question=bool(payload.speak_question),
        )
        if prompt_detail:
            detail = prompt_detail
        voice_state = _get_device_json(resolved_ip, "/voice/status", timeout_sec=4.0)
        transcribe = _post_device_json(
            resolved_ip,
            "/voice/transcribe_recent",
            {"window_ms": max(2500, int(payload.window_ms or 5000))},
            timeout_sec=12.0,
        )
        transcript_text = str(transcribe.get("transcript") or "").strip()
        session_payload["voice_runtime_state"] = voice_state
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if transcript_text and not _assessment_should_ignore_transcript(session_payload, transcript_text):
        session_payload["voice_last_consumed_transcript"] = transcript_text
        merged, updated = await _assessment_apply_turn(
            conn,
            user_id,
            int(session_id),
            session_payload,
            answer_text=transcript_text,
            transcript_text=transcript_text,
            device_id=payload.device_id,
            voice_mode="robot",
            surface="desktop",
        )
        session_payload = merged
        base_state = updated
        detail = "transcript_processed"
        if session_payload.get("status") == "completed":
            try:
                _assessment_speak_text(resolved_ip, "测评完成啦，我已经记住你的互动偏好了。")
                _post_device_json(resolved_ip, "/voice/session/stop", {"mode": "assessment"})
            except Exception:
                pass
            session_payload["voice_session_active"] = False
            session_payload["voice_mode"] = "text"
            _save_assessment_session(conn, user_id, session_payload, session_id=int(session_id))
            base_state = _assessment_response(session_payload, device_online=True, exists=True)
        else:
            try:
                session_payload, spoke_next, _next_detail = _assessment_sync_voice_prompt(
                    session_payload,
                    device_ip=resolved_ip,
                    speak_question=bool(payload.speak_question),
                )
                prompt_spoken = prompt_spoken or spoke_next
                _save_assessment_session(conn, user_id, session_payload, session_id=int(session_id))
            except Exception:
                pass
            base_state = _assessment_response(session_payload, device_online=True, exists=True)
    else:
        _save_assessment_session(conn, user_id, session_payload, session_id=int(session_id))
        if transcript_text:
            detail = "transcript_ignored"
        base_state = _assessment_response(session_payload, device_online=True, exists=True)

    return ActivationAssessmentVoicePollResponse(
        ok=True,
        device_online=True,
        transcript=transcript_text,
        transcript_processed=detail == "transcript_processed",
        prompt_spoken=prompt_spoken,
        detail=detail,
        state=base_state,
    )


@app.get("/api/activation/personality/state", response_model=ActivationPersonalityStateResponse)
def activation_personality_state(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationPersonalityStateResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    return _personality_response(_get_personality_profile(conn, int(user["id"])))


@app.post("/api/activation/personality/infer", response_model=ActivationPersonalityInferResponse)
async def activation_personality_infer(
    payload: ActivationPersonalityInferRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationPersonalityInferResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    answers = [str(item).strip() for item in payload.answers or [] if str(item).strip()]
    transcript = str(payload.transcript or "").strip()
    merged_lines = []
    if transcript:
        merged_lines.append(transcript)
    merged_lines.extend(answers)
    if not merged_lines:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="answers or transcript is required")
    recent_rows = _list_chat_messages(conn, int(user["id"]), 12, session_key=build_session_key("desktop", int(user["id"])))
    recent_history = [
        {
            "sender": str(row["sender"] or ""),
            "text": str(row["text"] or ""),
            "timestamp_ms": int(row["timestamp_ms"] or 0),
        }
        for row in recent_rows[-8:]
        if str(row["text"] or "").strip()
    ]
    inference_input = {
        "identity": _get_activation_profile(conn, int(user["id"])),
        "surface": str(payload.surface or "desktop").strip() or "desktop",
        "answers": answers,
        "transcript": transcript,
        "context": payload.context or {},
        "recent_history": recent_history,
    }
    session_key = f"activation-{int(user['id'])}-personality-{int(time.time() * 1000)}"
    prompt = (
        f"{PERSONALITY_SYSTEM_PROMPT}\n\n"
        f"{PERSONALITY_EXTRACTION_PROMPT}\n\n"
        f"输入数据：{json.dumps(inference_input, ensure_ascii=False)}"
    )
    try:
        raw = await assistant_service.gateway.send_message(session_key, prompt)
        parsed = _extract_json_block(raw)
    except Exception:
        raw = ""
        parsed = _heuristic_personality_profile("\n".join(merged_lines))
    summary = str(parsed.get("summary") or "").strip()
    response_style = str(parsed.get("response_style") or "").strip()
    care_style = str(parsed.get("care_style") or "").strip()
    traits = [str(item).strip() for item in parsed.get("traits") or [] if str(item).strip()]
    topics = [str(item).strip() for item in parsed.get("topics") or [] if str(item).strip()]
    boundaries = [str(item).strip() for item in parsed.get("boundaries") or [] if str(item).strip()]
    signals = [str(item).strip() for item in parsed.get("signals") or [] if str(item).strip()]
    confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    sample_count = max(len(answers), 1 if transcript else 0, int(parsed.get("sample_count") or 0))
    return ActivationPersonalityInferResponse(
        ok=True,
        summary=summary,
        response_style=response_style,
        care_style=care_style,
        traits=traits,
        topics=topics,
        boundaries=boundaries,
        signals=signals,
        confidence=confidence,
        sample_count=sample_count,
        inference_version=str(parsed.get("inference_version") or "v1").strip() or "v1",
        raw_json=parsed or {"raw_text": raw},
    )


@app.post("/api/activation/personality/complete", response_model=ActivationPersonalityStateResponse)
def activation_personality_complete(
    payload: ActivationPersonalityCompleteRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ActivationPersonalityStateResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    summary = str(payload.summary or "").strip()
    if not summary:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="summary is required")
    profile = _upsert_personality_profile(
        conn,
        int(user["id"]),
        {
            "summary": summary,
            "response_style": str(payload.response_style or "").strip(),
            "care_style": str(payload.care_style or "").strip(),
            "traits": payload.traits or [],
            "topics": payload.topics or [],
            "boundaries": payload.boundaries or [],
            "signals": payload.signals or [],
            "confidence": max(0.0, min(1.0, float(payload.confidence or 0.0))),
            "sample_count": max(0, int(payload.sample_count or 0)),
            "inference_version": str(payload.inference_version or "v1").strip() or "v1",
            "profile": payload.profile or {},
        },
    )
    assistant_service.store.append_memory(
        int(user["id"]),
        title="personality_profile",
        content=(
            f"人格画像已确认。摘要：{summary}；回复风格：{str(payload.response_style or '').strip()}；"
            f"陪伴风格：{str(payload.care_style or '').strip()}；特征：{', '.join(payload.traits or [])}"
        ),
        tags=["activation", "personality", "profile"],
    )
    return _personality_response(profile)


@app.get("/api/user/profile", response_model=ProfileResponse)
def user_profile(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ProfileResponse:
    user = _parse_access_token(credentials, conn)
    profile = _profile_from_user(user)
    return ProfileResponse(**profile)


@app.post("/api/user/profile", response_model=ProfileResponse)
async def update_user_profile(
    payload: ProfileUpdateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ProfileResponse:
    user = _parse_access_token(credentials, conn)
    current = _profile_from_user(user)
    display_name = current["display_name"]
    avatar_url = current["avatar_url"]
    bio = current.get("bio")
    location = current.get("location")

    if payload.display_name is not None:
        candidate = str(payload.display_name).strip()
        display_name = candidate or current["username"] or "鍏遍福鏃呬汉"

    if payload.avatar_url is not None:
        candidate = str(payload.avatar_url).strip()
        avatar_url = candidate or None

    if payload.bio is not None:
        candidate = str(payload.bio).strip()
        bio = candidate or None

    if payload.location is not None:
        candidate = str(payload.location).strip()
        location = candidate or None

    updated_at = int(time.time())
    conn.execute(
        "UPDATE users SET display_name = ?, avatar_url = ?, bio = ?, location = ?, updated_at = ? WHERE id = ?",
        (display_name, avatar_url, bio, location, updated_at, int(user["id"])),
    )
    conn.commit()
    response = ProfileResponse(
        id=int(user["id"]),
        username=current["username"],
        display_name=display_name,
        avatar_url=avatar_url,
        bio=bio,
        location=location,
        created_at=current.get("created_at"),
        updated_at=updated_at,
    )
    await event_manager.broadcast(
        {
            "type": "UserProfileUpdated",
            "timestamp_ms": int(time.time() * 1000),
            "payload": response.model_dump(),
        }
    )
    return response


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, conn: Connection = Depends(get_db)) -> TokenResponse:
    data = auth.decode_token(payload.refresh_token)
    if not data or data.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    if not _refresh_token_valid(conn, payload.refresh_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked")

    user_id = int(data.get("sub", 0))
    user = _get_user_by_id(conn, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    _revoke_refresh_token(conn, payload.refresh_token)
    access = auth.create_access_token(user["id"], user["username"])
    refresh = auth.create_refresh_token(user["id"], user["username"])
    expires_at = int(time.time()) + refresh["expires_in"]
    _insert_refresh_token(conn, user["id"], refresh["token"], expires_at)

    return TokenResponse(
        access_token=access["token"],
        refresh_token=refresh["token"],
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_SEC,
    )


@app.post("/api/auth/refresh", response_model=TokenResponse)
def refresh_api(payload: RefreshRequest, conn: Connection = Depends(get_db)) -> TokenResponse:
    return refresh(payload, conn)


@app.post("/auth/logout")
def logout(payload: LogoutRequest, conn: Connection = Depends(get_db)) -> Dict[str, str]:
    _revoke_refresh_token(conn, payload.refresh_token)
    return {"status": "ok"}


@app.post("/api/auth/logout")
def logout_api(payload: LogoutRequest, conn: Connection = Depends(get_db)) -> Dict[str, str]:
    return logout(payload, conn)


@app.get("/auth/me", response_model=UserResponse)
def me(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> UserResponse:
    user = _parse_access_token(credentials, conn)
    return UserResponse(id=user["id"], username=user["username"], created_at=user["created_at"])


@app.get("/api/emotion/realtime", response_model=RealtimeScoresResponse)
def emotion_realtime() -> RealtimeScoresResponse:
    scores = getattr(app.state, "scores", {"V": 0.0, "A": 0.0, "T": 0.0, "S": 0.0})
    return RealtimeScoresResponse(**scores)


@app.get("/api/emotion/realtime/detail", response_model=RealtimeRiskDetailResponse)
def emotion_realtime_detail() -> RealtimeRiskDetailResponse:
    scores = getattr(app.state, "scores", {"V": 0.0, "A": 0.0, "T": 0.0, "S": 0.0})
    detail = getattr(app.state, "risk_detail", {})
    if not isinstance(detail, dict):
        detail = {}
    v_sub = detail.get("V_sub")
    if not isinstance(v_sub, dict):
        v_sub = {}
    a_sub = detail.get("A_sub")
    if not isinstance(a_sub, dict):
        a_sub = {}
    t_sub = detail.get("T_sub")
    if not isinstance(t_sub, dict):
        t_sub = {}

    # Keep a stable contract for UI/CLI even when upstream engine is older or not ready.
    v_sub.setdefault("face_ok", 0.0)
    v_sub.setdefault("expression_class_id", -1.0)
    v_sub.setdefault("expression_confidence", 0.0)
    v_sub.setdefault("expression_risk", 0.0)
    v_sub.setdefault("expression_valid", 0.0)
    v_sub.setdefault("expr_reason", "unavailable")
    v_sub.setdefault("expr_source", "none")
    v_sub.setdefault("mp_expr_reason", "unavailable")
    v_sub.setdefault("frame_decode_ok", 0.0)
    v_sub.setdefault("fer_invoked", 0.0)
    v_sub.setdefault("expr_model_ready", 0.0)
    detail = {"V_sub": v_sub, "A_sub": a_sub, "T_sub": t_sub}
    ts_ms = int(getattr(app.state, "risk_timestamp_ms", int(time.time() * 1000)))
    mode = getattr(app.state, "risk_mode", None)
    return RealtimeRiskDetailResponse(
        V=float(scores.get("V", 0.0)),
        A=float(scores.get("A", 0.0)),
        T=float(scores.get("T", 0.0)),
        S=float(scores.get("S", 0.0)),
        timestamp_ms=ts_ms,
        mode=mode,
        detail=detail,
    )


@app.post("/api/emotion/realtime", response_model=RealtimeScoresResponse)
def emotion_realtime_update(payload: RealtimeScoresResponse) -> RealtimeScoresResponse:
    app.state.scores = payload.model_dump()
    app.state.risk_timestamp_ms = int(time.time() * 1000)
    return payload


def _camera_emotion_analyze_sync(
    payload: CameraEmotionAnalyzeRequest,
    *,
    timestamp_ms: int,
    width: int,
    height: int,
) -> CameraEmotionAnalyzeResponse:
    image_bytes = _decode_camera_image_payload(payload.image_data_url)
    with _camera_analysis_lock:
        runtime = _get_camera_vision_runtime()
        scorer = runtime.get("scorer")
        if scorer is None:
            raise HTTPException(status_code=503, detail="camera emotion runtime unavailable")

        frame = VideoFrame(
            format="jpeg",
            data=image_bytes,
            width=width,
            height=height,
            timestamp_ms=timestamp_ms,
            seq=0,
            device_id="desktop-camera",
        )

        face_count, primary_bbox, counter_source = _count_faces_for_camera(frame, runtime)
        bbox_percent = _bbox_to_percent(primary_bbox, width, height)
        model_ready = bool(
            getattr(scorer, "_expr", None) and getattr(scorer._expr, "ready", False)
        ) or bool(
            getattr(scorer, "_expr_mp", None) and getattr(scorer._expr_mp, "ready", False)
        )

        if face_count != 1:
            pause_reason = "multiple_faces" if face_count > 1 else "no_face"
            detail = {
                "face_ok": 0.0,
                "expression_class_id": -1.0,
                "expression_confidence": 0.0,
                "expression_risk": 0.0,
                "expression_valid": 0.0,
                "expr_reason": pause_reason,
                "expr_source": counter_source,
                "expr_model_ready": 1.0 if model_ready else 0.0,
                "face_count": float(face_count),
            }
            app.state.scores = {"V": 0.0, "A": 0.0, "T": 0.0, "S": 0.0}
            app.state.risk_detail = {"V_sub": detail, "A_sub": {}, "T_sub": {}}
            app.state.risk_timestamp_ms = timestamp_ms
            app.state.risk_mode = "desktop_camera"
            return CameraEmotionAnalyzeResponse(
                timestamp_ms=timestamp_ms,
                model_ready=model_ready,
                face_detected=bool(face_count > 0),
                face_count=face_count,
                focus_locked=False,
                recognition_paused=True,
                pause_reason=pause_reason,
                bbox=CameraEmotionFaceBox(**bbox_percent) if bbox_percent else None,
                detail=detail,
            )

        v_score, detail = scorer.score(frame, True)
        if not isinstance(detail, dict):
            detail = {}
        detail = dict(detail)
        detail["face_count"] = float(face_count)
        detail["counter_source"] = counter_source
        detail["subject_locked"] = 1.0

        expr_id = int(detail.get("expression_class_id", -1))
        expr_conf = float(detail.get("expression_confidence", 0.0) or 0.0)
        expr_label = {
            0: "neutral",
            1: "happiness",
            2: "surprise",
            3: "sadness",
            4: "anger",
            5: "disgust",
            6: "fear",
            7: "contempt",
        }.get(expr_id, "unknown")
        s_score = float(v_score)
        app.state.scores = {"V": float(v_score), "A": 0.0, "T": 0.0, "S": s_score}
        app.state.risk_detail = {"V_sub": detail, "A_sub": {}, "T_sub": {}}
        app.state.risk_timestamp_ms = timestamp_ms
        app.state.risk_mode = "desktop_camera"
        return CameraEmotionAnalyzeResponse(
            timestamp_ms=timestamp_ms,
            model_ready=model_ready,
            face_detected=True,
            face_count=face_count,
            focus_locked=True,
            recognition_paused=False,
            pause_reason="",
            emotion_label=expr_label,
            emotion_label_zh=_expression_label_to_zh(expr_label),
            confidence=expr_conf,
            V=float(v_score),
            A=0.0,
            T=0.0,
            S=s_score,
            bbox=CameraEmotionFaceBox(**bbox_percent) if bbox_percent else None,
            detail=detail,
        )


@app.post("/api/vision/camera/analyze", response_model=CameraEmotionAnalyzeResponse)
async def camera_emotion_analyze(
    payload: CameraEmotionAnalyzeRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> CameraEmotionAnalyzeResponse:
    _parse_access_token_for_local_desktop(credentials, conn, request)
    timestamp_ms = int(payload.timestamp_ms or int(time.time() * 1000))
    width = int(payload.width or 0)
    height = int(payload.height or 0)
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width/height are required")
    return await run_in_threadpool(
        _camera_emotion_analyze_sync,
        payload,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
    )


@app.get("/api/emotion/history", response_model=List[EmotionEventResponse])
def emotion_history(
    limit: int = 50,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> List[EmotionEventResponse]:
    user = _parse_access_token(credentials, conn)
    rows = _list_emotion_events(conn, int(user["id"]), limit, start_ms, end_ms)
    return [EmotionEventResponse(**_row_to_event(row)) for row in rows]


@app.post("/api/emotion/history", response_model=EmotionEventResponse)
def emotion_history_add(
    payload: EmotionEventRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> EmotionEventResponse:
    user = _parse_access_token(credentials, conn)
    event_id = _insert_emotion_event(conn, int(user["id"]), payload)
    data = _row_to_event(
        {
            "id": event_id,
            "timestamp_ms": payload.timestamp_ms,
            "type": payload.type,
            "description": payload.description,
            "v": payload.V,
            "a": payload.A,
            "t": payload.T,
            "s": payload.S,
            "intensity": payload.intensity,
            "source": payload.source,
        }
    )
    return EmotionEventResponse(**data)


@app.post("/api/client/session/heartbeat", response_model=ClientSessionHeartbeatResponse)
def client_session_heartbeat(
    payload: ClientSessionHeartbeatRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ClientSessionHeartbeatResponse:
    user = _parse_access_token(credentials, conn)
    client_ip = (payload.client_ip or "").strip()
    if not client_ip and request.client:
        client_ip = str(request.client.host or "").strip()
    _upsert_client_session(
        conn,
        int(user["id"]),
        str(payload.client_type or "").strip() or "mobile",
        str(payload.client_id or "").strip() or "unknown",
        (payload.current_ssid or "").strip() or None,
        client_ip or None,
        bool(payload.is_active),
    )
    selected = {}
    if payload.device_id:
        selected = _get_device(conn, int(user["id"]), payload.device_id)
    else:
        devices = _list_devices(conn, int(user["id"]))
        if devices:
            selected = devices[0]
    strategy = _compute_device_network_strategy(conn, int(user["id"]), selected or {"device_id": payload.device_id or ""})
    if selected:
        _apply_device_network_state(
            conn,
            int(user["id"]),
            str(selected.get("device_id") or ""),
            strategy["desired_ssid"],
            bool(strategy["network_mismatch"]),
            bool(strategy["missing_profile"]),
            strategy["last_switch_reason"],
        )
    return ClientSessionHeartbeatResponse(
        ok=True,
        desired_ssid=strategy["desired_ssid"],
        network_mismatch=bool(strategy["network_mismatch"]),
        missing_profile=bool(strategy["missing_profile"]),
        last_switch_reason=strategy["last_switch_reason"],
    )


@app.post("/api/device/heartbeat", response_model=DeviceHeartbeatResponse)
def device_heartbeat(
    payload: DeviceHeartbeatRequest,
    request: Request,
    conn: Connection = Depends(get_db),
) -> DeviceHeartbeatResponse:
    device_ip = (payload.device_ip or "").strip()
    if not device_ip and request.client:
        device_ip = str(request.client.host or "").strip()

    now_ms = int(time.time() * 1000)
    last_seen_ms = payload.last_seen_ms
    if not last_seen_ms or last_seen_ms < 1_000_000_000_000:
        last_seen_ms = now_ms
    status_payload = dict(payload.status or {})
    if device_ip and not status_payload.get("ip"):
        status_payload["ip"] = device_ip
    if payload.ssid and not status_payload.get("ssid"):
        status_payload["ssid"] = payload.ssid
    if payload.rssi is not None and status_payload.get("rssi") is None:
        status_payload["rssi"] = int(payload.rssi)

    updated = _update_devices_by_device_id(
        conn,
        payload.device_id,
        device_ip or None,
        (payload.device_mac or "").strip() or None,
        (payload.ssid or "").strip() or None,
        last_seen_ms,
        status_payload or None,
    )
    owner = _get_device_owner(conn, payload.device_id)
    strategy = {
        "desired_ssid": None,
        "network_mismatch": False,
        "missing_profile": False,
        "last_switch_reason": None,
        "profiles": [],
    }
    if owner:
        if payload.ssid:
            _mark_wifi_profile_success(conn, int(owner["user_id"]), payload.device_id, payload.ssid)
            owner = _get_device(conn, int(owner["user_id"]), payload.device_id)
        strategy = _compute_device_network_strategy(conn, int(owner["user_id"]), owner)
        _apply_device_network_state(
            conn,
            int(owner["user_id"]),
            payload.device_id,
            strategy["desired_ssid"],
            bool(strategy["network_mismatch"]),
            bool(strategy["missing_profile"]),
            strategy["last_switch_reason"],
        )
    profile_payload = [
        {
            "ssid": str(item.get("ssid") or "").strip(),
            "password": _decrypt_wifi_password(str(item.get("encrypted_password") or "")),
            "last_success_at": item.get("last_success_at"),
        }
        for item in strategy.get("profiles", [])
        if str(item.get("ssid") or "").strip()
    ]
    return DeviceHeartbeatResponse(
        ok=True,
        updated=updated,
        desired_ssid=strategy["desired_ssid"],
        network_mismatch=bool(strategy["network_mismatch"]),
        missing_profile=bool(strategy["missing_profile"]),
        last_switch_reason=strategy["last_switch_reason"],
        profiles=profile_payload,
    )


@app.post("/api/device/claim", response_model=DeviceClaimResponse)
def device_claim(
    payload: DeviceClaimRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceClaimResponse:
    user = _parse_access_token(credentials, conn)
    device = _upsert_device(
        conn,
        int(user["id"]),
        payload.device_id,
        (payload.device_ip or "").strip() or None,
        (payload.ssid or "").strip() or None,
        (payload.device_mac or "").strip() or None,
    )
    claim = _create_claim_session(conn, int(user["id"]), payload.device_id)
    return DeviceClaimResponse(
        ok=True,
        device_id=payload.device_id,
        claim_token=str(claim.get("claim_token") or ""),
        expires_at_ms=int(claim.get("expires_at_ms") or 0),
        onboarding_state=str(device.get("onboarding_state") or "") or None,
        identity_state=str(device.get("identity_state") or "") or None,
    )


@app.get("/api/device/claim/status", response_model=DeviceClaimStatusResponse)
def device_claim_status(
    device_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceClaimStatusResponse:
    user = _parse_access_token(credentials, conn)
    device = _get_device(conn, int(user["id"]), device_id)
    owner = _get_device_owner(conn, device_id)
    claim = _get_active_claim_session(conn, device_id)
    selected = device or owner
    return DeviceClaimStatusResponse(
        ok=True,
        device_id=device_id,
        claimed=bool(owner),
        claimed_user_id=int(owner.get("user_id") or 0) if owner else None,
        onboarding_state=(str(selected.get("onboarding_state") or "").strip() or None) if selected else None,
        identity_state=(str(selected.get("identity_state") or "").strip() or None) if selected else None,
        claim_active=bool(claim),
        expires_at_ms=int(claim.get("expires_at_ms") or 0) if claim else None,
    )


@app.post("/api/device/owner/enrollment", response_model=OwnerEnrollmentResponse)
def owner_enrollment(
    payload: OwnerEnrollmentRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> OwnerEnrollmentResponse:
    user_id: Optional[int] = None
    if credentials is not None:
        try:
            user = _parse_access_token(credentials, conn)
            user_id = int(user["id"])
        except HTTPException:
            user_id = None
    if user_id is None:
        claim = _get_claim_session_by_token(conn, payload.claim_token)
        if not claim:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid claim token")
        user_id = int(claim["claimed_user_id"])
    user_row = _get_user_by_id(conn, int(user_id))
    if not bool(user_row.get("is_configured", 0)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Activation must complete before owner enrollment")
    owner = _get_device_owner(conn, payload.device_id)
    if owner and int(owner.get("user_id") or 0) != int(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Device owned by another user")
    _upsert_device(conn, int(user_id), payload.device_id)
    _upsert_owner_profile(conn, int(user_id), payload)
    return OwnerEnrollmentResponse(
        ok=True,
        device_id=payload.device_id,
        embedding_version=payload.embedding_version,
        identity_state="ready",
    )


@app.get("/api/device/owner/status", response_model=OwnerStatusResponse)
def owner_status(
    device_id: str,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> OwnerStatusResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    profile = _get_owner_profile(conn, int(user["id"]), device_id)
    return OwnerStatusResponse(
        ok=True,
        device_id=device_id,
        enrolled=bool(profile),
        owner_label=(str(profile.get("owner_label") or "").strip() or None) if profile else None,
        embedding_version=(str(profile.get("embedding_version") or "").strip() or None) if profile else None,
        recognition_enabled=bool(int(profile.get("recognition_enabled") or 0)) if profile else True,
        last_sync_ms=int(profile.get("last_sync_ms") or 0) if profile else None,
        enrolled_at_ms=int(profile.get("enrolled_at_ms") or 0) if profile else None,
    )


@app.post("/api/device/owner/enrollment/start", response_model=OwnerEnrollmentStartResponse)
def owner_enrollment_start(
    payload: OwnerEnrollmentStartRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> OwnerEnrollmentStartResponse:
    user = _parse_access_token_for_local_activation(credentials, conn, request)
    if not bool(user.get("is_configured", 0)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Activation must complete before face enrollment")
    if not bool(_get_psychometric_profile(conn, int(user["id"]))):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Psychometric assessment must complete before face enrollment")
    selected = {}
    if payload.device_id:
        selected = _get_device(conn, int(user["id"]), payload.device_id)
    else:
        devices = _list_devices(conn, int(user["id"]))
        if devices:
            selected = devices[0]
    if not selected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    resolved_ip = str(selected.get("device_ip") or "").strip()
    if not resolved_ip:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Device IP missing")
    try:
        state = _post_device_json(
            resolved_ip,
            "/owner/enrollment/start",
            {"owner_label": str(payload.owner_label or "owner").strip() or "owner", "claim_token": ""},
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return OwnerEnrollmentStartResponse(
        ok=True,
        device_id=str(selected.get("device_id") or payload.device_id or "unknown"),
        started=True,
        detail="Face enrollment requested on robot",
        state=state,
    )


@app.post("/api/engine/event")
async def engine_event(
    payload: EngineEventRequest,
    conn: Connection = Depends(get_db),
) -> Dict[str, bool]:
    payload_data = payload.payload if isinstance(payload.payload, dict) else {}
    if payload.type == "RiskUpdate":
        scores = {
            "V": float(payload_data.get("V", 0.0)),
            "A": float(payload_data.get("A", 0.0)),
            "T": float(payload_data.get("T", 0.0)) if payload_data.get("T") is not None else 0.0,
            "S": float(payload_data.get("S", 0.0)),
        }
        app.state.scores = scores
        app.state.risk_detail = payload_data.get("detail") if isinstance(payload_data.get("detail"), dict) else {}
        app.state.risk_mode = payload_data.get("mode")
        app.state.risk_timestamp_ms = int(payload.timestamp_ms)

    if payload.type in {"TriggerFired", "CarePlanReady", "DailySummaryReady"}:
        event_payload = _event_to_emotion(payload)
        if event_payload:
            user_id = _get_default_user_id(conn)
            if user_id is not None:
                _insert_emotion_event(conn, user_id, event_payload)

    if payload.type in {"VoiceChatUser", "VoiceChatBot"}:
        text = str(payload_data.get("text", "")).strip()
        sender = "user" if payload.type == "VoiceChatUser" else "bot"
        rewritten = False
        if sender == "bot":
            text, rewritten = _sanitize_outbound_bot_text(text)
            payload_data["text"] = text
            payload.payload = payload_data
        user_id = _get_default_user_id(conn)
        if user_id is not None and text:
            req = ChatMessageRequest(
                sender=sender,
                text=text,
                content_type="text",
                attachments=[],
                timestamp_ms=int(payload.timestamp_ms),
            )
            msg_id = _insert_chat_message(conn, user_id, req)
            await event_manager.broadcast(
                {
                    "type": "ChatMessage",
                    "timestamp_ms": int(payload.timestamp_ms),
                    "payload": {
                        "id": msg_id,
                        "sender": sender,
                        "text": text,
                        "content_type": "text",
                        "attachments": [],
                        "timestamp_ms": int(payload.timestamp_ms),
                    },
                }
            )
        if rewritten:
            await event_manager.broadcast(
                {
                    "type": "WakeAudioState",
                    "timestamp_ms": int(payload.timestamp_ms),
                    "payload": {
                        "ok": True,
                        "state": "thinking",
                        "reason": "function_call_blocked_and_rewritten",
                    },
                }
            )

    await event_manager.broadcast(payload.model_dump())
    return {"ok": True}


@app.post("/api/engine/signal")
def engine_signal(
    payload: EngineSignalRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, bool]:
    _ = _parse_access_token(credentials, conn)
    signal = {
        "type": payload.type,
        "timestamp_ms": int(time.time() * 1000),
        "payload": payload.payload or {},
    }
    _enqueue_signal(signal)
    return {"ok": True}


@app.post("/api/engine/signal/local")
def engine_signal_local(
    payload: EngineSignalRequest,
    request: Request,
) -> Dict[str, bool]:
    if not _is_local_request(request):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local access only")
    signal = {
        "type": payload.type,
        "timestamp_ms": int(time.time() * 1000),
        "payload": payload.payload or {},
    }
    _enqueue_signal(signal)
    return {"ok": True}


@app.post("/api/engine/signal/pull", response_model=EngineSignalPullResponse)
def engine_signal_pull(
    request: Request,
    payload: Optional[EngineSignalPullRequest] = None,
) -> EngineSignalPullResponse:
    if not _is_local_request(request):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local access only")
    limit = payload.limit if payload else 10
    limit = max(1, min(50, int(limit)))
    signals = _drain_signals(limit)
    return EngineSignalPullResponse(signals=signals)


@app.get("/api/device/list", response_model=List[DeviceInfoResponse])
def device_list(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> List[DeviceInfoResponse]:
    user = _parse_access_token(credentials, conn)
    devices = _list_devices(conn, int(user["id"]))
    return [
        DeviceInfoResponse(
            device_id=row.get("device_id", ""),
            device_ip=row.get("device_ip"),
            device_mac=row.get("device_mac"),
            ssid=row.get("ssid"),
            last_seen_ms=row.get("last_seen_ms"),
        )
        for row in devices
    ]


@app.get("/api/device/status", response_model=DeviceStatusResponse)
def device_status(
    device_id: Optional[str] = None,
    device_ip: Optional[str] = None,
    probe: bool = False,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceStatusResponse:
    user = _parse_access_token(credentials, conn)
    selected = {}
    if device_id:
        selected = _get_device(conn, int(user["id"]), device_id)
    else:
        devices = _list_devices(conn, int(user["id"]))
        if devices:
            selected = devices[0]
    if not selected and not device_ip:
        return DeviceStatusResponse(
            device_id=device_id or "unbound",
            device_ip=None,
            device_mac=None,
            online=False,
            last_seen_ms=None,
            ssid=None,
            desired_ssid=None,
            network_mismatch=False,
            missing_profile=False,
            last_switch_reason=None,
            status=None,
            error="Device not bound yet",
        )

    resolved_id = device_id or selected.get("device_id", "unknown")
    resolved_ip = device_ip or selected.get("device_ip")
    resolved_mac = selected.get("device_mac")
    cached_status = None
    if selected and selected.get("status_json"):
        try:
            cached_status = json.loads(selected.get("status_json") or "{}")
        except Exception:
            cached_status = None
    last_seen_ms = selected.get("last_seen_ms")
    heartbeat_fresh = False
    if last_seen_ms is not None:
        try:
            heartbeat_fresh = int(time.time() * 1000) - int(last_seen_ms) <= HEARTBEAT_STALE_MS
        except Exception:
            heartbeat_fresh = False
    strategy = _compute_device_network_strategy(conn, int(user["id"]), selected or {"device_id": resolved_id, "ssid": None})
    if selected:
        _apply_device_network_state(
            conn,
            int(user["id"]),
            resolved_id,
            strategy["desired_ssid"],
            bool(strategy["network_mismatch"]),
            bool(strategy["missing_profile"]),
            strategy["last_switch_reason"],
        )
    if not resolved_ip:
        return DeviceStatusResponse(
            device_id=resolved_id,
            device_ip=None,
            device_mac=resolved_mac,
            online=heartbeat_fresh,
            last_seen_ms=last_seen_ms,
            ssid=selected.get("ssid") if selected else None,
            desired_ssid=strategy["desired_ssid"],
            network_mismatch=bool(strategy["network_mismatch"]),
            missing_profile=bool(strategy["missing_profile"]),
            last_switch_reason=strategy["last_switch_reason"],
            status=cached_status if heartbeat_fresh else None,
            error=None if heartbeat_fresh else "Device IP missing",
        )

    if not probe:
        return DeviceStatusResponse(
            device_id=resolved_id,
            device_ip=resolved_ip,
            device_mac=resolved_mac,
            online=heartbeat_fresh,
            last_seen_ms=last_seen_ms,
            ssid=selected.get("ssid") if selected else None,
            desired_ssid=strategy["desired_ssid"],
            network_mismatch=bool(strategy["network_mismatch"]),
            missing_profile=bool(strategy["missing_profile"]),
            last_switch_reason=strategy["last_switch_reason"],
            status=cached_status if heartbeat_fresh else None,
            error=None if heartbeat_fresh else "Device heartbeat stale",
        )

    try:
        status_payload = _fetch_device_status(resolved_ip)
        last_seen_ms = int(time.time() * 1000)
        if selected:
            _update_device_status(conn, int(user["id"]), resolved_id, last_seen_ms, status_payload)
        return DeviceStatusResponse(
            device_id=resolved_id,
            device_ip=resolved_ip,
            device_mac=resolved_mac,
            online=True,
            last_seen_ms=last_seen_ms,
            ssid=selected.get("ssid") if selected else None,
            desired_ssid=strategy["desired_ssid"],
            network_mismatch=bool(strategy["network_mismatch"]),
            missing_profile=bool(strategy["missing_profile"]),
            last_switch_reason=strategy["last_switch_reason"],
            status=status_payload,
            error=None,
        )
    except Exception as exc:
        if heartbeat_fresh:
            return DeviceStatusResponse(
                device_id=resolved_id,
                device_ip=resolved_ip,
                device_mac=resolved_mac,
                online=True,
                last_seen_ms=last_seen_ms,
                ssid=selected.get("ssid") if selected else None,
                desired_ssid=strategy["desired_ssid"],
                network_mismatch=bool(strategy["network_mismatch"]),
                missing_profile=bool(strategy["missing_profile"]),
                last_switch_reason=strategy["last_switch_reason"],
                status=cached_status,
                error=None,
            )
        return DeviceStatusResponse(
            device_id=resolved_id,
            device_ip=resolved_ip,
            device_mac=resolved_mac,
            online=False,
            last_seen_ms=last_seen_ms,
            ssid=selected.get("ssid") if selected else None,
            desired_ssid=strategy["desired_ssid"],
            network_mismatch=bool(strategy["network_mismatch"]),
            missing_profile=bool(strategy["missing_profile"]),
            last_switch_reason=strategy["last_switch_reason"],
            status=None,
            error=str(exc),
        )


@app.get("/api/device/settings", response_model=DeviceSettingsResponse)
def device_settings(
    device_id: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceSettingsResponse:
    user = _parse_access_token(credentials, conn)
    selected = _select_device_for_user(conn, int(user["id"]), device_id)
    resolved_device_id = str(device_id or selected.get("device_id") or "unbound")
    settings_state = _get_device_settings(conn, int(user["id"]), resolved_device_id)
    return DeviceSettingsResponse(
        ok=bool(selected),
        device_id=resolved_device_id,
        settings=dict(settings_state.get("settings") or {}),
        ui_state=_cached_ui_state(selected),
        updated_at_ms=settings_state.get("updated_at_ms"),
    )


@app.post("/api/device/settings", response_model=DeviceSettingsResponse)
async def update_device_settings(
    payload: DeviceSettingsUpdateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceSettingsResponse:
    user = _parse_access_token(credentials, conn)
    selected = _select_device_for_user(conn, int(user["id"]), payload.device_id)
    if not selected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not bound yet")
    device_id = str(selected.get("device_id") or "")
    updated = _upsert_device_settings(conn, int(user["id"]), device_id, dict(payload.settings or {}))
    now_ms = int(time.time() * 1000)
    signal_payload = {
        "device_id": device_id,
        "settings": dict(updated.get("settings") or {}),
        "updated_at_ms": int(updated.get("updated_at_ms") or now_ms),
    }
    _enqueue_signal({"type": "settings_apply", "timestamp_ms": now_ms, "payload": signal_payload})
    await event_manager.broadcast(
        {
            "type": "SettingsChanged",
            "timestamp_ms": now_ms,
            "payload": signal_payload,
        }
    )
    return DeviceSettingsResponse(
        ok=True,
        device_id=device_id,
        settings=dict(updated.get("settings") or {}),
        ui_state=_cached_ui_state(selected),
        updated_at_ms=updated.get("updated_at_ms"),
    )


@app.post("/api/device/settings/open", response_model=DeviceSettingsResponse)
async def open_device_settings_page(
    payload: DeviceSettingsPageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceSettingsResponse:
    user = _parse_access_token(credentials, conn)
    selected = _select_device_for_user(conn, int(user["id"]), payload.device_id)
    if not selected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not bound yet")
    device_id = str(selected.get("device_id") or "")
    settings_state = _get_device_settings(conn, int(user["id"]), device_id)
    now_ms = int(time.time() * 1000)
    ui_state = _merge_settings(
        _cached_ui_state(selected),
        {
            "page": "settings",
            "screen_awake": True,
            "source": str(payload.source or "desktop"),
            "opened_at_ms": now_ms,
        },
    )
    outbound_payload = {
        "device_id": device_id,
        "ui_state": ui_state,
        "settings": dict(settings_state.get("settings") or {}),
    }
    _enqueue_signal({"type": "settings_page_open", "timestamp_ms": now_ms, "payload": outbound_payload})
    await event_manager.broadcast(
        {
            "type": "SettingsPageOpened",
            "timestamp_ms": now_ms,
            "payload": outbound_payload,
        }
    )
    return DeviceSettingsResponse(
        ok=True,
        device_id=device_id,
        settings=dict(settings_state.get("settings") or {}),
        ui_state=ui_state,
        updated_at_ms=settings_state.get("updated_at_ms"),
    )


@app.post("/api/device/settings/close", response_model=DeviceSettingsResponse)
async def close_device_settings_page(
    payload: DeviceSettingsPageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DeviceSettingsResponse:
    user = _parse_access_token(credentials, conn)
    selected = _select_device_for_user(conn, int(user["id"]), payload.device_id)
    if not selected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not bound yet")
    device_id = str(selected.get("device_id") or "")
    settings_state = _get_device_settings(conn, int(user["id"]), device_id)
    now_ms = int(time.time() * 1000)
    ui_state = _merge_settings(
        _cached_ui_state(selected),
        {
            "page": "expression",
            "screen_awake": True,
            "source": str(payload.source or "desktop"),
            "last_closed_at_ms": now_ms,
        },
    )
    outbound_payload = {
        "device_id": device_id,
        "ui_state": ui_state,
        "settings": dict(settings_state.get("settings") or {}),
    }
    _enqueue_signal({"type": "settings_page_close", "timestamp_ms": now_ms, "payload": outbound_payload})
    await event_manager.broadcast(
        {
            "type": "SettingsPageClosed",
            "timestamp_ms": now_ms,
            "payload": outbound_payload,
        }
    )
    return DeviceSettingsResponse(
        ok=True,
        device_id=device_id,
        settings=dict(settings_state.get("settings") or {}),
        ui_state=ui_state,
        updated_at_ms=settings_state.get("updated_at_ms"),
    )


@app.get("/api/chat/history", response_model=List[ChatMessageResponse])
async def chat_history(
    limit: int = 100,
    surface: str = "desktop",
    session_key: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> List[ChatMessageResponse]:
    user = _parse_access_token(credentials, conn)
    resolved_surface = normalize_surface(surface)
    resolved_session_key = session_key or build_session_key(resolved_surface, int(user["id"]))
    inserted = _sync_wechat_transcript_history(conn, int(user["id"]), resolved_session_key, surface=resolved_surface)
    for item in inserted:
        await event_manager.broadcast(
            {
                "type": "ChatMessage",
                "timestamp_ms": item.timestamp_ms,
                "payload": item.model_dump(),
            }
        )
    rows = _list_chat_messages(conn, int(user["id"]), limit, session_key=resolved_session_key)
    return [_chat_response_from_row(row) for row in rows]


@app.post("/api/chat/history", response_model=ChatMessageResponse)
async def chat_history_add(
    payload: ChatMessageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> ChatMessageResponse:
    user = _parse_access_token(credentials, conn)
    if not payload.session_key:
        payload.session_key = build_session_key(normalize_surface(payload.surface), int(user["id"]))
    payload.surface = normalize_surface(payload.surface)
    msg_id = _insert_chat_message(conn, int(user["id"]), payload)
    response = ChatMessageResponse(
        id=msg_id,
        sender=payload.sender,
        text=payload.text,
        content_type=str(payload.content_type or "text"),
        attachments=payload.attachments or [],
        timestamp_ms=payload.timestamp_ms,
        surface=payload.surface,
        session_key=payload.session_key,
    )
    await event_manager.broadcast(
        {
            "type": "ChatMessage",
            "timestamp_ms": payload.timestamp_ms,
            "payload": response.model_dump(),
        }
    )
    return response


def _safe_upload_name(name: str) -> str:
    raw = Path(str(name or "upload.bin")).name
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})
    safe = safe.strip("._")
    return safe or "upload.bin"


@app.post("/api/chat/upload")
async def chat_upload(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, object]:
    user = _parse_access_token(credentials, conn)
    content_type = str(file.content_type or "").lower()
    if not (content_type.startswith("image/") or content_type.startswith("video/")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image/video uploads are supported")

    kind = "image" if content_type.startswith("image/") else "video"
    ts = int(time.time() * 1000)
    date_key = time.strftime("%Y%m%d", time.localtime(ts / 1000.0))
    target_dir = CHAT_UPLOAD_ROOT / str(int(user["id"])) / date_key
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = _safe_upload_name(file.filename or "upload.bin")
    stem = Path(original_name).stem or "upload"
    suffix = Path(original_name).suffix
    target_name = f"{stem}-{ts}{suffix}"
    target_path = target_dir / target_name

    with target_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    size = int(target_path.stat().st_size) if target_path.exists() else 0
    rel = target_path.relative_to(UPLOAD_ROOT).as_posix()
    return {
        "ok": True,
        "attachment": {
            "kind": kind,
            "url": f"/uploads/{rel}",
            "mime": content_type,
            "name": original_name,
            "size": size,
        },
    }


@app.post("/api/desktop/voice/transcribe", response_model=DesktopVoiceTranscribeResponse)
async def desktop_voice_transcribe(
    file: UploadFile = File(...),
    context: str = Form("chat"),
    request: Request = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DesktopVoiceTranscribeResponse:
    _parse_access_token_for_local_desktop(credentials, conn, request)
    content_type = str(file.content_type or "").lower()
    if not (content_type.startswith("audio/") or str(file.filename or "").lower().endswith(".wav")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only audio uploads are supported")
    try:
        audio_bytes = await file.read()
        payload = desktop_speech_service.transcribe_upload(
            audio_bytes=audio_bytes,
            filename=str(file.filename or ""),
            content_type=content_type,
            context=str(context or "chat"),
        )
        return DesktopVoiceTranscribeResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/api/desktop/voice/synthesize")
async def desktop_voice_synthesize(
    payload: dict = Body(...),
    request: Request = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Response:
    _parse_access_token_for_local_desktop(credentials, conn, request)
    text = str(payload.get("text", "") or "").strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty_text")
    provider = str(payload.get("provider", "voxcpm_distill") or "voxcpm_distill").strip() or "voxcpm_distill"
    voice_style = str(payload.get("voice_style", "gentle") or "gentle").strip() or "gentle"
    try:
        result = await _desktop_voice_fetch_json(
            DESKTOP_TTS_URL,
            method="POST",
            payload={
                "text": text,
                "provider": provider,
                "voice_style": voice_style,
                "cfg_value": float(payload.get("cfg_value", 1.28) or 1.28),
                "inference_timesteps": int(payload.get("inference_timesteps", 4) or 4),
            },
            timeout_s=240.0,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"remote_tts_unavailable: {exc}") from exc

    audio_b64 = str(result.get("audio_base64", "") or "")
    if not audio_b64:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="remote_tts_missing_audio")
    audio_bytes = base64.b64decode(audio_b64)
    headers = {
        "X-TTS-Provider": str(result.get("provider", provider)),
        "X-TTS-Model": str(result.get("model_name", "unknown")),
        "X-TTS-Sample-Rate": str(result.get("sample_rate", 0)),
    }
    return Response(
        content=audio_bytes,
        media_type=str(result.get("mime_type", "audio/wav")),
        headers=headers,
    )


async def _assistant_send_impl(
    conn: Connection,
    user_id: int,
    payload: AssistantSendRequest,
) -> AssistantSendResponse:
    raw_text = str(payload.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text is required")
    merged_metadata = dict(payload.metadata or {})
    assistant_runtime = _get_user_assistant_settings(conn, int(user_id), payload.device_id)
    if "assistant_mode" not in merged_metadata:
        merged_metadata["assistant_mode"] = str(assistant_runtime.get("mode") or "product")
    if "assistant_native_control" not in merged_metadata:
        merged_metadata["assistant_native_control"] = bool(assistant_runtime.get("native_control_enabled", True))
    merged_metadata["user_profile"] = _build_assistant_identity_context(conn, int(user_id))
    stored_memory_summary = assistant_service.get_profile_memory_summary(int(user_id), max_chars=1200)
    provided_memory_summary = str(merged_metadata.get("memory_summary") or "").strip()
    if stored_memory_summary:
        if provided_memory_summary:
            if stored_memory_summary not in provided_memory_summary:
                merged_metadata["memory_summary"] = f"{stored_memory_summary}\n\n最近上下文补充：{provided_memory_summary}"
        else:
            merged_metadata["memory_summary"] = stored_memory_summary
    response_payload = await assistant_service.send_message(
        conn,
        user_id=int(user_id),
        text=raw_text,
        surface=payload.surface,
        session_key=payload.session_key,
        device_id=payload.device_id,
        sender_id=payload.sender_id,
        attachments=payload.attachments or [],
        metadata=merged_metadata,
    )
    user_message = ChatMessageRequest(
        sender="user",
        text=raw_text,
        content_type="text",
        attachments=payload.attachments or [],
        timestamp_ms=int(response_payload["timestamp_ms"]) - 1,
        surface=str(response_payload["surface"]),
        session_key=str(response_payload["session_key"]),
    )
    bot_text = _sanitize_outbound_bot_text(str(response_payload.get("text") or ""))[0] or "我在。"
    bot_message = ChatMessageRequest(
        sender="assistant",
        text=bot_text,
        content_type="text",
        attachments=[],
        timestamp_ms=int(response_payload["timestamp_ms"]),
        surface=str(response_payload["surface"]),
        session_key=str(response_payload["session_key"]),
    )
    user_id_int = int(user_id)
    user_msg_id = _insert_chat_message(conn, user_id_int, user_message)
    bot_msg_id = _insert_chat_message(conn, user_id_int, bot_message)
    await event_manager.broadcast(
        {
            "type": "ChatMessage",
            "timestamp_ms": user_message.timestamp_ms,
            "payload": {
                "id": user_msg_id,
                "sender": user_message.sender,
                "text": user_message.text,
                "content_type": user_message.content_type,
                "attachments": user_message.attachments,
                "timestamp_ms": user_message.timestamp_ms,
                "surface": user_message.surface,
                "session_key": user_message.session_key,
            },
        }
    )
    await event_manager.broadcast(
        {
            "type": "ChatMessage",
            "timestamp_ms": bot_message.timestamp_ms,
            "payload": {
                "id": bot_msg_id,
                "sender": bot_message.sender,
                "text": bot_message.text,
                "content_type": bot_message.content_type,
                "attachments": [],
                "timestamp_ms": bot_message.timestamp_ms,
                "surface": bot_message.surface,
                "session_key": bot_message.session_key,
            },
        }
    )
    if normalize_surface(payload.surface) == "desktop":
        await _mirror_bot_reply_to_wechat(bot_text)
    return AssistantSendResponse(
        ok=True,
        surface=str(response_payload["surface"]),
        session_key=str(response_payload["session_key"]),
        text=bot_text,
        tool_results=response_payload.get("tool_results") or [],
        timestamp_ms=int(response_payload["timestamp_ms"]),
    )


@app.post("/api/assistant/send", response_model=AssistantSendResponse)
async def assistant_send(
    payload: AssistantSendRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantSendResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    try:
        return await _assistant_send_impl(conn, int(user["id"]), payload)
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/api/assistant/bridge/send", response_model=AssistantSendResponse)
async def assistant_bridge_send(
    payload: AssistantBridgeSendRequest,
    request: Request,
    conn: Connection = Depends(get_db),
) -> AssistantSendResponse:
    user = _bridge_user_from_request(request, conn)
    bridge_payload = AssistantSendRequest(
        text=payload.text,
        surface=payload.surface,
        session_key=payload.session_key,
        sender_id=payload.sender_id,
        metadata=payload.metadata,
    )
    try:
        return await _assistant_send_impl(conn, int(user["id"]), bridge_payload)
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/api/assistant/session/status", response_model=AssistantSessionStatusResponse)
def assistant_session_status(
    surface: str = "desktop",
    session_key: Optional[str] = None,
    device_id: Optional[str] = None,
    sender_id: Optional[str] = None,
    limit: int = 30,
    request: Request = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantSessionStatusResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    status_payload = assistant_service.get_session_status(
        conn,
        user_id=int(user["id"]),
        surface=surface,
        session_key=session_key,
        device_id=device_id,
        sender_id=sender_id,
    )
    rows = _list_chat_messages(conn, int(user["id"]), max(1, min(int(limit), 100)), session_key=str(status_payload["session_key"]))
    history = [_chat_response_from_row(row) for row in rows]
    return AssistantSessionStatusResponse(
        ok=True,
        surface=str(status_payload["surface"]),
        session_key=str(status_payload["session_key"]),
        last_message_ts_ms=status_payload.get("last_message_ts_ms"),
        message_count=int(status_payload.get("message_count") or 0),
        history=history,
    )


@app.post("/api/assistant/session/reset")
async def assistant_session_reset(
    payload: AssistantSessionResetRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, object]:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    try:
        data = await assistant_service.reset_session(
            int(user["id"]),
            surface=payload.surface,
            session_key=payload.session_key,
            device_id=payload.device_id,
            sender_id=payload.sender_id,
        )
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {"ok": True, **data}


@app.get("/api/assistant/todos", response_model=AssistantTodoListResponse)
def assistant_todos(
    state: Optional[str] = None,
    request: Request = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantTodoListResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    items = [AssistantTodoItem(**item) for item in assistant_service.list_todos(int(user["id"]), state=state)]
    return AssistantTodoListResponse(ok=True, items=items)


@app.get("/api/assistant/todos/due", response_model=AssistantTodoListResponse)
def assistant_todos_due(
    limit: int = 10,
    request: Request = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantTodoListResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    items = [
        AssistantTodoItem(**item)
        for item in assistant_service.claim_due_todos(int(user["id"]), limit=max(1, min(int(limit), 20)))
    ]
    return AssistantTodoListResponse(ok=True, items=items)


@app.post("/api/assistant/todos", response_model=AssistantTodoItem)
def assistant_todos_create(
    payload: AssistantTodoCreateRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantTodoItem:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    item = assistant_service.create_todo(
        int(user["id"]),
        title=payload.title,
        details=payload.details,
        due_at_ms=payload.due_at_ms,
        tags=payload.tags,
    )
    return AssistantTodoItem(**item)


@app.patch("/api/assistant/todos/{todo_id}", response_model=AssistantTodoItem)
def assistant_todos_patch(
    todo_id: str,
    payload: AssistantTodoUpdateRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantTodoItem:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    try:
        item = assistant_service.update_todo(int(user["id"]), todo_id, payload.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return AssistantTodoItem(**item)


@app.get("/api/assistant/memory/search", response_model=AssistantMemorySearchResponse)
def assistant_memory_search(
    q: str,
    limit: int = 10,
    request: Request = None,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantMemorySearchResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    return AssistantMemorySearchResponse(
        ok=True,
        query=q,
        results=assistant_service.search_memory(int(user["id"]), q, limit=max(1, min(int(limit), 20))),
    )


@app.get("/api/assistant/runtime/status", response_model=AssistantRuntimeStatusResponse)
def assistant_runtime_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> AssistantRuntimeStatusResponse:
    _ = _parse_access_token_for_local_desktop(credentials, conn, request)
    return AssistantRuntimeStatusResponse(ok=True, **assistant_service.runtime_status())


@app.post("/api/assistant/runtime/warmup")
async def assistant_runtime_warmup(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> Dict[str, object]:
    _ = _parse_access_token_for_local_desktop(credentials, conn, request)
    payload = await assistant_service.warmup_desktop_session()
    payload["ok"] = bool(payload.get("ok"))
    return payload


def _require_bearer(credentials: Optional[HTTPAuthorizationCredentials]) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    if not auth.decode_token(credentials.credentials):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def _evaluate_proactive_care_timing(
    payload: CareRequest,
    current_ts_ms: int,
    history_items: List[Dict[str, object]],
    expr_label: str,
    expr_conf: float,
) -> Dict[str, object]:
    raw_text = str(payload.context or "").strip()
    emotion = str(payload.current_emotion or "").strip().lower()
    lowered = raw_text.lower()
    hour = time.localtime(current_ts_ms / 1000).tm_hour

    high_risk_tokens = [
        "想死",
        "不想活",
        "活不下去",
        "结束掉",
        "不如死",
        "消失",
        "撑不住了",
        "崩溃了",
        "想结束",
    ]
    support_tokens = [
        "帮我",
        "怎么办",
        "好累",
        "累死",
        "难受",
        "烦死",
        "想哭",
        "崩溃",
        "压力",
        "焦虑",
        "害怕",
        "睡不着",
        "失眠",
        "撑不住",
        "受不了",
    ]
    busy_tokens = [
        "开会",
        "在忙",
        "忙着",
        "上班",
        "稍后",
        "回头",
        "等下",
        "待会",
        "先别",
        "别吵",
        "睡了",
        "休息了",
    ]
    negative_emotions = {
        "sad",
        "sadness",
        "down",
        "depressed",
        "anger",
        "angry",
        "frustrated",
        "irritated",
        "fear",
        "anxious",
        "anxiety",
        "stress",
        "stressed",
    }
    negative_expr_labels = {"sadness", "anger", "fear", "disgust", "contempt"}

    history_gap_ms: Optional[int] = None
    if history_items:
        history_gap_ms = max(0, int(current_ts_ms) - int(history_items[-1].get("timestamp_ms", current_ts_ms)))
    recent_user_count = sum(
        1
        for item in history_items[-4:]
        if str(item.get("sender", "user")).strip().lower() == "user"
    )

    state_score = 0.10
    state_reasons: List[str] = []
    is_high_risk = any(token in raw_text for token in high_risk_tokens)
    if is_high_risk:
        state_score += 0.62
        state_reasons.append("命中高风险表达")
    if emotion in negative_emotions:
        state_score += 0.22
        state_reasons.append(f"当前情绪为 {emotion}")
    if any(token in raw_text for token in support_tokens):
        state_score += 0.18
        state_reasons.append("文本出现求助/高压信号")
    if expr_label in negative_expr_labels and expr_conf >= 0.55:
        state_score += 0.14
        state_reasons.append(f"表情信号偏负向({expr_label}, {expr_conf:.2f})")
    if recent_user_count >= 2:
        state_score += 0.08
        state_reasons.append("近期连续多轮用户表达")
    state_score = max(0.0, min(1.0, state_score))
    state_active = is_high_risk or state_score >= 0.45

    receptivity_score = 0.35
    receptivity_reasons: List[str] = []
    busy_detected = any(token in raw_text for token in busy_tokens)
    if busy_detected:
        receptivity_score -= 0.36
        receptivity_reasons.append("文本出现忙碌/不便被打扰信号")
    explicit_invite_tokens = ["怎么办", "帮我", "可以吗", "能不能", "能否", "怎么做", "要不要"]
    if "?" in raw_text or "？" in raw_text or any(token in lowered for token in explicit_invite_tokens):
        receptivity_score += 0.20
        receptivity_reasons.append("当前文本含明确回应邀请")
    if history_gap_ms is None or history_gap_ms <= 3 * 60 * 1000:
        receptivity_score += 0.14
        receptivity_reasons.append("当前会话仍在活跃窗口内")
    elif history_gap_ms >= 15 * 60 * 1000:
        receptivity_score -= 0.14
        receptivity_reasons.append("距离上一轮交互较久")
    if hour >= 23 or hour < 7:
        receptivity_score -= 0.10
        receptivity_reasons.append("当前处于深夜/清晨时段")
    if is_high_risk:
        receptivity_score += 0.24
        receptivity_reasons.append("高风险场景优先安全介入")
    receptivity_score = max(0.0, min(1.0, receptivity_score))
    receptive = is_high_risk or receptivity_score >= 0.45

    if is_high_risk:
        recommendation = "intervene_now"
        intensity = "safety_check"
    elif state_active and receptive:
        recommendation = "intervene_now"
        intensity = "gentle_support"
    elif state_active:
        recommendation = "defer_softly"
        intensity = "brief_checkin"
    else:
        recommendation = "hold"
        intensity = "minimal"

    return {
        "current_hour": int(hour),
        "history_gap_ms": history_gap_ms,
        "state_stage": {
            "score": round(state_score, 3),
            "active": bool(state_active),
            "reasons": state_reasons,
        },
        "receptivity_stage": {
            "score": round(receptivity_score, 3),
            "receptive": bool(receptive),
            "reasons": receptivity_reasons,
        },
        "high_risk": bool(is_high_risk),
        "recommendation": recommendation,
        "intensity": intensity,
    }


def _build_care_context(payload: CareRequest) -> Dict[str, object]:
    expr_labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
    runtime_detail = app.state.risk_detail if isinstance(getattr(app.state, "risk_detail", None), dict) else {}
    runtime_v_sub = runtime_detail.get("V_sub") if isinstance(runtime_detail.get("V_sub"), dict) else {}
    runtime_expr_id = int(runtime_v_sub.get("expression_class_id", -1) or -1)
    runtime_expr_label = (
        expr_labels[runtime_expr_id]
        if runtime_expr_id >= 0 and runtime_expr_id < len(expr_labels)
        else str(runtime_v_sub.get("expression_label", "unknown") or "unknown")
    )
    runtime_expr_conf = float(runtime_v_sub.get("expression_confidence", 0.0) or 0.0)
    payload_expr_label = str(payload.expression_label or "unknown").strip() or "unknown"
    payload_expr_conf = float(payload.expression_confidence or 0.0)
    expr_label = payload_expr_label
    expr_conf = payload_expr_conf
    expr_source = "payload"
    if (expr_label == "unknown" and runtime_expr_label != "unknown") or (expr_conf <= 0.0 and runtime_expr_conf > 0.0):
        expr_label = runtime_expr_label or expr_label
        expr_conf = runtime_expr_conf if runtime_expr_conf > 0.0 else expr_conf
        expr_source = "runtime_risk_detail"

    history_items = []
    current_ts_ms = payload.current_ts_ms or int(time.time() * 1000)
    for item in payload.history or []:
        if isinstance(item, dict):
            history_items.append(
                {
                    "sender": str(item.get("sender", "user")),
                    "text": str(item.get("text", "")),
                    "timestamp_ms": int(item.get("timestamp_ms", current_ts_ms)),
                }
            )
        else:
            history_items.append(
                {
                    "sender": str(item.sender),
                    "text": str(item.text),
                    "timestamp_ms": int(item.timestamp_ms),
                }
            )

    history_items = [item for item in history_items if item.get("text")]
    history_items.sort(key=lambda item: item["timestamp_ms"])
    history_gap_ms = None
    if history_items:
        last_ts = int(history_items[-1]["timestamp_ms"])
        history_gap_ms = int(current_ts_ms) - last_ts

    history_stale_ms = 10 * 60 * 1000
    history_usable = history_gap_ms is None or history_gap_ms <= history_stale_ms
    if history_usable:
        model_history = history_items[-4:]
    else:
        model_history = []
    history_summary = ""
    if model_history:
        recent = model_history[-3:]
        history_summary = " | ".join([f"{item['sender']}: {item['text']}" for item in recent])
    memory_summary = str(payload.memory_summary or "").strip()
    if len(memory_summary) > 600:
        memory_summary = memory_summary[-600:]
    attachment_items: List[Dict[str, object]] = []
    for item in payload.attachments or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in {"image", "video"}:
            continue
        entry: Dict[str, object] = {
            "kind": kind,
            "url": str(item.get("url", "")).strip(),
            "mime": str(item.get("mime", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "size": int(item.get("size", 0) or 0),
        }
        image_data_url = str(item.get("image_data_url", "")).strip()
        if kind == "image" and image_data_url.startswith("data:image/"):
            entry["image_data_url"] = image_data_url
        attachment_items.append(entry)

    care_timing = _evaluate_proactive_care_timing(
        payload,
        int(current_ts_ms),
        history_items,
        expr_label,
        expr_conf,
    )

    return {
        "input_type": "user_text",
        "scene": "office",
        "current_emotion": payload.current_emotion,
        "current_message": {
            "text": payload.context,
            "timestamp_ms": int(current_ts_ms),
        },
        "history": [
            {
                "sender": item["sender"],
                "text": item["text"],
                "timestamp_ms": int(item["timestamp_ms"]),
            }
            for item in model_history
        ],
        "history_summary": history_summary,
        "memory_summary": memory_summary,
        "attachments": attachment_items,
        "history_gap_ms": history_gap_ms,
        "history_usable": history_usable,
        "care_timing": care_timing,
        "expression_modality": {
            "label": expr_label,
            "confidence": expr_conf,
            "source": expr_source,
            "note": "这是算法观测到的情绪信号，不是用户原话。",
        },
        "expression_label": expr_label,
        "expression_confidence": expr_conf,
        "constraints": (
            "优先回答当前问题；仅在表达难受或求助时给关怀建议；避免重复追问，不诊断，不说教；"
            "避免输出emoji、颜文字或表情包标签；"
            "不输出思考过程；新闻问题优先联网搜索；"
            "若发生联网搜索，先明确说“我帮你联网搜搜”；"
            "回复时要参考 expression_modality（算法信号，非用户原话）；"
            "回复时必须参考 care_timing：若 recommendation=hold，则只允许极轻量、非打扰式回应；"
            "若 recommendation=defer_softly，则最多一到两句，不要连续追问；"
            "不要向用户展示function_call/tool_call/json草稿。"
        ),
    }


def _normalize_care_runtime(runtime: Dict[str, object]) -> Dict[str, object]:
    gateway_ready = bool(runtime.get("gateway_ready"))
    provider_network_ok = bool(runtime.get("provider_network_ok"))
    gateway_error = str(runtime.get("gateway_error") or "").strip()
    provider_network_detail = str(runtime.get("provider_network_detail") or "").strip()
    if gateway_error:
        detail = gateway_error
    elif gateway_ready and not provider_network_ok:
        detail = provider_network_detail or "OpenClaw provider network unavailable"
    else:
        detail = "OpenClaw ready" if gateway_ready else "OpenClaw unavailable"
    return {
        "gateway_ready": gateway_ready,
        "provider_network_ok": provider_network_ok,
        "ai_ready": bool(gateway_ready and provider_network_ok),
        "ai_detail": detail,
    }


def _build_care_prompt(
    context: Dict[str, object],
    user_profile: Dict[str, object],
    runtime_status: Dict[str, object],
) -> str:
    payload = {
        "project": "共鸣连接",
        "channel": "主动关怀",
        "runtime": runtime_status,
        "user_profile": user_profile,
        "care_context": context,
    }
    return (
        f"{CARE_SYSTEM_PROMPT}\n\n"
        f"{CARE_RESPONSE_RULES}\n\n"
        "下面是本次需要回应的结构化上下文，请只输出给用户的话，不要复述这些字段：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _fallback_care_text(
    payload: CareRequest,
    user_profile: Dict[str, object],
    detail: str = "",
    timing: Optional[Dict[str, object]] = None,
) -> str:
    raw_text = str(payload.context or "").strip()
    emotion = str(payload.current_emotion or "").strip().lower()
    preferred_name = str((user_profile.get("identity") or {}).get("preferred_name") or "").strip()
    prefix = f"{preferred_name}，" if preferred_name else ""
    recommendation = str((timing or {}).get("recommendation") or "").strip()
    if recommendation == "hold":
        return f"{prefix}我先轻一点陪着你，不多打扰。等你想说的时候，直接叫我。"
    if recommendation == "defer_softly":
        return f"{prefix}我先不一下子问很多。你如果愿意，只回我一句现在最卡的是哪件事。"
    if emotion in {"sad", "sadness", "down", "depressed"}:
        base = f"{prefix}我听出来你现在有点低落，我们先别急着把所有事一次想完。先把眼前最压人的那一件事说清楚，我陪你一起拆开。"
    elif emotion in {"angry", "anger", "frustrated", "irritated"}:
        base = f"{prefix}你现在这股顶着的劲我接到了。先别急着硬扛，先停十秒，把最让你卡住的那一点拎出来，我们只处理那一件。"
    elif emotion in {"fear", "anxious", "anxiety", "stress", "stressed"}:
        base = f"{prefix}先稳一下，我们先只看当前这一分钟要处理的事。你可以先做一次慢呼气，然后告诉我最担心的是哪一块。"
    elif raw_text:
        base = f"{prefix}我在，刚刚这句话我接住了。我们先不求一下子理顺全部，只把现在最需要回应的那一点说清楚。"
    else:
        base = f"{prefix}我在这里，先陪你把现在的状态稳住。你可以直接说一句此刻最想解决的事。"
    if detail:
        return f"{base}"
    return base


def _build_policy_care_response(
    payload: CareRequest,
    user_profile: Dict[str, object],
    runtime_snapshot: Dict[str, object],
    timing: Optional[Dict[str, object]] = None,
) -> Optional[CareResponse]:
    timing = timing if isinstance(timing, dict) else {}
    recommendation = str(timing.get("recommendation") or "").strip()
    if recommendation not in {"hold", "defer_softly"}:
        return None
    stage_reasons = timing.get("receptivity_stage") if isinstance(timing.get("receptivity_stage"), dict) else {}
    reasons = stage_reasons.get("reasons") if isinstance(stage_reasons.get("reasons"), list) else []
    reason_text = "；".join(str(item).strip() for item in reasons if str(item).strip())
    detail = f"policy:{recommendation}"
    if reason_text:
        detail = f"{detail} ({reason_text})"
    return CareResponse(
        text=_fallback_care_text(payload, user_profile, detail=detail, timing=timing),
        followup_question="",
        style="warm",
        source=f"policy_{recommendation}",
        detail=detail,
        ai_ready=bool(runtime_snapshot.get("ai_ready")),
        timing=timing,
    )


def _persist_proactive_care_decision(
    conn: Connection,
    user_id: int,
    payload: CareRequest,
    entrypoint: str,
    response: CareResponse,
) -> None:
    timing = response.timing if isinstance(response.timing, dict) else {}
    state_stage = timing.get("state_stage") if isinstance(timing.get("state_stage"), dict) else {}
    receptivity_stage = timing.get("receptivity_stage") if isinstance(timing.get("receptivity_stage"), dict) else {}
    timestamp_ms = int(payload.current_ts_ms or int(time.time() * 1000))
    created_at = int(time.time() * 1000)
    preview = re.sub(r"\s+", " ", str(payload.context or "").strip())[:180]
    conn.execute(
        """
        INSERT INTO proactive_care_decisions (
            user_id, timestamp_ms, entrypoint, current_emotion, recommendation, intensity,
            source, ai_ready, high_risk, state_score, receptivity_score, context_preview,
            detail, timing_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            timestamp_ms,
            str(entrypoint or "llm_care"),
            str(payload.current_emotion or "").strip(),
            str(timing.get("recommendation") or "").strip(),
            str(timing.get("intensity") or "").strip(),
            str(response.source or "fallback").strip(),
            1 if bool(response.ai_ready) else 0,
            1 if bool(timing.get("high_risk")) else 0,
            float(state_stage.get("score", 0.0) or 0.0),
            float(receptivity_stage.get("score", 0.0) or 0.0),
            preview,
            str(response.detail or "").strip(),
            json.dumps(timing, ensure_ascii=False),
            created_at,
        ),
    )
    conn.commit()


def _finalize_care_response(
    conn: Connection,
    user_id: int,
    payload: CareRequest,
    entrypoint: str,
    response: CareResponse,
) -> CareResponse:
    try:
        _persist_proactive_care_decision(conn, user_id, payload, entrypoint, response)
    except Exception:
        pass
    return response


async def _run_care_reply(
    conn: Connection,
    user_id: int,
    payload: CareRequest,
    entrypoint: str,
) -> CareResponse:
    user_profile = _build_assistant_identity_context(conn, int(user_id))
    context = _build_care_context(payload)
    timing = context.get("care_timing") if isinstance(context.get("care_timing"), dict) else {}
    runtime_snapshot = _normalize_care_runtime(assistant_service.runtime_status())
    policy_response = _build_policy_care_response(payload, user_profile, runtime_snapshot, timing=timing)
    if policy_response is not None:
        return _finalize_care_response(conn, int(user_id), payload, entrypoint, policy_response)
    stored_memory_summary = assistant_service.get_profile_memory_summary(int(user_id), max_chars=1200)
    prompt = _build_care_prompt(context, user_profile, runtime_snapshot)
    if not bool(runtime_snapshot.get("ai_ready")):
        fallback_text = _fallback_care_text(
            payload,
            user_profile,
            str(runtime_snapshot.get("ai_detail") or ""),
            timing=timing,
        )
        response = CareResponse(
            text=fallback_text,
            followup_question="",
            style="warm",
            source="fallback",
            detail=str(runtime_snapshot.get("ai_detail") or ""),
            ai_ready=False,
            timing=timing,
        )
        return _finalize_care_response(conn, int(user_id), payload, entrypoint, response)
    try:
        assistant_reply = await assistant_service.send_message(
            conn,
            user_id=int(user_id),
            text=prompt,
            surface="desktop",
            session_key=f"desktop:{int(user_id)}:care",
            metadata={
                "entrypoint": entrypoint,
                "assistant_mode": "product",
                "assistant_native_control": False,
                "current_emotion": payload.current_emotion,
                "care_channel": "proactive_care",
                "care_runtime": runtime_snapshot,
                "care_timing": timing,
                "user_profile": user_profile,
                "memory_summary": stored_memory_summary,
            },
        )
        safe_text, _rewritten = _sanitize_outbound_bot_text(str(assistant_reply.get("text") or ""))
        if safe_text:
            response = CareResponse(
                text=safe_text,
                followup_question="",
                style="warm",
                source="ai",
                detail=str(runtime_snapshot.get("ai_detail") or ""),
                ai_ready=True,
                timing=timing,
            )
            return _finalize_care_response(conn, int(user_id), payload, entrypoint, response)
        raise RuntimeError("OpenClaw returned empty care reply")
    except Exception as exc:
        detail = str(exc).strip() or str(runtime_snapshot.get("ai_detail") or "")
        fallback_text = _fallback_care_text(payload, user_profile, detail, timing=timing)
        response = CareResponse(
            text=fallback_text,
            followup_question="",
            style="warm",
            source="fallback",
            detail=detail,
            ai_ready=bool(runtime_snapshot.get("ai_ready")),
            timing=timing,
        )
        return _finalize_care_response(conn, int(user_id), payload, entrypoint, response)


def _inject_tooling_budget(
    context: Dict[str, object],
    conn: Connection,
    user_id: int,
) -> Dict[str, object]:
    _ensure_llm_loaded()
    date_key = _usage_date_key()
    usage = _get_tool_usage_daily(conn, user_id, date_key)
    web_limit = int(getattr(_llm_config, "web_search_daily_limit", 5) if _llm_config else 5)
    web_limit = max(0, web_limit)
    web_used = int(usage.get("web_search_count", 0))
    web_remaining = max(0, web_limit - web_used)

    emotion_cap = int(getattr(_llm_config, "emotion_linked_search_daily_cap", 1) if _llm_config else 1)
    emotion_cap = max(0, emotion_cap)
    emotion_used = int(usage.get("emotion_auto_search_count", 0))
    emotion_remaining = max(0, emotion_cap - emotion_used)

    tooling = context.get("tooling") if isinstance(context.get("tooling"), dict) else {}
    tooling = dict(tooling)
    tooling.update(
        {
            "user_id": int(user_id),
            "date_key": date_key,
            "web_search_daily_limit": web_limit,
            "web_search_budget_remaining": web_remaining,
            "web_search_used": web_used,
            "emotion_auto_search_daily_cap": emotion_cap,
            "emotion_auto_search_remaining": emotion_remaining,
            "emotion_auto_search_used": emotion_used,
        }
    )
    context["tooling"] = tooling
    return context


def _sse(event: str, payload: Dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/llm/care", response_model=CareResponse)
async def llm_care(
    payload: CareRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> CareResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    return await _run_care_reply(conn, int(user["id"]), payload, entrypoint="llm_care")


@app.post("/api/llm/care/stream")
async def llm_care_stream(
    payload: CareRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
):
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    care_reply = await _run_care_reply(conn, int(user["id"]), payload, entrypoint="llm_care_stream")
    final_text = str(care_reply.text or "").strip() or "我在这里陪着你。"

    async def event_stream():
        try:
            yield _sse(
                "start",
                {
                    "ok": True,
                    "source": care_reply.source,
                    "ai_ready": care_reply.ai_ready,
                    "timing": care_reply.timing,
                },
            )
            sent_text = ""
            for char in final_text[:100]:
                delta = char
                sent_text += delta
                yield _sse("delta", {"text": delta})
                await asyncio.sleep(0)

            yield _sse(
                "done",
                {
                    "text": sent_text or final_text,
                    "followup_question": "",
                    "style": "warm",
                    "source": care_reply.source,
                    "detail": care_reply.detail,
                    "ai_ready": care_reply.ai_ready,
                    "timing": care_reply.timing,
                },
            )
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/llm/daily_summary", response_model=DailySummaryResponse)
async def llm_daily_summary(
    payload: DailySummaryRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    conn: Connection = Depends(get_db),
) -> DailySummaryResponse:
    user = _parse_access_token_for_local_desktop(credentials, conn, request)
    fallback_summary = "今天没有记录到明显情绪事件。需要的话，随时可以和我聊聊。"
    fallback_highlights = [
        "暂无明显触发事件",
        "整体状态较平稳",
        "需要时可随时记录感受",
    ]
    try:
        summary_text = await assistant_service.gateway.send_message(
            f"desktop:{int(user['id'])}:daily-summary",
            "请根据以下事件生成今天的情绪总结。第一行给 summary，后面 3 行每行一个 highlight。\n"
            f"{json.dumps(payload.events, ensure_ascii=False)}",
        )
    except Exception:
        summary_text = ""
    if not summary_text.strip():
        return DailySummaryResponse(summary=fallback_summary, highlights=fallback_highlights)
    lines = [line.strip("- ").strip() for line in summary_text.splitlines() if line.strip()]
    summary = lines[0] if lines else fallback_summary
    highlights = lines[1:4] if len(lines) > 1 else fallback_highlights
    return DailySummaryResponse(
        summary=summary,
        highlights=highlights or fallback_highlights,
    )

def _row_to_event(row: Dict) -> Dict:
    return {
        "id": row.get("id"),
        "timestamp_ms": row.get("timestamp_ms"),
        "type": row.get("type"),
        "description": row.get("description"),
        "V": row.get("v"),
        "A": row.get("a"),
        "T": row.get("t"),
        "S": row.get("s"),
        "intensity": row.get("intensity"),
        "source": row.get("source"),
    }


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await event_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("LOCAL_BACKEND_PORT", "8000")))

