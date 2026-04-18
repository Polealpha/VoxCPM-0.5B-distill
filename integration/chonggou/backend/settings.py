from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


AUTH_SECRET_KEY = _env("AUTH_SECRET_KEY", "change-this-secret")
AUTH_ALGORITHM = _env("AUTH_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_SEC = int(_env("ACCESS_TOKEN_EXPIRE_SEC", "900"))  # 15 min
REFRESH_TOKEN_EXPIRE_SEC = int(_env("REFRESH_TOKEN_EXPIRE_SEC", "1209600"))  # 14 days

DB_PATH = _env("AUTH_DB_PATH", "backend/auth.db")

ALLOWED_ORIGINS = _env("AUTH_CORS_ORIGINS", "*")

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent


def _resolve_default_openclaw_repo() -> Path:
    candidates = [
        _REPO_ROOT.parent / "openclaw",
        _REPO_ROOT / ".openclaw-latest" / "node_modules" / "openclaw",
        _REPO_ROOT / "app windows" / "vendor" / "openclaw-runtime",
    ]
    for candidate in candidates:
        try:
            if (candidate / "openclaw.mjs").exists():
                return candidate
        except Exception:
            continue
    return candidates[0]


_DEFAULT_OPENCLAW_REPO = _resolve_default_openclaw_repo()
_DEFAULT_OPENCLAW_WORKSPACE = _REPO_ROOT / "assistant_data" / "openclaw_workspace"
_DEFAULT_OPENCLAW_STATE = _REPO_ROOT / "assistant_data" / "openclaw_state"
_DEFAULT_OPENCLAW_GATEWAY_PORT = 18940
_DEFAULT_OPENCLAW_GATEWAY_URL = f"ws://127.0.0.1:{_DEFAULT_OPENCLAW_GATEWAY_PORT}"
_DEFAULT_OPENCLAW_GATEWAY_ORIGIN = f"http://127.0.0.1:{_DEFAULT_OPENCLAW_GATEWAY_PORT}"
_DEFAULT_OPENCLAW_DESKTOP_SHARED_SESSION_KEY = "agent:main:desktop:emoresonance"
_LOCAL_RUNTIME_ROOT = Path(
    _env(
        "EMORESONANCE_RUNTIME_HOME",
        str(
            Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(_REPO_ROOT / "assistant_data"))
            / "EmoResonance"
        ),
    )
)
_DEFAULT_OPENCLAW_CODEX_HOME = _LOCAL_RUNTIME_ROOT / "codex_home"

OPENCLAW_REPO_PATH = _env("OPENCLAW_REPO_PATH", str(_DEFAULT_OPENCLAW_REPO))
OPENCLAW_WORKSPACE_DIR = _env("OPENCLAW_WORKSPACE_DIR", str(_DEFAULT_OPENCLAW_WORKSPACE))
OPENCLAW_STATE_DIR = _env("OPENCLAW_STATE_DIR", str(_DEFAULT_OPENCLAW_STATE))
OPENCLAW_CODEX_HOME = _env("OPENCLAW_CODEX_HOME", str(_DEFAULT_OPENCLAW_CODEX_HOME))
OPENCLAW_GATEWAY_URL = _env("OPENCLAW_GATEWAY_URL", _DEFAULT_OPENCLAW_GATEWAY_URL)
OPENCLAW_GATEWAY_ORIGIN = _env("OPENCLAW_GATEWAY_ORIGIN", _DEFAULT_OPENCLAW_GATEWAY_ORIGIN)
OPENCLAW_PROXY_URL = _env("OPENCLAW_PROXY_URL", "")
OPENCLAW_TIMEOUT_MS = int(_env("OPENCLAW_TIMEOUT_MS", "120000"))
OPENCLAW_CLIENT_ID = _env("OPENCLAW_CLIENT_ID", "gateway-client")
OPENCLAW_CLIENT_MODE = _env("OPENCLAW_CLIENT_MODE", "backend")
OPENCLAW_DESKTOP_SHARED_SESSION_KEY = _env(
    "OPENCLAW_DESKTOP_SHARED_SESSION_KEY",
    _DEFAULT_OPENCLAW_DESKTOP_SHARED_SESSION_KEY,
)
ASSISTANT_BRIDGE_TOKEN = _env("ASSISTANT_BRIDGE_TOKEN", "change-this-bridge-token")
ASSISTANT_BRIDGE_USER_ID = int(_env("ASSISTANT_BRIDGE_USER_ID", "1"))
DESKTOP_APP_ALLOWLIST_JSON = _env(
    "DESKTOP_APP_ALLOWLIST_JSON",
    '{"notepad":["notepad"],"calc":["calc"],"explorer":["explorer"],"vscode":["code"],"chrome":["cmd","/c","start","chrome"],"edge":["cmd","/c","start","msedge"]}',
)
OPENCLAW_PREFERRED_MODE = _env("OPENCLAW_PREFERRED_MODE", "agent")
OPENCLAW_PREFERRED_CODE_MODEL = _env("OPENCLAW_PREFERRED_CODE_MODEL", "glm-5")
DESKTOP_STT_PROVIDER = _env("DESKTOP_STT_PROVIDER", "faster_whisper")
DESKTOP_STT_FALLBACK_PROVIDER = _env("DESKTOP_STT_FALLBACK_PROVIDER", "sherpa_onnx")
DESKTOP_STT_MODEL_NAME = _env("DESKTOP_STT_MODEL_NAME", "distil-large-v3")
DESKTOP_STT_LANGUAGE = _env("DESKTOP_STT_LANGUAGE", "zh")
DESKTOP_STT_DEVICE = _env("DESKTOP_STT_DEVICE", "cpu")
DESKTOP_STT_COMPUTE_TYPE = _env("DESKTOP_STT_COMPUTE_TYPE", "int8")
DESKTOP_STT_BEAM_SIZE = int(_env("DESKTOP_STT_BEAM_SIZE", "8"))
DESKTOP_STT_BEST_OF = int(_env("DESKTOP_STT_BEST_OF", "5"))
DESKTOP_STT_PATIENCE = float(_env("DESKTOP_STT_PATIENCE", "1.2"))
DESKTOP_STT_REPETITION_PENALTY = float(_env("DESKTOP_STT_REPETITION_PENALTY", "1.05"))
DESKTOP_STT_NO_SPEECH_THRESHOLD = float(_env("DESKTOP_STT_NO_SPEECH_THRESHOLD", "0.45"))
DESKTOP_STT_LOG_PROB_THRESHOLD = float(_env("DESKTOP_STT_LOG_PROB_THRESHOLD", "-0.7"))
DESKTOP_STT_COMPRESSION_RATIO_THRESHOLD = float(_env("DESKTOP_STT_COMPRESSION_RATIO_THRESHOLD", "2.2"))
DESKTOP_STT_CHUNK_LENGTH = int(_env("DESKTOP_STT_CHUNK_LENGTH", "18"))
DESKTOP_STT_INITIAL_PROMPT = _env(
    "DESKTOP_STT_INITIAL_PROMPT",
    "以下是中文陪伴机器人、电脑端助手与主人的自然口语对话，请尽量准确输出中文标点、人名、设备名和控制指令。",
)
DESKTOP_STT_HOTWORDS = _env(
    "DESKTOP_STT_HOTWORDS",
    "OpenClaw,共感智能,小念,树莓派,云台,主人,网易云音乐,机器人,设置页,主动关怀",
)
DESKTOP_STT_VAD_FILTER = _env("DESKTOP_STT_VAD_FILTER", "1").strip().lower() not in {"0", "false", "no"}
DESKTOP_STT_MAX_SEC = int(_env("DESKTOP_STT_MAX_SEC", "45"))
DESKTOP_STT_NUM_THREADS = int(_env("DESKTOP_STT_NUM_THREADS", "4"))
DESKTOP_STT_PREPROCESS = _env("DESKTOP_STT_PREPROCESS", "1").strip().lower() not in {"0", "false", "no"}
DESKTOP_STT_TRIM_SILENCE = _env("DESKTOP_STT_TRIM_SILENCE", "1").strip().lower() not in {"0", "false", "no"}
DESKTOP_STT_SILENCE_THRESHOLD = int(_env("DESKTOP_STT_SILENCE_THRESHOLD", "320"))
DESKTOP_STT_TARGET_PEAK = int(_env("DESKTOP_STT_TARGET_PEAK", "24000"))
DEFAULT_ROBOT_DEVICE_IP = _env("DEFAULT_ROBOT_DEVICE_IP", "192.168.137.50")
ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS = _env("ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
