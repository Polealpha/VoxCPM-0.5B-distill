from __future__ import annotations

import asyncio
import ctypes
import socket
import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlite3 import Connection
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.parse import urlparse

import httpx

from .assistant_store import AssistantWorkspaceStore
from .care_prompts import ASSISTANT_PRODUCT_PROMPT, CARE_RESPONSE_RULES, CARE_SYSTEM_PROMPT
from .openclaw_gateway import (
    OpenClawGatewayClient,
    OpenClawGatewayConfig,
    OpenClawGatewayError,
    discover_openclaw_state_dir,
    resolve_openclaw_proxy_url,
)
from .settings import (
    DEFAULT_ROBOT_DEVICE_IP,
    DESKTOP_APP_ALLOWLIST_JSON,
    OPENCLAW_CLIENT_ID,
    OPENCLAW_CLIENT_MODE,
    OPENCLAW_CODEX_HOME,
    OPENCLAW_DESKTOP_SHARED_SESSION_KEY,
    OPENCLAW_GATEWAY_ORIGIN,
    OPENCLAW_REPO_PATH,
    OPENCLAW_GATEWAY_URL,
    OPENCLAW_STATE_DIR,
    OPENCLAW_TIMEOUT_MS,
    OPENCLAW_WORKSPACE_DIR,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def normalize_surface(surface: str) -> str:
    raw = str(surface or "desktop").strip().lower()
    return raw if raw in {"desktop", "mobile", "wecom", "robot"} else "desktop"


def build_session_key(
    surface: str,
    user_id: int,
    explicit: Optional[str] = None,
    device_id: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    normalized = normalize_surface(surface)
    shared_desktop_session = str(OPENCLAW_DESKTOP_SHARED_SESSION_KEY or "").strip()
    if normalized == "desktop" and shared_desktop_session:
        return shared_desktop_session
    if normalized == "wecom":
        return f"wecom:{str(sender_id or user_id).strip()}"
    if normalized == "robot":
        return f"robot:{str(device_id or user_id).strip()}"
    return f"{normalized}:{int(user_id)}"


@dataclass
class ToolExecutionResult:
    name: str
    ok: bool
    detail: str
    data: Dict[str, object]


class AssistantService:
    def __init__(self) -> None:
        self.workspace_dir = self._resolve_workspace_dir()
        self.desktop_session_ready = False
        self.desktop_session_detail = "desktop_session_unwarmed"
        self.desktop_session_checked_ms = 0
        self.store = AssistantWorkspaceStore(self.workspace_dir)
        self.gateway = OpenClawGatewayClient(
            OpenClawGatewayConfig(
                state_dir=OPENCLAW_STATE_DIR,
                workspace_dir=self.workspace_dir,
                codex_home=OPENCLAW_CODEX_HOME,
                repo_path=OPENCLAW_REPO_PATH,
                url=OPENCLAW_GATEWAY_URL,
                origin=OPENCLAW_GATEWAY_ORIGIN,
                timeout_ms=OPENCLAW_TIMEOUT_MS,
                client_id=OPENCLAW_CLIENT_ID,
                client_mode=OPENCLAW_CLIENT_MODE,
            )
        )
        self.gateway_fast = OpenClawGatewayClient(
            OpenClawGatewayConfig(
                state_dir=OPENCLAW_STATE_DIR,
                workspace_dir=self.workspace_dir,
                codex_home=OPENCLAW_CODEX_HOME,
                repo_path=OPENCLAW_REPO_PATH,
                url=OPENCLAW_GATEWAY_URL,
                origin=OPENCLAW_GATEWAY_ORIGIN,
                timeout_ms=OPENCLAW_TIMEOUT_MS,
                client_id=OPENCLAW_CLIENT_ID,
                client_mode=OPENCLAW_CLIENT_MODE,
                prefer_websocket_first=True,
                allow_agent_fallback=False,
            )
        )
        try:
            self.app_allowlist = json.loads(DESKTOP_APP_ALLOWLIST_JSON)
        except Exception:
            self.app_allowlist = {}

    def _set_desktop_session_health(self, ok: bool, detail: str) -> None:
        self.desktop_session_ready = bool(ok)
        self.desktop_session_detail = str(detail or "").strip()
        self.desktop_session_checked_ms = _now_ms()

    def _resolve_workspace_dir(self) -> str:
        configured = Path(str(OPENCLAW_WORKSPACE_DIR or "")).expanduser().resolve()
        candidates = [configured]
        try:
            state_dir = discover_openclaw_state_dir(OPENCLAW_STATE_DIR, str(configured))
            runtime_workspace = (state_dir / "workspace").resolve()
            candidates.append(runtime_workspace)
        except Exception:
            pass
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(configured)

    async def _send_via_gateway(
        self,
        session_key: str,
        message: str,
        *,
        prefer_fast_path: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> str:
        if prefer_fast_path and timeout_ms is None:
            timeout_ms = 30000
        gateway = self.gateway_fast if prefer_fast_path else self.gateway
        if prefer_fast_path:
            gateway = OpenClawGatewayClient(
                OpenClawGatewayConfig(
                    state_dir=OPENCLAW_STATE_DIR,
                    workspace_dir=self.workspace_dir,
                    codex_home=OPENCLAW_CODEX_HOME,
                    repo_path=OPENCLAW_REPO_PATH,
                    url=OPENCLAW_GATEWAY_URL,
                    origin=OPENCLAW_GATEWAY_ORIGIN,
                    timeout_ms=OPENCLAW_TIMEOUT_MS,
                    client_id=OPENCLAW_CLIENT_ID,
                    client_mode=OPENCLAW_CLIENT_MODE,
                    prefer_websocket_first=True,
                    allow_agent_fallback=False,
                )
            )
        attempts = 2 if prefer_fast_path else 1
        last_error: Optional[OpenClawGatewayError] = None
        for attempt in range(attempts):
            try:
                return await gateway.send_message(session_key, message, timeout_ms=timeout_ms)
            except OpenClawGatewayError as exc:
                last_error = exc
                detail = str(exc).lower()
                is_retryable = (
                    "timed out during opening handshake" in detail
                    or "session id not found" in detail
                    or "not paired" in detail
                    or "gateway runtime probe failed" in detail
                )
                if (not is_retryable) or attempt >= attempts - 1:
                    raise
                await asyncio.sleep(1.0 + attempt * 1.5)
                if prefer_fast_path:
                    gateway = OpenClawGatewayClient(
                        OpenClawGatewayConfig(
                            state_dir=OPENCLAW_STATE_DIR,
                            workspace_dir=self.workspace_dir,
                            codex_home=OPENCLAW_CODEX_HOME,
                            repo_path=OPENCLAW_REPO_PATH,
                            url=OPENCLAW_GATEWAY_URL,
                            origin=OPENCLAW_GATEWAY_ORIGIN,
                            timeout_ms=OPENCLAW_TIMEOUT_MS,
                            client_id=OPENCLAW_CLIENT_ID,
                            client_mode=OPENCLAW_CLIENT_MODE,
                            prefer_websocket_first=True,
                            allow_agent_fallback=False,
                        )
                    )
        if last_error is not None:
            raise last_error
        raise OpenClawGatewayError("OpenClaw send failed without a captured error")

    async def send_message(
        self,
        conn: Connection,
        user_id: int,
        text: str,
        surface: str,
        session_key: Optional[str] = None,
        device_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        attachments: Optional[List[dict]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        normalized_surface = normalize_surface(surface)
        entrypoint = str((metadata or {}).get("entrypoint") or "").strip().lower() if isinstance(metadata, dict) else ""
        assistant_mode = self._resolve_assistant_mode(metadata)
        if normalized_surface == "desktop" and entrypoint == "desktop_chat" and assistant_mode != "agent":
            assistant_mode = "agent"
        native_control_enabled = self._resolve_native_control_enabled(metadata)
        prefer_fast_gateway = normalized_surface == "desktop" and entrypoint == "desktop_chat"
        exact_reply_target = self._extract_exact_reply_target(text)
        resolved_session_key = build_session_key(
            normalized_surface,
            user_id=user_id,
            explicit=session_key,
            device_id=device_id,
            sender_id=sender_id,
        )
        gateway_session_key = resolved_session_key
        tool_results = await self._run_explicit_tools(
            conn,
            user_id,
            text,
            device_id=device_id,
            assistant_mode=assistant_mode,
            native_control_enabled=native_control_enabled,
        )
        if assistant_mode != "agent" and tool_results and self._should_short_circuit_tool_reply(text):
            reply = self._compose_tool_only_reply(tool_results)
        else:
            try:
                if exact_reply_target and not tool_results and not attachments:
                    reply = await self._send_via_gateway(
                        f"{gateway_session_key}:exact",
                        self._compose_exact_reply_message(exact_reply_target),
                        prefer_fast_path=prefer_fast_gateway,
                    )
                else:
                    if prefer_fast_gateway and not tool_results and not attachments:
                        message = self._compose_desktop_chat_fast_message(text, metadata)
                    else:
                        message = self._compose_openclaw_message(
                            text,
                            normalized_surface,
                            resolved_session_key,
                            tool_results,
                            attachments,
                            metadata,
                            assistant_mode,
                            native_control_enabled,
                        )
                    reply = await self._send_via_gateway(
                        gateway_session_key,
                        message,
                        prefer_fast_path=prefer_fast_gateway,
                    )
            except OpenClawGatewayError as exc:
                detail = str(exc or "").strip().lower()
                retryable_desktop_error = (
                    prefer_fast_gateway
                    and not tool_results
                    and not attachments
                    and ("timed out" in detail or "replyrunalreadyactive" in detail or "already active" in detail)
                )
                if not retryable_desktop_error:
                    raise
                try:
                    await self.gateway.reset_session(resolved_session_key)
                except Exception:
                    pass
                reply = await self._send_via_gateway(
                    gateway_session_key,
                    self._compose_desktop_chat_fast_message(text, metadata),
                    prefer_fast_path=True,
                    timeout_ms=30000,
                )
            reply = self._sanitize_gateway_reply(reply)
            if not tool_results and not str(reply or "").strip():
                reply = await self._send_via_gateway(
                    f"{gateway_session_key}:rewrite",
                    self._compose_retry_message(
                        text,
                        normalized_surface,
                        assistant_mode,
                        native_control_enabled,
                    ),
                    prefer_fast_path=prefer_fast_gateway,
                )
                reply = self._sanitize_gateway_reply(reply)
            if exact_reply_target and reply != exact_reply_target:
                reply = await self._send_via_gateway(
                    f"{gateway_session_key}:exact-retry",
                    self._compose_exact_reply_message(exact_reply_target),
                    prefer_fast_path=prefer_fast_gateway,
                )
                reply = self._sanitize_gateway_reply(reply)
            if not tool_results and self._reply_is_false_heartbeat(reply, text):
                reply = await self._send_via_gateway(
                    f"{gateway_session_key}:retry",
                    self._compose_retry_message(
                        text,
                        normalized_surface,
                        assistant_mode,
                        native_control_enabled,
                    ),
                    prefer_fast_path=prefer_fast_gateway,
                )
                reply = self._sanitize_gateway_reply(reply)
            if (
                prefer_fast_gateway
                and not tool_results
                and not attachments
                and self._looks_like_underformed_desktop_reply(reply, text)
            ):
                reply = await self._send_via_gateway(
                    gateway_session_key,
                    self._compose_desktop_chat_repair_message(text),
                    prefer_fast_path=True,
                    timeout_ms=30000,
                )
                reply = self._sanitize_gateway_reply(reply)
            if assistant_mode == "agent" and self._looks_like_setup_or_internal_reply(reply):
                reply = await self._send_via_gateway(
                    f"{gateway_session_key}:repair",
                    self._compose_retry_message(
                        text,
                        normalized_surface,
                        assistant_mode,
                        native_control_enabled,
                    ),
                    prefer_fast_path=prefer_fast_gateway,
                )
                reply = self._sanitize_gateway_reply(reply)
            if assistant_mode == "agent" and (
                self._looks_like_setup_or_internal_reply(reply)
                or (
                    self._should_short_circuit_tool_reply(text)
                    and not tool_results
                    and (
                        self._reply_lacks_execution_signal(reply)
                        or self._reply_indicates_blocked_execution(reply)
                    )
                )
            ):
                fallback_results = await self._run_explicit_tools(
                    conn,
                    user_id,
                    text,
                    device_id=device_id,
                    assistant_mode="product",
                    native_control_enabled=True,
                )
                if fallback_results:
                    tool_results = fallback_results
                    reply = self._compose_tool_only_reply(tool_results)
        return {
            "surface": normalized_surface,
            "session_key": resolved_session_key,
            "text": reply.strip(),
            "tool_results": [result.__dict__ for result in tool_results],
            "timestamp_ms": _now_ms(),
        }

    async def reset_session(
        self,
        user_id: int,
        surface: str,
        session_key: Optional[str],
        device_id: Optional[str],
        sender_id: Optional[str],
    ) -> Dict[str, object]:
        resolved_surface = normalize_surface(surface)
        resolved_key = build_session_key(
            resolved_surface,
            user_id=user_id,
            explicit=session_key,
            device_id=device_id,
            sender_id=sender_id,
        )
        await self.gateway.reset_session(resolved_key)
        return {"surface": resolved_surface, "session_key": resolved_key}

    def get_session_status(
        self,
        conn: Connection,
        user_id: int,
        surface: str,
        session_key: Optional[str],
        device_id: Optional[str],
        sender_id: Optional[str],
    ) -> Dict[str, object]:
        resolved_surface = normalize_surface(surface)
        resolved_key = build_session_key(
            resolved_surface,
            user_id=user_id,
            explicit=session_key,
            device_id=device_id,
            sender_id=sender_id,
        )
        row = conn.execute(
            """
            SELECT COUNT(*) AS count, MAX(timestamp_ms) AS last_message_ts_ms
            FROM chat_messages
            WHERE user_id = ? AND session_key = ?
            """,
            (int(user_id), resolved_key),
        ).fetchone()
        return {
            "surface": resolved_surface,
            "session_key": resolved_key,
            "message_count": int(row["count"] or 0) if row else 0,
            "last_message_ts_ms": int(row["last_message_ts_ms"] or 0) if row and row["last_message_ts_ms"] is not None else None,
        }

    def list_todos(self, user_id: int, state: Optional[str] = None) -> List[Dict[str, object]]:
        return self.store.list_todos(user_id, state=state)

    def create_todo(
        self,
        user_id: int,
        title: str,
        details: str = "",
        due_at_ms: Optional[int] = None,
        tags: Optional[List[str]] = None,
        action: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        return self.store.create_todo(
            user_id,
            title=title,
            details=details,
            due_at_ms=due_at_ms,
            tags=tags,
            action=action,
        )

    def update_todo(self, user_id: int, todo_id: str, changes: Dict[str, object]) -> Dict[str, object]:
        return self.store.update_todo(user_id, todo_id, changes)

    def claim_due_todos(self, user_id: int, now_ms: Optional[int] = None, limit: int = 10) -> List[Dict[str, object]]:
        return self.store.claim_due_todos(user_id, now_ms=now_ms, limit=limit)

    def search_memory(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, object]]:
        return self.store.search_memory(user_id, query, limit=limit)

    def get_profile_memory_summary(self, user_id: int, max_chars: int = 1200) -> str:
        return self.store.get_profile_memory_summary(user_id, max_chars=max_chars)

    def runtime_status(self) -> Dict[str, object]:
        try:
            state_dir = discover_openclaw_state_dir(OPENCLAW_STATE_DIR, self.workspace_dir)
            gateway_ready, gateway_error = self.gateway.probe_connection_blocking(
                force=True,
                timeout_ms=4000,
            )
            if gateway_ready:
                gateway_error = ""
            resolved_state_dir = str(state_dir)
        except OpenClawGatewayError as exc:
            gateway_ready = False
            gateway_error = str(exc)
            resolved_state_dir = str(OPENCLAW_STATE_DIR or "")
        except Exception as exc:
            gateway_ready = False
            gateway_error = f"OpenClaw runtime probe failed: {exc}"
            resolved_state_dir = str(OPENCLAW_STATE_DIR or "")
        provider_network_ok, provider_network_detail = self._probe_provider_network()
        if provider_network_ok:
            provider_network_detail = ""
        desktop_session_ready = bool(self.desktop_session_ready)
        desktop_session_detail = str(self.desktop_session_detail or "")
        return {
            "gateway_ready": gateway_ready,
            "gateway_error": gateway_error,
            "provider_network_ok": provider_network_ok,
            "provider_network_detail": provider_network_detail,
            "desktop_session_ready": desktop_session_ready,
            "desktop_session_detail": desktop_session_detail,
            "desktop_session_checked_ms": int(self.desktop_session_checked_ms or 0),
            "ai_ready": bool(gateway_ready and provider_network_ok and desktop_session_ready),
            "state_dir": resolved_state_dir,
            "workspace_dir": str(self.workspace_dir),
            "desktop_tools": [
            "desktop.launch_app",
            "desktop.open_url",
            "desktop.web_search",
            "desktop.play_music",
            "desktop.music_pause",
            "desktop.music_play_pause",
            "desktop.music_next",
            "desktop.music_previous",
            "desktop.todo_create",
            "desktop.write_note",
                "robot.get_status",
                "robot.speak",
                "robot.pan_tilt",
                "robot.start_owner_enrollment",
                "robot.get_preview",
            ],
            "robot_bridge_ready": True,
        }

    async def warmup_desktop_session(self) -> Dict[str, object]:
        warmup_session_key = f"{build_session_key('desktop', user_id=0)}:warmup"
        last_detail = ""
        for attempt in range(2):
            try:
                reply = await self._send_via_gateway(
                    warmup_session_key,
                    self._compose_desktop_chat_fast_message("你好", {"current_emotion": "calm"}),
                    prefer_fast_path=True,
                    timeout_ms=30000,
                )
                reply = self._sanitize_gateway_reply(reply)
                if str(reply or "").strip():
                    self._set_desktop_session_health(True, "desktop_session_warmup_ready")
                    return {
                        "ok": True,
                        "session_key": warmup_session_key,
                        "reply": reply,
                        "attempt": attempt + 1,
                    }
                last_detail = "empty_reply"
            except Exception as exc:
                last_detail = str(exc)
            try:
                await self.gateway.reset_session(warmup_session_key)
            except Exception:
                pass
            await asyncio.sleep(1.5 + attempt * 1.5)
        self._set_desktop_session_health(False, last_detail or "desktop_session_warmup_failed")
        return {
            "ok": False,
            "session_key": warmup_session_key,
            "detail": last_detail,
        }

    def list_wechat_mirror_messages(self, session_key: str, limit: int = 200) -> List[Dict[str, object]]:
        try:
            state_dir = discover_openclaw_state_dir(OPENCLAW_STATE_DIR, self.workspace_dir)
            transcript_path = self._resolve_openclaw_session_file(state_dir, session_key)
            if transcript_path is None or not transcript_path.exists():
                return []
            items: List[Dict[str, object]] = []
            active_wechat_turn = False
            with transcript_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for raw_line in fh:
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if str(payload.get("type") or "") != "message":
                        continue
                    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
                    role = str(message.get("role") or "").strip().lower()
                    content = message.get("content") if isinstance(message.get("content"), list) else []
                    timestamp_ms = self._resolve_transcript_timestamp_ms(payload, message)
                    if role == "user":
                        text = self._extract_wechat_user_text(content)
                        active_wechat_turn = bool(text)
                        if text:
                            items.append(
                                {
                                    "sender": "user",
                                    "text": text,
                                    "timestamp_ms": timestamp_ms,
                                }
                            )
                        continue
                    if role != "assistant" or not active_wechat_turn:
                        continue
                    text = self._extract_transcript_text(content)
                    if not text:
                        continue
                    items.append(
                        {
                            "sender": "assistant",
                            "text": text,
                            "timestamp_ms": timestamp_ms,
                        }
                    )
            if len(items) > limit:
                items = items[-limit:]
            return items
        except Exception:
            return []

    def _resolve_openclaw_session_file(self, state_dir: Path, session_key: str) -> Optional[Path]:
        raw_key = str(session_key or "").strip()
        if not raw_key:
            return None
        agent_id = "main"
        if raw_key.startswith("agent:"):
            parts = raw_key.split(":")
            if len(parts) >= 2 and parts[1].strip():
                agent_id = parts[1].strip()
        sessions_path = state_dir / "agents" / agent_id / "sessions" / "sessions.json"
        if not sessions_path.exists():
            return None
        try:
            payload = json.loads(sessions_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        entry = payload.get(raw_key)
        if not isinstance(entry, dict):
            return None
        session_file = str(entry.get("sessionFile") or "").strip()
        if session_file:
            return Path(session_file).expanduser().resolve()
        session_id = str(entry.get("sessionId") or "").strip()
        if not session_id:
            return None
        return (sessions_path.parent / f"{session_id}.jsonl").resolve()

    @staticmethod
    def _resolve_transcript_timestamp_ms(payload: Dict[str, object], message: Dict[str, object]) -> int:
        try:
            nested = message.get("timestamp")
            if nested is not None:
                return int(nested)
        except Exception:
            pass
        stamp = str(payload.get("timestamp") or "").strip()
        if stamp:
            try:
                return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)
            except Exception:
                pass
        return _now_ms()

    @staticmethod
    def _extract_transcript_text(content: List[object]) -> str:
        texts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "") != "text":
                continue
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
        return "\n\n".join(texts).strip()

    def _extract_wechat_user_text(self, content: List[object]) -> str:
        raw = self._extract_transcript_text(content)
        if not raw:
            return ""
        if "openclaw-weixin:" not in raw:
            return ""
        stripped = re.sub(
            r"^Conversation info \(untrusted metadata\):\s*```json.*?```\s*",
            "",
            raw,
            flags=re.DOTALL,
        ).strip()
        return stripped

    def _probe_provider_network(self) -> Tuple[bool, str]:
        proxy_url = resolve_openclaw_proxy_url()
        if proxy_url:
            parsed = urlparse(proxy_url)
            proxy_host = parsed.hostname
            proxy_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if not proxy_host:
                return False, f"configured proxy is invalid: {proxy_url}"
            try:
                with socket.create_connection((proxy_host, int(proxy_port)), timeout=1.5):
                    return True, f"provider traffic routed via proxy: {proxy_url}"
            except OSError as exc:
                return False, f"configured proxy unreachable: {proxy_url} ({exc})"
        targets = [("open.bigmodel.cn", 443)]
        try:
            state_dir = discover_openclaw_state_dir(OPENCLAW_STATE_DIR, self.workspace_dir)
            cfg_path = state_dir / "openclaw.json"
            if cfg_path.exists():
                payload = json.loads(cfg_path.read_text(encoding="utf-8", errors="ignore"))
                primary = str((((payload.get("agents") or {}).get("defaults") or {}).get("model") or {}).get("primary") or "").strip().lower()
                if primary.startswith("zai/"):
                    targets = [("open.bigmodel.cn", 443)]
        except Exception:
            pass
        errors: List[str] = []
        for host, port in targets:
            try:
                with socket.create_connection((host, int(port)), timeout=1.5):
                    return True, f"provider endpoint reachable: {host}:{port}"
            except OSError as exc:
                errors.append(f"{host}:{port} ({exc})")
        return False, "provider endpoints unreachable: " + "; ".join(errors)

    def _probe_gateway_socket(self, url: str) -> Tuple[bool, str]:
        parsed = urlparse(str(url or "").strip())
        host = parsed.hostname
        port = parsed.port
        if not host or port is None:
            return False, f"Invalid OpenClaw gateway url: {url}"
        try:
            with socket.create_connection((host, int(port)), timeout=1.5):
                return True, ""
        except OSError as exc:
            return False, f"OpenClaw gateway unreachable at {host}:{port} ({exc})"

    def _compose_openclaw_message(
        self,
        text: str,
        surface: str,
        session_key: str,
        tool_results: List[ToolExecutionResult],
        attachments: Optional[List[dict]],
        metadata: Optional[Dict[str, object]],
        assistant_mode: str = "product",
        native_control_enabled: bool = True,
    ) -> str:
        lines = []
        care_channel = ""
        memory_summary = ""
        entrypoint = ""
        current_emotion = ""
        expression_label = ""
        expression_confidence = 0.0
        if isinstance(metadata, dict):
            care_channel = str(metadata.get("care_channel") or "").strip().lower()
            memory_summary = str(metadata.get("memory_summary") or "").strip()
            entrypoint = str(metadata.get("entrypoint") or "").strip().lower()
            current_emotion = str(metadata.get("current_emotion") or "").strip()
            expression_label = str(metadata.get("expression_label") or "").strip()
            try:
                expression_confidence = float(metadata.get("expression_confidence") or 0.0)
            except Exception:
                expression_confidence = 0.0
        if str(assistant_mode or "").strip().lower() == "agent":
            lines.append("[assistant_mode=agent]")
            lines.append(f"[assistant_native_control={str(bool(native_control_enabled)).lower()}]")
            if entrypoint == "desktop_chat":
                lines.append("Answer briefly and naturally in Chinese.")
                lines.append("Only use desktop/browser/robot tools if the user clearly asks you to operate something.")
            else:
                lines.append(
                    "Agent mode is enabled. Prefer native OpenClaw execution first. "
                    "For ordinary chat, answer briefly and naturally in Chinese. "
                    "Only use desktop/browser/robot tools when the user clearly asks for an operation."
                )
            if current_emotion:
                lines.append(f"Current emotion signal: {current_emotion}")
            if expression_label and expression_label.lower() != "unknown":
                lines.append(
                    f"Expression hint: {expression_label} (confidence={expression_confidence:.2f})"
                )
            if memory_summary:
                lines.append("Long-term memory summary:")
                lines.append(memory_summary[:240])
            if tool_results:
                lines.append(
                    "Available tools: desktop.launch_app, desktop.open_url, desktop.web_search, "
                    "desktop.play_music, desktop.music_pause, desktop.music_play_pause, desktop.music_next, "
                    "desktop.music_previous, desktop.todo_create, desktop.write_note, robot.get_status, "
                    "robot.speak, robot.pan_tilt, robot.start_owner_enrollment, robot.get_preview."
                )
        elif care_channel == "proactive_care":
            lines.extend(self._compose_proactive_care_block(text, metadata))
        else:
            lines.extend(
                [
                    ASSISTANT_PRODUCT_PROMPT,
                    "请直接处理这次用户请求；如果只是普通聊天，就正常聊天；如果用户明显难受，再轻量关怀。",
                    "不要要求用户重复发送，除非输入真的为空。",
                ]
            )
        for item in tool_results:
            lines.append(f"Tool result [{item.name}] ok={str(item.ok).lower()}: {item.detail}")
        if memory_summary and care_channel != "proactive_care" and assistant_mode != "agent":
            lines.append("Long-term memory summary:")
            lines.append(memory_summary)
        if attachments:
            lines.append(f"Attachments: {json.dumps(attachments, ensure_ascii=False)}")
        if metadata and care_channel != "proactive_care" and assistant_mode != "agent":
            lines.append(f"Metadata: {json.dumps(metadata, ensure_ascii=False)}")
        if care_channel != "proactive_care":
            lines.append(str(text or "").strip())
        if self._contains_high_risk_distress(text):
            lines.extend(
                [
                    "Important safety rule: the user's latest message contains clear high-risk distress wording.",
                    "Do not reply with a bare list of URLs, source links, or hotline pages.",
                    "First acknowledge the pain, then ask one short safety-check question, then give one small concrete next step.",
                    "If you mention support resources, mention them naturally in one sentence only after the emotional response.",
                ]
            )
        return "\n".join(lines).strip()

    def _compose_proactive_care_block(self, text: str, metadata: Optional[Dict[str, object]]) -> List[str]:
        payload = dict(metadata or {})
        context = {
            "project": "共鸣连接",
            "channel": "主动关怀",
            "current_emotion": str(payload.get("current_emotion") or "").strip(),
            "current_ts_ms": payload.get("current_ts_ms"),
            "history": payload.get("history") if isinstance(payload.get("history"), list) else [],
            "memory_summary": str(payload.get("memory_summary") or "").strip(),
            "expression_modality": {
                "label": str(payload.get("expression_label") or "unknown").strip() or "unknown",
                "confidence": float(payload.get("expression_confidence") or 0.0),
                "note": "这是算法观测信号，不等于用户明确确认。",
            },
            "user_profile": payload.get("user_profile") if isinstance(payload.get("user_profile"), dict) else {},
            "user_message": str(text or "").strip(),
        }
        return [
            "[care_channel=proactive_care]",
            CARE_SYSTEM_PROMPT,
            CARE_RESPONSE_RULES,
            "以下是这次主动关怀需要使用的上下文。只输出给用户的话，不要复述这些字段，不要输出内部说明：",
            json.dumps(context, ensure_ascii=False),
        ]

    def _compose_exact_reply_message(self, exact_reply_target: str) -> str:
        return (
            "You must obey the user's instruction exactly. "
            f"Return exactly this string and nothing else: {exact_reply_target}"
        )

    def _compose_retry_message(
        self,
        text: str,
        surface: str,
        assistant_mode: str,
        native_control_enabled: bool,
    ) -> str:
        lines = [
            "This is a normal end-user chat request, not a heartbeat check.",
            "Do not reply HEARTBEAT_OK.",
            "Answer the user's request directly in concise Chinese.",
            "Do not output a bare list of links or source URLs.",
            str(text or "").strip(),
        ]
        if self._contains_high_risk_distress(text):
            lines.extend(
                [
                    "The user is expressing high-risk distress.",
                    "Respond first with emotional acknowledgement and one short safety-check question.",
                    "Do not dump hotline websites or source links as the main reply.",
                ]
            )
        return "\n".join(lines).strip()

    def _contains_high_risk_distress(self, text: str) -> bool:
        raw = str(text or "").strip().lower()
        if not raw:
            return False
        markers = [
            "想死",
            "不想活",
            "活不下去",
            "结束生命",
            "自杀",
            "轻生",
            "不如死了",
            "i want to die",
            "kill myself",
            "suicide",
        ]
        return any(marker in raw for marker in markers)

    def _compose_desktop_chat_fast_message(
        self,
        text: str,
        metadata: Optional[Dict[str, object]],
    ) -> str:
        raw_text = str(text or "").strip()
        current_emotion = ""
        if isinstance(metadata, dict):
            current_emotion = str(metadata.get("current_emotion") or "").strip()
        lines = [
            "你是心念双灵里的本地关怀助手。",
            "请直接用一到两句、简短、自然、完整的中文回答。",
            "不要输出链接、内部状态、emoji、英文模板或多余解释。",
            "如果用户在确认你是谁、在不在、能做什么，就直接说明你是本地关怀助手并自然回答。",
        ]
        if current_emotion:
            lines.append(f"当前情绪参考：{current_emotion}")
        lines.append(f"用户：{raw_text}")
        return "\n".join(lines).strip()
        history_lines: List[str] = []
        if isinstance(metadata, dict):
            current_emotion = str(metadata.get("current_emotion") or "").strip()
            history = metadata.get("history")
            if isinstance(history, list):
                for item in history[-4:]:
                    if not isinstance(item, dict):
                        continue
                    sender = "用户" if str(item.get("sender") or "").strip().lower() == "user" else "助手"
                    snippet = str(item.get("text") or "").strip()
                    if not snippet:
                        continue
                    history_lines.append(f"{sender}: {snippet[:48]}")
        lines = [
            "你是心念双灵里的本地关怀助手。",
            "请用简短、自然、完整的中文直接回复用户。",
            "不要输出英文、链接、内部状态、心跳词、emoji 或波浪号。",
            "如果用户问你是谁、在不在、正在做什么，要直接说明你是本地关怀助手，并自然回答。",
        ]
        if current_emotion:
            lines.append(f"当前情绪参考：{current_emotion}")
        if history_lines:
            lines.append("最近对话：")
            lines.extend(history_lines)
        lines.append(f"用户：{raw_text}")
        return "\n".join(lines).strip()

    def _compose_desktop_chat_repair_message(self, text: str) -> str:
        raw_text = str(text or "").strip()
        return "\n".join(
            [
                "请直接给用户一句完整、自然的中文答复。",
                "不要只回两三个字，不要输出链接、英文、系统状态或内部提示。",
                f"用户：{raw_text}",
            ]
        ).strip()
        return "\n".join(
            [
                "请用一整句自然、完整的中文直接回复用户。",
                "不要只回复两个字，不要输出英文、链接、内部状态或心跳词。",
                f"用户：{raw_text}",
            ]
        ).strip()

    def _looks_like_underformed_desktop_reply(self, reply: str, user_text: str) -> bool:
        raw_reply = str(reply or "").strip()
        if not raw_reply:
            return True
        compact = re.sub(r"\s+", "", raw_reply)
        suspicious = {
            "我是",
            "在",
            "嗯",
            "哈",
            "哈哈",
            "收到",
            "看起来",
            "有什么",
            "在想",
            "你好",
        }
        if compact in suspicious:
            return True
        if len(compact) <= 2:
            return True
        if len(compact) <= 12 and not re.search(r"[。！？!?…~]$", compact):
            return True
        if re.search(r"(你|我|他|她|它|们|的|了|呢|吗)$", compact) and len(compact) <= 16:
            return True
        raw_user = str(user_text or "").strip()
        if len(compact) <= 4 and any(token in raw_user for token in ["谁", "什么", "怎么", "为什么", "？", "?"]):
            return True
        return False

    def _resolve_assistant_mode(self, metadata: Optional[Dict[str, object]]) -> str:
        raw = ""
        if isinstance(metadata, dict):
            raw = str(metadata.get("assistant_mode") or "").strip().lower()
        return "agent" if raw == "agent" else "product"

    def _resolve_native_control_enabled(self, metadata: Optional[Dict[str, object]]) -> bool:
        if not isinstance(metadata, dict):
            return True
        raw = metadata.get("assistant_native_control", True)
        if isinstance(raw, str):
            return raw.strip().lower() not in {"0", "false", "off", "no"}
        return bool(raw)

    def _sanitize_gateway_reply(self, reply: str) -> str:
        raw = str(reply or "").strip()
        if not raw:
            return raw
        if self._looks_like_link_dump(raw):
            return ""
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return raw
        internal_markers = (
            "BOOTSTRAP.md",
            "HEARTBEAT.md",
            "USER.md",
            "MEMORY.md",
            "workspace context",
            "read-only session",
            "loading the workspace",
            "bootstrap flow",
            "\u4f1a\u8bdd\u4e0a\u4e0b\u6587",
            "\u4e0a\u4e0b\u6587\u6587\u4ef6",
            "\u5f15\u5bfc\u6587\u4ef6",
            "\u957f\u671f\u8bb0\u5fc6",
            "\u5de5\u4f5c\u533a",
            "\u6211\u5148\u8bfb\u53d6",
            "\u6211\u5148\u770b\u770b",
            "\u6211\u5148\u786e\u8ba4",
            "\u63a5\u4e0b\u6765\u8bfb\u53d6",
            "\u65e5\u5fd7\u76ee\u5f55",
            "\u6b63\u5728\u786e\u8ba4\u5f53\u524d\u4f1a\u8bdd\u72b6\u6001",
            "\u6b63\u5728\u53d6\u56de\u5fc5\u8981\u4e0a\u4e0b\u6587",
            "\u5fc5\u8981\u4fe1\u606f\u5df2\u8db3\u591f",
            "\u8865\u4e00\u9879\u4f1a\u8bdd\u8bb0\u5f55\u68c0\u67e5",
            "\u6211\u5148\u6838\u5bf9",
            "\u518d\u786e\u8ba4\u4e00\u6b21",
        )
        filtered = [line for line in lines if not any(marker.lower() in line.lower() for marker in internal_markers)]
        if filtered:
            lines = filtered
        for line in lines:
            if re.fullmatch(r"[A-Z0-9_ -]{4,80}", line):
                return line.strip()
        if len(lines) > 1 and not self._looks_like_setup_or_internal_reply(lines[-1]):
            return lines[-1].strip()
        if len(lines) > 1:
            last = lines[-1]
            if re.fullmatch(r"[A-Z0-9_ -]{4,80}", last):
                return last.strip()
        return "\n".join(lines).strip()

    def _looks_like_link_dump(self, reply: str) -> bool:
        raw = str(reply or "").strip()
        if not raw:
            return False
        compact = re.sub(r"\s+", " ", raw)
        url_count = len(re.findall(r"https?://\S+", compact))
        non_url = re.sub(r"https?://\S+", "", compact)
        non_url = non_url.replace("来源:", "").replace("来源：", "").strip(" ，,。：:;；<>[]()")
        if url_count >= 2 and len(non_url) < 24:
            return True
        if compact.startswith("来源:") or compact.startswith("来源："):
            return url_count >= 1 and len(non_url) < 48
        return False

    def _extract_exact_reply_target(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        match = re.search(r"只回复\s+([A-Z0-9_ -]{4,80})", raw)
        if match:
            return match.group(1).strip()
        match = re.search(r"仅回复\s+([A-Z0-9_ -]{4,80})", raw)
        if match:
            return match.group(1).strip()
        match = re.search(r"return exactly\s+([A-Z0-9_ -]{4,80})", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _reply_is_false_heartbeat(self, reply: str, user_text: str) -> bool:
        raw_reply = str(reply or "").strip().upper()
        if raw_reply not in {"HE", "HEA", "HEAR", "HEART", "HEARTBEAT", "HEARTBEAT_OK"}:
            return False
        raw_user = str(user_text or "").strip().lower()
        heartbeat_markers = [
            "heartbeat",
            "心跳",
            "read heartbeat.md",
            "follow it strictly",
        ]
        return not any(marker in raw_user for marker in heartbeat_markers)

    def _looks_like_setup_or_internal_reply(self, reply: str) -> bool:
        raw = str(reply or "").strip().lower()
        if not raw:
            return False
        markers = [
            "\u600e\u4e48\u79f0\u547c\u6211",
            "\u600e\u4e48\u79f0\u547c\u4f60",
            "\u98ce\u683c\u5b9a\u4e0b\u6765",
            "\u56fa\u5b9a emoji",
            "\u7ee7\u7eed\u6536\u96c6\u5fc5\u8981\u4e0a\u4e0b\u6587",
            "\u5148\u6574\u7406\u5f53\u524d\u4f1a\u8bdd",
            "\u542f\u52a8\u8bf4\u660e\u6587\u4ef6",
            "\u672c\u5730\u58f3\u6709\u53d7\u9650\u6a21\u5f0f",
            "\u521d\u59cb\u5316\u548c\u5f53\u524d\u8eab\u4efd\u76f8\u5173\u6587\u4ef6",
            "\u8bfb\u53d6\u5fc5\u8981\u5185\u5bb9",
            "\u5982\u679c\u4f60\u61d2\u5f97\u8bbe",
            "\u6211\u5728\u3002\u5148\u628a\u79f0\u547c",
            "\u6ca1\u6709\u5b9e\u9645\u4efb\u52a1\u6307\u4ee4",
            "\u6211\u5728\u7ebf\u4e86",
            "\u5148\u5b9a\u4e24\u4e2a",
            "\u5148\u5b9a\u4e24\u4e2a\u6700\u6709\u7528\u7684\u4e1c\u897f",
            "\u600e\u4e48\u79f0\u547c",
            "\u8d77\u4ec0\u4e48\u540d\u5b57",
            "\u504f\u4ec0\u4e48\u98ce\u683c",
            "\u5148\u770b\u770b\u73af\u5883",
            "\u5148\u8bfb\u53d6\u4e00\u4e0b",
            "\u5148\u68c0\u67e5\u4e00\u4e0b",
            "\u5148\u786e\u8ba4\u4e00\u4e0b",
            "\u67e5\u770b\u5de5\u4f5c\u533a",
            "\u6574\u7406\u5f53\u524d\u4e0a\u4e0b\u6587",
            "\u8bfb\u53d6\u8bb0\u5fc6",
            "\u8bfb\u53d6\u5f15\u5bfc\u6587\u4ef6",
            "\u5148\u505a\u51c6\u5907",
            "\u6211\u5148\u51c6\u5907",
            "\u6b63\u5728\u786e\u8ba4\u5f53\u524d\u4f1a\u8bdd\u72b6\u6001",
            "\u6b63\u5728\u53d6\u56de\u5fc5\u8981\u4e0a\u4e0b\u6587",
            "\u5fc5\u8981\u4fe1\u606f\u5df2\u8db3\u591f",
            "\u8865\u4e00\u9879\u4f1a\u8bdd\u8bb0\u5f55\u68c0\u67e5",
            "\u8bf7\u91cd\u65b0\u53d1\u4e00\u6b21\u539f\u6587",
            "\u8bf7\u518d\u53d1\u4e00\u6b21\u539f\u6587",
            "\u8bf7\u91cd\u65b0\u53d1\u9001\u539f\u6587",
            "\u8bf7\u628a\u539f\u6587\u518d\u53d1\u4e00\u6b21",
        ]
        return any(marker in raw for marker in markers)

    def _reply_lacks_execution_signal(self, reply: str) -> bool:
        raw = str(reply or "").strip().lower()
        if not raw:
            return True
        positive_markers = [
            "\u5df2\u6253\u5f00",
            "\u5df2\u7ecf\u6253\u5f00",
            "\u5df2\u4e3a\u4f60",
            "\u5df2\u7ecf\u4e3a\u4f60",
            "\u5df2\u542f\u52a8",
            "\u5df2\u7ecf\u542f\u52a8",
            "\u5df2\u5f00\u59cb\u6267\u884c",
            "\u6b63\u5728\u6267\u884c",
            "opened",
            "launched",
            "started",
            "\u6267\u884c\u5b8c\u6210",
            "\u5904\u7406\u597d\u4e86",
        ]
        return not any(marker.lower() in raw for marker in positive_markers)

    def _reply_indicates_blocked_execution(self, reply: str) -> bool:
        raw = str(reply or "").strip().lower()
        if not raw:
            return False
        blocked_markers = [
            "被取消",
            "取消了",
            "刚被取消",
            "没打开",
            "还没打开",
            "未打开",
            "没执行",
            "未执行",
            "被拦截",
            "无权限",
            "权限不足",
            "not allowed",
            "permission denied",
            "blocked",
            "cancelled",
            "canceled",
            "aborted",
        ]
        return any(marker.lower() in raw for marker in blocked_markers)

    def _should_short_circuit_tool_reply(self, text: str) -> bool:
        raw = str(text or "").strip().lower()
        keywords = [
            "\u542c\u6b4c",
            "\u653e\u6b4c",
            "\u64ad\u653e",
            "\u6682\u505c\u64ad\u653e",
            "\u7ee7\u7eed\u64ad\u653e",
            "\u4e0b\u4e00\u9996",
            "\u4e0a\u4e00\u9996",
            "\u6253\u5f00\u7f51\u9875",
            "\u6253\u5f00\u7f51\u7ad9",
            "\u641c\u7d22",
            "\u63d0\u9192\u6211",
            "\u6dfb\u52a0\u5f85\u529e",
            "\u65b0\u589e\u5f85\u529e",
            "\u8bb0\u4e2a\u5f85\u529e",
            "\u8bb0\u7b14\u8bb0",
            "\u6253\u5f00",
            "\u542f\u52a8",
            "\u8ba9\u673a\u5668\u4eba",
            "\u673a\u5668\u4eba",
            "\u9884\u89c8",
            "\u5f00\u59cb\u4e3b\u4eba\u5efa\u6863",
        ]
        return any(keyword in raw for keyword in keywords)

    def _compose_tool_only_reply(self, tool_results: List[ToolExecutionResult]) -> str:
        lines: List[str] = []
        for item in tool_results:
            if item.name == "desktop.play_music":
                query = str(item.data.get("query") or "").strip()
                attempted = bool(item.data.get("attempted_search"))
                if attempted:
                    lines.append(f"\u5df2\u4e3a\u4f60\u62c9\u8d77\u7f51\u6613\u4e91\u97f3\u4e50\uff0c\u5e76\u5c1d\u8bd5\u641c\u7d22 {query}\u3002")
                else:
                    lines.append(f"\u5df2\u4e3a\u4f60\u62c9\u8d77\u7f51\u6613\u4e91\u97f3\u4e50\uff0c\u641c\u7d22 {query} \u8fd9\u4e00\u6b65\u8fd8\u6ca1\u53ef\u9760\u6ce8\u5165\u3002")
            elif item.name == "desktop.music_pause":
                lines.append("\u5df2\u53d1\u9001\u6682\u505c\u64ad\u653e\u3002")
            elif item.name == "desktop.music_play_pause":
                lines.append("\u5df2\u53d1\u9001\u7ee7\u7eed\u64ad\u653e\u3002")
            elif item.name == "desktop.music_next":
                lines.append("\u5df2\u5207\u5230\u4e0b\u4e00\u9996\u3002")
            elif item.name == "desktop.music_previous":
                lines.append("\u5df2\u5207\u5230\u4e0a\u4e00\u9996\u3002")
            elif item.name == "desktop.open_url":
                target_url = item.data.get("url") or "\u76ee\u6807\u7f51\u9875"
                lines.append(f"\u5df2\u6253\u5f00 {target_url}\u3002")
            elif item.name == "desktop.web_search":
                search_query = item.data.get("query") or "\u76ee\u6807\u5185\u5bb9"
                lines.append(f"\u5df2\u4e3a\u4f60\u641c\u7d22 {search_query}\u3002")
            elif item.name == "desktop.todo_create":
                title = str(item.data.get("title") or "").strip()
                todo_title = title or "\u65b0\u4efb\u52a1"
                lines.append(f"\u5df2\u8bb0\u4e0b\u5f85\u529e\uff1a{todo_title}\u3002")
            elif item.name == "desktop.write_note":
                lines.append("\u7b14\u8bb0\u5df2\u7ecf\u8bb0\u4e0b\u4e86\u3002")
            elif item.name == "desktop.launch_app":
                app_name = item.data.get("app") or "\u76ee\u6807\u5e94\u7528"
                lines.append(f"\u5df2\u542f\u52a8 {app_name}\u3002")
            elif item.name == "robot.get_status":
                lines.append("\u6211\u5df2\u7ecf\u8bfb\u5230\u673a\u5668\u4eba\u7684\u5f53\u524d\u72b6\u6001\u3002")
            elif item.name == "robot.speak":
                spoken = str(item.data.get("text") or "").strip()
                spoken_text = spoken or "\u597d\u7684"
                lines.append(f"\u6211\u5df2\u7ecf\u8ba9\u673a\u5668\u4eba\u8bf4\u4e86\uff1a{spoken_text}\u3002")
            elif item.name == "robot.pan_tilt":
                lines.append("\u6211\u5df2\u7ecf\u8ba9\u673a\u5668\u4eba\u52a8\u4e86\u4e00\u4e0b\u3002")
            elif item.name == "robot.start_owner_enrollment":
                lines.append("\u6211\u5df2\u7ecf\u5f00\u59cb\u4e3b\u4eba\u5efa\u6863\u3002")
            elif item.name == "robot.get_preview":
                lines.append("\u6211\u5df2\u7ecf\u51c6\u5907\u597d\u673a\u5668\u4eba\u9884\u89c8\u4e86\u3002")
        return "\n".join(lines).strip() or "\u5df2\u7ecf\u5904\u7406\u597d\u4e86\u3002"

    async def _run_explicit_tools(
        self,
        conn: Connection,
        user_id: int,
        text: str,
        device_id: Optional[str] = None,
        assistant_mode: str = "product",
        native_control_enabled: bool = True,
    ) -> List[ToolExecutionResult]:
        results: List[ToolExecutionResult] = []
        raw = str(text or "").strip()
        if not raw:
            return results
        desktop_side_effects_allowed = not (
            str(assistant_mode or "").strip().lower() == "agent" and bool(native_control_enabled)
        )

        reminder = self._parse_reminder(raw)
        if reminder is not None:
            title, due_at_ms, action = reminder
            item = self.create_todo(
                user_id,
                title=title,
                due_at_ms=due_at_ms,
                tags=["reminder", "assistant"],
                action=action,
            )
            due_label = self.store.format_due_label(due_at_ms)
            results.append(
                ToolExecutionResult(
                    name="desktop.todo_create",
                    ok=True,
                    detail=f"Reminder scheduled for {due_label}" if due_label else "Reminder scheduled",
                    data=item,
                )
            )

        todo_match = re.search(r"^(?:\u6dfb\u52a0\u5f85\u529e|\u65b0\u589e\u5f85\u529e|\u8bb0\u4e2a\u5f85\u529e)[:\uff1a]?\s*(.+)$", raw, flags=re.IGNORECASE)
        if todo_match:
            item = self.create_todo(user_id, title=todo_match.group(1).strip())
            results.append(ToolExecutionResult(name="desktop.todo_create", ok=True, detail=f"Added todo: {item['title']}", data=item))

        note_match = re.search(r"^(?:\u8bb0\u4e00\u4e2a\u7b14\u8bb0|\u8bb0\u4e2a\u7b14\u8bb0|\u5199\u4e2a\u7b14\u8bb0|\u8bb0\u7b14\u8bb0)[:\uff1a]?\s*(.+)$", raw, flags=re.IGNORECASE)
        if note_match:
            note = self.store.write_note(user_id, title="assistant-note", body=note_match.group(1).strip())
            results.append(ToolExecutionResult(name="desktop.write_note", ok=True, detail="Note written", data=note))

        direct_url_match = re.search(r"(https?://\S+)", raw, flags=re.IGNORECASE)
        if desktop_side_effects_allowed and direct_url_match:
            url = direct_url_match.group(1).strip()
            self._launch_url(url)
            results.append(ToolExecutionResult(name="desktop.open_url", ok=True, detail=f"Opened {url}", data={"url": url}))

        if desktop_side_effects_allowed and not direct_url_match:
            page_match = re.search(r"(?:\u6253\u5f00\u7f51\u9875|\u6253\u5f00\u7f51\u7ad9|\u6253\u5f00\u9875\u9762|open website|open page)\s*[:\uff1a]?\s*(.+)$", raw, flags=re.IGNORECASE)
            if page_match:
                target = self._trim_desktop_target(page_match.group(1).strip())
                url = self._normalize_web_target(target)
                self._launch_url(url)
                results.append(ToolExecutionResult(name="desktop.open_url", ok=True, detail=f"Opened {url}", data={"url": url}))
            else:
                generic_open_match = re.search(r"^(?:\u6253\u5f00)\s*(.+)$", raw, flags=re.IGNORECASE)
                if generic_open_match:
                    target = self._trim_desktop_target(generic_open_match.group(1).strip())
                    target_key = target.lower()
                    known_apps = {
                        "\u8bb0\u4e8b\u672c",
                        "\u8ba1\u7b97\u5668",
                        "\u8d44\u6e90\u7ba1\u7406\u5668",
                        "notepad",
                        "calc",
                        "explorer",
                        "vscode",
                        "chrome",
                        "edge",
                    }
                    if target_key not in known_apps:
                        url = self._normalize_web_target(target)
                        self._launch_url(url)
                        results.append(
                            ToolExecutionResult(
                                name="desktop.open_url",
                                ok=True,
                                detail=f"Opened {url}",
                                data={"url": url},
                            )
                        )

        search_match = re.search(r"(?:\u641c\u7d22|\u67e5\u4e00\u4e0b|\u641c\u4e00\u4e0b|search)\s*[:\uff1a]?\s*(.+)$", raw, flags=re.IGNORECASE)
        if desktop_side_effects_allowed and search_match:
            query = search_match.group(1).strip()
            if query:
                url = f"https://www.baidu.com/s?wd={quote_plus(query)}"
                self._launch_url(url)
                results.append(ToolExecutionResult(name="desktop.web_search", ok=True, detail=f"Searched {query}", data={"query": query, "url": url}))

        music_query = self._parse_music_request(raw)
        if desktop_side_effects_allowed and music_query:
            launch_result = self._launch_music_app(music_query)
            results.append(ToolExecutionResult(name="desktop.play_music", ok=True, detail=launch_result["detail"], data=launch_result))

        music_control = self._parse_music_control(raw)
        if desktop_side_effects_allowed and music_control:
            action_result = self._send_media_control(music_control)
            results.append(ToolExecutionResult(name=f"desktop.music_{music_control}", ok=True, detail=action_result["detail"], data=action_result))

        app_match = re.search(r"(?:\u6253\u5f00|\u542f\u52a8|launch)\s*(\u8bb0\u4e8b\u672c|\u8ba1\u7b97\u5668|\u8d44\u6e90\u7ba1\u7406\u5668|vscode|chrome|edge|notepad|calc|explorer)\b", raw, flags=re.IGNORECASE)
        if desktop_side_effects_allowed and app_match:
            alias_map = {
                "\u8bb0\u4e8b\u672c": "notepad",
                "notepad": "notepad",
                "\u8ba1\u7b97\u5668": "calc",
                "calc": "calc",
                "\u8d44\u6e90\u7ba1\u7406\u5668": "explorer",
                "explorer": "explorer",
                "vscode": "vscode",
                "chrome": "chrome",
                "edge": "edge",
            }
            label = str(app_match.group(1) or "").strip()
            key = alias_map.get(label.lower(), alias_map.get(label, label.lower()))
            self._launch_app(key)
            results.append(ToolExecutionResult(name="desktop.launch_app", ok=True, detail=f"Launched {label}", data={"app": key}))

        if self._contains_any(raw, ["\u673a\u5668\u4eba\u72b6\u6001", "\u673a\u5668\u4eba\u7684\u72b6\u6001", "robot status"]):
            status_payload = await self._robot_get_status(conn, user_id, device_id=device_id)
            results.append(ToolExecutionResult(name="robot.get_status", ok=True, detail="Fetched robot status", data=status_payload))

        speak_match = re.search(r"(?:\u8ba9\u673a\u5668\u4eba(?:\u8bf4|\u64ad\u62a5|\u8bb2\u8bdd)(?:\u4e00\u53e5|\u4e00\u4e0b)?|\u673a\u5668\u4eba(?:\u8bf4|\u64ad\u62a5|\u8bb2\u8bdd))\s*[:\uff1a]?\s*(.+)$", raw, flags=re.IGNORECASE)
        if speak_match:
            spoken = speak_match.group(1).strip()
            if spoken:
                payload = await self._robot_post(conn, user_id, "/speak", {"text": spoken}, device_id=device_id)
                results.append(ToolExecutionResult(name="robot.speak", ok=True, detail="Spoke via robot", data={"text": spoken, **payload}))

        pan, tilt = self._parse_robot_pan_tilt(raw)
        if pan is not None or tilt is not None:
            payload = {"pan": pan or 0.0, "tilt": tilt or 0.0}
            result = await self._robot_post(conn, user_id, "/pan_tilt", payload, device_id=device_id)
            results.append(ToolExecutionResult(name="robot.pan_tilt", ok=True, detail="Moved robot pan/tilt", data=result or payload))

        if self._contains_any(raw, ["\u5f00\u59cb\u4e3b\u4eba\u5efa\u6863", "\u5f00\u59cb\u626b\u8138", "start owner enrollment"]):
            payload = await self._robot_post(conn, user_id, "/owner/enrollment/start", {}, device_id=device_id)
            results.append(ToolExecutionResult(name="robot.start_owner_enrollment", ok=True, detail="Started owner enrollment", data=payload))

        if self._contains_any(raw, ["\u9884\u89c8", "\u770b\u770b\u76f8\u673a", "camera preview"]):
            preview = self._robot_preview(conn, user_id, device_id=device_id)
            results.append(ToolExecutionResult(name="robot.get_preview", ok=True, detail="Prepared robot preview", data=preview))

        return results

    def _parse_robot_pan_tilt(self, raw: str) -> Tuple[Optional[float], Optional[float]]:
        pan: Optional[float] = None
        tilt: Optional[float] = None
        if self._contains_any(raw, ["\u5de6\u8f6c", "\u5411\u5de6\u770b", "\u770b\u5de6\u8fb9", "turn left"]):
            pan = -0.35
        if self._contains_any(raw, ["\u53f3\u8f6c", "\u5411\u53f3\u770b", "\u770b\u53f3\u8fb9", "turn right"]):
            pan = 0.35
        if self._contains_any(raw, ["\u62ac\u5934", "\u770b\u4e0a\u9762", "look up"]):
            tilt = 0.35
        if self._contains_any(raw, ["\u4f4e\u5934", "\u770b\u4e0b\u9762", "look down"]):
            tilt = -0.35
        if self._contains_any(raw, ["\u52a8\u4e00\u52a8", "\u6d3b\u52a8\u4e00\u4e0b", "move a bit"]):
            pan = 0.25 if pan is None else pan
            tilt = 0.18 if tilt is None else tilt
        return pan, tilt

    def _normalize_web_target(self, target: str) -> str:
        value = str(target or "").strip()
        if re.match(r"^https?://", value, flags=re.IGNORECASE):
            return value
        if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", value, flags=re.IGNORECASE):
            return f"https://{value}"
        return f"https://www.baidu.com/s?wd={quote_plus(value)}"

    def _trim_desktop_target(self, target: str) -> str:
        value = str(target or "").strip()
        if not value:
            return value
        parts = re.split(r"(?:并且|并|然后|再|，并|, and | and then )", value, maxsplit=1, flags=re.IGNORECASE)
        trimmed = parts[0].strip()
        return trimmed or value

    def _parse_music_request(self, raw: str) -> Optional[str]:
        value = str(raw or "").strip()
        patterns = [
            r"(?:\u542c\u6b4c|\u653e\u9996\u6b4c|\u64ad\u653e\u97f3\u4e50|\u64ad\u653e\u6b4c\u66f2|play music)\s*[:\uff1a]?\s*(.+)$",
            r"(?:\u5e2e\u6211\u542c\u6b4c|\u5e2e\u6211\u653e\u6b4c|\u5e2e\u6211\u64ad\u653e)\s*[:\uff1a]?\s*(.+)$",
            r"(?:\u60f3\u542c|\u6211\u60f3\u542c)\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if not match:
                continue
            query = self._strip_punctuation(str(match.group(1) or ""))
            if query:
                return query
        return None

    def _parse_music_control(self, raw: str) -> Optional[str]:
        if self._contains_any(raw, ["\u6682\u505c\u64ad\u653e", "\u6682\u505c\u97f3\u4e50", "\u6682\u505c\u4e00\u4e0b", "pause music", "pause playback"]):
            return "pause"
        if self._contains_any(raw, ["\u7ee7\u7eed\u64ad\u653e", "\u6062\u590d\u64ad\u653e", "\u7ee7\u7eed\u97f3\u4e50", "resume music", "resume playback"]):
            return "play_pause"
        if self._contains_any(raw, ["\u4e0b\u4e00\u9996", "\u5207\u4e0b\u4e00\u9996", "next song", "next track"]):
            return "next"
        if self._contains_any(raw, ["\u4e0a\u4e00\u9996", "\u5207\u4e0a\u4e00\u9996", "previous song", "previous track"]):
            return "previous"
        return None

    def _parse_reminder(self, raw: str) -> Optional[Tuple[str, int, Dict[str, object]]]:
        value = str(raw or "").strip()
        relative_match = re.search(r"(?:\u63d0\u9192\u6211)\s*(.+?)\s*(?:\u5728|\u8fc7)\s*(\d+)\s*(\u79d2|\u5206\u949f|\u5206|\u5c0f\u65f6|\u65f6|\u5929)\u540e", value, flags=re.IGNORECASE)
        if relative_match:
            title = str(relative_match.group(1) or "").strip()
            amount = int(relative_match.group(2))
            unit = str(relative_match.group(3) or "")
            return title, _now_ms() + (self._unit_to_seconds(amount, unit) * 1000), {"type": "reminder"}

        relative_after_match = re.search(r"(?:\u63d0\u9192\u6211)\s*(\d+)\s*(\u79d2|\u5206\u949f|\u5206|\u5c0f\u65f6|\u65f6|\u5929)\u540e\s*(.+)$", value, flags=re.IGNORECASE)
        if relative_after_match:
            amount = int(relative_after_match.group(1))
            unit = str(relative_after_match.group(2) or "")
            title = str(relative_after_match.group(3) or "").strip()
            return title, _now_ms() + (self._unit_to_seconds(amount, unit) * 1000), {"type": "reminder"}

        english_match = re.search(r"(?:remind me to)\s*(.+?)\s*(?:in)\s*(\d+)\s*(seconds?|minutes?|hours?|days?)", value, flags=re.IGNORECASE)
        if english_match:
            title = str(english_match.group(1) or "").strip()
            amount = int(english_match.group(2))
            unit = str(english_match.group(3) or "")
            return title, _now_ms() + (self._unit_to_seconds(amount, unit) * 1000), {"type": "reminder"}

        absolute_match = re.search(r"(?:\u63d0\u9192\u6211)\s*(.+?)\s*(\u4eca\u5929|\u660e\u5929)?\s*(\u4e0a\u5348|\u4e0b\u5348|\u665a\u4e0a)?\s*(\d{1,2})[:\u70b9\u65f6]?(\d{1,2})?", value, flags=re.IGNORECASE)
        if absolute_match:
            title = str(absolute_match.group(1) or "").strip()
            if not title:
                return None
            day_hint = str(absolute_match.group(2) or "").strip()
            period = str(absolute_match.group(3) or "").strip()
            hour = int(absolute_match.group(4))
            minute = int(absolute_match.group(5) or 0)
            if period in {"\u4e0b\u5348", "\u665a\u4e0a"} and hour < 12:
                hour += 12
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if day_hint == "\u660e\u5929":
                target = target + timedelta(days=1)
            elif target <= now:
                target = target + timedelta(days=1)
            return title, int(target.timestamp() * 1000), {"type": "reminder"}
        return None

    def _unit_to_seconds(self, amount: int, unit: str) -> int:
        normalized = str(unit or "").strip().lower()
        if normalized in {"\u79d2", "second", "seconds"}:
            return max(1, amount)
        if normalized in {"\u5c0f\u65f6", "\u65f6", "hour", "hours"}:
            return max(1, amount) * 3600
        if normalized in {"\u5929", "day", "days"}:
            return max(1, amount) * 86400
        return max(1, amount) * 60

    def _contains_any(self, raw: str, keywords: List[str]) -> bool:
        lowered = raw.lower()
        return any(keyword.lower() in lowered for keyword in keywords)

    def _strip_punctuation(self, text: str) -> str:
        return str(text or "").strip().strip("\u3002\uff01\uff1f!?\uff0c,\uff1b;\uff1a:")

    def _robot_endpoint(self, value: str) -> str:
        raw = str(value or "").strip()
        if ":" in raw:
            return raw
        return f"{raw}:8090"

    async def _robot_get_status(self, conn: Connection, user_id: int, device_id: Optional[str] = None) -> Dict[str, object]:
        device = self._resolve_device(conn, user_id, device_id=device_id)
        endpoint = self._robot_endpoint(str(device["device_ip"]))
        url = f"http://{endpoint}/status"
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

    async def _robot_post(
        self,
        conn: Connection,
        user_id: int,
        path: str,
        payload: Dict[str, object],
        device_id: Optional[str] = None,
    ) -> Dict[str, object]:
        device = self._resolve_device(conn, user_id, device_id=device_id)
        endpoint = self._robot_endpoint(str(device["device_ip"]))
        url = f"http://{endpoint}{path}"
        async with httpx.AsyncClient(timeout=20.0, trust_env=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

    def _robot_preview(self, conn: Connection, user_id: int, device_id: Optional[str] = None) -> Dict[str, object]:
        device = self._resolve_device(conn, user_id, device_id=device_id)
        endpoint = self._robot_endpoint(str(device["device_ip"]))
        return {"preview_url": f"http://{endpoint}/camera/preview.jpg", "device_id": device["device_id"]}

    def _resolve_device(self, conn: Connection, user_id: int, device_id: Optional[str] = None) -> Dict[str, object]:
        params: List[Any] = [int(user_id)]
        query = "SELECT * FROM devices WHERE user_id = ?"
        if device_id:
            query += " AND device_id = ?"
            params.append(str(device_id))
        query += " ORDER BY updated_at DESC LIMIT 1"
        try:
            row = conn.execute(query, params).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row:
            device = dict(row)
            if str(device.get("device_ip") or "").strip():
                return device
        fallback_ip = str(DEFAULT_ROBOT_DEVICE_IP or "").strip()
        if fallback_ip:
            return {
                "device_id": str(device_id or "polealpha-zero2w"),
                "device_ip": fallback_ip,
            }
        raise RuntimeError("No bound robot device found")

    def _launch_url(self, url: str) -> None:
        subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)

    def _launch_app(self, alias: str) -> None:
        candidate = self.app_allowlist.get(alias)
        if not candidate:
            raise RuntimeError(f"App not allowed: {alias}")
        if isinstance(candidate, str):
            command = [candidate]
        else:
            command = [str(part) for part in list(candidate)]
        subprocess.Popen(command, shell=False)

    def _launch_music_app(self, query: str) -> Dict[str, object]:
        exe = self._resolve_cloudmusic_executable()
        if not exe:
            raise RuntimeError("CloudMusic not found on this computer")
        proc = subprocess.Popen([exe], shell=False)
        attempted_search = False
        if query:
            attempted_search = self._try_cloudmusic_search(query, proc.pid)
        detail = (
            f"Launched CloudMusic and attempted in-app search for {query}"
            if attempted_search
            else f"Launched CloudMusic for {query}"
        )
        return {
            "app": "cloudmusic",
            "path": exe,
            "query": query,
            "attempted_search": attempted_search,
            "detail": detail,
        }

    def _resolve_cloudmusic_executable(self) -> Optional[str]:
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\cloudmusic.exe",
            ) as key:
                value, _ = winreg.QueryValueEx(key, None)
                if value and Path(value).exists():
                    return str(Path(value))
        except Exception:
            pass
        candidates = [
            Path(r"C:\Program Files\NetEase\CloudMusic\cloudmusic.exe"),
            Path(r"C:\Program Files (x86)\NetEase\CloudMusic\cloudmusic.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _try_cloudmusic_search(self, query: str, pid: int) -> bool:
        escaped_query = query.replace("'", "''")
        script = (
            "Add-Type -AssemblyName Microsoft.VisualBasic; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            f"$pid={int(pid)}; "
            "$deadline=(Get-Date).AddSeconds(8); "
            "$target=$null; "
            "while((Get-Date) -lt $deadline) { "
            "$target=Get-Process -Id $pid -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1; "
            "if($target){break}; Start-Sleep -Milliseconds 250 } ; "
            "if(-not $target){ exit 2 }; "
            f"Set-Clipboard -Value '{escaped_query}'; "
            "[Microsoft.VisualBasic.Interaction]::AppActivate($pid) | Out-Null; "
            "Start-Sleep -Milliseconds 250; "
            "[System.Windows.Forms.SendKeys]::SendWait('^f'); "
            "Start-Sleep -Milliseconds 200; "
            "[System.Windows.Forms.SendKeys]::SendWait('^a'); "
            "Start-Sleep -Milliseconds 120; "
            "[System.Windows.Forms.SendKeys]::SendWait('^v'); "
            "Start-Sleep -Milliseconds 120; "
            "[System.Windows.Forms.SendKeys]::SendWait('{ENTER}'); "
            "exit 0"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def _send_media_control(self, action: str) -> Dict[str, object]:
        key_map = {
            "pause": 0xB3,
            "play_pause": 0xB3,
            "next": 0xB0,
            "previous": 0xB1,
        }
        vk = key_map.get(action)
        if vk is None:
            raise RuntimeError(f"Unsupported media action: {action}")
        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, 0x1, 0)
        user32.keybd_event(vk, 0, 0x1 | 0x2, 0)
        return {"action": action, "detail": f"Sent media control: {action}"}
