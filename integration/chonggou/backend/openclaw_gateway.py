from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key, load_pem_public_key

try:
    import websockets
except Exception:  # pragma: no cover - dependency handled at runtime
    websockets = None


class OpenClawGatewayError(RuntimeError):
    pass


_LOCAL_PROXY_CANDIDATES = (
    "http://127.0.0.1:7897",
    "http://127.0.0.1:7890",
    "http://127.0.0.1:10808",
    "http://127.0.0.1:1080",
)


@dataclass
class OpenClawGatewayConfig:
    state_dir: str
    workspace_dir: str
    codex_home: str
    repo_path: str
    url: str
    origin: str
    timeout_ms: int
    client_id: str
    client_mode: str
    prefer_websocket_first: bool = False
    allow_agent_fallback: bool = True


def _proxy_endpoint_reachable(proxy_url: str, timeout_s: float = 0.3) -> bool:
    parsed = urlparse(str(proxy_url or "").strip())
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def resolve_openclaw_proxy_url(env: Optional[Dict[str, str]] = None) -> str:
    source = env or os.environ
    for key in (
        "OPENCLAW_PROXY_URL",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        value = str(source.get(key, "") or "").strip()
        if value:
            return value
    for candidate in _LOCAL_PROXY_CANDIDATES:
        if _proxy_endpoint_reachable(candidate):
            return candidate
    return ""


def build_openclaw_proxy_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    proxy_url = resolve_openclaw_proxy_url(env)
    if not proxy_url:
        return {}
    return {
        "OPENCLAW_PROXY_URL": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "ALL_PROXY": proxy_url,
        "all_proxy": proxy_url,
    }


def discover_openclaw_state_dir(configured: str, workspace_dir: str) -> Path:
    configured_path = Path(configured).expanduser().resolve() if configured else None
    candidates: List[Path] = []
    if configured_path is not None:
        copied = _materialize_openclaw_state_dir(configured_path, workspace_dir)
        if copied is not None:
            return copied
        candidates.append(configured_path)
    candidates.extend(
        [
            Path.home() / ".openclaw",
            Path(os.environ.get("APPDATA", "")) / "Antigravity" / "openclaw",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Antigravity" / "openclaw",
            Path(os.environ.get("LOCALAPPDATA", "")) / "com.lbjlaq.antigravity-tools" / "openclaw",
            Path(workspace_dir).expanduser().resolve() / ".openclaw",
            Path(workspace_dir).expanduser().resolve() / ".." / ".openclaw",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "openclaw.json").exists() and (candidate / "identity" / "device.json").exists():
            return candidate
    raise OpenClawGatewayError("OpenClaw state dir not found; set OPENCLAW_STATE_DIR")


def _materialize_openclaw_state_dir(target: Path, workspace_dir: str) -> Optional[Path]:
    target = target.expanduser().resolve()
    if (target / "openclaw.json").exists() and (target / "identity" / "device.json").exists():
        return target
    fallback_candidates = [
        Path.home() / ".openclaw",
        Path(os.environ.get("APPDATA", "")) / "Antigravity" / "openclaw",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Antigravity" / "openclaw",
        Path(os.environ.get("LOCALAPPDATA", "")) / "com.lbjlaq.antigravity-tools" / "openclaw",
        Path(workspace_dir).expanduser().resolve() / ".openclaw",
        Path(workspace_dir).expanduser().resolve() / ".." / ".openclaw",
    ]
    source: Optional[Path] = None
    for candidate in fallback_candidates:
        candidate = candidate.expanduser().resolve()
        if candidate == target:
            continue
        if (candidate / "openclaw.json").exists() and (candidate / "identity" / "device.json").exists():
            source = candidate
            break
    if source is None:
        return None
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / "identity").mkdir(parents=True, exist_ok=True)
        for rel in ("openclaw.json", "identity/device.json", "identity/device-auth.json"):
            src = source / rel
            if src.exists():
                shutil.copy2(src, target / rel)
        return target if (target / "openclaw.json").exists() and (target / "identity" / "device.json").exists() else None
    except Exception:
        return None


class OpenClawGatewayClient:
    def __init__(self, config: OpenClawGatewayConfig) -> None:
        self.config = config
        self._probe_checked_at_ms = 0
        self._probe_ok = False
        self._probe_error = ""

    def _resolve_timeout_ms(self, timeout_ms: Optional[int] = None) -> int:
        try:
            raw = int(timeout_ms if timeout_ms is not None else self.config.timeout_ms)
        except Exception:
            raw = 20000
        raw = max(5000, raw)
        return min(raw, 120000)

    def _cache_probe_result(self, ok: bool, detail: str = "") -> tuple[bool, str]:
        self._probe_checked_at_ms = int(time.time() * 1000)
        self._probe_ok = bool(ok)
        self._probe_error = "" if ok else str(detail or "").strip()
        return self._probe_ok, self._probe_error

    @classmethod
    def _format_runtime_detail(cls, error: object) -> str:
        raw = str(error or "").strip()
        lowered = raw.lower()
        if "not_paired" in lowered or "pairing_required" in lowered or "pairing required" in lowered:
            return "OpenClaw 尚未配对，请先完成配对后再使用真实 AI 回复。"
        if "token missing" in lowered:
            return "OpenClaw 身份令牌缺失，请重新初始化本地运行环境。"
        if "state dir not found" in lowered:
            return "OpenClaw 状态目录缺失，请检查本地运行目录。"
        if "device id missing" in lowered:
            return "OpenClaw 设备身份缺失，请重新初始化本地运行环境。"
        if "websockets dependency missing" in lowered:
            return "OpenClaw 运行依赖缺失：websockets。"
        return raw or "OpenClaw 当前不可用"

    @classmethod
    def _describe_gateway_error(cls, context: str, error: object) -> str:
        detail = cls._format_runtime_detail(error)
        if detail.startswith("OpenClaw ") or detail.startswith("OpenClaw尚未"):
            return detail
        if not detail:
            return context
        return f"{context}: {detail}"

    async def _probe_connection_once(self, runtime: Dict[str, str]) -> None:
        async with self._connect(runtime) as ws:
            inbox: List[Dict[str, object]] = []
            await self._connect_session(ws, runtime, inbox)

    async def probe_connection(
        self,
        runtime: Optional[Dict[str, str]] = None,
        *,
        force: bool = False,
        max_age_ms: int = 5000,
        timeout_ms: int = 5000,
    ) -> tuple[bool, str]:
        now_ms = int(time.time() * 1000)
        if (
            not force
            and self._probe_checked_at_ms
            and (now_ms - self._probe_checked_at_ms) <= max(0, int(max_age_ms))
        ):
            return self._probe_ok, self._probe_error
        try:
            resolved_runtime = runtime or self._load_runtime()
            await asyncio.wait_for(
                self._probe_connection_once(resolved_runtime),
                timeout=max(2.0, min(8.0, self._resolve_timeout_ms(timeout_ms) / 1000.0)),
            )
            return self._cache_probe_result(True, "")
        except OpenClawGatewayError as exc:
            return self._cache_probe_result(False, self._format_runtime_detail(exc))
        except Exception as exc:
            return self._cache_probe_result(False, self._format_runtime_detail(exc))

    def probe_connection_blocking(
        self,
        *,
        force: bool = False,
        max_age_ms: int = 5000,
        timeout_ms: int = 5000,
    ) -> tuple[bool, str]:
        now_ms = int(time.time() * 1000)
        if (
            not force
            and self._probe_checked_at_ms
            and (now_ms - self._probe_checked_at_ms) <= max(0, int(max_age_ms))
        ):
            return self._probe_ok, self._probe_error
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.probe_connection(force=force, max_age_ms=max_age_ms, timeout_ms=timeout_ms)
            )

        result: Dict[str, tuple[bool, str]] = {}
        error_box: Dict[str, Exception] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(
                    self.probe_connection(force=force, max_age_ms=max_age_ms, timeout_ms=timeout_ms)
                )
            except Exception as exc:  # pragma: no cover - defensive path
                error_box["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join(max(1.0, self._resolve_timeout_ms(timeout_ms) / 1000.0 + 0.5))
        if thread.is_alive():
            return self._cache_probe_result(False, "OpenClaw 连接校验超时。")
        if "error" in error_box:
            return self._cache_probe_result(False, self._format_runtime_detail(error_box["error"]))
        return result.get("value", self._cache_probe_result(False, "OpenClaw 连接校验失败。"))

    async def send_message(self, session_key: str, text: str, timeout_ms: Optional[int] = None) -> str:
        normalized_session_key = self._normalize_agent_session_key(str(session_key))
        runtime = self._load_runtime()
        timeout_ms = self._resolve_timeout_ms(timeout_ms)
        websocket_error: Optional[Exception] = None
        agent_error: Optional[Exception] = None
        direct_cli_error: Optional[Exception] = None
        direct_cli_available = self._direct_cli_fallback_available(runtime)
        if self.config.prefer_websocket_first:
            try:
                return await self._send_message_via_websocket(runtime, normalized_session_key, str(text), timeout_ms)
            except Exception as exc:
                websocket_error = exc
                if not self.config.allow_agent_fallback and not direct_cli_available:
                    detail = self._format_runtime_detail(exc)
                    if "pair" in detail.lower():
                        self._cache_probe_result(False, detail)
                    raise OpenClawGatewayError(detail) from exc
        if self.config.allow_agent_fallback:
            cli_session_id = self._resolve_cli_session_id(runtime, normalized_session_key)
            if cli_session_id:
                try:
                    return await self._send_message_via_agent(runtime, cli_session_id, str(text), timeout_ms)
                except Exception as exc:
                    agent_error = exc
            else:
                agent_error = OpenClawGatewayError(
                    f"OpenClaw CLI session id not found for session key: {normalized_session_key}"
                )
        if websocket_error is None:
            try:
                return await self._send_message_via_websocket(runtime, normalized_session_key, str(text), timeout_ms)
            except Exception as exc:
                websocket_error = exc
        if direct_cli_available:
            try:
                return await self._send_message_via_direct_cli(runtime, str(text), timeout_ms)
            except Exception as exc:
                direct_cli_error = exc
        if websocket_error is not None and agent_error is not None and direct_cli_error is not None:
            websocket_detail = self._format_runtime_detail(websocket_error)
            agent_detail = str(agent_error).strip() or agent_error.__class__.__name__
            direct_detail = str(direct_cli_error).strip() or direct_cli_error.__class__.__name__
            raise OpenClawGatewayError(
                "OpenClaw provider failed via "
                f"agent ({agent_detail}), websocket ({websocket_detail}), and direct CLI ({direct_detail})"
            ) from websocket_error
        if websocket_error is not None and direct_cli_error is not None and agent_error is None:
            websocket_detail = self._format_runtime_detail(websocket_error)
            direct_detail = str(direct_cli_error).strip() or direct_cli_error.__class__.__name__
            raise OpenClawGatewayError(
                f"OpenClaw provider failed via websocket ({websocket_detail}) and direct CLI ({direct_detail})"
            ) from websocket_error
        if agent_error is not None and direct_cli_error is not None and websocket_error is None:
            agent_detail = str(agent_error).strip() or agent_error.__class__.__name__
            direct_detail = str(direct_cli_error).strip() or direct_cli_error.__class__.__name__
            raise OpenClawGatewayError(
                f"OpenClaw provider failed via agent ({agent_detail}) and direct CLI ({direct_detail})"
            ) from agent_error
        if websocket_error is not None:
            detail = self._format_runtime_detail(websocket_error)
            if "pair" in detail.lower():
                self._cache_probe_result(False, detail)
            raise OpenClawGatewayError(detail) from websocket_error
        if agent_error is not None:
            raise OpenClawGatewayError(str(agent_error).strip() or "OpenClaw agent path failed") from agent_error
        if direct_cli_error is not None:
            raise OpenClawGatewayError(str(direct_cli_error).strip() or "OpenClaw direct CLI path failed") from direct_cli_error
        raise OpenClawGatewayError("OpenClaw send failed without a usable transport")
        try:
            async with self._connect(runtime) as ws:
                inbox: List[Dict[str, object]] = []
                await self._connect_session(ws, runtime, inbox)
                baseline = None
                run_id = f"assistant-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
                response = await self._rpc_request(
                    ws,
                    "chat.send",
                    {
                        "sessionKey": normalized_session_key,
                        "message": str(text),
                        "deliver": False,
                        "timeoutMs": timeout_ms,
                        "idempotencyKey": run_id,
                    },
                    inbox,
                    timeout_ms=max(timeout_ms, 15000),
                )
                if not response.get("ok"):
                    raise OpenClawGatewayError(f"OpenClaw chat.send failed: {response.get('error')}")
                return await self._wait_for_reply(ws, normalized_session_key, run_id, baseline, inbox, timeout_ms)
        except OpenClawGatewayError as exc:
            detail = self._format_runtime_detail(exc)
            if "配对" in detail or "pair" in detail.lower():
                self._cache_probe_result(False, detail)
            raise OpenClawGatewayError(detail) from exc
        except Exception as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            if agent_error is not None:
                agent_detail = str(agent_error).strip() or agent_error.__class__.__name__
                raise OpenClawGatewayError(
                    f"OpenClaw provider failed via agent ({agent_detail}) and websocket fallback ({detail})"
                ) from exc
            raise OpenClawGatewayError(f"OpenClaw websocket fallback failed: {detail}") from exc

    async def reset_session(self, session_key: str) -> None:
        normalized_session_key = self._normalize_agent_session_key(str(session_key))
        runtime = self._load_runtime()
        self._clear_cli_resume_state(runtime, normalized_session_key)
        async with self._connect(runtime) as ws:
            inbox: List[Dict[str, object]] = []
            await self._connect_session(ws, runtime, inbox)
            response = await self._rpc_request(
                ws,
                "sessions.reset",
                {"key": normalized_session_key},
                inbox,
                timeout_ms=10000,
            )
            if not response.get("ok"):
                raise OpenClawGatewayError(f"OpenClaw sessions.reset failed: {response.get('error')}")
        self._clear_cli_resume_state(runtime, normalized_session_key)

    def _load_runtime(self) -> Dict[str, str]:
        state_dir = discover_openclaw_state_dir(self.config.state_dir, self.config.workspace_dir)
        openclaw_json = self._read_json(state_dir / "openclaw.json")
        device_json = self._read_json(state_dir / "identity" / "device.json")
        auth_path = state_dir / "identity" / "device-auth.json"
        auth_json = self._read_json(auth_path) if auth_path.exists() else {}
        gateway_port = int(openclaw_json.get("gateway", {}).get("port", 18789))
        url = self.config.url.strip() or f"ws://127.0.0.1:{gateway_port}"
        origin = self.config.origin.strip() or f"http://127.0.0.1:{gateway_port}"
        gateway_token = str(openclaw_json.get("gateway", {}).get("auth", {}).get("token", "") or "").strip()
        device_token = str(auth_json.get("tokens", {}).get("operator", {}).get("token", "") or "").strip()
        token = gateway_token or device_token
        if not token:
            raise OpenClawGatewayError("OpenClaw token missing")
        return {
            "state_dir": str(state_dir),
            "url": url,
            "origin": origin,
            "token": token,
            "device_id": str(device_json.get("deviceId", "") or "").strip(),
            "private_key_pem": str(device_json.get("privateKeyPem", "") or ""),
            "public_key_pem": str(device_json.get("publicKeyPem", "") or ""),
        }

    @staticmethod
    def _normalize_agent_session_key(session_key: str) -> str:
        raw = str(session_key or "").strip() or "session"
        if re.fullmatch(r"[A-Za-z0-9._:-]+", raw):
            return raw
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-") or "session"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"{cleaned[:48]}-{digest}"

    def _resolve_cli_session_id(self, runtime: Dict[str, str], session_key: str) -> Optional[str]:
        raw = str(session_key or "").strip()
        if not raw:
            return None
        if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", raw):
            return raw
        sessions_path = Path(str(runtime["state_dir"])) / "agents" / "main" / "sessions" / "sessions.json"
        try:
            sessions = self._read_json(sessions_path)
        except Exception:
            return None
        if not isinstance(sessions, dict):
            return None
        candidates: List[str] = []
        normalized = self._normalize_agent_session_key(raw)
        for candidate in (
            raw,
            normalized,
            f"agent:main:{raw}",
            f"agent:main:{normalized}",
        ):
            clean = str(candidate or "").strip()
            if clean and clean not in candidates:
                candidates.append(clean)
        for candidate in candidates:
            row = sessions.get(candidate)
            if not isinstance(row, dict):
                continue
            session_id = str(row.get("sessionId") or "").strip()
            if session_id:
                return session_id
        return None

    def _clear_cli_resume_state(self, runtime: Dict[str, str], session_key: str) -> None:
        sessions_path = Path(str(runtime["state_dir"])) / "agents" / "main" / "sessions" / "sessions.json"
        try:
            payload = self._read_json(sessions_path)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        raw = str(session_key or "").strip()
        normalized = self._normalize_agent_session_key(raw)
        candidates: List[str] = []
        for candidate in (raw, normalized, f"agent:main:{raw}", f"agent:main:{normalized}"):
            clean = str(candidate or "").strip()
            if clean and clean not in candidates:
                candidates.append(clean)
        changed = False
        now_ms = int(time.time() * 1000)
        for candidate in candidates:
            row = payload.get(candidate)
            if not isinstance(row, dict):
                continue
            if row.pop("cliSessionIds", None) is not None:
                changed = True
            if row.pop("activeRunId", None) is not None:
                changed = True
            if row.pop("pendingRunId", None) is not None:
                changed = True
            if row.get("abortedLastRun") is not False:
                row["abortedLastRun"] = False
                changed = True
            if row.get("updatedAt") != now_ms:
                row["updatedAt"] = now_ms
                changed = True
        if not changed:
            return
        with contextlib.suppress(Exception):
            sessions_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _send_message_via_websocket(
        self,
        runtime: Dict[str, str],
        session_key: str,
        text: str,
        timeout_ms: int,
    ) -> str:
        async with self._connect(runtime) as ws:
            inbox: List[Dict[str, object]] = []
            await self._connect_session(ws, runtime, inbox)
            baseline = None
            with contextlib.suppress(Exception):
                baseline = await self._latest_assistant_message(
                    ws,
                    session_key,
                    inbox,
                    limit=4,
                    timeout_ms=1500,
                )
            run_id = f"assistant-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
            response = await self._rpc_request(
                ws,
                "chat.send",
                {
                    "sessionKey": session_key,
                    "message": str(text),
                    "deliver": False,
                    "timeoutMs": timeout_ms,
                    "idempotencyKey": run_id,
                },
                inbox,
                timeout_ms=max(timeout_ms, 15000),
            )
            if not response.get("ok"):
                raise OpenClawGatewayError(f"OpenClaw chat.send failed: {response.get('error')}")
            return await self._wait_for_reply(ws, session_key, run_id, baseline, inbox, timeout_ms)

    async def _send_message_via_agent(
        self,
        runtime: Dict[str, str],
        session_key: str,
        text: str,
        timeout_ms: int,
    ) -> str:
        repo_root = Path(str(self.config.repo_path or "")).expanduser().resolve()
        launcher = repo_root / "scripts" / "run-node.mjs"
        if launcher.exists():
            launcher_args = ["node", str(launcher)]
        else:
            launcher = repo_root / "openclaw.mjs"
            if not launcher.exists():
                raise OpenClawGatewayError(f"OpenClaw launcher not found: {launcher}")
            launcher_args = ["node", str(launcher)]
        env = {
            **os.environ,
            "OPENCLAW_STATE_DIR": str(runtime["state_dir"]),
        }
        env["OPENCLAW_GATEWAY_URL"] = str(runtime.get("url") or self.config.url)
        env["OPENCLAW_GATEWAY_ORIGIN"] = str(runtime.get("origin") or self.config.origin)
        env.update(build_openclaw_proxy_env(env))
        codex_home = self._prepare_codex_home(runtime)
        env["CODEX_HOME"] = str(codex_home)
        codex_tmp = codex_home / "tmp"
        codex_tmp.mkdir(parents=True, exist_ok=True)
        env["TMP"] = str(codex_tmp)
        env["TEMP"] = str(codex_tmp)
        env["TMPDIR"] = str(codex_tmp)
        command_timeout_s = min(180.0, max(15.0, timeout_ms / 1000.0))
        if os.name == "nt":
            stdout, stderr, payload_text, returncode = await self._run_windows_command(
                [
                    *launcher_args,
                    "agent",
                    "--session-id",
                    session_key,
                    "--message",
                    text,
                    "--thinking",
                    "low",
                    "--json",
                ],
                cwd=str(repo_root),
                env=env,
                timeout_s=command_timeout_s,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *launcher_args,
                "agent",
                "--session-id",
                session_key,
                "--message",
                text,
                "--thinking",
                "low",
                "--json",
                cwd=str(repo_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr, payload_text = await self._collect_agent_output(
                process, timeout_s=command_timeout_s
            )
            returncode = process.returncode
        stderr_payload = self._extract_agent_payload_text(stderr)
        if payload_text:
            return payload_text
        if stderr_payload:
            return stderr_payload
        direct_cli_markers = (
            "spawn EPERM",
            "spawn eperm",
            "gateway closed",
            "returned no text payload",
        )
        agent_failed = returncode != 0 or not stdout.strip()
        direct_cli_available = self._direct_cli_fallback_available(runtime)
        if direct_cli_available and (
            agent_failed or any(marker in stderr.lower() for marker in [m.lower() for m in direct_cli_markers])
        ):
            return await self._send_message_via_direct_cli(runtime, text, timeout_ms)
        if returncode != 0:
            raise OpenClawGatewayError(
                "OpenClaw agent command failed: "
                f"stderr={self._summarize_cli_output(stderr)} stdout={self._summarize_cli_output(stdout)}"
            )
        output = stdout.strip()
        payload_text = self._extract_agent_payload_text(output)
        if payload_text:
            return payload_text
        if not output:
            raise OpenClawGatewayError(
                "OpenClaw agent command produced no output: "
                f"stderr={self._summarize_cli_output(stderr)}"
            )
        raise OpenClawGatewayError(
            "OpenClaw agent command returned no text payload: "
            f"stdout={self._summarize_cli_output(output)} stderr={self._summarize_cli_output(stderr)}"
        )

    def _direct_cli_fallback_available(self, runtime: Dict[str, str]) -> bool:
        try:
            defaults = self._load_agent_defaults(runtime)
        except Exception:
            return False
        cli_backends = defaults.get("cliBackends") or {}
        backend = cli_backends.get("codex-cli") if isinstance(cli_backends, dict) else None
        if not isinstance(backend, dict):
            return False
        primary_model = str((defaults.get("model") or {}).get("primary") or "").strip()
        return primary_model.lower().startswith("codex-cli/")

    def _load_agent_defaults(self, runtime: Dict[str, str]) -> Dict[str, object]:
        openclaw_json = self._read_json(Path(str(runtime["state_dir"])) / "openclaw.json")
        defaults = ((openclaw_json.get("agents") or {}).get("defaults") or {}) if isinstance(openclaw_json, dict) else {}
        return defaults if isinstance(defaults, dict) else {}

    @staticmethod
    def _summarize_cli_output(text: str, limit: int = 400) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        return re.sub(r"\s+", " ", raw)[:limit]

    async def _send_message_via_direct_cli(
        self,
        runtime: Dict[str, str],
        text: str,
        timeout_ms: int,
    ) -> str:
        repo_root = Path(str(self.config.repo_path or "")).expanduser().resolve()
        defaults = self._load_agent_defaults(runtime)
        cli_backends = (defaults.get("cliBackends") or {}) if isinstance(defaults, dict) else {}
        backend = cli_backends.get("codex-cli") if isinstance(cli_backends, dict) else None
        if not isinstance(backend, dict):
            raise OpenClawGatewayError("OpenClaw direct CLI fallback unavailable: codex-cli backend missing")
        command = str(backend.get("command") or "").strip()
        args = [str(item) for item in (backend.get("args") or []) if str(item).strip()]
        if not command or not args:
            raise OpenClawGatewayError("OpenClaw direct CLI fallback unavailable: codex-cli command incomplete")
        primary_model = str(((defaults.get("model") or {}) if isinstance(defaults, dict) else {}).get("primary") or "").strip()
        if "/" in primary_model:
            provider_id, model_id = primary_model.split("/", 1)
            if provider_id.strip().lower() == "codex-cli" and model_id.strip():
                model_arg = str(backend.get("modelArg") or "").strip()
                if model_arg and model_arg not in args:
                    args.extend([model_arg, model_id.strip()])
        env = {
            **os.environ,
            "OPENCLAW_STATE_DIR": str(runtime["state_dir"]),
        }
        env["OPENCLAW_GATEWAY_URL"] = str(runtime.get("url") or self.config.url)
        env["OPENCLAW_GATEWAY_ORIGIN"] = str(runtime.get("origin") or self.config.origin)
        env.update(build_openclaw_proxy_env(env))
        codex_home = self._prepare_codex_home(runtime)
        env["CODEX_HOME"] = str(codex_home)
        codex_tmp = codex_home / "tmp"
        codex_tmp.mkdir(parents=True, exist_ok=True)
        env["TMP"] = str(codex_tmp)
        env["TEMP"] = str(codex_tmp)
        env["TMPDIR"] = str(codex_tmp)
        command_timeout_s = min(180.0, max(15.0, timeout_ms / 1000.0))
        prompt_arg = str(text)
        payload_text: Optional[str] = None
        if os.name == "nt":
            stdout_text, stderr_text, payload_text, returncode = await self._run_windows_command(
                [command, *args, prompt_arg],
                cwd=str(repo_root),
                env=env,
                timeout_s=command_timeout_s,
            )
        else:
            try:
                process = await asyncio.create_subprocess_exec(
                    command,
                    *args,
                    prompt_arg,
                    cwd=str(repo_root),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as exc:
                raise OpenClawGatewayError(f"OpenClaw direct CLI spawn failed: {exc}") from exc
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=command_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                await self._stop_agent_process(process)
                raise OpenClawGatewayError("OpenClaw direct CLI command timed out") from exc
            stdout_text = stdout.decode("utf-8", errors="ignore").strip()
            stderr_text = stderr.decode("utf-8", errors="ignore").strip()
            returncode = process.returncode
        payload_text = payload_text or self._extract_agent_payload_text(stdout_text)
        if payload_text:
            return payload_text
        if returncode == 0 and stdout_text:
            return stdout_text
        detail = stderr_text or stdout_text or f"exit code {returncode}"
        raise OpenClawGatewayError(f"OpenClaw direct CLI command failed: {detail}")

    async def _run_windows_command(
        self,
        argv: List[str],
        cwd: str,
        env: Dict[str, str],
        timeout_s: float,
        input_text: Optional[str] = None,
    ) -> tuple[str, str, Optional[str], int]:
        def _invoke() -> tuple[str, str, int]:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                input=input_text.encode("utf-8") if input_text is not None else None,
                capture_output=True,
                timeout=timeout_s,
            )
            stdout_text = completed.stdout.decode("utf-8", errors="ignore").strip()
            stderr_text = completed.stderr.decode("utf-8", errors="ignore").strip()
            return stdout_text, stderr_text, int(completed.returncode or 0)

        try:
            stdout_text, stderr_text, returncode = await asyncio.to_thread(_invoke)
        except subprocess.TimeoutExpired as exc:
            raise OpenClawGatewayError("OpenClaw agent command timed out") from exc
        payload_text = self._extract_agent_payload_text(stdout_text)
        return stdout_text, stderr_text, payload_text, returncode

    def _prepare_codex_home(self, runtime: Dict[str, str]) -> Path:
        configured = str(os.environ.get("CODEX_HOME", "") or "").strip()
        if configured:
            codex_home = Path(configured).expanduser()
        elif str(self.config.codex_home or "").strip():
            codex_home = Path(str(self.config.codex_home)).expanduser()
        else:
            codex_home = Path(str(self.config.workspace_dir)).expanduser().resolve().parent / "codex_home" / "runtime"
        codex_home.mkdir(parents=True, exist_ok=True)
        source_home = Path.home() / ".codex"
        for name in ("auth.json", "cap_sid"):
            src = source_home / name
            dest = codex_home / name
            if src.exists():
                try:
                    if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
                        shutil.copy2(src, dest)
                except Exception:
                    pass
        self._repair_codex_home_state(codex_home)
        (codex_home / "config.toml").write_text(self._build_codex_home_config(runtime), encoding="utf-8")
        return codex_home

    def _repair_codex_home_state(self, codex_home: Path) -> None:
        for candidate in [
            codex_home / ".tmp" / "plugins",
            codex_home / ".tmp" / "plugins.sha",
            codex_home / "cache" / "codex_apps_tools",
        ]:
            with contextlib.suppress(Exception):
                if candidate.exists():
                    if candidate.is_dir():
                        shutil.rmtree(candidate, ignore_errors=True)
                    else:
                        candidate.unlink()
        with contextlib.suppress(Exception):
            tmp_dir = codex_home / ".tmp"
            if tmp_dir.exists():
                for child in tmp_dir.iterdir():
                    if child.is_dir() and child.name.lower().startswith("plugins-clone-"):
                        shutil.rmtree(child, ignore_errors=True)
        state_db = codex_home / "state_5.sqlite"
        journal_candidates = [
            codex_home / "state_5.sqlite-journal",
            codex_home / "state_5.sqlite-shm",
            codex_home / "state_5.sqlite-wal",
        ]
        # Keep the OAuth/token files but always rebuild Codex's transient state db.
        # The desktop environment tends to leave this sqlite runtime in a corrupted
        # or locked state, while a clean home reliably boots the provider.
        if state_db.exists():
            with contextlib.suppress(Exception):
                state_db.unlink()
            for candidate in journal_candidates:
                with contextlib.suppress(Exception):
                    candidate.unlink()
            return
        for candidate in journal_candidates:
            with contextlib.suppress(Exception):
                candidate.unlink()

    def _build_codex_home_config(self, runtime: Dict[str, str]) -> str:
        model_name = "glm-5"
        try:
            openclaw_json = self._read_json(Path(str(runtime["state_dir"])) / "openclaw.json")
            primary = str(((openclaw_json.get("agents") or {}).get("defaults") or {}).get("model", {}).get("primary") or "").strip()
            if primary.startswith("codex-cli/"):
                model_name = primary.split("/", 1)[1] or "glm-5"
            elif "/" in primary:
                model_name = primary.split("/", 1)[1] or model_name
            elif primary:
                model_name = primary
        except Exception:
            pass
        return (
            f'model = "{model_name}"\n'
            'model_reasoning_effort = "low"\n'
            'personality = "pragmatic"\n\n'
            '[features]\n'
            'plugins = false\n'
            'shell_snapshot = false\n\n'
        )

    async def _collect_agent_output(
        self,
        process: asyncio.subprocess.Process,
        timeout_s: float,
    ) -> tuple[str, str, Optional[str]]:
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        pending: Dict[asyncio.Task[bytes], str] = {}
        if process.stdout is not None:
            pending[asyncio.create_task(process.stdout.read(4096))] = "stdout"
        if process.stderr is not None:
            pending[asyncio.create_task(process.stderr.read(4096))] = "stderr"
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    await self._stop_agent_process(process)
                    raise OpenClawGatewayError("OpenClaw agent command timed out")
                done, _ = await asyncio.wait(
                    pending.keys(),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    await self._stop_agent_process(process)
                    raise OpenClawGatewayError("OpenClaw agent command timed out")
                for task in done:
                    stream_name = pending.pop(task)
                    chunk = task.result()
                    if not chunk:
                        continue
                    text = chunk.decode("utf-8", errors="ignore")
                    if stream_name == "stdout":
                        stdout_parts.append(text)
                        payload_text = self._extract_agent_payload_text("".join(stdout_parts))
                        if payload_text:
                            await self._stop_agent_process(process)
                            return "".join(stdout_parts), "".join(stderr_parts), payload_text
                        if process.stdout is not None:
                            pending[asyncio.create_task(process.stdout.read(4096))] = "stdout"
                    else:
                        stderr_parts.append(text)
                        if process.stderr is not None:
                            pending[asyncio.create_task(process.stderr.read(4096))] = "stderr"
            await asyncio.wait_for(process.wait(), timeout=1.0)
            return "".join(stdout_parts), "".join(stderr_parts), None
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending.keys(), return_exceptions=True)

    async def _stop_agent_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=1.0)

    @classmethod
    def _extract_agent_payload_text(cls, output: str) -> Optional[str]:
        for parsed in reversed(cls._extract_agent_json_candidates(output)):
            if isinstance(parsed, dict):
                item = parsed.get("item")
                if isinstance(item, dict):
                    item_type = str(item.get("type") or "").strip().lower()
                    item_text = str(item.get("text") or "").strip()
                    if item_type == "agent_message" and item_text:
                        return item_text
            payloads = (((parsed.get("result") or {}).get("payloads")) if isinstance(parsed, dict) else None) or []
            for payload in payloads:
                text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
                if text:
                    return text
        return None

    @classmethod
    def _extract_agent_json_candidates(cls, output: str) -> List[Dict[str, object]]:
        raw = str(output or "").strip()
        if not raw:
            return []
        parsed_objects: List[Dict[str, object]] = []
        for line in raw.splitlines():
            parsed = cls._try_extract_agent_json(line)
            if parsed is not None:
                parsed_objects.append(parsed)
        parsed_whole = cls._try_extract_agent_json(raw)
        if parsed_whole is not None and (not parsed_objects or parsed_objects[-1] != parsed_whole):
            parsed_objects.append(parsed_whole)
        return parsed_objects

    @staticmethod
    def _try_extract_agent_json(raw: str) -> Optional[Dict[str, object]]:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_agent_json(output: str) -> Dict[str, object]:
        raw = str(output or "").strip()
        if not raw:
            raise OpenClawGatewayError("OpenClaw agent command produced no output")
        parsed = OpenClawGatewayClient._try_extract_agent_json(raw)
        if parsed is None:
            raise OpenClawGatewayError(f"OpenClaw agent command returned non-JSON output: {raw[:200]}")
        return parsed

    @staticmethod
    def _read_json(path: Path) -> Dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def _connect(self, runtime: Dict[str, str]):
        if websockets is None:
            raise OpenClawGatewayError("websockets dependency missing")
        return websockets.connect(
            runtime["url"],
            origin=runtime["origin"],
            max_size=20_000_000,
            proxy=None,
        )

    async def _connect_session(self, ws, runtime: Dict[str, str], inbox: List[Dict[str, object]]) -> None:
        private_key = load_pem_private_key(runtime["private_key_pem"].encode("utf-8"), password=None)
        public_key = load_pem_public_key(runtime["public_key_pem"].encode("utf-8"))
        public_raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        public_b64 = self._b64url_no_pad(public_raw)
        if not runtime["device_id"]:
            raise OpenClawGatewayError("OpenClaw device id missing")
        nonce: Optional[str] = None
        try:
            first, _ = await self._recv_json(ws, inbox, timeout=min(1.2, max(0.1, 1.2)))
            if first.get("type") == "event" and first.get("event") == "connect.challenge":
                nonce = str((first.get("payload") or {}).get("nonce") or "") or None
        except Exception:
            nonce = None
        scopes = ["operator.admin", "operator.approvals", "operator.pairing", "operator.read", "operator.write"]
        signed_at = int(time.time() * 1000)
        version = "v2" if nonce else "v1"
        sign_input = self._make_sign_input(
            version=version,
            device_id=runtime["device_id"],
            client_id=self.config.client_id,
            client_mode=self.config.client_mode,
            role="operator",
            scopes=scopes,
            signed_at_ms=signed_at,
            token=runtime["token"],
            nonce=nonce,
        ).encode("utf-8")
        signature = self._b64url_no_pad(private_key.sign(sign_input))
        response = await self._rpc_request(
            ws,
            "connect",
            {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": self.config.client_id,
                    "version": "dev",
                    "platform": "Win32",
                    "mode": self.config.client_mode,
                },
                "role": "operator",
                "scopes": scopes,
                "caps": [],
                "auth": {"token": runtime["token"]},
                "device": {
                    "id": runtime["device_id"],
                    "publicKey": public_b64,
                    "signature": signature,
                    "signedAt": signed_at,
                    **({"nonce": nonce} if nonce else {}),
                },
                "userAgent": "chonggou-backend",
                "locale": "zh-CN",
            },
            inbox,
            timeout_ms=3500,
        )
        if not response.get("ok"):
            raise OpenClawGatewayError(
                self._describe_gateway_error("OpenClaw connect failed", response.get("error"))
            )

    async def _wait_for_reply(
        self,
        ws,
        session_key: str,
        run_id: str,
        baseline: Optional[Dict[str, object]],
        inbox: List[Dict[str, object]],
        timeout_ms: int,
    ) -> str:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                msg, _ = await self._recv_json(ws, inbox, timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                msg = None
            if msg and msg.get("type") == "event" and msg.get("event") == "chat":
                payload = msg.get("payload") or {}
                payload_session_key = str(payload.get("sessionKey") or "").strip()
                if payload_session_key and payload_session_key != session_key:
                    continue
                state = str(payload.get("state") or "")
                assistant_message = self._extract_assistant_message(payload.get("message") or payload)
                if assistant_message and self._is_new_assistant_message(assistant_message, baseline):
                    if state in {"final", "completed", "done"}:
                        final_text = str(assistant_message.get("text") or "").strip()
                        if final_text:
                            return final_text
                if payload.get("runId") == run_id and state in {"error", "aborted"}:
                    raise OpenClawGatewayError(f"OpenClaw chat {state}: {payload.get('errorMessage')}")
        with contextlib.suppress(Exception):
            latest = await self._latest_assistant_message(
                ws,
                session_key,
                inbox,
                limit=4,
                timeout_ms=1500,
            )
            if latest and self._is_new_assistant_message(latest, baseline):
                latest_text = str(latest.get("text") or "").strip()
                if latest_text:
                    return latest_text
        try:
            await self._rpc_request(
                ws,
                "chat.abort",
                {"sessionKey": session_key, "runId": run_id},
                inbox,
                timeout_ms=3000,
            )
        except Exception:
            pass
        raise OpenClawGatewayError("OpenClaw chat timed out")

    async def _latest_assistant_message(
        self,
        ws,
        session_key: str,
        inbox: List[Dict[str, object]],
        *,
        limit: int = 20,
        timeout_ms: int = 8000,
    ) -> Optional[Dict[str, object]]:
        response = await self._rpc_request(
            ws,
            "chat.history",
            {"sessionKey": session_key, "limit": int(limit)},
            inbox,
            timeout_ms=int(timeout_ms),
        )
        payload = response.get("payload") if isinstance(response, dict) else None
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list):
            return None
        for message in reversed(messages):
            parsed = self._extract_assistant_message(message)
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _extract_assistant_message(cls, message_obj: object) -> Optional[Dict[str, object]]:
        if not isinstance(message_obj, dict):
            return None
        if str(message_obj.get("role") or "") != "assistant":
            return None
        text = cls._extract_text_from_message(message_obj).strip()
        if not text:
            return None
        timestamp = int(message_obj.get("timestamp") or 0)
        return {"timestamp": timestamp, "text": text}

    @staticmethod
    def _is_new_assistant_message(
        latest: Optional[Dict[str, object]],
        baseline: Optional[Dict[str, object]],
    ) -> bool:
        if latest is None:
            return False
        if baseline is None:
            return bool(str(latest.get("text") or "").strip())
        latest_ts = int(latest.get("timestamp") or 0)
        baseline_ts = int(baseline.get("timestamp") or 0)
        if latest_ts > baseline_ts:
            return True
        return str(latest.get("text") or "").strip() != str(baseline.get("text") or "").strip()

    async def _rpc_request(
        self,
        ws,
        method: str,
        params: Dict[str, object],
        inbox: List[Dict[str, object]],
        timeout_ms: int,
    ) -> Dict[str, object]:
        req_id = str(uuid.uuid4())
        await ws.send(json.dumps({"type": "req", "id": req_id, "method": method, "params": params}, ensure_ascii=False))
        deadline = time.monotonic() + max(1.0, timeout_ms / 1000.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenClawGatewayError(f"OpenClaw rpc timeout: {method}")
            msg, from_inbox = await self._recv_json(ws, inbox, timeout=min(remaining, 2.0))
            if msg.get("type") == "res" and msg.get("id") == req_id:
                return msg
            if not from_inbox:
                inbox.append(msg)

    async def _recv_json(
        self,
        ws,
        inbox: List[Dict[str, object]],
        timeout: float,
    ) -> tuple[Dict[str, object], bool]:
        if inbox:
            return inbox.pop(0), True
        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.05, float(timeout)))
        return json.loads(raw), False

    @staticmethod
    def _extract_text_from_message(message_obj: Dict[str, object]) -> str:
        parts = message_obj.get("content") if isinstance(message_obj, dict) else None
        if not isinstance(parts, list):
            return ""
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
        return "".join(chunks).strip()

    @staticmethod
    def _b64url_no_pad(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    @staticmethod
    def _make_sign_input(
        version: str,
        device_id: str,
        client_id: str,
        client_mode: str,
        role: str,
        scopes: List[str],
        signed_at_ms: int,
        token: str,
        nonce: Optional[str],
    ) -> str:
        parts = [version, device_id, client_id, client_mode, role, ",".join(scopes), str(signed_at_ms), token or ""]
        if version == "v2":
            parts.append(nonce or "")
        return "|".join(parts)
