const path = require("path");
const fs = require("fs");
const os = require("os");
const { spawn, spawnSync } = require("child_process");
const net = require("net");
const { app, BrowserWindow, shell, ipcMain, Tray, Menu, nativeImage, screen, Notification, session } = require("electron");
const { DeviceSyncManager } = require("./deviceSync.cjs");

let mainWindow = null;
let tray = null;
let isQuitting = false;
let floatWindow = null;
let chatWindow = null;
let floatDragState = null;
let backendProc = null;
let openClawGatewayProc = null;
let openClawAssistantReady = false;
let localTtsProc = null;
let localSttProc = null;
let backendSupervisorTimer = null;
let ensureLocalBackendPromise = null;
let assistantRuntimeWarmAtMs = 0;
const LOCAL_BACKEND_PORT = 8000;
const LOCAL_BACKEND_URL = `http://127.0.0.1:${LOCAL_BACKEND_PORT}`;
const LOCAL_OPENCLAW_GATEWAY_PORT = 18940;
const LOCAL_OPENCLAW_GATEWAY_TOKEN = "chonggou-emoresonance-openclaw-bridge";
const LOCAL_TTS_PORT = 18884;
const LOCAL_STT_PORT = 18885;
const LOCAL_TTS_BASE_URL = `http://127.0.0.1:${LOCAL_TTS_PORT}`;
const LOCAL_STT_BASE_URL = `http://127.0.0.1:${LOCAL_STT_PORT}`;
const OPENCLAW_DESKTOP_SESSION_KEY = "agent:main:desktop:emoresonance";
const ASSISTANT_RUNTIME_WARM_MAX_AGE_MS = 120000;
const LOCAL_OPENCLAW_PROVIDER = {
  providerId: "zai",
  profileId: "zai:default",
  endpoint: "coding-cn",
  baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4/",
  modelId: "glm-5",
  modelRef: "zai/glm-5",
  thinkingDefault: "low",
};
const deviceSyncManager = new DeviceSyncManager({
  onStatus: (payload) => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send("bridge-status", payload);
      }
    }
  },
});

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const probeTcpPort = (host, port, timeoutMs = 500) =>
  new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      try {
        socket.destroy();
      } catch {}
      resolve(value);
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    try {
      socket.connect(port, host);
    } catch {
      finish(false);
    }
  });

const getListeningPidForPort = (port) => {
  if (process.platform !== "win32") return null;
  try {
    const probe = spawnSync(
      "cmd",
      ["/c", `netstat -ano | findstr LISTENING | findstr :${port}`],
      { encoding: "utf8", windowsHide: true }
    );
    const output = String(probe.stdout || "").trim();
    if (!output) return null;
    for (const line of output.split(/\r?\n/)) {
      const parts = String(line || "").trim().split(/\s+/).filter(Boolean);
      const pid = Number(parts[parts.length - 1]);
      if (Number.isFinite(pid) && pid > 0) {
        return pid;
      }
    }
  } catch {}
  return null;
};

const stopLegacyOpenClawGateway = () => {
  if (process.platform !== "win32") return;
  try {
    spawnSync("schtasks", ["/End", "/TN", "OpenClaw Gateway"], { windowsHide: true });
  } catch {}
  const pid = getListeningPidForPort(18890);
  if (!pid) return;
  try {
    spawnSync("taskkill", ["/PID", String(pid), "/F"], { windowsHide: true });
    appendStartupLog(`stopLegacyOpenClawGateway killed legacy pid=${pid} on port=18890`);
  } catch (err) {
    appendStartupLog(`stopLegacyOpenClawGateway failed pid=${pid}: ${err?.message || err}`);
  }
};

const getStartupLogPath = () => {
  try {
    const base = app?.getPath ? app.getPath("userData") : path.join(os.homedir(), "AppData", "Roaming", "emoresonance---dual-robot-companion");
    const logsDir = path.join(base, "logs");
    fs.mkdirSync(logsDir, { recursive: true });
    return path.join(logsDir, "bridge-startup.log");
  } catch {
    return path.join(os.tmpdir(), "emoresonance-bridge-startup.log");
  }
};

const appendStartupLog = (message) => {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  try {
    fs.appendFileSync(getStartupLogPath(), line, "utf8");
  } catch {}
};

const resolveExistingPath = (candidates) => {
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {}
  }
  return "";
};

const pathEquals = (left, right) => {
  try {
    return path.resolve(String(left || "")).toLowerCase() === path.resolve(String(right || "")).toLowerCase();
  } catch {
    return String(left || "").trim().toLowerCase() === String(right || "").trim().toLowerCase();
  }
};

const isValidOpenClawStateDir = (candidate) => {
  const target = String(candidate || "").trim();
  if (!target) return false;
  try {
    return (
      fs.existsSync(path.join(target, "openclaw.json")) &&
      fs.existsSync(path.join(target, "identity", "device.json"))
    );
  } catch {
    return false;
  }
};

const resolveNativeOpenClawStateDir = () => {
  for (const candidate of [process.env.OPENCLAW_NATIVE_STATE_DIR, path.join(os.homedir(), ".openclaw")]) {
    if (isValidOpenClawStateDir(candidate)) {
      return path.resolve(candidate);
    }
  }
  return "";
};

const readOpenClawConfig = (stateDir) =>
  readJsonIfExists(path.join(String(stateDir || "").trim(), "openclaw.json")) || {};

const resolveOpenClawDesktopSessionKey = (stateDir) => {
  const explicit = String(process.env.OPENCLAW_DESKTOP_SHARED_SESSION_KEY || "").trim();
  if (explicit) return explicit;
  return OPENCLAW_DESKTOP_SESSION_KEY;
};

const resolveRuntimeRoot = () => {
  if (process.env.EMOTION_BRIDGE_ROOT) {
    return process.env.EMOTION_BRIDGE_ROOT;
  }
  if (!app.isPackaged) {
    return path.resolve(__dirname, "..", "..");
  }
  return path.join(process.resourcesPath, "bridge-runtime");
};

const resolveOpenClawRepo = (runtimeRoot) => {
  if (process.env.OPENCLAW_REPO_PATH) {
    return process.env.OPENCLAW_REPO_PATH;
  }
  const candidates = [
    "E:\\Desktop\\openclaw",
    path.resolve(runtimeRoot, "..", "openclaw"),
    path.join(runtimeRoot, ".openclaw-latest", "node_modules", "openclaw"),
    path.join(runtimeRoot, "app windows", "vendor", "openclaw-runtime"),
  ];
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(path.join(candidate, "openclaw.mjs"))) {
        return candidate;
      }
    } catch {}
  }
  return candidates[0];
};

const resolveSenseVoiceModelDir = () =>
  resolveExistingPath([
    process.env.SENSEVOICE_MODEL_DIR,
    process.env.LOCAL_STT_MODEL_DIR,
    path.join(os.homedir(), ".cache", "modelscope", "hub", "iic", "SenseVoiceSmall"),
    path.join("E:\\model_cache", "SenseVoiceSmall"),
  ]);

const resolveVoxBaseModelDir = (runtimeRoot) =>
  resolveExistingPath([
    process.env.VOXCPM_DISTILL_BASE,
    process.env.VOXCPM_BASE_MODEL,
    path.join("E:\\model_cache", "VoxCPM-0.5B"),
    path.join(runtimeRoot, "assistant_data", "models", "VoxCPM-0.5B"),
    path.join(runtimeRoot, "models", "VoxCPM-0.5B"),
  ]);

const resolveVoxLoraWeightsPath = (runtimeRoot) =>
  resolveExistingPath([
    process.env.VOXCPM_DISTILL_LORA,
    path.join(runtimeRoot, "assistant_data", "models", "voxcpm_distill", "lora_weights.safetensors"),
  ]);

const resolveVoxLoraConfigPath = (runtimeRoot) =>
  resolveExistingPath([
    process.env.VOXCPM_DISTILL_LORA_CONFIG,
    path.join(runtimeRoot, "assistant_data", "models", "voxcpm_distill", "lora_config.json"),
  ]);

const resolveVoxSweetPromptPath = (runtimeRoot) =>
  resolveExistingPath([
    process.env.VOXCPM_SWEET_PROMPT_WAV,
    path.join(runtimeRoot, "assistant_data", "voice_refs", "sweet_female_prompt.wav"),
  ]);

const resolveBridgeScriptPath = (runtimeRoot, filename) =>
  resolveExistingPath([
    path.join(runtimeRoot, "scripts", filename),
    path.resolve(runtimeRoot, "..", "scripts", filename),
  ]);

const resolveOpenClawWorkspace = (runtimeRoot) => {
  const configured = String(process.env.OPENCLAW_WORKSPACE_DIR || "").trim();
  if (configured) {
    return configured;
  }
  return path.join(runtimeRoot, "assistant_data", "openclaw_workspace");
};

const resolveOpenClawStateDir = (runtimeRoot) => {
  const configured = String(process.env.OPENCLAW_STATE_DIR || "").trim();
  if (isValidOpenClawStateDir(configured)) {
    return path.resolve(configured);
  }
  const preferNative = String(process.env.OPENCLAW_USE_NATIVE_STATE || "").trim().toLowerCase();
  if (preferNative === "1" || preferNative === "true" || preferNative === "yes") {
    const nativeState = resolveNativeOpenClawStateDir();
    if (nativeState) {
      return nativeState;
    }
  }
  return path.join(runtimeRoot, "assistant_data", "openclaw_state");
};

const resolveOpenClawCodexHome = (runtimeRoot) => {
  if (process.env.OPENCLAW_CODEX_HOME) {
    return process.env.OPENCLAW_CODEX_HOME;
  }
  if (process.env.LOCALAPPDATA) {
    return path.join(process.env.LOCALAPPDATA, "EmoResonance", "codex_home");
  }
  return path.join(runtimeRoot, "assistant_data", "codex_home");
};

const resolveOpenClawProxyUrl = () => {
  for (const key of ["OPENCLAW_PROXY_URL", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"]) {
    if (process.env[key]) {
      return process.env[key];
    }
  }
  if (process.platform === "win32") {
    for (const port of [7897, 7890, 10808, 1080]) {
      try {
        const probe = spawnSync(
          "cmd",
          ["/c", `netstat -ano | findstr LISTENING | findstr 127.0.0.1:${port}`],
          { stdio: "ignore", windowsHide: true }
        );
        if (probe.status === 0) {
          return `http://127.0.0.1:${port}`;
        }
      } catch {}
    }
  }
  return "";
};

const buildOpenClawProxyEnv = () => {
  const proxyUrl = resolveOpenClawProxyUrl();
  if (!proxyUrl) {
    return {};
  }
  return {
    OPENCLAW_PROXY_URL: proxyUrl,
    HTTPS_PROXY: proxyUrl,
    HTTP_PROXY: proxyUrl,
    ALL_PROXY: proxyUrl,
    https_proxy: proxyUrl,
    http_proxy: proxyUrl,
    all_proxy: proxyUrl,
  };
};

const LOCAL_NO_PROXY = "127.0.0.1,localhost";

const buildLocalNoProxyEnv = () => ({
  OPENCLAW_PROXY_URL: "",
  HTTPS_PROXY: "",
  HTTP_PROXY: "",
  ALL_PROXY: "",
  https_proxy: "",
  http_proxy: "",
  all_proxy: "",
  NO_PROXY: LOCAL_NO_PROXY,
  no_proxy: LOCAL_NO_PROXY,
});

const resolveAppDataRoot = () => {
  try {
    if (app?.getPath) {
      const userData = app.getPath("userData");
      if (userData) {
        fs.mkdirSync(userData, { recursive: true });
        return userData;
      }
    }
  } catch {}
  const fallback = path.join(
    process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming"),
    "emoresonance---dual-robot-companion"
  );
  fs.mkdirSync(fallback, { recursive: true });
  return fallback;
};

const resolveOpenClawProviderConfigPath = () => {
  const primary = path.join(resolveAppDataRoot(), "openclaw-provider.json");
  const candidates = [
    primary,
    path.join(process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming"), "emoresonance---dual-robot-companion", "openclaw-provider.json"),
    path.join(process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming"), "EmoResonance", "openclaw-provider.json"),
  ];
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {}
  }
  return primary;
};

const readJsonIfExists = (pathname) => {
  try {
    if (!fs.existsSync(pathname)) {
      return null;
    }
    const raw = fs.readFileSync(pathname, "utf8").replace(/^\uFEFF/, "");
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

const loadOpenClawProviderConfig = () => {
  const fileConfig = readJsonIfExists(resolveOpenClawProviderConfigPath()) || {};
  const modelId = String(fileConfig.modelId || LOCAL_OPENCLAW_PROVIDER.modelId).trim() || LOCAL_OPENCLAW_PROVIDER.modelId;
  const apiKey = String(
    process.env.ZAI_API_KEY ||
      process.env.Z_AI_API_KEY ||
      fileConfig.apiKey ||
      ""
  ).trim();
  return {
    providerId: LOCAL_OPENCLAW_PROVIDER.providerId,
    profileId: LOCAL_OPENCLAW_PROVIDER.profileId,
    endpoint: String(fileConfig.endpoint || LOCAL_OPENCLAW_PROVIDER.endpoint).trim() || LOCAL_OPENCLAW_PROVIDER.endpoint,
    baseUrl: String(fileConfig.baseUrl || LOCAL_OPENCLAW_PROVIDER.baseUrl).trim() || LOCAL_OPENCLAW_PROVIDER.baseUrl,
    modelId,
    modelRef: `zai/${modelId}`,
    thinkingDefault: String(fileConfig.thinkingDefault || LOCAL_OPENCLAW_PROVIDER.thinkingDefault).trim() || LOCAL_OPENCLAW_PROVIDER.thinkingDefault,
    apiKey,
  };
};

const buildOpenClawProviderEnv = () => {
  const providerConfig = loadOpenClawProviderConfig();
  if (!providerConfig || !providerConfig.apiKey) {
    return {};
  }
  return {
    ZAI_API_KEY: providerConfig.apiKey,
    Z_AI_API_KEY: providerConfig.apiKey,
  };
};

const loadOpenClawModelPreference = (stateDir) => {
  const cfg = readOpenClawConfig(stateDir);
  const primary = String(cfg?.agents?.defaults?.model?.primary || "").trim();
  const thinkingDefault =
    String(cfg?.agents?.defaults?.thinkingDefault || "").trim() || LOCAL_OPENCLAW_PROVIDER.thinkingDefault;
  if (primary.startsWith("codex-cli/")) {
    return { model: primary.split("/", 2)[1] || LOCAL_OPENCLAW_PROVIDER.modelId, thinkingDefault };
  }
  if (primary.includes("/")) {
    return { model: primary.split("/", 2)[1] || LOCAL_OPENCLAW_PROVIDER.modelId, thinkingDefault };
  }
  return { model: primary || LOCAL_OPENCLAW_PROVIDER.modelId, thinkingDefault };
};

const parseLatestActivationProfile = (workspaceDir) => {
  const usersRoot = path.join(workspaceDir, "assistant_data", "users");
  if (!fs.existsSync(usersRoot)) {
    return null;
  }
  let latest = null;
  for (const entry of fs.readdirSync(usersRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const userId = String(entry.name || "").trim();
    if (!/^\d+$/.test(userId)) continue;
    const memoryPath = path.join(usersRoot, userId, "memory.md");
    if (!fs.existsSync(memoryPath)) continue;
    let text = "";
    try {
      text = fs.readFileSync(memoryPath, "utf8");
    } catch {
      continue;
    }
    if (!text.trim()) continue;
    const chunks = text
      .split(/^##\s/m)
      .map((chunk) => chunk.trim())
      .filter(Boolean);
    for (const chunk of chunks) {
      const normalized = `## ${chunk}`;
      if (!normalized.includes("activation_profile")) continue;
      const match = normalized.match(
        /称呼：([^；\n]+)；角色：([^；\n]+)；关系：([^；\n]+)；摘要：([^\n]+)/
      );
      if (!match) continue;
      let mtimeMs = 0;
      try {
        mtimeMs = fs.statSync(memoryPath).mtimeMs;
      } catch {}
      const candidate = {
        userId,
        preferredName: match[1].trim(),
        roleLabel: match[2].trim(),
        relationToRobot: match[3].trim(),
        identitySummary: match[4].trim(),
        memoryPath,
        mtimeMs,
      };
      if (!latest || candidate.mtimeMs >= latest.mtimeMs) {
        latest = candidate;
      }
    }
  }
  return latest;
};

const writeIfChanged = (targetPath, content) => {
  const next = String(content || "");
  try {
    if (fs.existsSync(targetPath)) {
      const current = fs.readFileSync(targetPath, "utf8");
      if (current === next) {
        return;
      }
    }
  } catch {}
  fs.writeFileSync(targetPath, next, "utf8");
};

const ensureAgentModelsConfig = (agentDir, providerConfig) => {
  const target = path.join(agentDir, "models.json");
  const current = readJsonIfExists(target) || {};
  current.providers = current.providers || {};
  current.providers[providerConfig.providerId] = {
    ...(current.providers[providerConfig.providerId] || {}),
    baseUrl: providerConfig.baseUrl,
    apiKey: providerConfig.apiKey || (current.providers[providerConfig.providerId] || {}).apiKey || "",
    api: "openai-completions",
    models: [
      {
        id: providerConfig.modelId,
        name: "GLM-5",
        reasoning: true,
        input: ["text"],
        contextWindow: 204800,
        maxTokens: 131072,
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
        },
      },
    ],
  };
  fs.writeFileSync(target, JSON.stringify(current, null, 2), "utf8");
};

const syncOpenClawWorkspaceProfile = (workspaceDir, runtimeRoot) => {
  const profile = parseLatestActivationProfile(workspaceDir);
  const repoSummary = `- Canonical project root: ${runtimeRoot}
- Canonical workspace memory: ${path.join(workspaceDir, "assistant_data", "users")}
- Do not assume USER.md/IDENTITY.md are placeholders; they are synced from current product data when available.
`;
  if (!profile) {
    const fallbackUser = `# USER.md

- Preferred Name: 当前桌面用户
- Role: owner
- Relation To Robot: primary_user
- Timezone: Asia/Shanghai
- Product: 心念双灵
- Identity Summary: 当前为本地桌面默认用户；如无明确建档信息，请正常聊天，不要主动讨论未建档或未激活。

## Repo Sync
${repoSummary}`;
    const fallbackIdentityNeutral = `# IDENTITY.md

- Status: local_desktop_default
- Preferred Name: 当前桌面用户
- Role: owner
- Relation To Robot: primary_user
- Summary: 当前为本地桌面默认用户；若没有明确的人物档案，请保持中性陪伴，不要主动提未建档、未激活或身份缺失。

## Repo Sync
${repoSummary}`;
    writeIfChanged(path.join(workspaceDir, "USER.md"), fallbackUser);
    writeIfChanged(path.join(workspaceDir, "IDENTITY.md"), fallbackIdentityNeutral);
    return;
    const fallbackIdentity = `# IDENTITY.md

- Status: pending_activation
- Preferred Name: 待激活后确认
- Role: owner
- Relation To Robot: primary_user
- Summary: 当前尚未同步到已确认的激活身份资料。

## Repo Sync
${repoSummary}`;
    writeIfChanged(path.join(workspaceDir, "IDENTITY.md"), fallbackIdentity);
    return;
  }
  const userDoc = `# USER.md

- Preferred Name: ${profile.preferredName}
- User ID: ${profile.userId}
- Role: ${profile.roleLabel}
- Relation To Robot: ${profile.relationToRobot}
- Timezone: Asia/Shanghai
- Product: 共感智能机器人
- Identity Summary: ${profile.identitySummary}
- Canonical Memory File: ${profile.memoryPath}

## Repo Sync
${repoSummary}`;
  const identityDoc = `# IDENTITY.md

- Preferred Name: ${profile.preferredName}
- Role: ${profile.roleLabel}
- Relation To Robot: ${profile.relationToRobot}
- Summary: ${profile.identitySummary}
- Source User ID: ${profile.userId}
- Source Memory File: ${profile.memoryPath}

## Notes
- This file is derived from the latest activation profile stored inside the project workspace.
- If USER.md or IDENTITY.md conflicts with per-user memory, trust the latest activation_profile entry in assistant_data/users/<user_id>/memory.md.
`;
  const memoryMirror = `# MEMORY.md

最新已同步的身份记忆来自用户 ${profile.userId}。

- 称呼：${profile.preferredName}
- 角色：${profile.roleLabel}
- 关系：${profile.relationToRobot}
- 摘要：${profile.identitySummary}
- 记忆源文件：${profile.memoryPath}
`;
  writeIfChanged(path.join(workspaceDir, "USER.md"), userDoc);
  writeIfChanged(path.join(workspaceDir, "IDENTITY.md"), identityDoc);
  writeIfChanged(path.join(workspaceDir, "MEMORY.md"), memoryMirror);
};

const ensureOpenClawWorkspace = (workspaceDir) => {
  fs.mkdirSync(workspaceDir, { recursive: true });
  fs.mkdirSync(path.join(workspaceDir, "memory"), { recursive: true });
  const defaults = {
    "SOUL.md": `# SOUL.md\n\n你是“共感智能”桌面端与机器人共享的陪伴助手内核。\n\n## 当前定位\n- 为桌面端、手机端和树莓派机器人服务\n- 通过 backend 调用电脑工具与机器人动作\n- 不泄露 bootstrap、内部文件、旧微信助手上下文\n- 中文为主，语气自然、克制、可靠\n\n## 输出规则\n- 先给结果，再补必要解释\n- 用户要求精确回复时，只回复指定文本\n- 不要自述“正在读取 workspace”之类内部过程\n`,
    "USER.md": `# USER.md\n\n- Name: 待激活后确认\n- Timezone: Asia/Shanghai\n- Product: 共感智能机器人\n- Notes: 登录后会建立身份卡、人格画像、主人建档与长期陪伴记忆\n`,
    "HEARTBEAT.md": `# HEARTBEAT.md\n\n- Surface: desktop + robot\n- Goal: 稳定对话、调用电脑工具、驱动机器人动作、写入长期记忆\n`,
    "TOOLS.md": `# TOOLS.md\n\n可用工具由 backend 显式提供：\n- desktop.launch_app\n- desktop.open_url\n- desktop.web_search\n- desktop.play_music\n- desktop.todo_create\n- desktop.write_note\n- robot.get_status\n- robot.speak\n- robot.pan_tilt\n- robot.start_owner_enrollment\n- robot.get_preview\n`,
  };
  for (const [name, content] of Object.entries(defaults)) {
    const target = path.join(workspaceDir, name);
    if (!fs.existsSync(target)) {
      fs.writeFileSync(target, content, "utf8");
    }
  }
  syncOpenClawWorkspaceProfile(workspaceDir, path.resolve(workspaceDir, "..", ".."));
};

const ensureOpenClawCodexHome = (runtimeRoot, workspaceDir, openClawRepo) => {
  const codexHome = resolveOpenClawCodexHome(runtimeRoot);
  const sourceCodexHome = path.join(os.homedir(), ".codex");
  fs.mkdirSync(codexHome, { recursive: true });
  fs.mkdirSync(path.join(codexHome, ".tmp"), { recursive: true });
  fs.mkdirSync(path.join(codexHome, "tmp"), { recursive: true });
  fs.mkdirSync(path.join(codexHome, "cache"), { recursive: true });
  for (const name of ["auth.json", "cap_sid"]) {
    const src = path.join(sourceCodexHome, name);
    const dest = path.join(codexHome, name);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
    }
  }
  for (const stalePath of [
    path.join(codexHome, ".tmp", "plugins"),
    path.join(codexHome, ".tmp", "plugins.sha"),
    path.join(codexHome, "cache", "codex_apps_tools"),
  ]) {
    try {
      if (fs.existsSync(stalePath)) {
        fs.rmSync(stalePath, { recursive: true, force: true });
      }
    } catch {}
  }
  try {
    const tmpEntries = fs.readdirSync(path.join(codexHome, ".tmp"), { withFileTypes: true });
    for (const entry of tmpEntries) {
      if (entry.isDirectory() && /^plugins-clone-/i.test(entry.name)) {
        fs.rmSync(path.join(codexHome, ".tmp", entry.name), { recursive: true, force: true });
      }
    }
  } catch {}
  for (const stale of ["state_5.sqlite", "state_5.sqlite-journal", "state_5.sqlite-shm", "state_5.sqlite-wal"]) {
    const target = path.join(codexHome, stale);
    try {
      if (fs.existsSync(target) && fs.statSync(target).size === 0) {
        fs.rmSync(target, { force: true });
      }
    } catch {}
  }
  const escapeTomlPath = (value) => String(value || "").replace(/\\/g, "\\\\");
  const modelPreference = loadOpenClawModelPreference(resolveOpenClawStateDir(runtimeRoot));
  const config = [
    `model = "${modelPreference.model}"`,
    `model_reasoning_effort = "${modelPreference.thinkingDefault}"`,
    'personality = "pragmatic"',
    "",
    "[features]",
    "plugins = false",
    "shell_snapshot = false",
    "",
    `[projects.'${escapeTomlPath(workspaceDir)}']`,
    'trust_level = "trusted"',
    "",
    `[projects.'${escapeTomlPath(runtimeRoot)}']`,
    'trust_level = "trusted"',
    "",
    `[projects.'${escapeTomlPath(openClawRepo)}']`,
    'trust_level = "trusted"',
    "",
    "[windows]",
    'sandbox = "unelevated"',
    "",
  ].join("\n");
  fs.writeFileSync(path.join(codexHome, "config.toml"), config, "utf8");
  return codexHome;
};

const ensureOpenClawState = (stateDir, workspaceDir = "") => {
  const sourceState = path.join(os.homedir(), ".openclaw");
  const providerConfig = loadOpenClawProviderConfig();
  const nativeState = resolveNativeOpenClawStateDir();
  const isNativeState = nativeState && pathEquals(stateDir, nativeState);
  fs.mkdirSync(stateDir, { recursive: true });
  fs.mkdirSync(path.join(stateDir, "identity"), { recursive: true });
  fs.mkdirSync(path.join(stateDir, "devices"), { recursive: true });
  fs.mkdirSync(path.join(stateDir, "logs"), { recursive: true });
  const authAgentDirs = [
    path.join(stateDir, "agents", "default", "agent"),
    path.join(stateDir, "agents", "main", "agent"),
  ];
  for (const authAgentDir of authAgentDirs) {
    fs.mkdirSync(authAgentDir, { recursive: true });
  }
  const copyIfMissing = (src, dest) => {
    if (!fs.existsSync(dest) && fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
    }
  };
  copyIfMissing(path.join(sourceState, "identity", "device.json"), path.join(stateDir, "identity", "device.json"));
  copyIfMissing(
    path.join(sourceState, "identity", "device-auth.json"),
    path.join(stateDir, "identity", "device-auth.json")
  );
  copyIfMissing(path.join(sourceState, "devices", "paired.json"), path.join(stateDir, "devices", "paired.json"));
  copyIfMissing(path.join(sourceState, "devices", "pending.json"), path.join(stateDir, "devices", "pending.json"));
  const targetConfig = path.join(stateDir, "openclaw.json");
  const sourceConfigPath = path.join(sourceState, "openclaw.json");
  const cfg = readJsonIfExists(targetConfig) || readJsonIfExists(sourceConfigPath) || {};
  cfg.gateway = cfg.gateway || {};
  cfg.gateway.mode = "local";
  cfg.gateway.port = LOCAL_OPENCLAW_GATEWAY_PORT;
  cfg.gateway.auth = cfg.gateway.auth || {};
  cfg.gateway.auth.mode = "token";
  cfg.gateway.auth.token = LOCAL_OPENCLAW_GATEWAY_TOKEN;
  cfg.agents = cfg.agents || {};
  cfg.agents.defaults = cfg.agents.defaults || {};
  if (String(workspaceDir || "").trim()) {
    cfg.agents.defaults.workspace = workspaceDir;
  }
  if (!isNativeState && providerConfig) {
    cfg.auth = cfg.auth || {};
    cfg.auth.profiles = {
      ...(cfg.auth.profiles || {}),
      [providerConfig.profileId]: {
        provider: providerConfig.providerId,
        mode: "api_key",
      },
    };
    cfg.auth.order = {
      ...(cfg.auth.order || {}),
      [providerConfig.providerId]: [providerConfig.profileId],
    };
    cfg.models = cfg.models || {};
    cfg.models.providers = {
      ...(cfg.models.providers || {}),
      [providerConfig.providerId]: {
        ...((cfg.models.providers || {})[providerConfig.providerId] || {}),
        api: "openai-completions",
        baseUrl: providerConfig.baseUrl,
        apiKey: providerConfig.apiKey || ((cfg.models.providers || {})[providerConfig.providerId] || {}).apiKey || "",
        models: [
          {
            id: providerConfig.modelId,
            name: "GLM-5",
            reasoning: true,
            input: ["text"],
            contextWindow: 204800,
            maxTokens: 131072,
          },
        ],
      },
    };
    cfg.agents.defaults.model = cfg.agents.defaults.model || {};
    cfg.agents.defaults.model.primary = providerConfig.modelRef;
    cfg.agents.defaults.thinkingDefault = providerConfig.thinkingDefault;
    delete cfg.agents.defaults.cliBackends;
    if (cfg.plugins && typeof cfg.plugins === "object") {
      const entries = cfg.plugins.entries && typeof cfg.plugins.entries === "object" ? { ...cfg.plugins.entries } : {};
      delete entries.openai;
      if (Object.keys(entries).length > 0) {
        cfg.plugins.entries = entries;
      } else {
        delete cfg.plugins;
      }
    }
  } else if (cfg.plugins && typeof cfg.plugins === "object") {
    const entries = cfg.plugins.entries && typeof cfg.plugins.entries === "object" ? { ...cfg.plugins.entries } : {};
    delete entries.openai;
    if (Object.keys(entries).length > 0) {
      cfg.plugins.entries = entries;
    } else {
      delete cfg.plugins;
    }
  }
  fs.writeFileSync(targetConfig, JSON.stringify(cfg, null, 2), "utf8");
  if (!isNativeState && providerConfig?.apiKey) {
    const authStoreTargets = [
      ...authAgentDirs.map((dir) => path.join(dir, "auth-profiles.json")),
      path.join(stateDir, "auth-profiles.json"),
    ];
    for (const authStorePath of authStoreTargets) {
      const authStore = readJsonIfExists(authStorePath) || { version: 1, profiles: {} };
      authStore.version = 1;
      authStore.profiles = {
        ...(authStore.profiles || {}),
        [providerConfig.profileId]: {
          type: "api_key",
          provider: providerConfig.providerId,
          key: providerConfig.apiKey,
          endpoint: providerConfig.endpoint,
          baseUrl: providerConfig.baseUrl,
          model: providerConfig.modelId,
        },
      };
      authStore.order = {
        ...(authStore.order || {}),
        [providerConfig.providerId]: [
          providerConfig.profileId,
        ],
      };
      authStore.lastGood = {
        ...(authStore.lastGood || {}),
        [providerConfig.providerId]: providerConfig.profileId,
      };
      fs.writeFileSync(authStorePath, JSON.stringify(authStore, null, 2), "utf8");
    }
    for (const authAgentDir of authAgentDirs) {
      ensureAgentModelsConfig(authAgentDir, providerConfig);
    }
  }
};

const stopListeningProcessOnPort = (port, label) => {
  const pid = getListeningPidForPort(port);
  if (!pid) return false;
  try {
    spawnSync("taskkill", ["/PID", String(pid), "/F"], { windowsHide: true });
    appendStartupLog(`stopListeningProcessOnPort label=${label} port=${port} pid=${pid}`);
    return true;
  } catch (error) {
    appendStartupLog(`stopListeningProcessOnPort failed label=${label} port=${port}: ${error?.message || error}`);
  }
  return false;
};

const stopManagedPortProcesses = () => {
  stopListeningProcessOnPort(LOCAL_BACKEND_PORT, "backend");
  stopListeningProcessOnPort(LOCAL_OPENCLAW_GATEWAY_PORT, "openclaw-gateway");
  stopListeningProcessOnPort(LOCAL_TTS_PORT, "tts");
  stopListeningProcessOnPort(LOCAL_STT_PORT, "stt");
};

const commandExists = (command) => {
  try {
    const probe = process.platform === "win32" ? ["cmd", ["/c", "where", command]] : ["which", [command]];
    const result = spawnSync(probe[0], probe[1], { stdio: "ignore" });
    return result.status === 0;
  } catch {
    return false;
  }
};

const resolvePythonCommand = (runtimeRoot) => {
  if (process.env.EMOTION_BRIDGE_PYTHON) {
    return { command: process.env.EMOTION_BRIDGE_PYTHON, args: [] };
  }
  const gpuVenvCandidates = process.platform === "win32"
    ? [
        path.join(runtimeRoot, ".venv-gpu", "Scripts", "python.exe"),
        path.resolve(runtimeRoot, "..", "lunwen", ".venv-gpu", "Scripts", "python.exe"),
        "E:\\Desktop\\lunwen\\.venv-gpu\\Scripts\\python.exe",
      ]
    : [
        path.join(runtimeRoot, ".venv-gpu", "bin", "python"),
      ];
  for (const gpuPython of gpuVenvCandidates) {
    if (fs.existsSync(gpuPython)) {
      return { command: gpuPython, args: [] };
    }
  }
  const bundledPythonCandidates = process.platform === "win32"
    ? [
        path.join(runtimeRoot, "python", "python.exe"),
        path.join(runtimeRoot, "vendor", "python-runtime", "python.exe"),
        path.join(runtimeRoot, "app windows", "vendor", "python-runtime", "python.exe"),
      ]
    : [
        path.join(runtimeRoot, "python", "bin", "python3"),
        path.join(runtimeRoot, "vendor", "python-runtime", "bin", "python3"),
      ];
  for (const bundledPython of bundledPythonCandidates) {
    if (fs.existsSync(bundledPython)) {
      return { command: bundledPython, args: [] };
    }
  }
  const venvPython = process.platform === "win32"
    ? path.join(runtimeRoot, ".venv", "Scripts", "python.exe")
    : path.join(runtimeRoot, ".venv", "bin", "python");
  if (fs.existsSync(venvPython)) {
    return { command: venvPython, args: [] };
  }
  if (process.platform === "win32" && commandExists("py")) {
    return { command: "py", args: ["-3"] };
  }
  if (commandExists("python")) {
    return { command: "python", args: [] };
  }
  if (commandExists("python3")) {
    return { command: "python3", args: [] };
  }
  return null;
};

const buildPythonPath = (runtimeRoot, pythonCommand = "") => {
  const delimiter = process.platform === "win32" ? ";" : ":";
  const useBundled = String(pythonCommand || "").includes("vendor\\python-runtime") || String(pythonCommand || "").includes("vendor/python-runtime");
  const candidates = [runtimeRoot];
  if (useBundled) {
    candidates.push(
      path.join(runtimeRoot, "vendor", "python-site-packages"),
      path.join(runtimeRoot, "app windows", "vendor", "python-site-packages")
    );
  }
  candidates.push(process.env.PYTHONPATH || "");
  const deduped = candidates.filter((value, index, list) => value && list.indexOf(value) === index);
  return deduped.join(delimiter);
};

const buildLocalVoiceBridgeEnv = (runtimeRoot) => {
  const env = {
    DESKTOP_TTS_BASE_URL: process.env.DESKTOP_TTS_BASE_URL || LOCAL_TTS_BASE_URL,
    DESKTOP_TTS_URL: process.env.DESKTOP_TTS_URL || `${LOCAL_TTS_BASE_URL}/api/tts/synthesize`,
    DESKTOP_TTS_HEALTH: process.env.DESKTOP_TTS_HEALTH || `${LOCAL_TTS_BASE_URL}/api/tts/health`,
    DESKTOP_STT_BASE_URL: process.env.DESKTOP_STT_BASE_URL || LOCAL_STT_BASE_URL,
    DESKTOP_STT_URL: process.env.DESKTOP_STT_URL || `${LOCAL_STT_BASE_URL}/api/stt/transcribe`,
    DESKTOP_STT_HEALTH: process.env.DESKTOP_STT_HEALTH || `${LOCAL_STT_BASE_URL}/api/stt/health`,
  };
  const senseVoiceModelDir = resolveSenseVoiceModelDir();
  if (senseVoiceModelDir) {
    env.SENSEVOICE_MODEL_DIR = process.env.SENSEVOICE_MODEL_DIR || senseVoiceModelDir;
  }
  const voxBaseModelDir = resolveVoxBaseModelDir(runtimeRoot);
  if (voxBaseModelDir) {
    env.VOXCPM_DISTILL_BASE = process.env.VOXCPM_DISTILL_BASE || voxBaseModelDir;
    env.VOXCPM_BASE_MODEL = process.env.VOXCPM_BASE_MODEL || voxBaseModelDir;
  }
  const voxLoraWeightsPath = resolveVoxLoraWeightsPath(runtimeRoot);
  if (voxLoraWeightsPath) {
    env.VOXCPM_DISTILL_LORA = process.env.VOXCPM_DISTILL_LORA || voxLoraWeightsPath;
  }
  const voxLoraConfigPath = resolveVoxLoraConfigPath(runtimeRoot);
  if (voxLoraConfigPath) {
    env.VOXCPM_DISTILL_LORA_CONFIG = process.env.VOXCPM_DISTILL_LORA_CONFIG || voxLoraConfigPath;
  }
  const voxSweetPromptPath = resolveVoxSweetPromptPath(runtimeRoot);
  if (voxSweetPromptPath) {
    env.VOXCPM_SWEET_PROMPT_WAV = process.env.VOXCPM_SWEET_PROMPT_WAV || voxSweetPromptPath;
  }
  env.VOXCPM_SWEET_PROMPT_TEXT =
    process.env.VOXCPM_SWEET_PROMPT_TEXT ||
    "你好呀，我会一直温柔地陪着你，我们慢慢聊就好。";
  return env;
};

const startLocalVoiceSidecar = ({
  currentProc,
  runtimeRoot,
  python,
  codexHome,
  scriptPath,
  port,
  env,
  logLabel,
}) => {
  if (currentProc) return currentProc;
  if (!fs.existsSync(scriptPath)) {
    appendStartupLog(`${logLabel} missing script=${scriptPath}`);
    return null;
  }
  const proc = spawn(
    python.command,
    [...python.args, scriptPath],
    {
      cwd: runtimeRoot,
      env: {
        ...process.env,
        ...env,
        PYTHONPATH: buildPythonPath(runtimeRoot, python.command),
        TMP: path.join(codexHome, "tmp"),
        TEMP: path.join(codexHome, "tmp"),
        TMPDIR: path.join(codexHome, "tmp"),
      },
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  proc.stdout?.on("data", (chunk) => {
    const textOut = String(chunk || "").trim();
    if (textOut) appendStartupLog(`[${logLabel}][stdout] ${textOut}`);
  });
  proc.stderr?.on("data", (chunk) => {
    const textErr = String(chunk || "").trim();
    if (textErr) appendStartupLog(`[${logLabel}][stderr] ${textErr}`);
  });
  proc.on("exit", (code, signal) => {
    appendStartupLog(`${logLabel} exited code=${code} signal=${signal}`);
    if (logLabel === "local-tts") {
      localTtsProc = null;
      if (!isQuitting) {
        setTimeout(() => {
          if (!localTtsProc) {
            ensureLocalVoiceSidecars(runtimeRoot, python, codexHome).catch(() => {});
          }
        }, 1500);
      }
    } else if (logLabel === "local-stt") {
      localSttProc = null;
      if (!isQuitting) {
        setTimeout(() => {
          if (!localSttProc) {
            ensureLocalVoiceSidecars(runtimeRoot, python, codexHome).catch(() => {});
          }
        }, 1500);
      }
    }
  });
  appendStartupLog(`${logLabel} spawned port=${port}`);
  return proc;
};

const ensureLocalVoiceSidecars = async (runtimeRoot, python, codexHome) => {
  const voiceEnv = buildLocalVoiceBridgeEnv(runtimeRoot);
  if (!(await probeTcpPort("127.0.0.1", LOCAL_STT_PORT, 400))) {
    const modelDir = String(voiceEnv.SENSEVOICE_MODEL_DIR || "").trim();
    if (modelDir) {
      localSttProc = startLocalVoiceSidecar({
        currentProc: localSttProc,
        runtimeRoot,
        python,
        codexHome,
        scriptPath: resolveBridgeScriptPath(runtimeRoot, "serve_sensevoice_stt_api.py"),
        port: LOCAL_STT_PORT,
        env: {
          ...voiceEnv,
          SENSEVOICE_HOST: "127.0.0.1",
          SENSEVOICE_PORT: String(LOCAL_STT_PORT),
        },
        logLabel: "local-stt",
      });
    }
  }
  if (!(await probeTcpPort("127.0.0.1", LOCAL_TTS_PORT, 400))) {
    const modelDir = String(voiceEnv.VOXCPM_DISTILL_BASE || "").trim();
    if (modelDir) {
      localTtsProc = startLocalVoiceSidecar({
        currentProc: localTtsProc,
        runtimeRoot,
        python,
        codexHome,
        scriptPath: resolveBridgeScriptPath(runtimeRoot, "serve_voxcpm_tts_api.py"),
        port: LOCAL_TTS_PORT,
        env: {
          ...voiceEnv,
          VOXCPM_TTS_HOST: "127.0.0.1",
          VOXCPM_TTS_PORT: String(LOCAL_TTS_PORT),
        },
        logLabel: "local-tts",
      });
    }
  }
};

const isLocalBackendHealthy = async () => {
  try {
    const response = await fetch(`${LOCAL_BACKEND_URL}/api/runtime/version`);
    return response.ok;
  } catch {
    return false;
  }
};

const isLocalOpenClawGatewayHealthy = async () =>
  probeTcpPort("127.0.0.1", LOCAL_OPENCLAW_GATEWAY_PORT, 500);

const warmLocalAssistantRuntime = async () => {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetch(`${LOCAL_BACKEND_URL}/api/assistant/runtime/warmup`, {
        method: "POST",
      });
      const payload = await response.json().catch(() => ({}));
      const ok = Boolean(response.ok && payload?.ok);
      appendStartupLog(
        `warmLocalAssistantRuntime attempt=${attempt + 1} status=${response.status} ok=${String(ok)} detail=${String(payload?.detail || payload?.reply || "")}`
      );
      if (ok) {
        assistantRuntimeWarmAtMs = Date.now();
        return true;
      }
    } catch (error) {
      appendStartupLog(`warmLocalAssistantRuntime attempt=${attempt + 1} failed=${String(error || "")}`);
    }
    await delay(2000 + attempt * 2000);
  }
  return false;
};

const waitForLocalOpenClawGateway = async (maxAttempts = 260) => {
  for (let i = 0; i < maxAttempts; i += 1) {
    if ((await isLocalOpenClawGatewayHealthy()) && openClawAssistantReady) {
      appendStartupLog(`waitForLocalOpenClawGateway ready attempts=${i + 1}`);
      return true;
    }
    if (!openClawGatewayProc && i >= 4) {
      appendStartupLog(`waitForLocalOpenClawGateway aborted attempts=${i + 1} reason=process-exited`);
      return false;
    }
    await delay(500);
  }
  appendStartupLog(`waitForLocalOpenClawGateway timeout attempts=${maxAttempts}`);
  return false;
};

const startLocalBackend = () => {
  if (backendProc) return;
  const runtimeRoot = resolveRuntimeRoot();
  const openClawRepo = resolveOpenClawRepo(runtimeRoot);
  const workspaceDir = resolveOpenClawWorkspace(runtimeRoot);
  const stateDir = resolveOpenClawStateDir(runtimeRoot);
  appendStartupLog(`startLocalBackend runtimeRoot=${runtimeRoot}`);
  ensureOpenClawWorkspace(workspaceDir);
  ensureOpenClawState(stateDir, workspaceDir);
  const codexHome = ensureOpenClawCodexHome(
    runtimeRoot,
    workspaceDir,
    openClawRepo
  );
  const voiceBridgeEnv = buildLocalVoiceBridgeEnv(runtimeRoot);
  const backendEntry = path.join(runtimeRoot, "backend", "main.py");
  if (!fs.existsSync(backendEntry)) {
    appendStartupLog(`startLocalBackend missing backendEntry=${backendEntry}`);
    console.error(`[main] backend bundle missing: ${backendEntry}`);
    return;
  }
  const python = resolvePythonCommand(runtimeRoot);
  if (!python) {
    appendStartupLog("startLocalBackend no python resolved");
    console.error("[main] no Python runtime available for local backend");
    return;
  }
  appendStartupLog(`startLocalBackend spawn python=${python.command} args=${python.args.join(" ")}`);
  stopListeningProcessOnPort(LOCAL_BACKEND_PORT, "backend-pre-spawn");
  const desktopSessionKey = resolveOpenClawDesktopSessionKey(stateDir);
  backendProc = spawn(
    python.command,
    [...python.args, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", String(LOCAL_BACKEND_PORT)],
    {
      cwd: runtimeRoot,
      env: {
        ...process.env,
        ...buildLocalNoProxyEnv(),
        ...buildOpenClawProxyEnv(),
        ...buildOpenClawProviderEnv(),
        ...voiceBridgeEnv,
        PYTHONPATH: buildPythonPath(runtimeRoot, python.command),
        OPENCLAW_WORKSPACE_DIR: workspaceDir,
        OPENCLAW_STATE_DIR: stateDir,
        OPENCLAW_REPO_PATH: openClawRepo,
        OPENCLAW_GATEWAY_URL: `ws://127.0.0.1:${LOCAL_OPENCLAW_GATEWAY_PORT}`,
        OPENCLAW_GATEWAY_ORIGIN: `http://127.0.0.1:${LOCAL_OPENCLAW_GATEWAY_PORT}`,
        OPENCLAW_DESKTOP_SHARED_SESSION_KEY: desktopSessionKey,
        OPENCLAW_CODEX_HOME: codexHome,
        CODEX_HOME: codexHome,
        TMP: path.join(codexHome, "tmp"),
        TEMP: path.join(codexHome, "tmp"),
        TMPDIR: path.join(codexHome, "tmp"),
        ALLOW_UNVERIFIED_LOCAL_DESKTOP_TOKENS: "1",
        DEFAULT_ROBOT_DEVICE_IP: process.env.DEFAULT_ROBOT_DEVICE_IP || "192.168.137.50",
      },
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  backendProc.stdout?.on("data", (chunk) => {
    const textOut = String(chunk || "").trim();
    if (textOut) appendStartupLog(`[local-backend][stdout] ${textOut}`);
    if (textOut) console.log(`[local-backend] ${textOut}`);
  });
  backendProc.stderr?.on("data", (chunk) => {
    const textErr = String(chunk || "").trim();
    if (textErr) appendStartupLog(`[local-backend][stderr] ${textErr}`);
    if (textErr) console.error(`[local-backend] ${textErr}`);
  });
  backendProc.on("exit", (code, signal) => {
    appendStartupLog(`startLocalBackend exited code=${code} signal=${signal}`);
    console.error(`[main] local backend exited code=${code} signal=${signal}`);
    assistantRuntimeWarmAtMs = 0;
    backendProc = null;
  });
};

const startLocalOpenClawGateway = () => {
  if (openClawGatewayProc) return;
  openClawAssistantReady = false;
  const runtimeRoot = resolveRuntimeRoot();
  const openClawRepo = resolveOpenClawRepo(runtimeRoot);
  const gatewayEntry = path.join(openClawRepo, "openclaw.mjs");
  appendStartupLog(`startLocalOpenClawGateway repo=${openClawRepo}`);
  if (!fs.existsSync(gatewayEntry)) {
    appendStartupLog(`startLocalOpenClawGateway missing gatewayEntry=${gatewayEntry}`);
    console.error(`[main] OpenClaw repo missing: ${gatewayEntry}`);
    return;
  }
  const workspaceDir = resolveOpenClawWorkspace(runtimeRoot);
  const stateDir = resolveOpenClawStateDir(runtimeRoot);
  const codexHome = ensureOpenClawCodexHome(runtimeRoot, workspaceDir, openClawRepo);
  ensureOpenClawWorkspace(workspaceDir);
  ensureOpenClawState(stateDir, workspaceDir);
  stopLegacyOpenClawGateway();
  stopListeningProcessOnPort(LOCAL_OPENCLAW_GATEWAY_PORT, "openclaw-gateway-pre-spawn");
  appendStartupLog(`startLocalOpenClawGateway spawn entry=${gatewayEntry}`);
  openClawGatewayProc = spawn(
    "node",
    [
      gatewayEntry,
      "gateway",
      "run",
      "--auth",
      "token",
      "--token",
      LOCAL_OPENCLAW_GATEWAY_TOKEN,
      "--port",
      String(LOCAL_OPENCLAW_GATEWAY_PORT),
      "--bind",
      "loopback",
      "--ws-log",
      "compact",
    ],
    {
      cwd: workspaceDir,
      env: {
        ...process.env,
        ...buildLocalNoProxyEnv(),
        ...buildOpenClawProxyEnv(),
        ...buildOpenClawProviderEnv(),
        OPENCLAW_WORKSPACE_DIR: workspaceDir,
        OPENCLAW_STATE_DIR: stateDir,
        OPENCLAW_REPO_PATH: openClawRepo,
        OPENCLAW_CODEX_HOME: codexHome,
        CODEX_HOME: codexHome,
        TMP: path.join(codexHome, "tmp"),
        TEMP: path.join(codexHome, "tmp"),
        TMPDIR: path.join(codexHome, "tmp"),
      },
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  openClawGatewayProc.stdout?.on("data", (chunk) => {
    const textOut = String(chunk || "").trim();
    if (textOut && textOut.includes("[plugins] embedded acpx runtime backend ready")) {
      openClawAssistantReady = true;
    }
    if (textOut) appendStartupLog(`[openclaw-gateway][stdout] ${textOut}`);
    if (textOut) console.log(`[openclaw-gateway] ${textOut}`);
  });
  openClawGatewayProc.stderr?.on("data", (chunk) => {
    const textErr = String(chunk || "").trim();
    if (textErr) appendStartupLog(`[openclaw-gateway][stderr] ${textErr}`);
    if (textErr) console.error(`[openclaw-gateway] ${textErr}`);
  });
  openClawGatewayProc.on("exit", (code, signal) => {
    appendStartupLog(`startLocalOpenClawGateway exited code=${code} signal=${signal}`);
    console.error(`[main] openclaw gateway exited code=${code} signal=${signal}`);
    openClawAssistantReady = false;
    assistantRuntimeWarmAtMs = 0;
    openClawGatewayProc = null;
  });
};

const ensureLocalBackend = async () => {
  if (ensureLocalBackendPromise) {
    return ensureLocalBackendPromise;
  }
  ensureLocalBackendPromise = (async () => {
    appendStartupLog("ensureLocalBackend begin");
    const runtimeRoot = resolveRuntimeRoot();
    const openClawRepo = resolveOpenClawRepo(runtimeRoot);
    const gatewayEntry = path.join(openClawRepo, "openclaw.mjs");
    const shouldWaitForGateway = fs.existsSync(gatewayEntry);
    const workspaceDir = resolveOpenClawWorkspace(runtimeRoot);
    const codexHome = ensureOpenClawCodexHome(runtimeRoot, workspaceDir, openClawRepo);
    const python = resolvePythonCommand(runtimeRoot);

    if (python) {
      await ensureLocalVoiceSidecars(runtimeRoot, python, codexHome);
    }

    let backendHealthy = await isLocalBackendHealthy();
    if (!backendHealthy) {
      startLocalOpenClawGateway();
      startLocalBackend();
      for (let i = 0; i < 60; i += 1) {
        await delay(500);
        if (await isLocalBackendHealthy()) {
          backendHealthy = true;
          break;
        }
      }
    } else if (shouldWaitForGateway && !(await isLocalOpenClawGatewayHealthy())) {
      startLocalOpenClawGateway();
    }

    if (!backendHealthy) {
      appendStartupLog("ensureLocalBackend timeout");
      return false;
    }

    if (shouldWaitForGateway && !(await isLocalOpenClawGatewayHealthy())) {
      appendStartupLog("ensureLocalBackend waiting for OpenClaw gateway");
      await waitForLocalOpenClawGateway();
    }

    const warmExpired =
      !assistantRuntimeWarmAtMs || Date.now() - assistantRuntimeWarmAtMs >= ASSISTANT_RUNTIME_WARM_MAX_AGE_MS;
    if (shouldWaitForGateway && backendHealthy && warmExpired) {
      appendStartupLog("ensureLocalBackend warming assistant runtime");
      const warmed = await warmLocalAssistantRuntime();
      appendStartupLog(`ensureLocalBackend assistant runtime warmed=${String(warmed)}`);
    }

    return true;
  })();
  try {
    return await ensureLocalBackendPromise;
  } finally {
    ensureLocalBackendPromise = null;
  }
};

const scheduleBackendSupervisor = () => {
  if (backendSupervisorTimer || isQuitting) return;
  backendSupervisorTimer = setInterval(() => {
    if (isQuitting) return;
    ensureLocalBackend().catch((error) => {
      appendStartupLog(`backendSupervisor ensureLocalBackend failed=${String(error || "")}`);
    });
  }, 15000);
};

const stopBackendSupervisor = () => {
  if (!backendSupervisorTimer) return;
  clearInterval(backendSupervisorTimer);
  backendSupervisorTimer = null;
};

const stopLocalBackend = () => {
  if (!backendProc) return;
  try {
    backendProc.kill();
  } catch {}
  assistantRuntimeWarmAtMs = 0;
  backendProc = null;
};

const stopLocalOpenClawGateway = () => {
  if (!openClawGatewayProc) return;
  try {
    openClawGatewayProc.kill();
  } catch {}
  assistantRuntimeWarmAtMs = 0;
  openClawGatewayProc = null;
};

const stopLocalVoiceSidecars = () => {
  if (localTtsProc) {
    try {
      localTtsProc.kill();
    } catch {}
    localTtsProc = null;
  }
  if (localSttProc) {
    try {
      localSttProc.kill();
    } catch {}
    localSttProc = null;
  }
};

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (!mainWindow.isVisible()) mainWindow.show();
      mainWindow.focus();
    }
    if (floatWindow && !floatWindow.isVisible()) {
      floatWindow.show();
    }
  });
}

const defaultTitleBar = {
  color: "#070b14",
  symbolColor: "#cbd5f5",
  height: 36,
};

const applyTitleBarTheme = (win, theme = {}) => {
  if (!win || typeof win.setTitleBarOverlay !== "function") return;
  const color = typeof theme.color === "string" ? theme.color : defaultTitleBar.color;
  const symbolColor =
    typeof theme.symbolColor === "string" ? theme.symbolColor : defaultTitleBar.symbolColor;
  const height = typeof theme.height === "number" ? theme.height : defaultTitleBar.height;
  try {
    win.setTitleBarOverlay({ color, symbolColor, height });
  } catch (_err) {
    // Ignore when titleBarOverlay is not enabled on this window.
  }
};

const escapeHtml = (text) =>
  String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const showWindowFatal = (win, title, message) => {
  if (!win || win.isDestroyed()) return;
  const html = `<!doctype html><html><head><meta charset="utf-8"/><title>Error</title></head><body style="margin:0;background:#070b14;color:#e2e8f0;font-family:Segoe UI,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;padding:20px;box-sizing:border-box;"><div style="max-width:760px;width:100%;border:1px solid rgba(148,163,184,.35);border-radius:14px;background:rgba(15,23,42,.85);padding:18px;"><h2 style="margin:0 0 10px 0;font-size:18px;">${escapeHtml(
    title
  )}</h2><div style="font-size:13px;line-height:1.6;color:#94a3b8;margin-bottom:10px;">应用未能正常加载。请重启软件；如果持续失败，请把此信息反馈给开发。</div><pre style="margin:0;padding:12px;border-radius:8px;background:rgba(2,6,23,.8);color:#f8fafc;white-space:pre-wrap;word-break:break-word;font-size:12px;">${escapeHtml(
    message
  )}</pre></div></body></html>`;
  win
    .loadURL(`data:text/html;charset=UTF-8,${encodeURIComponent(html)}`)
    .catch((err) => console.error("failed to load fallback html:", err));
};

const MEDIA_PERMISSIONS = new Set(["media", "camera", "microphone", "display-capture"]);

const configureMediaPermissions = () => {
  const ses = session.defaultSession;
  if (!ses) return;

  try {
    ses.setPermissionCheckHandler((_webContents, permission) =>
      MEDIA_PERMISSIONS.has(String(permission || "").trim().toLowerCase())
    );
  } catch (err) {
    appendStartupLog(`setPermissionCheckHandler failed: ${err?.message || err}`);
  }

  try {
    ses.setPermissionRequestHandler((_webContents, permission, callback) => {
      callback(MEDIA_PERMISSIONS.has(String(permission || "").trim().toLowerCase()));
    });
  } catch (err) {
    appendStartupLog(`setPermissionRequestHandler failed: ${err?.message || err}`);
  }

  try {
    if (typeof ses.setDevicePermissionHandler === "function") {
      ses.setDevicePermissionHandler((details) => {
        const deviceType = String(details?.deviceType || "").trim().toLowerCase();
        return deviceType === "media" || deviceType === "video" || deviceType === "audio";
      });
    }
  } catch (err) {
    appendStartupLog(`setDevicePermissionHandler failed: ${err?.message || err}`);
  }
};

const attachWindowDiagnostics = (win, name) => {
  if (!win || win.isDestroyed()) return;
  win.webContents.on("did-fail-load", (_event, code, desc, validatedURL) => {
    const msg = `[${name}] did-fail-load code=${code} desc=${desc} url=${validatedURL}`;
    console.error(msg);
    showWindowFatal(win, "页面加载失败", msg);
  });
  win.webContents.on("render-process-gone", (_event, details) => {
    const msg = `[${name}] render-process-gone reason=${details?.reason} code=${details?.exitCode}`;
    console.error(msg);
    showWindowFatal(win, "渲染进程已退出", msg);
  });
  win.webContents.on("unresponsive", () => {
    const msg = `[${name}] renderer unresponsive`;
    console.error(msg);
    showWindowFatal(win, "页面无响应", msg);
  });
  win.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    if (level >= 2) {
      console.error(`[renderer:${name}] ${sourceId}:${line} ${message}`);
    }
  });
};

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 640,
    backgroundColor: "#0F172A",
    icon: path.join(__dirname, "..", "assets", "app-icon.png"),
    titleBarStyle: "hidden",
    titleBarOverlay: {
      color: defaultTitleBar.color,
      symbolColor: defaultTitleBar.symbolColor,
      height: defaultTitleBar.height,
    },
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });
  attachWindowDiagnostics(win, "main");

  const devUrl = process.env.ELECTRON_DEV_URL;
  if (devUrl) {
    win.loadURL(devUrl).catch((err) => {
      const msg = `failed to load dev url ${devUrl}\n${err?.stack || err}`;
      console.error(msg);
      showWindowFatal(win, "开发模式加载失败", msg);
    });
    if (process.env.ELECTRON_OPEN_DEVTOOLS === "1") {
      win.webContents.openDevTools({ mode: "detach" });
    }
  } else {
    const indexPath = path.join(__dirname, "..", "dist", "index.html");
    win.loadFile(indexPath).catch((err) => {
      const msg = `failed to load index file ${indexPath}\n${err?.stack || err}`;
      console.error(msg);
      showWindowFatal(win, "桌面页面加载失败", msg);
    });
  }

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  win.on("close", (event) => {
    if (isQuitting) return;
    event.preventDefault();
    win.hide();
    if (tray) {
      tray.setToolTip("EmoResonance (running)");
    }
  });

  return win;
};

const loadWindowWithQuery = (win, query) => {
  const devUrl = process.env.ELECTRON_DEV_URL;
  if (devUrl) {
    return win.loadURL(`${devUrl}?${query}`);
  }
  const indexPath = path.join(__dirname, "..", "dist", "index.html");
  return win.loadFile(indexPath, { query: { [query.split("=")[0]]: query.split("=")[1] } });
};

const positionChatWindow = () => {
  if (!floatWindow || !chatWindow) return;
  const floatBounds = floatWindow.getBounds();
  const display = screen.getDisplayNearestPoint({ x: floatBounds.x, y: floatBounds.y });
  const workArea = display.workArea;
  const chatBounds = chatWindow.getBounds();
  const padding = 12;
  let x = floatBounds.x - chatBounds.width - padding;
  if (x < workArea.x) {
    x = floatBounds.x + floatBounds.width + padding;
  }
  let y = floatBounds.y + floatBounds.height - chatBounds.height;
  if (y < workArea.y + padding) y = workArea.y + padding;
  if (y + chatBounds.height > workArea.y + workArea.height - padding) {
    y = workArea.y + workArea.height - chatBounds.height - padding;
  }
  chatWindow.setPosition(Math.round(x), Math.round(y), false);
};

const createChatWindow = () => {
  const win = new BrowserWindow({
    width: 460,
    height: 560,
    resizable: false,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    backgroundColor: "#00000000",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });
  attachWindowDiagnostics(win, "float-chat");
  loadWindowWithQuery(win, "float=chat").catch((err) => {
    const msg = `failed to load float chat window\n${err?.stack || err}`;
    console.error(msg);
    showWindowFatal(win, "悬浮对话加载失败", msg);
  });
  win.on("close", (event) => {
    if (isQuitting) return;
    event.preventDefault();
    win.hide();
  });
  return win;
};

const createFloatWindow = () => {
  const win = new BrowserWindow({
    width: 64,
    height: 64,
    resizable: false,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    backgroundColor: "#00000000",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });
  attachWindowDiagnostics(win, "float-widget");
  loadWindowWithQuery(win, "float=widget").catch((err) => {
    const msg = `failed to load float widget window\n${err?.stack || err}`;
    console.error(msg);
    showWindowFatal(win, "悬浮窗加载失败", msg);
  });
  // Let clicks pass through except when explicitly enabled from renderer.
  win.setIgnoreMouseEvents(true, { forward: true });
  const primary = screen.getPrimaryDisplay();
  const { x, y, width, height } = primary.workArea;
  win.setPosition(Math.round(x + width - 88), Math.round(y + height - 132), false);
  win.webContents.on("context-menu", () => {
    const menu = Menu.buildFromTemplate([
      {
        label: "设置",
        click: () => openMainTab("CONTROL"),
      },
      {
        label: "关闭悬浮窗",
        click: () => {
          if (chatWindow) chatWindow.hide();
          win.hide();
        },
      },
      {
        label: "问题反馈",
        click: () => openMainTab("CHAT"),
      },
      { type: "separator" },
      {
        label: "退出",
        click: () => {
          isQuitting = true;
          app.quit();
        },
      },
    ]);
    menu.popup({ window: win });
  });
  win.on("closed", () => {
    floatWindow = null;
  });
  return win;
};

const openMainTab = (tab) => {
  if (!mainWindow || mainWindow.isDestroyed()) {
    mainWindow = createWindow();
  }
  mainWindow.show();
  mainWindow.focus();
  mainWindow.webContents.send("navigate-tab", tab);
};

const ensureChatWindow = () => {
  if (!chatWindow || chatWindow.isDestroyed()) {
    chatWindow = createChatWindow();
  }
  positionChatWindow();
  chatWindow.show();
  chatWindow.focus();
};

const toggleChatWindow = () => {
  if (!chatWindow || chatWindow.isDestroyed()) {
    ensureChatWindow();
    return;
  }
  if (chatWindow.isVisible()) {
    chatWindow.hide();
  } else {
    ensureChatWindow();
  }
};

const ensureTray = () => {
  if (tray) return tray;
  const trayIconPath = path.join(__dirname, "..", "assets", "app-icon.png");
  const icon = nativeImage.createFromPath(trayIconPath);
  tray = new Tray(icon);
  tray.setToolTip("EmoResonance");
  const contextMenu = Menu.buildFromTemplate([
    {
      label: "显示主窗口",
      click: () => {
        if (!mainWindow) return;
        mainWindow.show();
        mainWindow.focus();
      },
    },
    {
      label: "显示/隐藏悬浮窗",
      click: () => {
        if (!floatWindow || floatWindow.isDestroyed()) {
          floatWindow = createFloatWindow();
          return;
        }
        if (floatWindow.isVisible()) {
          floatWindow.hide();
          if (chatWindow) chatWindow.hide();
        } else {
          floatWindow.show();
        }
      },
    },
    {
      label: "退出",
      click: () => {
        isQuitting = true;
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(contextMenu);
  tray.on("click", () => {
    if (!mainWindow) return;
    if (mainWindow.isVisible()) {
      mainWindow.focus();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });
  return tray;
};

app.whenReady().then(async () => {
  configureMediaPermissions();
  stopManagedPortProcesses();
  await ensureLocalBackend();
  scheduleBackendSupervisor();
  mainWindow = createWindow();
  ensureTray();
  floatWindow = createFloatWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      mainWindow = createWindow();
    }
  });
});

ipcMain.handle("backend-session:set", async (_event, payload) => {
  await deviceSyncManager.setSession(payload || {});
  return { ok: true };
});

ipcMain.handle("backend-session:clear", async () => {
  await deviceSyncManager.clearSession();
  return { ok: true };
});

ipcMain.on("set-titlebar-theme", (event, theme) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (!win || win !== mainWindow) return;
  applyTitleBarTheme(win, theme);
  if (theme && typeof theme.backgroundColor === "string") {
    win.setBackgroundColor(theme.backgroundColor);
  }
});

ipcMain.on("float-toggle-chat", () => {
  toggleChatWindow();
});

ipcMain.on("open-main-tab", (_event, tab) => {
  openMainTab(tab);
});

ipcMain.on("float-hide", () => {
  if (floatWindow) floatWindow.hide();
  if (chatWindow) chatWindow.hide();
});

ipcMain.on("float-show", () => {
  if (!floatWindow || floatWindow.isDestroyed()) {
    floatWindow = createFloatWindow();
  }
  floatWindow.show();
});

ipcMain.on("float-set-interactive", (_event, enabled) => {
  if (!floatWindow || floatWindow.isDestroyed()) return;
  const shouldEnable = Boolean(enabled);
  // When interactive, capture mouse. When not, pass through.
  floatWindow.setIgnoreMouseEvents(!shouldEnable, { forward: true });
});

ipcMain.on("float-drag-move", (_event, payload) => {
  if (!floatWindow || floatWindow.isDestroyed() || !floatDragState) return;
  const cursor = screen.getCursorScreenPoint();
  const display = screen.getDisplayNearestPoint(cursor);
  const workArea = display.workArea;
  const bounds = floatWindow.getBounds();
  const padding = 0;
  const offsetX = floatDragState.offsetX ?? 0;
  const offsetY = floatDragState.offsetY ?? 0;
  let nextX = Math.round(cursor.x - offsetX);
  let nextY = Math.round(cursor.y - offsetY);
  nextX = Math.max(workArea.x + padding, Math.min(nextX, workArea.x + workArea.width - bounds.width - padding));
  nextY = Math.max(workArea.y + padding, Math.min(nextY, workArea.y + workArea.height - bounds.height - padding));
  floatWindow.setPosition(nextX, nextY, false);
  if (chatWindow && chatWindow.isVisible()) {
    positionChatWindow();
  }
});

ipcMain.on("float-drag-end", () => {
  floatDragState = null;
  if (floatWindow && !floatWindow.isDestroyed()) {
    floatWindow.setIgnoreMouseEvents(true, { forward: true });
  }
  if (chatWindow && chatWindow.isVisible()) {
    positionChatWindow();
  }
});

ipcMain.on("float-drag-start", (_event, payload) => {
  if (!floatWindow || floatWindow.isDestroyed()) return;
  const cursor = screen.getCursorScreenPoint();
  const [winX, winY] = floatWindow.getPosition();
  const bounds = floatWindow.getBounds();
  const offsetX = Math.max(0, Math.min(bounds.width, cursor.x - winX));
  const offsetY = Math.max(0, Math.min(bounds.height, cursor.y - winY));
  floatDragState = {
    startX: cursor.x,
    startY: cursor.y,
    winX,
    winY,
    offsetX: Math.round(offsetX),
    offsetY: Math.round(offsetY),
  };
});

ipcMain.on("notify-system", (_event, payload) => {
  if (!payload || !Notification.isSupported()) return;
  const title = String(payload.title || "心念双灵");
  const body = String(payload.body || "");
  const silent = Boolean(payload.silent);
  const notification = new Notification({
    title,
    body,
    silent,
    icon: path.join(__dirname, "..", "assets", "app-icon.png"),
  });
  notification.show();
});

process.on("uncaughtException", (error) => {
  console.error("[main] uncaughtException:", error);
  showWindowFatal(mainWindow, "主进程异常", error?.stack || String(error));
});

process.on("unhandledRejection", (reason) => {
  const text =
    reason instanceof Error ? reason.stack || reason.message : JSON.stringify(reason, null, 2);
  console.error("[main] unhandledRejection:", reason);
  showWindowFatal(mainWindow, "主进程未处理异常", text);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  isQuitting = true;
  stopBackendSupervisor();
  deviceSyncManager.dispose();
  stopLocalBackend();
  stopLocalOpenClawGateway();
  stopLocalVoiceSidecars();
});
