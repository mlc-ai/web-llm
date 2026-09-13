import * as webllm from "@mlc-ai/web-llm";
import "./style.css";

type EngineMode = "worker" | "main";

type ResumeResultLike = webllm.ResumeResult;

const SESSION_KEY = "webllm-resumable-example/session-id";
const OUTPUT_PREFIX = "webllm-resumable-example/output/";
const MODE_KEY = "webllm-resumable-example/mode";
const PROMPT_KEY = "webllm-resumable-example/prompt";
const MODEL_ID = "Qwen3-0.6B-q4f16_1-MLC";
const MODEL_LIB_URL = import.meta.env.VITE_WEBLLM_MODEL_LIB_URL?.trim() ?? "";

const modelNameEl = element<HTMLDivElement>("model-name");
const modelLibEl = element<HTMLAnchorElement>("model-lib");
const engineModeSelect = element<HTMLSelectElement>("engine-mode");
const sessionInput = element<HTMLInputElement>("session-id");
const checkpointIntervalInput = element<HTMLInputElement>(
  "checkpoint-interval",
);
const durabilitySelect = element<HTMLSelectElement>("durability");
const checkpointPromptInput = element<HTMLInputElement>("checkpoint-prompt");
const strictPersistenceInput = element<HTMLInputElement>("strict-persistence");
const autoReloadEnabledInput = element<HTMLInputElement>("auto-reload-enabled");
const autoReloadChunksInput = element<HTMLInputElement>("auto-reload-chunks");
const promptInput = element<HTMLTextAreaElement>("prompt");
const outputPre = element<HTMLPreElement>("output");
const sessionsPre = element<HTMLPreElement>("sessions");
const metricsPre = element<HTMLPreElement>("metrics");
const validationPre = element<HTMLPreElement>("validation");
const logPre = element<HTMLPreElement>("log");
const statusEl = element<HTMLDivElement>("status");
const environmentEl = element<HTMLParagraphElement>("environment");

let engine: any;
let worker: Worker | undefined;
let loadedModel: string | undefined;
let loadedMode: EngineMode | undefined;
let streaming = false;
let validationRunning = false;
let currentOutput = "";

type ValidationStatus = "pass" | "fail" | "skip";

interface ValidationCaseResult {
  name: string;
  status: ValidationStatus;
  elapsedMs: number;
  details?: unknown;
}

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node === null) {
    throw new Error(`Missing element: ${id}`);
  }
  return node as T;
}

function setStatus(status: string): void {
  statusEl.textContent = status;
}

function logLine(line: string, data?: unknown): void {
  const timestamp = new Date().toLocaleTimeString();
  const suffix = data === undefined ? "" : `\n${JSON.stringify(data, null, 2)}`;
  logPre.textContent = `${timestamp} ${line}${suffix}\n${logPre.textContent}`;
}

function currentSessionId(): string {
  const sessionId = sessionInput.value.trim();
  if (sessionId === "") {
    throw new Error("Session ID is required.");
  }
  return sessionId;
}

function selectedMode(): EngineMode {
  return engineModeSelect.value === "main" ? "main" : "worker";
}

function configuredSize(
  value: string | undefined,
  name: string,
): number | undefined {
  if (!value?.trim()) return undefined;
  const size = Number(value);
  if (!Number.isSafeInteger(size) || size <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return size;
}

function resumableAppConfig(): webllm.AppConfig {
  if (!MODEL_LIB_URL) {
    throw new Error(
      "Set VITE_WEBLLM_MODEL_LIB_URL to a compatible Qwen3 model WASM URL before loading. See README.md.",
    );
  }
  const baseConfig = webllm.prebuiltAppConfig;
  const modelRecord = baseConfig.model_list.find(
    (record) => record.model_id === MODEL_ID,
  );
  if (modelRecord === undefined) {
    throw new Error(`Cannot find model record for ${MODEL_ID}.`);
  }
  // The selected model library differs from the prebuilt artifact. Keep the
  // weight/tokenizer integrity metadata, but do not apply the old WASM hash.
  const integrity = modelRecord.integrity
    ? { ...modelRecord.integrity, model_lib: undefined }
    : undefined;
  const context = configuredSize(
    import.meta.env.VITE_WEBLLM_CONTEXT_WINDOW_SIZE,
    "VITE_WEBLLM_CONTEXT_WINDOW_SIZE",
  );
  const prefill = configuredSize(
    import.meta.env.VITE_WEBLLM_PREFILL_CHUNK_SIZE,
    "VITE_WEBLLM_PREFILL_CHUNK_SIZE",
  );
  return {
    ...baseConfig,
    model_list: [
      {
        ...modelRecord,
        integrity,
        model_lib: MODEL_LIB_URL,
        overrides: {
          ...modelRecord.overrides,
          ...(context === undefined ? {} : { context_window_size: context }),
          ...(prefill === undefined ? {} : { prefill_chunk_size: prefill }),
        },
      },
    ],
  };
}

function saveSessionState(): void {
  const sessionId = sessionInput.value.trim();
  if (sessionId !== "") {
    localStorage.setItem(SESSION_KEY, sessionId);
    localStorage.setItem(`${OUTPUT_PREFIX}${sessionId}`, currentOutput);
  }
  localStorage.setItem(MODE_KEY, selectedMode());
  localStorage.setItem(PROMPT_KEY, promptInput.value);
}

function restoreSessionState(): void {
  const sessionId = localStorage.getItem(SESSION_KEY) ?? `manual-${Date.now()}`;
  sessionInput.value = sessionId;
  currentOutput = localStorage.getItem(`${OUTPUT_PREFIX}${sessionId}`) ?? "";
  outputPre.textContent = currentOutput;

  const savedPrompt = localStorage.getItem(PROMPT_KEY);
  promptInput.value =
    savedPrompt ??
    "Write a long numbered list of practical browser debugging steps. Include at least 300 short items.";

  const savedMode = localStorage.getItem(MODE_KEY);
  if (savedMode === "main" || savedMode === "worker") {
    engineModeSelect.value = savedMode;
  }
}

function renderModelInfo(): void {
  modelNameEl.textContent = MODEL_ID;
  modelLibEl.href = MODEL_LIB_URL;
  modelLibEl.textContent = MODEL_LIB_URL || "Not configured (see README.md)";
}

async function updateEnvironment(): Promise<void> {
  const webgpu = "gpu" in navigator ? "WebGPU available" : "No WebGPU";
  const context = window.isSecureContext ? "secure context" : "not secure";
  let storage = "storage estimate unavailable";
  try {
    const estimate = await navigator.storage?.estimate?.();
    if (
      typeof estimate?.quota === "number" &&
      typeof estimate.usage === "number"
    ) {
      const free = Math.max(0, estimate.quota - estimate.usage);
      storage = `${formatBytes(free)} free of ${formatBytes(estimate.quota)}`;
    }
  } catch {
    storage = "storage estimate failed";
  }
  environmentEl.textContent = `${webgpu}, ${context}, ${storage}`;
}

function formatBytes(bytes: number): string {
  const mib = bytes / (1024 * 1024);
  if (mib < 1024) {
    return `${mib.toFixed(1)} MiB`;
  }
  return `${(mib / 1024).toFixed(2)} GiB`;
}

async function loadModel(): Promise<void> {
  const mode = selectedMode();
  if (engine !== undefined && loadedModel === MODEL_ID && loadedMode === mode) {
    return;
  }

  await unloadEngine();
  setStatus("Loading model");
  logLine(`Loading ${MODEL_ID} in ${mode} mode`, {
    modelLib: MODEL_LIB_URL,
  });

  const initProgressCallback = (progress: unknown) => {
    const text =
      typeof (progress as { text?: unknown })?.text === "string"
        ? (progress as { text: string }).text
        : "Loading model";
    setStatus(text);
  };

  if (mode === "worker") {
    worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    engine = await (webllm as any).CreateWebWorkerMLCEngine(worker, MODEL_ID, {
      appConfig: resumableAppConfig(),
      initProgressCallback,
    });
  } else {
    engine = await (webllm as any).CreateMLCEngine(MODEL_ID, {
      appConfig: resumableAppConfig(),
      initProgressCallback,
    });
  }

  loadedModel = MODEL_ID;
  loadedMode = mode;
  setStatus("Model loaded");
  logLine("Model loaded", {
    model: MODEL_ID,
    modelLib: MODEL_LIB_URL,
    mode,
  });
  saveSessionState();
  renderMetrics();
}

async function unloadEngine(): Promise<void> {
  try {
    await engine?.unload?.();
  } finally {
    worker?.terminate();
    worker = undefined;
    engine = undefined;
    loadedModel = undefined;
    loadedMode = undefined;
  }
}

async function startGeneration(): Promise<void> {
  if (streaming) {
    return;
  }
  await loadModel();

  const sessionId = currentSessionId();
  currentOutput = "";
  outputPre.textContent = "";
  saveSessionState();

  const checkpointIntervalTokens = Number(checkpointIntervalInput.value);
  const reloadAfterChunks = Number(autoReloadChunksInput.value);
  const request = {
    model: MODEL_ID,
    messages: [{ role: "user", content: promptInput.value }],
    stream: true,
    seed: 1234,
    max_tokens: 512,
    extra_body: {
      resumable: {
        enabled: true,
        sessionId,
        checkpointIntervalTokens,
        checkpointPrompt: checkpointPromptInput.checked,
        durabilityMode: durabilitySelect.value,
        strictPersistence: strictPersistenceInput.checked,
      },
    },
  };

  streaming = true;
  setStatus("Streaming");
  logLine("Starting resumable generation", request.extra_body.resumable);
  try {
    const chunks = await engine.chat.completions.create(request);
    let chunkCount = 0;
    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      if (delta !== "") {
        currentOutput += delta;
        outputPre.textContent = currentOutput;
        outputPre.scrollTop = outputPre.scrollHeight;
        saveSessionState();
      }
      chunkCount += 1;
      if (
        autoReloadEnabledInput.checked &&
        Number.isFinite(reloadAfterChunks) &&
        chunkCount >= reloadAfterChunks
      ) {
        logLine(`Reloading after ${chunkCount} chunks`);
        saveSessionState();
        location.reload();
        return;
      }
    }
    setStatus("Finished");
    logLine("Generation finished");
  } catch (err) {
    setStatus("Generation failed");
    logLine("Generation failed", errorJson(err));
    throw err;
  } finally {
    streaming = false;
    renderMetrics();
    await listSessions().catch(() => undefined);
  }
}

async function interruptGeneration(): Promise<void> {
  if (engine === undefined) {
    return;
  }
  await engine.interruptGenerate?.();
  logLine("Interrupt requested");
}

async function listSessions(): Promise<void> {
  await loadModel();
  const sessions = await engine.listResumableSessions();
  sessionsPre.textContent = JSON.stringify(sessions, null, 2);
  logLine("Listed resumable sessions");
}

async function restoreTextOnly(): Promise<void> {
  await loadModel();
  const result = (await engine.resumeChatCompletion(
    currentSessionId(),
  )) as ResumeResultLike;
  renderResumeResult(result);
  logLine("Text restore complete", result);
}

async function resumeAndContinue(): Promise<void> {
  await loadModel();
  setStatus("Resuming");
  const restored = (await engine.resumeChatCompletion(
    currentSessionId(),
  )) as ResumeResultLike;
  renderResumeResult(restored);
  const result = await engine.resumeChatCompletion(currentSessionId(), {
    continueGeneration: true,
    stream: true,
  });
  if (isAsyncIterable(result)) {
    for await (const chunk of result) {
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      currentOutput += delta;
      outputPre.textContent = currentOutput;
      saveSessionState();
    }
  } else {
    renderResumeResult(result as ResumeResultLike);
  }
  setStatus("Resume complete");
  renderMetrics();
  await listSessions().catch(() => undefined);
}

interface ValidationRequestOptions {
  maxTokens?: number;
  checkpointIntervalTokens?: number;
  checkpointPrompt?: boolean;
  strictPersistence?: boolean;
}

interface InterruptedValidationSession {
  firstOutput: string;
  chunks: number;
  deltas: number;
}

interface ResumeValidationSession {
  recoveredText: string;
  deltaText: string;
  chunks: number;
  deltas: number;
  interrupted: boolean;
  recoveryMode: string;
  streamed: boolean;
  nonStreamResult?: unknown;
}

const LOW_QUOTA_THRESHOLD_BYTES = 512 * 1024 * 1024;

function validationSessionId(label: string): string {
  return `validation-${label}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
}

function validationPrompt(label: string): string {
  return [
    `Validation case: ${label}.`,
    "Write a long numbered checklist for debugging browser-based LLM inference.",
    "Use short concrete items, continue until you have produced at least 180 items, and do not stop early.",
  ].join(" ");
}

function validationRequest(
  sessionId: string,
  prompt: string,
  options: ValidationRequestOptions = {},
): Record<string, unknown> {
  return {
    model: MODEL_ID,
    messages: [{ role: "user", content: prompt }],
    stream: true,
    seed: 1234,
    max_tokens: options.maxTokens ?? 384,
    extra_body: {
      resumable: {
        enabled: true,
        sessionId,
        checkpointIntervalTokens: options.checkpointIntervalTokens ?? 32,
        checkpointPrompt: options.checkpointPrompt ?? true,
        durabilityMode: "exact",
        strictPersistence: options.strictPersistence ?? false,
      },
    },
  };
}

async function resetValidationSession(sessionId: string): Promise<void> {
  await engine?.deleteResumableSession?.(sessionId).catch(() => undefined);
  localStorage.removeItem(`${OUTPUT_PREFIX}${sessionId}`);
}

async function drainIterator(iterator: AsyncIterator<any>): Promise<void> {
  while (!(await iterator.next()).done) {
    // Keep consuming so generator finally blocks release session/model locks.
  }
}

async function startInterruptedValidationSession(
  sessionId: string,
  prompt: string,
  chunksBeforeInterrupt = 1,
): Promise<InterruptedValidationSession> {
  const chunks = await engine.chat.completions.create(
    validationRequest(sessionId, prompt),
  );
  if (!isAsyncIterable(chunks)) {
    throw new Error("Expected streaming validation generation.");
  }
  const iterator = chunks[Symbol.asyncIterator]();
  let firstOutput = "";
  let chunkCount = 0;
  let deltaCount = 0;

  while (true) {
    const next = await iterator.next();
    if (next.done) {
      throw new Error("Generation finished before validation interrupt.");
    }
    chunkCount += 1;
    const delta = next.value.choices?.[0]?.delta?.content ?? "";
    if (delta === "") {
      continue;
    }
    firstOutput += delta;
    deltaCount += 1;
    if (deltaCount >= chunksBeforeInterrupt) {
      await engine.interruptGenerate?.();
      await drainIterator(iterator);
      return { firstOutput, chunks: chunkCount, deltas: deltaCount };
    }
  }
}

async function resumeValidationSession(
  sessionId: string,
  options: { interruptAfterDeltas?: number } = {},
): Promise<ResumeValidationSession> {
  const restored = (await engine.resumeChatCompletion(
    sessionId,
  )) as ResumeResultLike;
  const continuation = await engine.resumeChatCompletion(sessionId, {
    continueGeneration: true,
    stream: true,
  });
  if (!isAsyncIterable(continuation)) {
    return {
      recoveredText: restored.recoveredText,
      deltaText: "",
      chunks: 0,
      deltas: 0,
      interrupted: false,
      recoveryMode: restored.recoveryMode,
      streamed: false,
      nonStreamResult: continuation,
    };
  }

  const iterator = continuation[Symbol.asyncIterator]();
  let deltaText = "";
  let chunkCount = 0;
  let deltaCount = 0;
  let interrupted = false;

  while (true) {
    const next = await iterator.next();
    if (next.done) {
      break;
    }
    chunkCount += 1;
    const delta = next.value.choices?.[0]?.delta?.content ?? "";
    if (delta === "") {
      continue;
    }
    deltaText += delta;
    deltaCount += 1;
    if (
      options.interruptAfterDeltas !== undefined &&
      deltaCount >= options.interruptAfterDeltas
    ) {
      await engine.interruptGenerate?.();
      interrupted = true;
      break;
    }
  }
  if (interrupted) {
    await drainIterator(iterator);
  }

  return {
    recoveredText: restored.recoveredText,
    deltaText,
    chunks: chunkCount,
    deltas: deltaCount,
    interrupted,
    recoveryMode: restored.recoveryMode,
    streamed: true,
  };
}

async function findValidationSession(sessionId: string): Promise<unknown> {
  const sessions = await engine.listResumableSessions();
  return sessions.find(
    (session: { sessionId?: unknown }) => session.sessionId === sessionId,
  );
}

async function ensureWorkerModelLoaded(): Promise<void> {
  engineModeSelect.value = "worker";
  await loadModel();
}

async function validateWorkerStreamingResume(): Promise<unknown> {
  await ensureWorkerModelLoaded();
  const sessionId = validationSessionId("worker-stream");
  let passed = false;
  await resetValidationSession(sessionId);
  try {
    const interrupted = await startInterruptedValidationSession(
      sessionId,
      validationPrompt("worker streaming resume"),
    );
    const resumed = await resumeValidationSession(sessionId);
    if (!resumed.streamed) {
      throw new Error("Resume continuation did not return a stream.");
    }
    if (resumed.deltaText.length === 0) {
      throw new Error("Resume stream produced no continuation text.");
    }
    passed = true;
    return {
      sessionId,
      interrupted,
      resumed: {
        chunks: resumed.chunks,
        deltas: resumed.deltas,
        recoveryMode: resumed.recoveryMode,
        recoveredChars: resumed.recoveredText.length,
        deltaChars: resumed.deltaText.length,
      },
      session: await findValidationSession(sessionId),
    };
  } finally {
    if (passed) {
      await resetValidationSession(sessionId);
    }
  }
}

async function validateRepeatedResume(): Promise<unknown> {
  await ensureWorkerModelLoaded();
  const sessionId = validationSessionId("repeat-resume");
  let passed = false;
  await resetValidationSession(sessionId);
  try {
    const interrupted = await startInterruptedValidationSession(
      sessionId,
      validationPrompt("same-session repeated resume"),
    );
    const firstResume = await resumeValidationSession(sessionId, {
      interruptAfterDeltas: 1,
    });
    if (!firstResume.streamed) {
      throw new Error("First resume continuation did not return a stream.");
    }
    if (!firstResume.interrupted || firstResume.deltaText.length === 0) {
      throw new Error("First resume did not interrupt after streaming text.");
    }
    const afterFirstResume = await findValidationSession(sessionId);
    const secondResume = await resumeValidationSession(sessionId);
    if (!secondResume.streamed) {
      throw new Error("Second resume continuation did not return a stream.");
    }
    if (secondResume.deltaText.length === 0) {
      throw new Error("Second resume produced no continuation text.");
    }
    passed = true;
    return {
      sessionId,
      interrupted,
      firstResume: {
        chunks: firstResume.chunks,
        deltas: firstResume.deltas,
        deltaChars: firstResume.deltaText.length,
        recoveryMode: firstResume.recoveryMode,
      },
      afterFirstResume,
      secondResume: {
        chunks: secondResume.chunks,
        deltas: secondResume.deltas,
        deltaChars: secondResume.deltaText.length,
        recoveryMode: secondResume.recoveryMode,
      },
    };
  } finally {
    if (passed) {
      await resetValidationSession(sessionId);
    }
  }
}

async function validateWorkerLockExclusion(): Promise<unknown> {
  await ensureWorkerModelLoaded();
  const sessionId = validationSessionId("worker-lock");
  let passed = false;
  await resetValidationSession(sessionId);
  const chunks = await engine.chat.completions.create(
    validationRequest(sessionId, validationPrompt("worker lock exclusion"), {
      maxTokens: 384,
    }),
  );
  if (!isAsyncIterable(chunks)) {
    throw new Error("Expected streaming validation generation.");
  }

  const iterator = chunks[Symbol.asyncIterator]();
  const first = await iterator.next();
  if (first.done) {
    throw new Error("Generation finished before lock exclusion probe.");
  }

  const contenderWorker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
  const contender = new (webllm as any).WebWorkerMLCEngine(contenderWorker, {
    appConfig: resumableAppConfig(),
  });

  try {
    let rejectedMessage = "";
    try {
      await contender.resumeChatCompletion(sessionId, {
        continueGeneration: true,
      });
    } catch (err) {
      rejectedMessage = err instanceof Error ? err.message : String(err);
    }
    if (!rejectedMessage.includes("Resumable session is already active")) {
      throw new Error(
        `Expected active-session lock rejection, got: ${rejectedMessage}`,
      );
    }
    passed = true;
    return {
      sessionId,
      firstChunkHadText:
        (first.value.choices?.[0]?.delta?.content ?? "").length > 0,
      rejectedMessage,
    };
  } finally {
    await engine.interruptGenerate?.();
    await drainIterator(iterator).catch((err) => {
      logLine("Validation lock cleanup drain failed", errorJson(err));
    });
    await contender.unload?.().catch(() => undefined);
    contenderWorker.terminate();
    if (passed) {
      await resetValidationSession(sessionId);
    }
  }
}

async function validateLowQuotaObserved(): Promise<ValidationCaseResult> {
  const name = "low quota behavior";
  const started = performance.now();
  try {
    const estimate = await navigator.storage?.estimate?.();
    if (
      typeof estimate?.quota !== "number" ||
      typeof estimate.usage !== "number"
    ) {
      return {
        name,
        status: "skip",
        elapsedMs: performance.now() - started,
        details: "navigator.storage.estimate() is unavailable.",
      };
    }
    const freeBytes = Math.max(0, estimate.quota - estimate.usage);
    if (freeBytes >= LOW_QUOTA_THRESHOLD_BYTES) {
      return {
        name,
        status: "skip",
        elapsedMs: performance.now() - started,
        details: {
          freeBytes,
          quotaBytes: estimate.quota,
          thresholdBytes: LOW_QUOTA_THRESHOLD_BYTES,
          reason:
            "Browser quota is not low enough to exercise KV checkpoint skipping.",
        },
      };
    }

    await ensureWorkerModelLoaded();
    const sessionId = validationSessionId("low-quota");
    let passed = false;
    await resetValidationSession(sessionId);
    try {
      const interrupted = await startInterruptedValidationSession(
        sessionId,
        validationPrompt("low quota"),
      );
      const session = await findValidationSession(sessionId);
      const restored = (await engine.resumeChatCompletion(
        sessionId,
      )) as ResumeResultLike;
      if (restored.recoveredText.length === 0) {
        throw new Error("Low-quota recovery produced no restored text.");
      }
      passed = true;
      return {
        name,
        status: "pass",
        elapsedMs: performance.now() - started,
        details: {
          sessionId,
          freeBytes,
          quotaBytes: estimate.quota,
          interrupted,
          recoveryMode: restored.recoveryMode,
          restoredChars: restored.recoveredText.length,
          session,
        },
      };
    } finally {
      if (passed) {
        await resetValidationSession(sessionId);
      }
    }
  } catch (err) {
    return {
      name,
      status: "fail",
      elapsedMs: performance.now() - started,
      details: errorJson(err),
    };
  }
}

async function runValidationCase(
  name: string,
  fn: () => Promise<unknown>,
): Promise<ValidationCaseResult> {
  const started = performance.now();
  try {
    const details = await fn();
    return {
      name,
      status: "pass",
      elapsedMs: performance.now() - started,
      details,
    };
  } catch (err) {
    return {
      name,
      status: "fail",
      elapsedMs: performance.now() - started,
      details: errorJson(err),
    };
  }
}

function renderValidationResults(results: ValidationCaseResult[]): void {
  validationPre.textContent = JSON.stringify(results, null, 2);
}

async function runValidationSuite(): Promise<void> {
  if (streaming || validationRunning) {
    return;
  }
  validationRunning = true;
  streaming = true;
  const results: ValidationCaseResult[] = [];
  try {
    setStatus("Running validation");
    validationPre.textContent = "Running validation...";

    results.push(
      await runValidationCase(
        "worker mode streaming resume",
        validateWorkerStreamingResume,
      ),
    );
    renderValidationResults(results);

    results.push(
      await runValidationCase(
        "same-session repeated resume",
        validateRepeatedResume,
      ),
    );
    renderValidationResults(results);

    results.push(
      await runValidationCase(
        "worker lock exclusion",
        validateWorkerLockExclusion,
      ),
    );
    renderValidationResults(results);

    results.push(await validateLowQuotaObserved());
    renderValidationResults(results);

    setStatus(
      results.some((result) => result.status === "fail")
        ? "Validation failed"
        : "Validation complete",
    );
    logLine("Validation complete", results);
  } finally {
    streaming = false;
    validationRunning = false;
    renderMetrics();
    await listSessions().catch(() => undefined);
  }
}

async function deleteSession(): Promise<void> {
  await loadModel();
  const sessionId = currentSessionId();
  await engine.deleteResumableSession(sessionId);
  localStorage.removeItem(`${OUTPUT_PREFIX}${sessionId}`);
  currentOutput = "";
  outputPre.textContent = "";
  sessionsPre.textContent = "";
  renderMetrics();
  logLine("Deleted session", { sessionId });
}

function isAsyncIterable(value: unknown): value is AsyncIterable<any> {
  return (
    value !== null && typeof value === "object" && Symbol.asyncIterator in value
  );
}

function renderResumeResult(result: ResumeResultLike): void {
  currentOutput = result.recoveredText;
  outputPre.textContent = currentOutput;
  saveSessionState();
  renderMetrics();
}

function renderMetrics(): void {
  const metrics = engine?.lastResumableMetrics;
  if (metrics !== undefined) {
    metricsPre.textContent = JSON.stringify(metrics, null, 2);
    return;
  }
  metricsPre.textContent =
    selectedMode() === "worker"
      ? "Internal timing metrics are stored inside the worker engine. Switch to Main Thread mode if you need to inspect lastResumableMetrics from this page."
      : "No resumable metrics recorded yet.";
}

function errorJson(err: unknown): Record<string, unknown> {
  if (err instanceof Error) {
    return {
      name: err.name,
      message: err.message,
      stack: err.stack,
    };
  }
  return { value: String(err) };
}

function bind(id: string, handler: () => Promise<void> | void): void {
  element<HTMLButtonElement>(id).addEventListener("click", () => {
    Promise.resolve(handler()).catch((err) => {
      setStatus("Error");
      logLine("Action failed", errorJson(err));
    });
  });
}

function bindStatePersistence(): void {
  for (const node of [engineModeSelect, sessionInput, promptInput]) {
    node.addEventListener("change", saveSessionState);
  }
  promptInput.addEventListener("input", saveSessionState);
  sessionInput.addEventListener("input", () => {
    const sessionId = sessionInput.value.trim();
    currentOutput =
      sessionId === ""
        ? ""
        : (localStorage.getItem(`${OUTPUT_PREFIX}${sessionId}`) ?? "");
    outputPre.textContent = currentOutput;
    saveSessionState();
  });
}

async function init(): Promise<void> {
  renderModelInfo();
  restoreSessionState();
  bindStatePersistence();
  bind("load-model", loadModel);
  bind("start", startGeneration);
  bind("interrupt", interruptGeneration);
  bind("reload-page", () => {
    saveSessionState();
    location.reload();
  });
  bind("list-sessions", listSessions);
  bind("resume-text", restoreTextOnly);
  bind("resume-continue", resumeAndContinue);
  bind("delete-session", deleteSession);
  bind("run-validation", runValidationSuite);
  validationPre.textContent = "Not run.";
  await updateEnvironment();
  renderMetrics();
}

init().catch((err) => {
  setStatus("Init failed");
  logLine("Init failed", errorJson(err));
});
