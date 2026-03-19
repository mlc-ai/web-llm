import * as webllm from "@mlc-ai/web-llm";

function log(msg: string, type: "info" | "cos" | "error" | "warn" = "info") {
  const logs = document.getElementById("logs");
  if (!logs) return;
  const entry = document.createElement("div");
  entry.className = `log-entry ${type === "cos" ? "log-cos" : type === "error" ? "log-error" : type === "warn" ? "log-warn" : ""}`;
  entry.innerText = `${new Date().toLocaleTimeString()} - ${msg}`;
  logs.appendChild(entry);
  logs.scrollTop = logs.scrollHeight;

  // Use originalLog to avoid recursion
  originalLog(`[${type.toUpperCase()}] ${msg}`);
}

// Intercept console.log to show [COS] messages in our UI log
const originalLog = console.log;
console.log = (...args) => {
  originalLog(...args);
  const msg = args
    .map((arg) => (typeof arg === "object" ? JSON.stringify(arg) : String(arg)))
    .join(" ");
  // Only log to UI if it's a COS message and NOT already from our own 'log' call
  if (msg.includes("[COS]") && !msg.includes("[COS]]")) {
    // Our log uses [COS] (upper), WebLLM uses [COS]
    log(msg.replace("[COS] ", ""), "cos");
  }
};

const originalWarn = console.warn;
console.warn = (...args) => {
  originalWarn(...args);
  const msg = args
    .map((arg) => (typeof arg === "object" ? JSON.stringify(arg) : String(arg)))
    .join(" ");
  if (msg.includes("[COS]")) {
    log(msg.replace("[COS] ", ""), "warn");
  }
};

async function checkStatus() {
  // 1. Check WebGPU
  const webgpuStatus = document.getElementById("webgpu-status");
  if (webgpuStatus) {
    if ("gpu" in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          webgpuStatus.className = "status-card status-ok";
          webgpuStatus.querySelector(".text")!.textContent = "Available";
        } else {
          webgpuStatus.className = "status-card status-error";
          webgpuStatus.querySelector(".text")!.textContent = "No Adapter";
        }
      } catch (e) {
        webgpuStatus.className = "status-card status-error";
        webgpuStatus.querySelector(".text")!.textContent = "Error Requesting";
      }
    } else {
      webgpuStatus.className = "status-card status-error";
      webgpuStatus.querySelector(".text")!.textContent = "Not Supported";
    }
  }

  // 2. Check COS
  const cosStatus = document.getElementById("cos-status");
  const hasCOS = "crossOriginStorage" in navigator;
  if (cosStatus) {
    if (hasCOS) {
      cosStatus.className = "status-card status-ok";
      cosStatus.querySelector(".text")!.textContent = "API Available";
      log(
        "Experimental Cross-Origin Storage API detected in navigator.",
        "cos",
      );
    } else {
      cosStatus.className = "status-card status-warn";
      cosStatus.querySelector(".text")!.textContent = "Not Found (Using Mock)";
      log(
        "Cross-Origin Storage API not found. Mocking for demonstration purposes...",
        "warn",
      );
      mockCOS();
    }
  }
}

function mockCOS() {
  // Provide a simple mock for navigator.crossOriginStorage to show the protocol
  (navigator as any).crossOriginStorage = {
    async requestFileHandles(hashes: any[], options: any = {}) {
      log(
        `MOCK: requestFileHandles called for ${hashes.length} hashes (create: ${!!options.create})`,
        "cos",
      );
      // In a real environment, this would talk to the extension.
      // Here we just simulate a miss to let WebLLM fall back to standard cache.
      return hashes.map(() => null);
    },
  };
}

let engine: webllm.MLCEngineInterface | null = null;

async function initEngine() {
  const modelId = (document.getElementById("model-select") as HTMLSelectElement)
    .value;
  const initBtn = document.getElementById("init-btn") as HTMLButtonElement;
  const chatBtn = document.getElementById("chat-btn") as HTMLButtonElement;

  initBtn.disabled = true;
  log(`Initializing engine for ${modelId}...`);
  log("Enabling useCrossOriginStorageCache in AppConfig.", "cos");

  const appConfig: webllm.AppConfig = {
    ...webllm.prebuiltAppConfig,
    useCrossOriginStorageCache: true,
    useIndexedDBCache: true, // Primary cache fallback
  };

  try {
    engine = await webllm.CreateMLCEngine(modelId, {
      appConfig,
      initProgressCallback: (report) => {
        log(report.text);
      },
    });
    log("Engine initialized successfully!", "info");
    chatBtn.disabled = false;
  } catch (err: any) {
    log(`Initialization failed: ${err.message}`, "error");
    initBtn.disabled = false;
  }
}

async function runChat() {
  if (!engine) return;
  const chatBtn = document.getElementById("chat-btn") as HTMLButtonElement;
  chatBtn.disabled = true;

  try {
    log("Running simple chat completion...");
    const messages: webllm.ChatCompletionMessageParam[] = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Say hello!" },
    ];

    const chunks = await engine.chat.completions.create({
      messages,
      stream: true,
    });

    let reply = "";
    for await (const chunk of chunks) {
      const content = chunk.choices[0]?.delta?.content || "";
      reply += content;
      if (content) {
        // Periodically update if it's long, but here it's short
      }
    }
    log(`Assistant: ${reply}`);
  } catch (err: any) {
    log(`Chat failed: ${err.message}`, "error");
  } finally {
    chatBtn.disabled = false;
  }
}

async function clearCache() {
  const modelId = (document.getElementById("model-select") as HTMLSelectElement)
    .value;
  log(`Deleting model ${modelId} from cache...`);
  await webllm.deleteModelAllInfoInCache(modelId);
  log("Cache cleared.");
}

document.getElementById("init-btn")?.addEventListener("click", initEngine);
document.getElementById("chat-btn")?.addEventListener("click", runChat);
document
  .getElementById("clear-cache-btn")
  ?.addEventListener("click", clearCache);

checkStatus();
