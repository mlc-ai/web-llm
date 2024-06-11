import appConfig from "./app-config";
import * as webllm from "@mlc-ai/web-llm";

function getElementAndCheck(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element == null) {
    throw Error("Cannot find element " + id);
  }
  return element;
}

class ChatUI {
  private uiChat: HTMLElement;
  private uiChatInput: HTMLInputElement;
  private uiChatInfoLabel: HTMLLabelElement;
  private engine: webllm.MLCEngineInterface | webllm.WebWorkerMLCEngine;
  private config: webllm.AppConfig = appConfig;
  private selectedModel: string;
  private chatLoaded = false;
  private requestInProgress = false;
  private chatHistory: webllm.ChatCompletionMessageParam[] = [];
  // We use a request chain to ensure that
  // all requests send to chat are sequentialized
  private chatRequestChain: Promise<void> = Promise.resolve();

  /**
   * An asynchronous factory constructor since we need to await getMaxStorageBufferBindingSize();
   * this is not allowed in a constructor (which cannot be asynchronous).
   */
  public static CreateAsync = async (engine: webllm.MLCEngineInterface) => {
    const chatUI = new ChatUI();
    chatUI.engine = engine;
    // get the elements
    chatUI.uiChat = getElementAndCheck("chatui-chat");
    chatUI.uiChatInput = getElementAndCheck("chatui-input") as HTMLInputElement;
    chatUI.uiChatInfoLabel = getElementAndCheck(
      "chatui-info-label",
    ) as HTMLLabelElement;
    // register event handlers
    getElementAndCheck("chatui-reset-btn").onclick = () => {
      chatUI.onReset();
    };
    getElementAndCheck("chatui-send-btn").onclick = () => {
      chatUI.onGenerate();
    };
    // TODO: find other alternative triggers
    getElementAndCheck("chatui-input").onkeypress = (event) => {
      if (event.keyCode === 13) {
        chatUI.onGenerate();
      }
    };

    // When we detect low maxStorageBufferBindingSize, we assume that the device (e.g. an Android
    // phone) can only handle small models and make all other models unselectable. Otherwise, the
    // browser may crash. See https://github.com/mlc-ai/web-llm/issues/209.
    // Also use GPU vendor to decide whether it is a mobile device (hence with limited resources).
    const androidMaxStorageBufferBindingSize = 1 << 27; // 128MB
    const mobileVendors = new Set<string>(["qualcomm", "arm"]);
    let restrictModels = false;
    let maxStorageBufferBindingSize: number;
    let gpuVendor: string;
    try {
      [maxStorageBufferBindingSize, gpuVendor] = await Promise.all([
        engine.getMaxStorageBufferBindingSize(),
        engine.getGPUVendor(),
      ]);
    } catch (err) {
      chatUI.appendMessage("error", "Init error, " + err.toString());
      console.log(err.stack);
      return;
    }
    if (
      (gpuVendor.length != 0 && mobileVendors.has(gpuVendor)) ||
      maxStorageBufferBindingSize <= androidMaxStorageBufferBindingSize
    ) {
      chatUI.appendMessage(
        "init",
        "Your device seems to have " +
          "limited resources, so we restrict the selectable models.",
      );
      restrictModels = true;
    }

    // Populate modelSelector
    const modelSelector = getElementAndCheck(
      "chatui-select",
    ) as HTMLSelectElement;
    for (let i = 0; i < chatUI.config.model_list.length; ++i) {
      const item = chatUI.config.model_list[i];
      const opt = document.createElement("option");
      opt.value = item.model_id;
      opt.innerHTML = item.model_id;
      opt.selected = i == 0;
      if (
        (restrictModels &&
          (item.low_resource_required === undefined ||
            !item.low_resource_required)) ||
        (item.buffer_size_required_bytes &&
          maxStorageBufferBindingSize < item.buffer_size_required_bytes)
      ) {
        // Either on a low-resource device and not a low-resource model
        // Or device's maxStorageBufferBindingSize does not satisfy the model's need (if specified)
        const params = new URLSearchParams(location.search);
        opt.disabled = !params.has("bypassRestrictions");
        opt.selected = false;
      }
      if (
        !modelSelector.lastChild?.textContent?.startsWith(
          opt.value.split("-")[0],
        )
      ) {
        modelSelector.appendChild(document.createElement("hr"));
      }
      modelSelector.appendChild(opt);
    }
    modelSelector.appendChild(document.createElement("hr"));

    chatUI.selectedModel = modelSelector.value;
    modelSelector.onchange = () => {
      chatUI.onSelectChange(modelSelector);
    };

    return chatUI;
  };

  /**
   * Push a task to the execution queue.
   *
   * @param task The task to be executed;
   */
  private pushTask(task: () => Promise<void>) {
    const lastEvent = this.chatRequestChain;
    this.chatRequestChain = lastEvent.then(task);
  }
  // Event handlers
  // all event handler pushes the tasks to a queue
  // that get executed sequentially
  // the tasks previous tasks, which causes them to early stop
  // can be interrupted by engine.interruptGenerate
  private async onGenerate() {
    if (this.requestInProgress) {
      return;
    }
    this.pushTask(async () => {
      await this.asyncGenerate();
    });
  }

  private async onSelectChange(modelSelector: HTMLSelectElement) {
    if (this.requestInProgress) {
      // interrupt previous generation if any
      this.engine.interruptGenerate();
    }
    // try reset after previous requests finishes
    this.pushTask(async () => {
      await this.engine.resetChat();
      this.resetChatHistory();
      await this.unloadChat();
      this.selectedModel = modelSelector.value;
      await this.asyncInitChat();
    });
  }

  private async onReset() {
    if (this.requestInProgress) {
      // interrupt previous generation if any
      this.engine.interruptGenerate();
    }
    // try reset after previous requests finishes
    this.pushTask(async () => {
      await this.engine.resetChat();
      this.resetChatHistory();
    });
  }

  // Internal helper functions
  private appendMessage(kind, text) {
    if (kind == "init") {
      text = "[System Initalize] " + text;
    }
    if (this.uiChat === undefined) {
      throw Error("cannot find ui chat");
    }
    const msg = `
      <div class="msg ${kind}-msg">
        <div class="msg-bubble">
          <div class="msg-text">${text}</div>
        </div>
      </div>
    `;
    this.uiChat.insertAdjacentHTML("beforeend", msg);
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  private updateLastMessage(kind, text) {
    if (kind == "init") {
      text = "[System Initalize] " + text;
    }
    if (this.uiChat === undefined) {
      throw Error("cannot find ui chat");
    }
    const matches = this.uiChat.getElementsByClassName(`msg ${kind}-msg`);
    if (matches.length == 0) throw Error(`${kind} message do not exist`);
    const msg = matches[matches.length - 1];
    const msgText = msg.getElementsByClassName("msg-text");
    if (msgText.length != 1) throw Error("Expect msg-text");
    if (msgText[0].innerHTML == text) return;
    const list = text.split("\n").map((t) => {
      const item = document.createElement("div");
      item.textContent = t;
      return item;
    });
    msgText[0].innerHTML = "";
    list.forEach((item) => msgText[0].append(item));
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  private resetChatHistory() {
    this.chatHistory = [];
    const clearTags = ["left", "right", "init", "error"];
    for (const tag of clearTags) {
      // need to unpack to list so the iterator don't get affected by mutation
      const matches = [...this.uiChat.getElementsByClassName(`msg ${tag}-msg`)];
      for (const item of matches) {
        this.uiChat.removeChild(item);
      }
    }
    if (this.uiChatInfoLabel !== undefined) {
      this.uiChatInfoLabel.innerHTML = "";
    }
  }

  private async asyncInitChat() {
    if (this.chatLoaded) return;
    this.requestInProgress = true;
    this.appendMessage("init", "");
    const initProgressCallback = (report) => {
      this.updateLastMessage("init", report.text);
    };
    this.engine.setInitProgressCallback(initProgressCallback);

    try {
      await this.engine.reload(this.selectedModel);
    } catch (err) {
      this.appendMessage("error", "Init error, " + err.toString());
      console.log(err.stack);
      this.unloadChat();
      this.requestInProgress = false;
      return;
    }
    this.requestInProgress = false;
    this.chatLoaded = true;
  }

  private async unloadChat() {
    await this.engine.unload();
    this.chatLoaded = false;
  }

  /**
   * Run generate
   */
  private async asyncGenerate() {
    await this.asyncInitChat();
    this.requestInProgress = true;
    const prompt = this.uiChatInput.value;
    if (prompt == "") {
      this.requestInProgress = false;
      return;
    }

    this.appendMessage("right", prompt);
    this.uiChatInput.value = "";
    this.uiChatInput.setAttribute("placeholder", "Generating...");

    this.appendMessage("left", "");
    this.chatHistory.push({ role: "user", content: prompt });

    try {
      let curMessage = "";
      let usage: webllm.CompletionUsage | undefined = undefined;
      const completion = await this.engine.chat.completions.create({
        stream: true,
        messages: this.chatHistory,
        stream_options: { include_usage: true },
      });
      // TODO(Charlie): Processing of � requires changes
      for await (const chunk of completion) {
        const curDelta = chunk.choices[0]?.delta.content;
        if (curDelta) {
          curMessage += curDelta;
        }
        this.updateLastMessage("left", curMessage);
        if (chunk.usage) {
          usage = chunk.usage;
        }
      }
      if (usage) {
        this.uiChatInfoLabel.innerHTML =
          `prompt_tokens: ${usage.prompt_tokens}, ` +
          `completion_tokens: ${usage.completion_tokens}, ` +
          `prefill: ${usage.extra.prefill_tokens_per_s.toFixed(4)} tokens/sec, ` +
          `decoding: ${usage.extra.decode_tokens_per_s.toFixed(4)} tokens/sec`;
      }
      const finalMessage = await this.engine.getMessage();
      this.updateLastMessage("left", finalMessage); // TODO: Remove this after � issue is fixed
      this.chatHistory.push({ role: "assistant", content: finalMessage });
    } catch (err) {
      this.appendMessage("error", "Generate error, " + err.toString());
      console.log(err.stack);
      await this.unloadChat();
    }
    this.uiChatInput.setAttribute("placeholder", "Enter your message...");
    this.requestInProgress = false;
  }
}

const useWebWorker = appConfig.use_web_worker;
let engine: webllm.MLCEngineInterface;

// Here we do not use `CreateMLCEngine()` but instantiate an engine that is not loaded with model
if (useWebWorker) {
  engine = new webllm.WebWorkerMLCEngine(
    new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
    { appConfig },
  );
} else {
  engine = new webllm.MLCEngine({ appConfig });
}
ChatUI.CreateAsync(engine);

function getFileType(file: File) {
  if (file.name.includes("wasm")) {
    return "webllm/wasm";
  } else if (
    file.name.includes(".bin") ||
    file.name.includes("ndarray-cache.json")
  ) {
    return "webllm/model";
  } else if (file.name.includes("mlc-chat-config.json")) {
    return "webllm/config";
  } else {
    console.log("No model file suffix found");
    return "file-cache";
  }
}

async function uploadToIndexedDB(file: File) {
  let db;
  const request = indexedDB.open(getFileType(file), 1);
  request.onupgradeneeded = (event) => {
    db = (event.target as IDBOpenDBRequest).result;
    if (!db.objectStoreNames.contains("urls")) {
      db.createObjectStore("urls", { keyPath: "url" });
    }
  };
  request.onsuccess = (event) => {
    db = (event.target as IDBOpenDBRequest).result;
  };
  request.onerror = (event) => {
    console.error("Database error: ", (event.target as IDBOpenDBRequest).error);
  };
  const transaction = db.transaction("files", "readwrite");
  const store = transaction.objectStore("files");
  const reader = new FileReader();
  reader.onload = async (e) => {
    if (e.target === null || e.target.result === null) {
      console.error("Do not read any files");
      return;
    }
    const url = file.name;
    store.add(e.target.result, url);
  };
  transaction.oncomplete = function () {
    alert("All files have been uploaded to IndexedDB.");
  };
  transaction.onerror = function (event) {
    console.error("Error uploading files:", event);
  };
}

async function cacheFile(file: File, response: Response) {
  try {
    const cache = await caches.open(getFileType(file)); // Ensure getFileType is a synchronous function or awaited if async
    console.log("Put response into cache:", response);
    await cache.put(file.name, response);
  } catch (error) {
    console.error("Failed to cache the file:", error);
  }
}

async function uploadFiles(): Promise<void> {
  const input = document.getElementById("file-input") as HTMLInputElement;
  if (!input.files || input.files.length === 0) {
    alert("No files selected.");
    return;
  }
  if (appConfig.useIndexedDBCache) {
    for (const file of input.files) {
      uploadToIndexedDB(file);
    }
  } else {
    for (const file of input.files) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        if (e.target === null || e.target.result === null) {
          console.error("Do not read any files");
          return;
        }
        const arrayBuffer = e.target.result as ArrayBuffer;
        const response = new Response(arrayBuffer, {
          status: 200,
          statusText: "OK",
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-Length": arrayBuffer.byteLength.toString(),
          },
        });
        await cacheFile(file, response);
      };
      if (
        file.name.includes("mlc-chat-config.json") ||
        file.name.includes("ndarray-cache.json")
      ) {
        reader.readAsText(file);
      } else {
        reader.readAsArrayBuffer(file);
      }
    }
  }
}

(window as any).uploadFiles = uploadFiles;
