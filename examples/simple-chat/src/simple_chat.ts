import appConfig from "./app-config";
import { ChatInterface, ChatModule, ChatRestModule, ChatWorkerClient, AppConfig } from "@mlc-ai/web-llm";

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
  private chat: ChatInterface;
  private localChat: ChatInterface;
  private config: AppConfig = appConfig;
  private selectedModel: string;
  private chatLoaded = false;
  private requestInProgress = false;
  // We use a request chain to ensure that
  // all requests send to chat are sequentialized
  private chatRequestChain: Promise<void> = Promise.resolve();

  private constructor() {
  }

  /**
   * An asynchronous factory constructor since we need to await getMaxStorageBufferBindingSize();
   * this is not allowed in a constructor (which cannot be asynchronous).
   */
  public static CreateAsync = async (chat: ChatInterface, localChat: ChatInterface) => {
    const chatUI = new ChatUI();
    // use web worker to run chat generation in background
    chatUI.chat = chat;
    chatUI.localChat = localChat;
    // get the elements
    chatUI.uiChat = getElementAndCheck("chatui-chat");
    chatUI.uiChatInput = getElementAndCheck("chatui-input") as HTMLInputElement;
    chatUI.uiChatInfoLabel = getElementAndCheck("chatui-info-label") as HTMLLabelElement;
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
    const androidMaxStorageBufferBindingSize = 1 << 27;  // 128MB
    const mobileVendors = new Set<string>(["qualcomm", "arm"])
    let restrictModels = false;
    let maxStorageBufferBindingSize: number;
    let gpuVendor: string;
    try {
      [maxStorageBufferBindingSize, gpuVendor] = await Promise.all([
        chat.getMaxStorageBufferBindingSize(),
        chat.getGPUVendor(),
      ]);
    } catch (err) {
      chatUI.appendMessage("error", "Init error, " + err.toString());
      console.log(err.stack);
      return;
    }
    if ((gpuVendor.length != 0 && mobileVendors.has(gpuVendor)) ||
      (maxStorageBufferBindingSize <= androidMaxStorageBufferBindingSize)) {
      chatUI.appendMessage("init", "Your device seems to have " +
        "limited resources, so we restrict the selectable models.");
      restrictModels = true;
    }

    // Populate modelSelector
    const modelSelector = getElementAndCheck("chatui-select") as HTMLSelectElement;
    for (let i = 0; i < chatUI.config.model_list.length; ++i) {
      const item = chatUI.config.model_list[i];
      const opt = document.createElement("option");
      opt.value = item.local_id;
      opt.innerHTML = item.local_id;
      opt.selected = (i == 0);
      if (
        (restrictModels && (item.low_resource_required === undefined || !item.low_resource_required)) ||
        (item.buffer_size_required_bytes && maxStorageBufferBindingSize < item.buffer_size_required_bytes)
      ) {
        // Either on a low-resource device and not a low-resource model
        // Or device's maxStorageBufferBindingSize does not satisfy the model's need (if specified)
        const params = new URLSearchParams(location.search);
        opt.disabled = !params.has("bypassRestrictions");
        opt.selected = false;
      }
      if (!modelSelector.lastChild?.textContent?.startsWith(opt.value.split('-')[0])) {
        modelSelector.appendChild(document.createElement("hr"));
      }
      modelSelector.appendChild(opt);
    }
    modelSelector.appendChild(document.createElement("hr"));

    // Append local server option to the model selector
    const localServerOpt = document.createElement("option");
    localServerOpt.value = "Local Server";
    localServerOpt.innerHTML = "Local Server";
    modelSelector.append(localServerOpt);
    chatUI.selectedModel = modelSelector.value;
    modelSelector.onchange = () => {
      chatUI.onSelectChange(modelSelector);
    };

    return chatUI;
  }

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
  // can be interrupted by chat.interruptGenerate
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
      this.chat.interruptGenerate();
    }
    // try reset after previous requests finishes
    this.pushTask(async () => {
      await this.chat.resetChat();
      this.resetChatHistory();
      await this.unloadChat();
      this.selectedModel = modelSelector.value;
      await this.asyncInitChat();
    });
  }

  private async onReset() {
    if (this.requestInProgress) {
      // interrupt previous generation if any
      this.chat.interruptGenerate();
    }
    // try reset after previous requests finishes
    this.pushTask(async () => {
      await this.chat.resetChat();
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
    const list = text.split('\n').map((t) => {
      const item = document.createElement('div');
      item.textContent = t;
      return item;
    });
    msgText[0].innerHTML = '';
    list.forEach((item) => msgText[0].append(item));
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  private resetChatHistory() {
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
    }
    this.chat.setInitProgressCallback(initProgressCallback);

    try {
      if (this.selectedModel != "Local Server") {
        await this.chat.reload(this.selectedModel, undefined, this.config);
      }
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
    await this.chat.unload();
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
    const callbackUpdateResponse = (step, msg) => {
      this.updateLastMessage("left", msg);
    };

    try {
      if (this.selectedModel == "Local Server") {
        const output = await this.localChat.generate(prompt, callbackUpdateResponse);
        this.updateLastMessage("left", output);
        this.uiChatInfoLabel.innerHTML = await this.localChat.runtimeStatsText();
      } else {
        const output = await this.chat.generate(prompt, callbackUpdateResponse);
        this.updateLastMessage("left", output);
        this.uiChatInfoLabel.innerHTML = await this.chat.runtimeStatsText();
      }
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
let chat: ChatInterface;
let localChat: ChatInterface;

if (useWebWorker) {
  chat = new ChatWorkerClient(new Worker(
    new URL('./worker.ts', import.meta.url),
    { type: 'module' }
  ));
  localChat = new ChatRestModule();
} else {
  chat = new ChatModule();
  localChat = new ChatRestModule();
}
ChatUI.CreateAsync(chat, localChat);
