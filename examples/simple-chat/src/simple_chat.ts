import appConfig from "./app-config";
import { ChatInterface, ChatModule, ChatRestModule, ChatWorkerClient, ModelRecord } from "@mlc-ai/web-llm";

function getElementAndCheck(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element == null) {
    throw Error("Cannot find element " + id);
  }
  return element;
}

interface AppConfig {
  model_list: Array<ModelRecord>;
  model_lib_map?: Record<string, string>;
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

  constructor(chat: ChatInterface, localChat: ChatInterface) {
    // use web worker to run chat generation in background
    this.chat = chat;
    this.localChat = localChat;
    // get the elements
    this.uiChat = getElementAndCheck("chatui-chat");
    this.uiChatInput = getElementAndCheck("chatui-input") as HTMLInputElement;
    this.uiChatInfoLabel = getElementAndCheck("chatui-info-label") as HTMLLabelElement;
    // register event handlers
    getElementAndCheck("chatui-reset-btn").onclick = () => {
      this.onReset();
    };
    getElementAndCheck("chatui-send-btn").onclick = () => {
      this.onGenerate();
    };
    // TODO: find other alternative triggers
    getElementAndCheck("chatui-input").onkeypress = (event) => {
      if (event.keyCode === 13) {
        this.onGenerate();
      }
    };

    const modelSelector = getElementAndCheck("chatui-select") as HTMLSelectElement;
    for (let i = 0; i < this.config.model_list.length; ++i) {
      const item = this.config.model_list[i];
      const opt = document.createElement("option");
      opt.value = item.local_id;
      opt.innerHTML = item.local_id;
      opt.selected = (i == 0);
      modelSelector.appendChild(opt);
    }
    // Append local server option to the model selector
    const localServerOpt = document.createElement("option");
    localServerOpt.value = "Local Server";
    localServerOpt.innerHTML = "Local Server";
    modelSelector.append(localServerOpt);
    this.selectedModel = modelSelector.value;
    modelSelector.onchange = () => {
      this.onSelectChange(modelSelector);
    };
  }
  /**
   * Push a task to the execution queue.
   *
   * @param task The task to be executed;
   */
  private pushTask(task: ()=>Promise<void>) {
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
    {type: 'module'}
  ));
  localChat = new ChatRestModule();
} else {
  chat = new ChatModule();
  localChat = new ChatRestModule();
}
new ChatUI(chat, localChat);
