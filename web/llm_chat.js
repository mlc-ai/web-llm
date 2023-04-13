class LLMChatPipeline {
  constructor(tokenizer) {
    this.tokenizer = tokenizer;
  }
  dispose() {
    // note: tvm instance is not owned by this class
  }
}

/**
 * A instance that can be used to facilitate deployment.
 */
class LLMChatInstance {
  constructor() {
    this.requestInProgress = false;
    this.config = undefined;
    this.tvm = undefined;
    this.pipeline = undefined;
    this.uiChat = undefined;
    this.uiChatInput = undefined;
  }

  /**
   * Async initialize instance.
   */
  async asyncInit() {
    if (this.pipeline !== undefined) return;
    await this.#asyncInitConfig();
    await this.#asyncInitPipeline(this.config.tokenizer);
    this.uiChat = document.getElementById("chatui-chat");
    this.uiChatInput = document.getElementById("chatui-input");
  }

  /**
   * Async initialize config
   */
  async #asyncInitConfig() {
    if (this.config !== undefined) return;
    this.config = await (await fetch("llm-chat-config.json")).json();
  }

  /**
   * Initialize the pipeline
   *
   * @param tokenizerModel The url to tokenizer model.
   */
  async #asyncInitPipeline(tokenizerUrl) {
    if (this.pipeline !== undefined) return;
    const tokenizer = await tvmjsGlobalEnv.sentencePieceProcessor(tokenizerUrl);
    this.pipeline = new LLMChatPipeline(tokenizer);
  }

  appendMessage(kind, text) {
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

  updateLastMessage(kind, text) {
    const matches = this.uiChat.getElementsByClassName(`msg ${kind}-msg`);
    if (matches.length == 0) throw Error(`${kind} message do not exist`);
    const msg = matches[matches.length - 1];
    const msgText = msg.getElementsByClassName("msg-text");
    if (msgText.length != 1) throw Error("Expect msg-text");
    msgText[0].innerHTML = text;
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  async respondTestMessage(repeat) {
    this.appendMessage("left", "");
    const testMessage = "I am a friendly bot. Please ask questions.";
    const encodedResult = await this.pipeline.tokenizer.encodeIds(testMessage);

    const currentIds = [];
    for (let k = 0; k < repeat; ++k) {
      for (let i = 0; i < encodedResult.length; ++i) {
        currentIds.push(encodedResult[i]);
        const msg = this.pipeline.tokenizer.decodeIds(currentIds);
        this.updateLastMessage("left", msg);
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    }
  }

  /**
   * Run generate
   */
  async generate() {
    if (this.requestInProgress) {
      return;
    }

    this.requestInProgress = true;

    try {
      await this.asyncInit();
    } catch(err) {
      this.appendMessage("error", "Init error, " + err.toString());
      console.log(err.stack);
      this.reset();
      this.requestInProgress = false;
      return;
    }

    const prompt = this.uiChatInput.value;
    this.appendMessage("right", prompt);
    this.uiChatInput.value = "";
    this.uiChatInput.setAttribute("placeholder", "Generating...");

    try {
      await this.respondTestMessage(10);
    } catch (err) {
      this.appendMessage("error", "Generate error, " + err.toString());
      console.log(err.stack);
      this.reset();
    }

    this.uiChatInput.setAttribute("placeholder", "Enter your message...");
    this.requestInProgress = false;
  }

  /**
   * Reset the instance;
   */
  reset() {
    this.tvm = undefined;
    if (this.pipeline !== undefined) {
      this.pipeline.dispose();
    }
    this.pipeline = undefined;
  }
}

localLLMChatIntance = new LLMChatInstance();

tvmjsGlobalEnv.asyncOnGenerate = async function () {
  await localLLMChatIntance.generate();
};
