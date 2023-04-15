/**
 * Helper to keep track of history conversations.
 */
class Conversation {
  constructor(config) {
    this.system = config.system;
    this.roles = config.roles;
    this.offset = config.offset;
    this.seps = config.seps;
    this.convId = null;
    this.messages = [];
    this.contextWindowStart = 0;
  }

  /**
   * Get prompt arrays with the first one as system.
   *
   * @returns The prompt array.
   */
  getPromptArray() {
    if (this.seps.length == 0) {
      throw Error("Need seps to work")
    }
    let ret = [this.system + this.seps[0]];

    for (let i = 0; i < this.messages.length; ++i) {
      const item = this.messages[i];
      const role = item[0];
      const message = item[1];
      if (message !== undefined && message != "") {
        ret.push(role + ": " + message + this.seps[i % this.seps.length]);
      } else {
        ret.push(role + ":");
      }
    }
    return ret;
  }

  reset() {
    this.messages = [];
  }

  getStopStr() {
    return this.seps[this.seps.length - 1];
  }

  appendMessage(role, message) {
    this.messages.push([role, message]);
  }
}

function defaultConversation(maxWindowLength = 512) {
  return new Conversation({
    system: "A chat between a curious human and an artificial intelligence assistant. " +
            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles: ["Human", "Assistant"],
    maxWindowLength: maxWindowLength,
    messages: [
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        [ "Assistant",
          "Renewable energy sources are those that can be replenished naturally in a relatively " +
          "short amount of time, such as solar, wind, hydro, geothermal, and biomass. " +
          "Non-renewable energy sources, on the other hand, are finite and will eventually be " +
          "depleted, such as coal, oil, and natural gas. Here are some key differences between " +
          "renewable and non-renewable energy sources:\n" +
          "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable " +
          "energy sources are finite and will eventually run out.\n" +
          "2. Environmental impact: Renewable energy sources have a much lower environmental impact " +
          "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, " +
          "and other negative effects.\n" +
          "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically " +
          "have lower operational costs than non-renewable sources.\n" +
          "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote " +
          "locations than non-renewable sources.\n" +
          "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different " +
          "situations and needs, while non-renewable sources are more rigid and inflexible.\n" +
          "6. Sustainability: Renewable energy sources are more sustainable over the long term, while " +
          "non-renewable sources are not, and their depletion can lead to economic and social instability.\n"
        ]
    ],
    offset: 2,
    seps:["###"],
  });
};

class LLMChatPipeline {
  constructor(tvm, tokenizer, cacheMetadata, config) {
    if (cacheMetadata == undefined) {
      throw Error("Expect cacheMetadata");
    }
    this.tvm = tvm;
    this.logger = console.log;
    this.tokenizer = tokenizer;
    this.bosTokenId = 1;
    this.eosTokenId = 2;

    this.maxWindowLength = config.maxWindowLength;
    this.maxGenLength = config.maxGenLength;
    this.meanGenLength = config.meanGenLength;
    this.streamInterval = 1;

    this.decodingTotalTime = 0;
    this.decodingTotalTokens = 0;
    this.encodingTotalTime = 0;
    this.encodingTotalTokens = 0;

    this.conversation = defaultConversation();

    this.device = this.tvm.webgpu();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );
    this.encoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("encoding")
    );
    this.decoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("decoding")
    );
    this.encodingWithoutCache = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("encoding_without_cache")
    );
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("param", cacheMetadata.ParamSize)
    );
    const fcreateCache = this.tvm.getGlobalFunc("vm.builtin.attention_kv_cache_create");
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.attention_kv_cache_array_clear")
    );

    // use extern config for now
    // move to kv generation vm function
    const kvList = [];
    const kvConfig = config.kvConfig;
    for (let i = 0; i < kvConfig.numLayers; ++i) {
      const item = fcreateCache(
        this.tvm.empty(kvConfig.shape, kvConfig.dtype, this.device),
        this.tvm.makeShapeTuple(kvConfig.shape),
        this.tvm.scalar(0, "int")
      );
      kvList.push(item);
    }
    this.kvCache = this.tvm.detachFromCurrentScope(this.tvm.makeTVMArray(kvList));
    // fill with pad token
    this.logitsOnCPU = undefined;
  }


  dispose() {
    // note: tvm instance is not owned by this class
    this.params.dispose();
    this.encodingWithoutCache.dispose();
    this.decoding.dispose();
    this.encoding.dispose();
    this.vm.dispose();
    this.kvCache.dispose();
    this.fclearKVCaches.dispose();
    if (this.logitsOnCPU != undefined) {
      this.logitsOnCPU.dispose();
    }
  }

  #clearKVCache() {
    this.fclearKVCaches(this.kvCache);
  }

  #forward(inputs, curPos) {
    this.tvm.beginScope();
    var retValue;
    const seqLenShape = this.tvm.makeShapeTuple([curPos]);
    if (inputs.shape[1] > 1) {
      retValue = this.encoding(
        inputs, seqLenShape, this.kvCache, this.params
      );
    } else {
      retValue = this.decoding(
        inputs, seqLenShape, this.kvCache, this.params
      );
    }
    const logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  // NOTE: caller must call device.sync()
  #updateLogitsOnCPU(logits) {
    if (this.logitsOnCPU == undefined) {
      this.logitsOnCPU = this.tvm.detachFromCurrentScope(
        this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu())
      );
    } else {
      if(logits.shape[0] != this.logitsOnCPU.shape[0]) {
        throw Error("We expect the size of logits to remain unchanged");
      }
    }
    this.logitsOnCPU.copyFrom(logits);
  }

  async sampleTokenFromLogits(logits, temperature = 0.8, top_p = 0.95) {
    this.tvm.beginScope();
    this.#updateLogitsOnCPU(logits);
    this.tvm.endScope();
    await this.device.sync();
    return this.tvm.sampleTopPFromLogits(this.logitsOnCPU, temperature, top_p);
  }

  async getInputTokens() {
    const tokens = [this.bosTokenId];
    const prompts = this.conversation.getPromptArray();
    tokens.push(...await this.tokenizer.encodeIds(prompts[0]));

    let ctxLength = tokens.length;
    const context = [];
    for (let i = prompts.length - 1; i > 0; --i) {
      const encoded = this.tokenizer.encodeIds(prompts[i]);
      ctxLength += encoded.length;
      if (ctxLength + this.meanGenLength >= this.maxWindowLength && i + 2 < prompts.length) {
        this.logger("Shift window at " + i);
        break;
      }
      context.unshift(encoded);
    }
    const followMessage = [];
    for (const ctx of context) {
      followMessage.push(...ctx);
    }

    if (followMessage.length + tokens.length + this.meanGenLength >= this.maxWindowLength) {
      const maxMsgLen = this.maxWindowLength - tokens.length - this.meanGenLength;
      if (maxMsgLen < this.meanGenLength) {
        throw Error("Too small window config tokens.length=" + tokens.length);
      }
      this.logger("Slice message " + followMessage.length + " to " + maxMsgLen);
      followMessage = followMessage.slice(followMessage.length - maxMsgLen);
    }
    tokens.push(...followMessage);
    if (tokens.length + this.meanGenLength >= this.maxWindowLength) {
      throw Error("Exceed max window length curr=" + tokens.length);
    }
    return tokens;
  }

  resetChat() {
    this.conversation.reset();
    this.#clearKVCache();
    this.decodingTotalTime = 0;
    this.encodingTotalTime = 0;
    this.decodingTotalTokens = 0;
    this.encodingTotalTokens = 0;
  }

  async generate(inputPrompt, callbackUpdateResponse) {
    this.conversation.appendMessage(this.conversation.roles[0], inputPrompt);
    this.conversation.appendMessage(this.conversation.roles[1], "");
    const stopStr = this.conversation.getStopStr();
    const tokens = await this.getInputTokens();
    const inputTokenLength = tokens.length;

    var outputPrompt = "";
    this.#clearKVCache();
    const maxGenLen = Math.min(this.maxGenLength, this.maxWindowLength - tokens.length);
    if (maxGenLen < this.meanGenLength) {
      throw Error("Too small window size config");
    }

    for (let step = 0; step < maxGenLen; ++step) {
      this.tvm.beginScope();
      var inputData;

      let tstart = performance.now();
      if (step == 0) {
        inputData = this.tvm.empty([1, tokens.length], "int32", this.device);
        inputData.copyFrom(tokens);
      } else {
        inputData = this.tvm.empty([1, 1], "int32", this.device);
        inputData.copyFrom(tokens.slice(tokens.length - 1));
      }
      const logits = this.tvm.detachFromCurrentScope(
        this.#forward(inputData, inputTokenLength + step)
      );
      this.tvm.endScope();

      const nextToken = await this.sampleTokenFromLogits(logits);
      logits.dispose();

      tokens.push(nextToken);
      const outputTokens = tokens.slice(inputTokenLength);
      outputPrompt = this.tokenizer.decodeIds(outputTokens);

      if (nextToken == this.eosTokenId) break;

      const stopPos = outputPrompt.lastIndexOf(stopStr);
      if (stopPos != -1) {
        outputPrompt = outputPrompt.substring(0, stopPos);
        break;
      }
      let tend = performance.now();
      if (step != 0) {
        this.decodingTotalTokens += 1;
        this.decodingTotalTime += (tend - tstart) / 1000;
      } else {
        this.encodingTotalTime += (tend - tstart) / 1000;
        this.encodingTotalTokens += inputTokenLength;
      }

      if (step % this.streamInterval == 0) {
        callbackUpdateResponse(step, outputPrompt);
      }
    }
    return outputPrompt;
  }

  async evaluate() {
    // run a canonicla evaluateion fo the flow
    this.#clearKVCache();
    const testPrompt = "The capital of Canada is";
    const ids = await this.tokenizer.encodeIds(testPrompt);
    const inputPromptSize = ids.length;
    const tokens = Array.from(ids);
    tokens.unshift(this.bosTokenId);
    if (tokens.length == 0) {
      throw Error("empty token");
    }

    this.tvm.beginScope();
    const inputData = this.tvm.empty([1, tokens.length], "int32", this.device);
    inputData.copyFrom(tokens);
    const encodingStart = performance.now();
    this.#forward(inputData, tokens.length);
    this.tvm.endScope();
    await this.device.sync();

    const decodingStart = performance.now();

    this.tvm.beginScope();
    const firstSampleToken = this.tvm.empty([1, 1], "int32", this.device).copyFrom([6234]);
    this.#updateLogitsOnCPU(this.#forward(firstSampleToken, tokens.length + 1));
    await this.device.sync();
    this.tvm.endScope();

    const decodingEnd = performance.now();
    const msg = (
      `encoding-time=${((decodingStart - encodingStart) / 1000).toFixed(4)} sec` +
      `decoding-time=${((decodingEnd - decodingStart) / 1000).toFixed(4)} sec`
    );

    // simply log tokens for eyeballing.
    console.log("Logits:");
    console.log(this.logitsOnCPU.toArray());
    console.log(msg);
  }

  /**
   * async preload webgpu pipelines when possible.
   */
  async asyncLoadWebGPUPiplines() {
    await this.tvm.asyncLoadWebGPUPiplines(this.vm.getInternalModule());
  }

  runtimeStatsText() {
    return (
      `encoding: ${(this.encodingTotalTokens / this.encodingTotalTime).toFixed(4)} tokens/sec, ` +
      `decoding: ${(this.decodingTotalTokens / this.decodingTotalTime).toFixed(4)} tokens/sec`
    )
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
    this.logger = console.log;
    this.debugTest = false;
  }
 /**
   * Initialize TVM
   * @param wasmUrl URL to wasm source.
   * @param cacheUrl URL to NDArray cache.
   * @param logger Custom logger.
   */
  async #asyncInitTVM(wasmUrl, cacheUrl) {
    if (this.tvm !== undefined) {
      return;
    }
    this.logger = console.log;

    const wasmSource = await (
      await fetch(wasmUrl)
    ).arrayBuffer();
    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      new EmccWASI(),
      this.logger
    );
    // intialize WebGPU
    try {
      const output = await tvmjs.detectGPUDevice();
      if (output !== undefined) {
        var label = "WebGPU";
        if (output.adapterInfo.description.length != 0) {
          label += " - " + output.adapterInfo.description;
        } else {
          label += " - " + output.adapterInfo.vendor;
        }
        this.appendMessage("init", "Initialize GPU device: " + label);
        tvm.initWebGPU(output.device);
      } else {
        this.appendMessage("error", "This browser env do not support WebGPU");
        this.reset();
        throw Error("This browser env do not support WebGPU");
      }
    } catch(err) {
      this.appendMessage("error", "Find an error initializing the WebGPU device " + err.toString());
      console.log(err.stack);
      this.reset();
      throw Error("Find an error initializing WebGPU: " + err.toString());
    }
    this.appendMessage("init", "");
    this.tvm = tvm;
    self = this;
    function initProgressCallback(report) {
      self.updateLastMessage("init", report.text);
    }
    tvm.registerInitProgressCallback(initProgressCallback);
    if (!cacheUrl.startsWith("http")) {
      cacheUrl = new URL(cacheUrl, document.URL).href;
    }
    await tvm.fetchNDArrayCache(cacheUrl, tvm.webgpu());
  }
  /**
   * Async initialize instance.
   */
  async asyncInit() {
    if (this.pipeline !== undefined) return;
    await this.#asyncInitConfig();
    await this.#asyncInitTVM(this.config.wasmUrl, this.config.cacheUrl);
    await this.#asyncInitPipeline();
  }

  /**
   * Async initialize config
   */
  async #asyncInitConfig() {
    if (this.config !== undefined) return;
    this.config = await (await fetch("llm-chat-config.json")).json();
    this.uiChat = document.getElementById("chatui-chat");
    this.uiChatInput = document.getElementById("chatui-input");
    this.uiChatInfoLabel = document.getElementById("chatui-info-label");
  }

  /**
   * Initialize the pipeline
   *
   * @param tokenizerModel The url to tokenizer model.
   */
  async #asyncInitPipeline() {
    if (this.pipeline !== undefined) return;
    // initialize UX and tokenizer
    const tokenizer = await tvmjsGlobalEnv.sentencePieceProcessor(this.config.tokenizer);
    this.pipeline = this.tvm.withNewScope(() => {
      return new LLMChatPipeline(this.tvm, tokenizer,  this.tvm.cacheMetadata, this.config);
    });
    await this.pipeline.asyncLoadWebGPUPiplines();
    this.updateLastMessage("init", "All initialization finished.");
  }

  appendMessage(kind, text) {
    if (kind == "init") {
      text = "[System Initalize] " + text;
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

  updateLastMessage(kind, text) {
    if (kind == "init") {
      text = "[System Initalize] " + text;
    }
    const matches = this.uiChat.getElementsByClassName(`msg ${kind}-msg`);
    if (matches.length == 0) throw Error(`${kind} message do not exist`);
    const msg = matches[matches.length - 1];
    const msgText = msg.getElementsByClassName("msg-text");
    if (msgText.length != 1) throw Error("Expect msg-text");
    if (msgText[0].innerHTML == text) return;
    text = text.replaceAll("\n", "<br>");
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

  resetChat() {
    if (this.requestInProgress) return;
    const clearTags = ["left", "right"];
    for (const tag of clearTags) {
      const matches = this.uiChat.getElementsByClassName(`msg ${tag}-msg`);
      for (const item of matches) {
        item.remove();
      }
    }
    this.uiChatInfoLabel.innerHTML = "";
    this.pipeline.resetChat();
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

    if (this.debugTest) {
      await this.pipeline.evaluate();
      this.requestInProgress = false;
      return;
    }

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
      if (msg.endsWith("##")) {
        msg = msg.substring(0, msg.length - 2);
      } else if (msg.endsWith("#")) {
        msg = msg.substring(0, msg.length - 1);
      }
      this.updateLastMessage("left", msg);
    };
    try {
      const output = await this.pipeline.generate(prompt, callbackUpdateResponse);
      this.updateLastMessage("left", output);
      this.uiChatInfoLabel.innerHTML = this.pipeline.runtimeStatsText();
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

tvmjsGlobalEnv.asyncOnReset = async function () {
  await localLLMChatIntance.resetChat();
};
