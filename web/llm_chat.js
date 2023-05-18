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
    this.separator_style = config.separator_style;
    this.add_bos = config.add_bos;
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
    if (this.separator_style == "Two") {
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
    } else if (this.separator_style == "RedPajamaChat") {
      let ret = [this.system];

      for (let i = 0; i < this.messages.length; ++i) {
        const item = this.messages[i];
        const role = item[0];
        const message = item[1];
        if (message !== undefined && message != "") {
          ret.push(role + ": " + message + this.seps[i % this.seps.length] + "\n");
        } else {
          ret.push(role + ":");
        }
      }
      return ret;
    }
    throw Error("Unknown separator style " + this.separator_style);
  }

  /**
   * Get prompt arrays that has not been fed as input
   *
   * @returns The prompt array.
   */
  getPromptArrayUnproccessed() {
    if (this.seps.length == 0) {
      throw Error("Need seps to work")
    }
    if (this.messages.length < 3) {
      throw Error("needs to call getPromptArray for the first message");
    }
    if (this.separator_style == "Two") {
      let ret = [this.seps[this.seps.length - 1]];
      for (let i = this.messages.length - 2; i < this.messages.length; ++i) {
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
    } else if (this.separator_style == "RedPajamaChat") {
      let ret = [];
      for (let i = this.messages.length - 2; i < this.messages.length; ++i) {
        const item = this.messages[i];
        const role = item[0];
        const message = item[1];
        if (message !== undefined && message != "") {
          ret.push(message + this.seps[i % this.seps.length]+"\n");
        } else {
          ret.push(role + ":");
        }
      }
      return ret;
    }
    throw Error("Unknown separator style " + this.separator_style);
  }


  reset() {
    this.messages = [];
  }

  getStopStr() {
    if (this.separator_style == "Two") {
      return this.seps[this.seps.length - 1];
    } else if (this.separator_style == "RedPajamaChat") {
      return "<human>:";
    }
    throw Error("Unknown separator style " + this.separator_style);
  }

  appendMessage(role, message) {
    this.messages.push([role, message]);
  }
}

function getConversation(conv_template) {
  if (conv_template == "vicuna-v1.1") {
    return new Conversation({
      system: "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
      roles: ["USER", "ASSISTANT"],
      messages: [],
      offset: 0,
      seps: [" ", "</s>"],
      separator_style: "Two",
      add_bos: true,
    });
  } else if (conv_template == "wizardlm") {
    return new Conversation({
      system: "You are an AI assistant that gives helpful, detailed, and polite answers to the user's questions.",
      roles: ["", "### Response"],
      messages: [],
      offset: 0,
      seps: ["\n\n", "</s>"],
      separator_style: "Two",
      add_bos: true,
    })
  } else if (conv_template == "redpajama_chat") {
    return new Conversation({
      system: "",
      roles: ["<human>", "<bot>"],
      messages: [],
      offset: 0,
      seps: ["",""],
      separator_style: "RedPajamaChat",
      add_bos: false,
    })
  } else {
    throw Error("Unknown conv template "+ conv_template);
  }
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

    this.temperature = config.temperature;
    this.top_p = config.top_p;

    this.meanGenLength = config.mean_gen_len;
    this.streamInterval = 1;
    this.shiftFillFactor = config.shift_fill_factor;

    this.decodingTotalTime = 0;
    this.decodingTotalTokens = 0;
    this.encodingTotalTime = 0;
    this.encodingTotalTokens = 0;
    this.conversation = getConversation(config.conv_template);
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
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("param", cacheMetadata.ParamSize)
    );
    const fgetMetadata = this.vm.getFunction("get_metadata");
    var ret_value = fgetMetadata();
    const metadataStr = this.tvm.detachFromCurrentScope(ret_value).toString();
    const metadata = JSON.parse(metadataStr);
    this.maxWindowLength = metadata.max_window_size;
    this.stopTokens = metadata.stop_tokens;

    const fcreateCache = this.vm.getFunction("create_kv_cache");
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.attention_kv_cache_array_clear")
    );

    // use extern config for now
    this.kvCache = this.tvm.detachFromCurrentScope(fcreateCache());
    // fill with pad token
    this.logitsOnCPU = undefined;

    this.kvCacheLength = 0;
    this.clearCache = true
  }


  dispose() {
    // note: tvm instance is not owned by this class
    this.params.dispose();
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
    this.kvCacheLength = 0;
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
      if (logits.shape[0] != this.logitsOnCPU.shape[0]) {
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
    let tokens = [];
    let prompts = ""
    if (this.conversation.messages.length <= 2) {
      if (this.conversation.add_bos) {
        tokens.push(this.bosTokenId);
      }
      prompts = this.conversation.getPromptArray();
    } else {
      tokens.pop();
      prompts = this.conversation.getPromptArrayUnproccessed();
    }
    tokens.push(...await this.tokenizer.encode(prompts[0]));
    let ctxLength = tokens.length;
    let context = [];
    let need_shift_window = false;
    for (let i = prompts.length - 1; i > 0; --i) {
      const encoded = this.tokenizer.encode(prompts[i]);
      ctxLength += encoded.length;
      if (this.kvCacheLength + ctxLength + this.meanGenLength >= this.maxWindowLength) {
        need_shift_window = true;
        break;
      }
      context.unshift(encoded);
    }
    if (!need_shift_window) {
      for (const ctx of context) {
        tokens.push(...ctx);
      }
      return tokens;
    }
    // need shift window and re-encode
    this.logger("need shift window")
    this.kvCacheLength = 0;
    this.clearCache = true;
    // abandon all tokens we collected
    if (this.conversation.add_bos) {
      tokens = [this.bosTokenId];
    } else {
      tokens = [];
    }
    let all_prompts = this.conversation.getPromptArray();
    tokens.push(...await this.tokenizer.encode(all_prompts[0]));
    context = [];
    ctxLength = tokens.length;
    //only keep 10% of the window context
    for (let i = all_prompts.length - 1; i > 0; --i) {
      const encoded = this.tokenizer.encode(all_prompts[i]);
      ctxLength += encoded.length;
      if (ctxLength >= this.shiftFillFactor * this.maxWindowLength && i + 2 < all_prompts.length) {
        break;
      }
      context.unshift(encoded);
    }
    for (const ctx of context) {
      tokens.push(...ctx);
    }
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
    if (this.clearCache) {
      this.#clearKVCache();
      this.clearCache = false;
    }
    const maxGenLen = this.maxWindowLength - tokens.length;
    if (maxGenLen < this.meanGenLength) {
      throw Error("Too small window size config");
    }
    let step = 0;
    for (; step < maxGenLen && this.kvCacheLength + inputTokenLength + step < this.maxWindowLength; ++step) {
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
        this.#forward(inputData, this.kvCacheLength + inputTokenLength + step)
      );
      this.tvm.endScope();

      const nextToken = await this.sampleTokenFromLogits(logits, this.temperature, this.top_p);
      logits.dispose();

      tokens.push(nextToken);
      const outputTokens = tokens.slice(inputTokenLength);
      outputPrompt = this.tokenizer.decode(outputTokens);

      if (this.stopTokens.includes(nextToken)) {
        break;
      }

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
    this.kvCacheLength += tokens.length - 1;
    this.conversation.messages[this.conversation.messages.length - 1][1] = outputPrompt;
    return outputPrompt;
  }

  async evaluate() {
    // run a canonical evaluation of the flow
    this.#clearKVCache();
    const testPrompt = "The capital of Canada is";
    const ids = await this.tokenizer.encode(testPrompt);
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
    this.model = "vicuna-v1-7b-q4f32_0";

  }

  reboot() {
    this.config = undefined;
    this.pipeline = undefined;
    if (this.tvm !== undefined) {
      this.tvm.dispose();
      this.tvm = undefined;
    }
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
    } catch (err) {
      this.appendMessage("error", "Find an error initializing the WebGPU device " + err.toString());
      console.log(err.stack);
      this.reset();
      throw Error("Find an error initializing WebGPU: " + err.toString());
    }
    this.appendMessage("init", "");
    this.tvm = tvm;
    const initProgressCallback = (report) => {
      this.updateLastMessage("init", report.text);
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
    this.uiChat = document.getElementById("chatui-chat");
    this.uiChatInput = document.getElementById("chatui-input");
    this.uiChatInfoLabel = document.getElementById("chatui-info-label");
    var global_config = await (await fetch("global_config.json")).json();
    
    var model_config_url = undefined;
    if (global_config.url_dict[this.model] === undefined) {
      model_config_url = this.model;
    } else {
      var model_config_url = global_config.url_dict[this.model];
    }
    this.config = await (
      await fetch(model_config_url)
    ).json();
    this.logger(this.config)
    this.config.wasmUrl = global_config.model_lib_map[this.config.model_lib]
    var last_slash = model_config_url.lastIndexOf("/");
    var base_url = model_config_url.substring(0, last_slash + 1);
    if (this.config.model_url !== undefined) {
      this.config.cacheUrl = base_url + this.config.model_url;
    } else {
      this.config.cacheUrl = base_url;
    }
  }

  async findTokenizerPath(base_url) {
    const tokenizer_model_path = new URL("tokenizer.model", base_url);
    var tokenizer_model = await fetch(tokenizer_model_path);
    if (tokenizer_model.ok) {
      return await tvmjsGlobalEnv.tokenizerFromSentencePiece(await tokenizer_model.arrayBuffer())
    }
    const tokenizer_json_path = new URL("tokenizer.json", base_url);
    var tokenizer_json = await fetch(tokenizer_json_path);
    if (tokenizer_json.ok) {
      return await tvmjsGlobalEnv.tokenizerFromJSON(await tokenizer_json.arrayBuffer())
    }
    throw Error("Cannot find tokenizer model or json");
  }

  /**
   * Initialize the pipeline
   *
   */
  async #asyncInitPipeline() {
    if (this.pipeline !== undefined) return;
    // initialize UX and tokenizer
    var tokenizer = await this.findTokenizerPath(this.config.cacheUrl);
    this.pipeline = this.tvm.withNewScope(() => {
      return new LLMChatPipeline(this.tvm, tokenizer, this.tvm.cacheMetadata, this.config);
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
    const list = text.split('\n').map((t) => {
      const item = document.createElement('div');
      item.textContent = t;
      return item;
    });
    msgText[0].innerHTML = '';
    list.forEach((item) => msgText[0].append(item));
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  async respondTestMessage(repeat) {
    this.appendMessage("left", "");
    const testMessage = "I am a friendly bot. Please ask questions.";
    const encodedResult = await this.pipeline.tokenizer.encode(testMessage);

    const currentIds = [];
    for (let k = 0; k < repeat; ++k) {
      for (let i = 0; i < encodedResult.length; ++i) {
        currentIds.push(encodedResult[i]);
        const msg = this.pipeline.tokenizer.decode(currentIds);
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
    } catch (err) {
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

function handle_model_change() {
  var e = document.getElementById("model");
  function onChange() {
    localLLMChatIntance.reboot();
    localLLMChatIntance.model = e.value;
    localLLMChatIntance.logger("model changed to " +e.value)
  }
  e.onchange = onChange;
}

handle_model_change()

