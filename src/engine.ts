import * as tvmjs from "tvmjs";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import * as API from "./openai_api_protocols/apis";
import {
  ChatConfig,
  ChatOptions,
  AppConfig,
  prebuiltAppConfig,
  GenerationConfig,
  postInitAndCheckGenerationConfigValues,
  Role,
  EngineConfig,
} from "./config";
import { LLMChatPipeline } from "./llm_chat";
import {
  ChatCompletionRequest,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionFinishReason,
  ChatCompletionMessageParam,
  ChatCompletionRequestNonStreaming,
  ChatCompletionRequestStreaming,
  ChatCompletionRequestBase,
  CompletionUsage,
  ChatCompletionUserMessageParam,
} from "./openai_api_protocols/index";
import * as ChatCompletionAPI from "./openai_api_protocols/index";
import {
  InitProgressCallback,
  EngineInterface,
  GenerateProgressCallback,
  LogitProcessor
} from "./types";
import { Conversation, compareConversationObject, getConversation } from "./conversation";


/**
 * Creates `Engine`, and loads `modelId` onto WebGPU.
 * 
 * Equivalent to `new webllm.Engine().reload(...)`.
 * 
 * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
 * `engineConfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.EngineConfig`.
 * @returns An intialized `WebLLM.Engine` with `modelId` loaded.
 */
export async function CreateEngine(
  modelId: string,
  engineConfig?: EngineConfig,
): Promise<Engine> {
  const engine = new Engine();
  engine.setInitProgressCallback(engineConfig?.initProgressCallback);
  engine.setLogitProcessorRegistry(engineConfig?.logitProcessorRegistry);
  await engine.reload(modelId, engineConfig?.chatOpts, engineConfig?.appConfig);
  return engine;
}

/**
 * The main interface of Engine, which loads a model and performs tasks.
 * 
 * You can either initialize one with `webllm.CreateEngine(modelId)`, or `webllm.Engine().reload(modelId)`.
 */
export class Engine implements EngineInterface {
  public chat: API.Chat;

  private currentModelId?: string = undefined;  // Model current loaded, undefined if nothing is loaded
  private logger: (msg: string) => void = console.log;
  private logitProcessorRegistry?: Map<string, LogitProcessor>;
  private logitProcessor?: LogitProcessor;
  private pipeline?: LLMChatPipeline;
  private initProgressCallback?: InitProgressCallback;
  private interruptSignal = false;
  private deviceLostIsError = false;  // whether device.lost is due to actual error or model reload
  private config?: ChatConfig;

  constructor() {
    this.chat = new API.Chat(this);
  }

  setInitProgressCallback(initProgressCallback?: InitProgressCallback) {
    this.initProgressCallback = initProgressCallback;
  }

  getInitProgressCallback() {
    return this.initProgressCallback;
  }

  setLogitProcessorRegistry(logitProcessorRegistry?: Map<string, LogitProcessor>) {
    this.logitProcessorRegistry = logitProcessorRegistry;
  }

  async reload(modelId: string, chatOpts?: ChatOptions, appConfig?: AppConfig): Promise<void> {
    this.deviceLostIsError = false;  // so that unload() does not trigger device.lost warning
    this.unload();

    this.logitProcessor = this.logitProcessorRegistry?.get(modelId);
    const tstart = performance.now();
    if (appConfig === undefined) {
      appConfig = prebuiltAppConfig;
    }

    const findModelRecord = () => {
      const matchedItem = appConfig?.model_list.find(
        item => item.model_id == modelId
      );
      if (matchedItem !== undefined) return matchedItem;
      throw Error("Cannot find model_url for " + modelId);
    }

    const modelRecord = findModelRecord();
    const baseUrl = typeof document !== "undefined" ? document.URL : globalThis.location.origin;
    let modelUrl = modelRecord.model_url;
    if (!modelUrl.startsWith("http")) {
      modelUrl = new URL(modelUrl, baseUrl).href;
    }

    let configCache: tvmjs.ArtifactCacheTemplate;
    if (appConfig.useIndexedDBCache) {
      configCache = new tvmjs.ArtifactIndexedDBCache("webllm/config");
    } else {
      configCache = new tvmjs.ArtifactCache("webllm/config");
    }

    // load config
    const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
    this.config = {
      ...(await configCache.fetchWithCache(configUrl, "json")),
      ...chatOpts
    } as ChatConfig;

    // load tvm wasm
    let wasmCache: tvmjs.ArtifactCacheTemplate;
    if (appConfig.useIndexedDBCache) {
      wasmCache = new tvmjs.ArtifactIndexedDBCache("webllm/wasm");
    } else {
      wasmCache = new tvmjs.ArtifactCache("webllm/wasm");
    }

    const wasmUrl = modelRecord.model_lib_url;
    if (wasmUrl === undefined) {
      throw Error("You need to specify `model_lib_url` for each model in `model_list` " +
        "so that we can download the model library (i.e. wasm file).")
    }
    const fetchWasmSource = async () => {
      if (wasmUrl.includes("localhost")) {
        // do not cache wasm on local host as we might update code frequently
        return await fetch(wasmUrl);
      } else if (!wasmUrl.startsWith("http")) {
        // do not cache wasm on the same server as it can also refresh
        // rely on the normal caching strategy
        return await fetch(new URL(wasmUrl, baseUrl).href);
      } else {
        // use cache
        return await wasmCache.fetchWithCache(wasmUrl, "arraybuffer");
      }
    };
    const wasmSource = await fetchWasmSource();

    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      tvmjs.createPolyfillWASI(),
      this.logger
    );

    if (this.initProgressCallback !== undefined) {
      tvm.registerInitProgressCallback(this.initProgressCallback);
    }

    // detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw Error("Cannot find WebGPU in the environment");
    }
    let gpuLabel = "WebGPU";
    if (gpuDetectOutput.adapterInfo.description.length != 0) {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.description;
    } else {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.vendor;
    }
    if (modelRecord.required_features !== undefined) {
      for (const feature of modelRecord.required_features) {
        if (!gpuDetectOutput.device.features.has(feature)) {
          if (feature == "shader-f16") {
            throw Error(
              "This model requires WebGPU extension shader-f16, " +
              "which is not enabled in this browser. " +
              "You can try to launch Chrome Canary in command line with flag \"--enable-dawn-features=allow_unsafe_apis\"."
            );
          }
          throw Error(
            "This model requires feature " + feature +
            ", which is not yet supported by this browser. "
          );
        }
      }
    }

    tvm.initWebGPU(gpuDetectOutput.device);
    gpuDetectOutput.device.lost.then((info: any) => {
      // `fetchNDArrayCache` may exceed available memory; use `lost.then` to prevent crashing
      if (this.deviceLostIsError) {
        console.error("Device was lost, please try to initialize again. ", info);
        this.unload();
      }
    });
    this.deviceLostIsError = true;
    const tokenizer = await this.asyncLoadTokenizer(modelUrl, this.config, appConfig);
    const cacheType = appConfig.useIndexedDBCache ? "indexeddb" : "cache";
    await tvm.fetchNDArrayCache(modelUrl, tvm.webgpu(), "webllm/model", cacheType);
    this.pipeline = new LLMChatPipeline(tvm, tokenizer, this.config, this.logitProcessor);
    await this.pipeline?.asyncLoadWebGPUPipelines();
    const tend = performance.now();

    if (this.initProgressCallback !== undefined) {
      const text = "Finish loading on " + gpuLabel;
      this.initProgressCallback({
        progress: 1,
        timeElapsed: (tend - tstart) / 1e3,
        text: text
      })
    }
    this.currentModelId = modelId;
  }

  async generate(
    input: string | ChatCompletionRequestNonStreaming,
    progressCallback?: GenerateProgressCallback,
    streamInterval = 1,
    genConfig?: GenerationConfig,
  ): Promise<string> {
    console.log(
      "WARNING: `generate()` will soon be deprecated. " +
      "Please use `engine.chat.completions.create()` instead. " +
      "For multi-round chatting, see `examples/multi-round-chat` on how to use " +
      "`engine.chat.completions.create()` to achieve the same effect."
    );
    return this._generate(input, progressCallback, streamInterval, genConfig);
  }

  private async _generate(
    input: string | ChatCompletionRequestNonStreaming,
    progressCallback?: GenerateProgressCallback,
    streamInterval = 1,
    genConfig?: GenerationConfig,
  ): Promise<string> {
    this.interruptSignal = false;
    if (genConfig !== undefined) {
      postInitAndCheckGenerationConfigValues(genConfig);
    }
    await this.prefill(input, genConfig);

    let counter = 1;
    while (!this.stopped()) {
      if (this.interruptSignal) {
        this.getPipeline().triggerStop();
        break;
      }
      counter += 1;
      await this.decode(genConfig);
      if (counter % streamInterval == 0 && progressCallback !== undefined) {
        progressCallback(counter, await this.getMessage());
      }
    }
    return await this.getMessage();
  }

  /**
   * Similar to `generate()`; but instead of using callback, we use an async iterable.
   * @param request Request for chat completion.
   * @param genConfig Generation config extraced from `request`.
   */
  async* chatCompletionAsyncChunkGenerator(
    request: ChatCompletionRequestStreaming,
    genConfig: GenerationConfig
  ): AsyncGenerator<ChatCompletionChunk, void, void> {
    postInitAndCheckGenerationConfigValues(genConfig);
    if (request.seed !== null && request.seed !== undefined) {
      this.getPipeline().setSeed(request.seed);
    }
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const model = this.currentModelId!;
    const created = Date.now();
    const id = crypto.randomUUID();
    this.interruptSignal = false;
    let prevMessageLength = 0;  // to know where to start slicing the delta; does not count �

    function _countTrailingReplacementChar(curMessage: string): number {
      let cntr = 0;
      for (let i = curMessage.length - 1; i >= 0; i--) {
        if (curMessage.charAt(i) === "�") {
          cntr += 1;
        } else {
          return cntr;
        }
      }
      return cntr;
    }

    async function _getChunk(thisModule: Engine): Promise<ChatCompletionChunk | undefined> {
      // Remove the replacement character (U+FFFD) from the response to handle emojis.
      // Each emoji is made up of multiples of 4 tokens; when truncated, it is displayed as �, so
      // we skip this delta until a full emoji is rendered
      // TODO(Charlie): This does not consider cases of � not being emoji, need to fix with Streamer
      const curMessage = await thisModule.getMessage();
      const numTrailingReplacementChar = _countTrailingReplacementChar(curMessage);
      if (numTrailingReplacementChar % 4 !== 0) {
        return undefined;
      }

      const deltaMessage = curMessage.slice(prevMessageLength);
      prevMessageLength = curMessage.length;
      const chunk: ChatCompletionChunk = {
        id: id,
        choices: [{
          delta: { content: deltaMessage, role: "assistant" },
          finish_reason: null,  // not finished yet
          index: 0,
          logprobs: request.logprobs ? {
            content: thisModule.getPipeline().getTokenLogprobArray().slice(-1)  // always the last entry
          } as ChatCompletionChunk.Choice.Logprobs : null,
        }],
        model: model,
        object: "chat.completion.chunk",
        created: created
      }
      return chunk;
    }

    await this.prefill(request, genConfig);
    let curChunk = await _getChunk(this);  // prefill produces a chunk
    if (curChunk) {
      yield curChunk;
    }

    while (!this.stopped()) {
      if (this.interruptSignal) {
        this.getPipeline().triggerStop();
        break;
      }
      await this.decode(genConfig);
      curChunk = await _getChunk(this);
      if (curChunk) {
        yield curChunk;
      }
    }

    // Reset seed -- we do not want this seed to affect future requests
    if (request.seed !== null && request.seed !== undefined) {
      this.getPipeline().setSeed(Date.now());
    }

    const lastChunk: ChatCompletionChunk = {
      id: id,
      choices: [{
        delta: {},
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        finish_reason: this.getFinishReason()!,
        index: 0,
      }],
      model: model,
      object: "chat.completion.chunk",
      created: created
    }
    yield lastChunk;
  }

  /**
   * Completes a single ChatCompletionRequest.
   * 
   * @param request A OpenAI-style ChatCompletion request.
   * 
   * @note For each choice (i.e. `n`), a request is defined by a single `prefill()` and mulitple
   * `decode()`. This is important as it determines the behavior of various fields including `seed`.
   */
  async chatCompletion(
    request: ChatCompletionRequestNonStreaming
  ): Promise<ChatCompletion>;
  async chatCompletion(
    request: ChatCompletionRequestStreaming
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  async chatCompletion(
    request: ChatCompletionRequestBase
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;
  async chatCompletion(
    request: ChatCompletionRequest
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    // 0. Preprocess inputs
    if (!this.currentModelId) {
      throw new Error("Please call `Engine.reload(model)` first, or initialize with CreateEngine().");
    }
    ChatCompletionAPI.postInitAndCheckFields(request);
    const genConfig: GenerationConfig = {
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      max_gen_len: request.max_gen_len,
      stop: request.stop,
      top_p: request.top_p,
      temperature: request.temperature,
      logit_bias: request.logit_bias,
      logprobs: request.logprobs,
      top_logprobs: request.top_logprobs,
      response_format: request.response_format,
    }

    // 1. If request is streaming, return an AsyncIterable (an iterable version of `generate()`)
    if (request.stream) {
      return this.chatCompletionAsyncChunkGenerator(request, genConfig);
    }

    if (request.seed !== null && request.seed !== undefined) {
      this.getPipeline().setSeed(request.seed);
    }

    // 2. If request is non-streaming, directly reuse `generate()`
    const n = request.n ? request.n : 1;
    const choices: Array<ChatCompletion.Choice> = [];
    let completion_tokens = 0;
    let prompt_tokens = 0;
    for (let i = 0; i < n; i++) {
      let outputMessage: string;
      if (this.interruptSignal) {
        // A single interrupt signal should stop all choices' generations
        this.getPipeline().triggerStop();
        outputMessage = "";
      } else {
        outputMessage = await this._generate(
          request,
          /*progressCallback=*/undefined,
          /*streamInterval=*/1,
          /*genConfig=*/genConfig
        );
      }
      choices.push({
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        finish_reason: this.getFinishReason()!,
        index: i,
        logprobs: request.logprobs ? {
          content: this.getPipeline().getTokenLogprobArray()
        } as ChatCompletion.Choice.Logprobs : null,
        message: {
          content: outputMessage,
          role: "assistant",
        }
      });
      completion_tokens += this.getPipeline().getCurRoundDecodingTotalTokens();
      prompt_tokens += this.getPipeline().getCurRoundPrefillTotalTokens();
    }

    const response: ChatCompletion = {
      id: crypto.randomUUID(),
      choices: choices,
      model: this.currentModelId,
      object: "chat.completion",
      created: Date.now(),
      usage: {
        completion_tokens: completion_tokens,
        prompt_tokens: prompt_tokens,
        total_tokens: completion_tokens + prompt_tokens,
      } as CompletionUsage,
    }

    // Reset seed -- we do not want this seed to affect future requests
    if (request.seed !== null && request.seed !== undefined) {
      this.getPipeline().setSeed(Date.now());
    }
    return response;
  }

  async interruptGenerate() {
    this.interruptSignal = true;
  }

  async runtimeStatsText(): Promise<string> {
    return this.getPipeline().runtimeStatsText();
  }

  async resetChat(keepStats = false) {
    this.pipeline?.resetChat(keepStats);
  }

  async unload() {
    this.pipeline?.dispose();
    this.pipeline = undefined;
    this.currentModelId = undefined;
  }

  async getMaxStorageBufferBindingSize(): Promise<number> {
    // First detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw Error("Cannot find WebGPU in the environment");
    }

    const computeMB = (value: number) => {
      return Math.ceil(value / (1 << 20)) + "MB";
    }
    const maxStorageBufferBindingSize = gpuDetectOutput.device.limits.maxStorageBufferBindingSize;
    const defaultMaxStorageBufferBindingSize = 1 << 30;  // 1GB
    if (maxStorageBufferBindingSize < defaultMaxStorageBufferBindingSize) {
      console.log(
        `WARNING: the current maxStorageBufferBindingSize ` +
        `(${computeMB(maxStorageBufferBindingSize)}) ` +
        `may only work for a limited number of models, e.g.: \n` +
        `- Llama-2-7b-chat-hf-q4f16_1-1k \n` +
        `- RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k \n` +
        `- RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k \n` +
        `- TinyLlama-1.1B-Chat-v0.4-q4f16_1-1k \n` +
        `- TinyLlama-1.1B-Chat-v0.4-q4f32_1-1k`
      );
    }
    return maxStorageBufferBindingSize;
  }

  async getGPUVendor(): Promise<string> {
    // First detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw Error("Cannot find WebGPU in the environment");
    }
    return gpuDetectOutput.adapterInfo.vendor;
  }

  //--------------------------
  // Lower level API
  //--------------------------
  async forwardTokensAndSample(inputIds: Array<number>, isPrefill: boolean): Promise<number> {
    return this.getPipeline().forwardTokensAndSample(inputIds, isPrefill);
  }

  /**
   * @returns Whether the generation stopped.
   */
  stopped(): boolean {
    return this.getPipeline().stopped();
  }

  /**
   * @returns Finish reason; undefined if generation not started/stopped yet.
  */
  getFinishReason(): ChatCompletionFinishReason | undefined {
    return this.getPipeline().getFinishReason();
  }

  /**
   * Get the current generated response.
   *
   * @returns The current output message.
   */
  async getMessage(): Promise<string> {
    return this.getPipeline().getMessage();
  }

  /**
   * Get a new Conversation object based on the chat completion request.
   * 
   * @param request The incoming ChatCompletionRequest
   * @note `request.messages[-1]` is not included as it would be treated as a normal input to
   * `prefill()`.
   */
  private getConversationFromChatCompletionRequest(
    request: ChatCompletionRequest,
    config: ChatConfig
  ): Conversation {
    // 0. Instantiate a new Conversation object
    const conversation = getConversation(config.conv_template, config.conv_config);

    // 1. Populate function-calling-related fields
    const functionCallUsage = this.getFunctionCallUsage(request);
    conversation.function_string = functionCallUsage;
    conversation.use_function_calling = functionCallUsage === "";

    // 2. Populate conversation.messages
    const input = request.messages;
    const lastId = input.length - 1;
    if (input[lastId].role !== "user" || typeof input[lastId].content !== "string") {
      // TODO(Charlie): modify condition after we support multimodal inputs
      throw Error("The last message should be a string from the `user`.")
    }
    for (let i = 0; i < input.length - 1; i++) {
      const message: ChatCompletionMessageParam = input[i];
      if (message.role === "system") {
        if (i !== 0) {
          throw new Error("System prompt should always be the first one in `messages`.");
        }
        conversation.override_system_message = message.content;
      } else if (message.role === "user") {
        if (typeof message.content !== "string") {
          // TODO(Charlie): modify condition after we support multimodal inputs
          throw new Error("Last messages should be a string from the `user`.");
        }
        conversation.appendMessage(
          Role.user,
          message.content,
          message.name
        );
      } else if (message.role === "assistant") {
        if (typeof message.content !== "string") {
          throw new Error("Assistant message should have string content.");
        }
        conversation.appendMessage(
          Role.assistant,
          message.content,
          message.name
        );
      } else {
        throw new Error("Unsupported role: " + message.role);
      }
    }
    return conversation;
  }

  /**
   * Returns the function string based on the request.tools and request.tool_choice, raises erros if
   * encounter invalid request.
   * 
   * @param request The chatCompletionRequest we are about to prefill for.
   * @returns The string used to set Conversatoin.function_string
   */
  private getFunctionCallUsage(request: ChatCompletionRequest): string {
    if (request.tools == undefined ||
      (typeof request.tool_choice == "string" && request.tool_choice == "none")) {
      return "";
    }
    if (typeof request.tool_choice == "string" && request.tool_choice !== "auto") {
      throw Error(`Invalid tool choice value: ${request.tool_choice}`);
    }
    if (typeof request.tool_choice !== "string" && request.tool_choice?.type !== "function") {
      throw Error("Only 'function' tool choice is supported");
    }

    const singleFunctionToCall = typeof request.tool_choice !== "string" && request.tool_choice?.function?.name;
    if (singleFunctionToCall) {
      for (const f of request.tools) {
        if (singleFunctionToCall == f.function.name) {
          return JSON.stringify([f.function]);
        }
      }
      throw Error(`The tool choice function ${singleFunctionToCall} is not found in the tools list`);
    }

    const function_list = [];
    for (const f of request.tools) {
      if (f.type !== "function") {
        throw Error("Only 'function' tool type is supported");
      }
      function_list.push(f.function);
    }
    return JSON.stringify(function_list);
  }

  /**
   * Run a prefill step with a given input.
   * 
   * If `input` is a chatCompletionRequest, we treat `input.messages[-1]` as the usual user input.
   * We then convert `input.messages[:-1]` to a `Conversation` object, representing a conversation
   * history.
   * 
   * If the new `Conversation` object matches the current one loaded, it means we are
   * performing multi-round chatting, so we do not reset, hence reusing KV cache. Otherwise, we
   * reset every thing, treating the request as something completely new.
   * 
   * @param input The input prompt, or `messages` in OpenAI-like APIs.
   */
  async prefill(
    input: string | ChatCompletionRequest,
    genConfig?: GenerationConfig
  ) {
    if (this.config === undefined) {
      throw Error("Expect this.config to be initialized. Did you call `reload()`?");
    }
    let input_str: string;
    let input_role_str: string | undefined;
    if (typeof input === "string") {
      input_str = input;
    } else {
      // 1. Get new conversation based on request, determine if we are in multiround chatting
      const oldConv = this.getPipeline().getConversationObject();
      const newConv = this.getConversationFromChatCompletionRequest(input, this.config);
      if (!compareConversationObject(oldConv, newConv)) {
        // Not the same conversation, so not multiround chatting, reset everything (KV cache, etc.)
        this.resetChat();
        this.getPipeline().setConversation(newConv);
      } else {
        console.log("Multiround chatting, reuse KVCache.");
      }

      // 2. Treat the last message as the usual input
      const last_msg = input.messages[input.messages.length - 1] as ChatCompletionUserMessageParam;
      input_str = last_msg.content as string;
      input_role_str = last_msg.name ? last_msg.name : undefined;
    }
    return this.getPipeline().prefillStep(input_str, input_role_str, genConfig);
  }

  /**
   * Run a decode step to decode the next token.
   */
  async decode(genConfig?: GenerationConfig) {
    return this.getPipeline().decodeStep(genConfig);
  }

  private getPipeline(): LLMChatPipeline {
    if (this.pipeline === undefined) {
      throw Error("Chat module not yet initialized, did you call chat.reload?");
    }
    return this.pipeline;
  }

  private async asyncLoadTokenizer(
    baseUrl: string,
    config: ChatConfig,
    appConfig: AppConfig,
  ): Promise<Tokenizer> {
    let modelCache: tvmjs.ArtifactCacheTemplate;
    if (appConfig.useIndexedDBCache) {
      modelCache = new tvmjs.ArtifactIndexedDBCache("webllm/model");
    } else {
      modelCache = new tvmjs.ArtifactCache("webllm/model");
    }

    if (config.tokenizer_files.includes("tokenizer.json")) {
      const url = new URL("tokenizer.json", baseUrl).href;
      const model = await modelCache.fetchWithCache(url, "arraybuffer");
      return Tokenizer.fromJSON(model);
    }
    else if (config.tokenizer_files.includes("tokenizer.model")) {
      this.logger("Using `tokenizer.model` since we cannot locate `tokenizer.json`.\n" +
        "It is recommended to use `tokenizer.json` to ensure all token mappings are included, " +
        "since currently, files like `added_tokens.json`, `tokenizer_config.json` are ignored.\n" +
        "Consider converting `tokenizer.model` to `tokenizer.json` by compiling the model " +
        "with MLC again, or see if MLC's huggingface provides this file.");
      const url = new URL("tokenizer.model", baseUrl).href;
      const model = await modelCache.fetchWithCache(url, "arraybuffer");
      return Tokenizer.fromSentencePiece(model);
    }
    throw Error("Cannot handle tokenizer files " + config.tokenizer_files)
  }
}
