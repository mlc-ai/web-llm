import * as tvmjs from "tvmjs";
import log from "loglevel";
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
  MLCEngineConfig,
  DefaultLogLevel,
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
  ChatCompletionMessageToolCall,
} from "./openai_api_protocols/index";
import * as ChatCompletionAPI from "./openai_api_protocols/index";
import {
  InitProgressCallback,
  MLCEngineInterface,
  GenerateProgressCallback,
  LogitProcessor,
  LogLevel,
} from "./types";
import {
  Conversation,
  compareConversationObject,
  getConversation,
} from "./conversation";
import { cleanModelUrl } from "./support";
import {
  ChatModuleNotInitializedError,
  ConfigurationNotInitializedError,
  ContentTypeError,
  DeviceLostError,
  FeatureSupportError,
  FunctionNotFoundError,
  InvalidToolChoiceError,
  MessageOrderError,
  MissingModelWasmError,
  ModelNotFoundError,
  ModelNotLoadedError,
  ShaderF16SupportError,
  SystemMessageOrderError,
  ToolCallOutputInvalidTypeError,
  ToolCallOutputMissingFieldsError,
  ToolCallOutputParseError,
  UnsupportedRoleError,
  UnsupportedTokenizerFilesError,
  UnsupportedToolChoiceTypeError,
  UnsupportedToolTypeError,
  WebGPUNotAvailableError,
} from "./error";

/**
 * Creates `MLCEngine`, and loads `modelId` onto WebGPU.
 *
 * Equivalent to `new webllm.MLCEngine().reload(...)`.
 *
 * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
 * `engineConfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig`.
 * @param chatOpts Extra options to override chat behavior specified in `mlc-chat-config.json`.
 * @returns An initialized `WebLLM.MLCEngine` with `modelId` loaded.
 * @throws Throws error when device lost (mostly due to OOM); users should re-call `CreateMLCEngine()`,
 *   potentially with a smaller model or smaller context window size.
 */
export async function CreateMLCEngine(
  modelId: string,
  engineConfig?: MLCEngineConfig,
  chatOpts?: ChatOptions,
): Promise<MLCEngine> {
  const engine = new MLCEngine(engineConfig);
  await engine.reload(modelId, chatOpts);
  return engine;
}

/**
 * The main interface of MLCEngine, which loads a model and performs tasks.
 *
 * You can either initialize one with `webllm.CreateMLCEngine(modelId)`, or `webllm.MLCEngine().reload(modelId)`.
 */
export class MLCEngine implements MLCEngineInterface {
  public chat: API.Chat;

  private currentModelId?: string = undefined; // Model current loaded, undefined if nothing is loaded
  private logger: (msg: string) => void = log.info;
  private logitProcessorRegistry?: Map<string, LogitProcessor>;
  private logitProcessor?: LogitProcessor;
  private pipeline?: LLMChatPipeline;
  private initProgressCallback?: InitProgressCallback;
  private interruptSignal = false;
  private deviceLostIsError = true; // whether device.lost is due to actual error or model reload
  private config?: ChatConfig;
  private appConfig: AppConfig;

  constructor(engineConfig?: MLCEngineConfig) {
    this.appConfig = engineConfig?.appConfig || prebuiltAppConfig;
    this.setLogLevel(engineConfig?.logLevel || DefaultLogLevel);
    this.setInitProgressCallback(engineConfig?.initProgressCallback);
    this.setLogitProcessorRegistry(engineConfig?.logitProcessorRegistry);

    this.chat = new API.Chat(this);
  }

  setAppConfig(appConfig: AppConfig) {
    this.appConfig = appConfig;
  }

  setInitProgressCallback(initProgressCallback?: InitProgressCallback) {
    this.initProgressCallback = initProgressCallback;
  }

  getInitProgressCallback() {
    return this.initProgressCallback;
  }

  setLogitProcessorRegistry(
    logitProcessorRegistry?: Map<string, LogitProcessor>,
  ) {
    this.logitProcessorRegistry = logitProcessorRegistry;
  }

  /**
   * Reload model `modelId`.
   * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
   * `engineConfig.appConfig`.
   * @param chatOpts To optionally override the `mlc-chat-config.json` of `modelId`.
   * @throws Throws error when device lost (mostly due to OOM); users should re-call reload(),
   *   potentially with a smaller model or smaller context window size.
   */
  async reload(modelId: string, chatOpts?: ChatOptions): Promise<void> {
    await this.unload();

    this.logitProcessor = this.logitProcessorRegistry?.get(modelId);
    const tstart = performance.now();

    const findModelRecord = () => {
      const matchedItem = this.appConfig?.model_list.find(
        (item) => item.model_id == modelId,
      );
      if (matchedItem !== undefined) return matchedItem;
      throw new ModelNotFoundError(modelId);
    };

    const modelRecord = findModelRecord();
    const baseUrl =
      typeof document !== "undefined"
        ? document.URL
        : globalThis.location.origin;
    let modelUrl = cleanModelUrl(modelRecord.model);
    if (!modelUrl.startsWith("http")) {
      modelUrl = new URL(modelUrl, baseUrl).href;
    }

    let configCache: tvmjs.ArtifactCacheTemplate;
    if (this.appConfig.useIndexedDBCache) {
      configCache = new tvmjs.ArtifactIndexedDBCache("webllm/config");
    } else {
      configCache = new tvmjs.ArtifactCache("webllm/config");
    }

    // load config
    const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
    this.config = {
      ...(await configCache.fetchWithCache(configUrl, "json")),
      ...modelRecord.overrides,
      ...chatOpts,
    } as ChatConfig;

    // load tvm wasm
    let wasmCache: tvmjs.ArtifactCacheTemplate;
    if (this.appConfig.useIndexedDBCache) {
      wasmCache = new tvmjs.ArtifactIndexedDBCache("webllm/wasm");
    } else {
      wasmCache = new tvmjs.ArtifactCache("webllm/wasm");
    }

    const wasmUrl = modelRecord.model_lib;
    if (wasmUrl === undefined) {
      throw new MissingModelWasmError(modelRecord.model_id);
    }
    const fetchWasmSource = async () => {
      if (wasmUrl.includes("localhost")) {
        // do not cache wasm on local host as we might update code frequently
        return (await fetch(wasmUrl)).arrayBuffer();
      } else if (!wasmUrl.startsWith("http")) {
        // do not cache wasm on the same server as it can also refresh
        // rely on the normal caching strategy
        return (await fetch(new URL(wasmUrl, baseUrl).href)).arrayBuffer();
      } else {
        // use cache
        return await wasmCache.fetchWithCache(wasmUrl, "arraybuffer");
      }
    };
    const wasmSource = await fetchWasmSource();

    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      tvmjs.createPolyfillWASI(),
      this.logger,
    );

    if (this.initProgressCallback !== undefined) {
      tvm.registerInitProgressCallback(this.initProgressCallback);
    }

    // detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
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
            throw new ShaderF16SupportError();
          }
          throw new FeatureSupportError(feature);
        }
      }
    }

    // Most device lost happens in `reload()` since we allocate memory ahead of time. So we can
    // use this flag at the end of `reload()` to make the error handling synchronous.
    // This `.then()` exists throughout the lifetime of the device. Though we have not
    // experienced device error outside of `reload()`, it is still possible this `.then()` is
    // triggered outside of `reload()`. TODO: does this cause unexpected behavior?
    let deviceLostInReload = false;
    gpuDetectOutput.device.lost.then((info: any) => {
      if (this.deviceLostIsError) {
        log.error(
          `Device was lost during reload. This can happen due to insufficient memory or other GPU constraints. Detailed error: ${info}. Please try to reload WebLLM with a less resource-intensive model.`,
        );
        this.unload();
        deviceLostInReload = true;
      }
    });
    tvm.initWebGPU(gpuDetectOutput.device);

    const tokenizer = await this.asyncLoadTokenizer(
      modelUrl,
      this.config,
      this.appConfig,
    );
    const cacheType = this.appConfig.useIndexedDBCache ? "indexeddb" : "cache";
    await tvm.fetchNDArrayCache(
      modelUrl,
      tvm.webgpu(),
      "webllm/model",
      cacheType,
    );
    this.pipeline = new LLMChatPipeline(
      tvm,
      tokenizer,
      this.config,
      this.logitProcessor,
    );
    await this.pipeline?.asyncLoadWebGPUPipelines();
    const tend = performance.now();

    if (this.initProgressCallback !== undefined) {
      const text = "Finish loading on " + gpuLabel;
      this.initProgressCallback({
        progress: 1,
        timeElapsed: (tend - tstart) / 1e3,
        text: text,
      });
    }
    this.currentModelId = modelId;

    if (deviceLostInReload) {
      throw new DeviceLostError();
    }
  }

  async generate(
    input: string | ChatCompletionRequestNonStreaming,
    progressCallback?: GenerateProgressCallback,
    streamInterval = 1,
    genConfig?: GenerationConfig,
  ): Promise<string> {
    log.warn(
      "WARNING: `generate()` will soon be deprecated. " +
        "Please use `engine.chat.completions.create()` instead. " +
        "For multi-round chatting, see `examples/multi-round-chat` on how to use " +
        "`engine.chat.completions.create()` to achieve the same effect.",
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
  async *chatCompletionAsyncChunkGenerator(
    request: ChatCompletionRequestStreaming,
    genConfig: GenerationConfig,
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
    let prevMessageLength = 0; // to know where to start slicing the delta; does not count �

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

    async function _getChunk(
      thisModule: MLCEngine,
    ): Promise<ChatCompletionChunk | undefined> {
      // Remove the replacement character (U+FFFD) from the response to handle emojis.
      // Each emoji is made up of multiples of 4 tokens; when truncated, it is displayed as �, so
      // we skip this delta until a full emoji is rendered
      // TODO(Charlie): This does not consider cases of � not being emoji, need to fix with Streamer
      const curMessage = await thisModule.getMessage();
      const numTrailingReplacementChar =
        _countTrailingReplacementChar(curMessage);
      if (numTrailingReplacementChar % 4 !== 0) {
        return undefined;
      }

      const deltaMessage = curMessage.slice(prevMessageLength);
      prevMessageLength = curMessage.length;
      const chunk: ChatCompletionChunk = {
        id: id,
        choices: [
          {
            delta: { content: deltaMessage, role: "assistant" },
            finish_reason: null, // not finished yet
            index: 0,
            logprobs: request.logprobs
              ? ({
                  content: thisModule
                    .getPipeline()
                    .getTokenLogprobArray()
                    .slice(-1), // always the last entry
                } as ChatCompletionChunk.Choice.Logprobs)
              : null,
          },
        ],
        model: model,
        object: "chat.completion.chunk",
        created: created,
      };
      return chunk;
    }

    await this.prefill(request, genConfig);
    let curChunk = await _getChunk(this); // prefill produces a chunk
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

    // If function calling, use the last chunk to return tool_calls
    let finish_reason = this.getFinishReason()!;
    const isFunctionCalling =
      request.tools !== undefined && request.tools !== null;
    let tool_calls:
      | Array<ChatCompletionChunk.Choice.Delta.ToolCall>
      | undefined;

    if (this.getFinishReason()! == "stop" && isFunctionCalling) {
      // If stopped due to length or abort, cannot output return tool_calls field
      finish_reason = "tool_calls";
      const outputMessage = await this.getMessage();
      tool_calls = this.getToolCallFromOutputMessage(
        outputMessage,
        /*isStreaming=*/ true,
      ) as Array<ChatCompletionChunk.Choice.Delta.ToolCall>;
    }

    const lastChunk: ChatCompletionChunk = {
      id: id,
      choices: [
        {
          delta: isFunctionCalling
            ? {
                role: "assistant",
                tool_calls: tool_calls,
              }
            : {},
          finish_reason: finish_reason,
          index: 0,
        },
      ],
      model: model,
      object: "chat.completion.chunk",
      created: created,
    };
    yield lastChunk;

    if (request.stream_options?.include_usage) {
      const completion_tokens =
        this.getPipeline().getCurRoundDecodingTotalTokens();
      const prompt_tokens = this.getPipeline().getCurRoundPrefillTotalTokens();
      const prefill_tokens_per_s =
        this.getPipeline().getCurRoundPrefillTokensPerSec();
      const decode_tokens_per_s =
        this.getPipeline().getCurRoundDecodingTokensPerSec();
      const usageChunk: ChatCompletionChunk = {
        id: id,
        choices: [],
        usage: {
          completion_tokens: completion_tokens,
          prompt_tokens: prompt_tokens,
          total_tokens: completion_tokens + prompt_tokens,
          extra: {
            prefill_tokens_per_s: prefill_tokens_per_s,
            decode_tokens_per_s: decode_tokens_per_s,
          },
        } as CompletionUsage,
        model: model,
        object: "chat.completion.chunk",
        created: created,
      };
      yield usageChunk;
    }
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
    request: ChatCompletionRequestNonStreaming,
  ): Promise<ChatCompletion>;
  async chatCompletion(
    request: ChatCompletionRequestStreaming,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  async chatCompletion(
    request: ChatCompletionRequestBase,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;
  async chatCompletion(
    request: ChatCompletionRequest,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    // 0. Preprocess inputs
    if (!this.currentModelId) {
      throw new ModelNotLoadedError();
    }
    ChatCompletionAPI.postInitAndCheckFields(request, this.currentModelId);
    const genConfig: GenerationConfig = {
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      max_tokens: request.max_tokens,
      stop: request.stop,
      top_p: request.top_p,
      temperature: request.temperature,
      logit_bias: request.logit_bias,
      logprobs: request.logprobs,
      top_logprobs: request.top_logprobs,
      response_format: request.response_format,
    };

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
    let prefill_time = 0;
    let decode_time = 0;
    for (let i = 0; i < n; i++) {
      let outputMessage: string;
      if (this.interruptSignal) {
        // A single interrupt signal should stop all choices' generations
        this.getPipeline().triggerStop();
        outputMessage = "";
      } else {
        outputMessage = await this._generate(
          request,
          /*progressCallback=*/ undefined,
          /*streamInterval=*/ 1,
          /*genConfig=*/ genConfig,
        );
      }
      let finish_reason = this.getFinishReason()!;

      // 3. Post processing for function calling
      const isFunctionCalling =
        request.tools !== undefined && request.tools !== null;
      let tool_calls: Array<ChatCompletionMessageToolCall> | undefined;
      if (this.getFinishReason()! == "stop" && isFunctionCalling) {
        // If stopped due to length or abort, cannot output return tool_calls field
        finish_reason = "tool_calls";
        tool_calls = this.getToolCallFromOutputMessage(
          outputMessage,
          /*isStreaming=*/ false,
        );
      }

      choices.push({
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        finish_reason: finish_reason,
        index: i,
        logprobs: request.logprobs
          ? ({
              content: this.getPipeline().getTokenLogprobArray(),
            } as ChatCompletion.Choice.Logprobs)
          : null,
        message: isFunctionCalling
          ? {
              content: null,
              tool_calls: tool_calls,
              role: "assistant",
            }
          : {
              content: outputMessage,
              role: "assistant",
            },
      });
      completion_tokens += this.getPipeline().getCurRoundDecodingTotalTokens();
      prompt_tokens += this.getPipeline().getCurRoundPrefillTotalTokens();
      prefill_time += this.getPipeline().getCurRoundPrefillTotalTime();
      decode_time += this.getPipeline().getCurRoundDecodingTotalTime();
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
        extra: {
          prefill_tokens_per_s: prompt_tokens / prefill_time,
          decode_tokens_per_s: completion_tokens / decode_time,
        },
      } as CompletionUsage,
    };

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
    log.warn(
      "WARNING: `runtimeStatsText()` will soon be deprecated. " +
        "Please use `ChatCompletion.usage` for non-streaming requests, or " +
        "`ChatCompletionChunk.usage` for streaming requests, enabled by `stream_options`. " +
        "The only flow that expects to use `runtimeStatsText()` as of now is `forwardTokensAndSample()`.",
    );
    return this.getPipeline().runtimeStatsText();
  }

  async resetChat(keepStats = false) {
    this.pipeline?.resetChat(keepStats);
  }

  async unload() {
    this.deviceLostIsError = false; // so that unload() does not trigger device.lost error
    this.pipeline?.dispose();
    this.pipeline = undefined;
    this.currentModelId = undefined;
    this.deviceLostIsError = true;
  }

  async getMaxStorageBufferBindingSize(): Promise<number> {
    // First detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
    }

    const computeMB = (value: number) => {
      return Math.ceil(value / (1 << 20)) + "MB";
    };
    const maxStorageBufferBindingSize =
      gpuDetectOutput.device.limits.maxStorageBufferBindingSize;
    const defaultMaxStorageBufferBindingSize = 1 << 30; // 1GB
    if (maxStorageBufferBindingSize < defaultMaxStorageBufferBindingSize) {
      log.warn(
        `WARNING: the current maxStorageBufferBindingSize ` +
          `(${computeMB(maxStorageBufferBindingSize)}) ` +
          `may only work for a limited number of models, e.g.: \n` +
          `- Llama-3.1-8B-Instruct-q4f16_1-MLC-1k \n` +
          `- Llama-2-7b-chat-hf-q4f16_1-MLC-1k \n` +
          `- RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC-1k \n` +
          `- RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC-1k \n` +
          `- TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC-1k \n` +
          `- TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC-1k`,
      );
    }
    return maxStorageBufferBindingSize;
  }

  async getGPUVendor(): Promise<string> {
    // First detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
    }
    return gpuDetectOutput.adapterInfo.vendor;
  }

  //--------------------------
  // Lower level API
  //--------------------------
  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
  ): Promise<number> {
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
   * Set MLCEngine logging output level
   *
   * @param logLevel The new log level
   */
  setLogLevel(logLevel: LogLevel) {
    log.setLevel(logLevel);
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
    config: ChatConfig,
  ): Conversation {
    // 0. Instantiate a new Conversation object
    const conversation = getConversation(
      config.conv_template,
      config.conv_config,
    );

    // 1. Populate function-calling-related fields
    // TODO: either remove these or support gorilla-like function calling models.
    // These commented code was used to support gorilla, but we could not use grammar to
    // guarantee its output, nor make it conform to OpenAI's function calling output. Kept for now.
    // const functionCallUsage = this.getFunctionCallUsage(request);
    // conversation.function_string = functionCallUsage;
    // conversation.use_function_calling = functionCallUsage !== "";

    // 2. Populate conversation.messages
    const input = request.messages;
    const lastId = input.length - 1;
    if (
      input[lastId].role !== "user" ||
      typeof input[lastId].content !== "string"
    ) {
      // TODO(Charlie): modify condition after we support multimodal inputs
      throw new MessageOrderError(
        "The last message should be a string from the `user`.",
      );
    }
    for (let i = 0; i < input.length - 1; i++) {
      const message: ChatCompletionMessageParam = input[i];
      if (message.role === "system") {
        if (i !== 0) {
          throw new SystemMessageOrderError();
        }
        conversation.override_system_message = message.content;
      } else if (message.role === "user") {
        if (typeof message.content !== "string") {
          // TODO(Charlie): modify condition after we support multimodal inputs
          throw new ContentTypeError(message.role + "'s message");
        }
        conversation.appendMessage(Role.user, message.content, message.name);
      } else if (message.role === "assistant") {
        if (typeof message.content !== "string") {
          throw new ContentTypeError(message.role + "'s message");
        }
        conversation.appendMessage(
          Role.assistant,
          message.content,
          message.name,
        );
      } else {
        throw new UnsupportedRoleError(message.role);
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
    if (
      request.tools == undefined ||
      (typeof request.tool_choice == "string" && request.tool_choice == "none")
    ) {
      return "";
    }
    if (
      typeof request.tool_choice == "string" &&
      request.tool_choice !== "auto"
    ) {
      throw new InvalidToolChoiceError(request.tool_choice);
    }
    if (
      typeof request.tool_choice !== "string" &&
      request.tool_choice?.type !== "function"
    ) {
      throw new UnsupportedToolChoiceTypeError();
    }

    const singleFunctionToCall =
      typeof request.tool_choice !== "string" &&
      request.tool_choice?.function?.name;
    if (singleFunctionToCall) {
      for (const f of request.tools) {
        if (singleFunctionToCall == f.function.name) {
          return JSON.stringify([f.function]);
        }
      }
      throw new FunctionNotFoundError(singleFunctionToCall);
    }

    const function_list = [];
    for (const f of request.tools) {
      if (f.type !== "function") {
        throw new UnsupportedToolTypeError();
      }
      function_list.push(f.function);
    }
    return JSON.stringify(function_list);
  }

  /**
   * Given a string outputMessage, parse it as a JSON object and return an array of tool calls.
   *
   * Expect outputMessage to be a valid JSON string, and expect it to be an array of Function with
   * fields `arguments` and `name`.
   */
  private getToolCallFromOutputMessage(
    outputMessage: string,
    isStreaming: false,
  ): Array<ChatCompletionMessageToolCall>;
  private getToolCallFromOutputMessage(
    outputMessage: string,
    isStreaming: true,
  ): Array<ChatCompletionChunk.Choice.Delta.ToolCall>;
  private getToolCallFromOutputMessage(
    outputMessage: string,
    isStreaming: boolean,
  ):
    | Array<ChatCompletionMessageToolCall>
    | Array<ChatCompletionChunk.Choice.Delta.ToolCall> {
    // 1. Parse outputMessage to JSON object
    let toolCallsObject;
    try {
      toolCallsObject = JSON.parse(outputMessage);
    } catch (err) {
      throw new ToolCallOutputParseError(outputMessage, err as Error);
    }

    // 2. Expect to be an array
    if (!(toolCallsObject instanceof Array)) {
      throw new ToolCallOutputInvalidTypeError("array");
    }

    // 3. Parse each tool call and populate tool_calls
    const numToolCalls = toolCallsObject.length;
    const tool_calls = [];
    for (let id = 0; id < numToolCalls; id++) {
      const curToolCall = toolCallsObject[id];
      if (
        curToolCall.name === undefined ||
        curToolCall.arguments === undefined
      ) {
        throw new ToolCallOutputMissingFieldsError(
          ["name", "arguments"],
          curToolCall,
        );
      }
      tool_calls.push({
        name: curToolCall.name,
        arguments: JSON.stringify(curToolCall.arguments),
      });
    }

    // 4. Return based on whether it is streaming or not
    if (isStreaming) {
      const tool_calls_result: Array<ChatCompletionChunk.Choice.Delta.ToolCall> =
        [];
      for (let id = 0; id < numToolCalls; id++) {
        const curToolCall = tool_calls[id];
        tool_calls_result.push({
          index: id,
          function: {
            name: curToolCall.name,
            arguments: curToolCall.arguments,
          },
          type: "function",
        });
      }
      return tool_calls_result;
    } else {
      const tool_calls_result: Array<ChatCompletionMessageToolCall> = [];
      for (let id = 0; id < numToolCalls; id++) {
        const curToolCall = tool_calls[id];
        tool_calls_result.push({
          id: id.toString(),
          function: {
            name: curToolCall.name,
            arguments: curToolCall.arguments,
          },
          type: "function",
        });
      }
      return tool_calls_result;
    }
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
    genConfig?: GenerationConfig,
  ) {
    if (this.config === undefined) {
      throw new ConfigurationNotInitializedError();
    }
    let input_str: string;
    let input_role_str: string | undefined;
    if (typeof input === "string") {
      input_str = input;
    } else {
      // 1. Get new conversation based on request, determine if we are in multiround chatting
      const oldConv = this.getPipeline().getConversationObject();
      const newConv = this.getConversationFromChatCompletionRequest(
        input,
        this.config,
      );
      if (!compareConversationObject(oldConv, newConv)) {
        // Not the same conversation, so not multiround chatting, reset everything (KV cache, etc.)
        this.resetChat();
        this.getPipeline().setConversation(newConv);
      } else if (newConv.messages.length === 0) {
        // Empty oldConv, and no chat history in newConv, so reset and setConversation
        this.resetChat();
        this.getPipeline().setConversation(newConv);
      } else {
        log.info("Multiround chatting, reuse KVCache.");
      }

      // 2. Treat the last message as the usual input
      const last_msg = input.messages[
        input.messages.length - 1
      ] as ChatCompletionUserMessageParam;
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
      throw new ChatModuleNotInitializedError();
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
    } else if (config.tokenizer_files.includes("tokenizer.model")) {
      this.logger(
        "Using `tokenizer.model` since we cannot locate `tokenizer.json`.\n" +
          "It is recommended to use `tokenizer.json` to ensure all token mappings are included, " +
          "since currently, files like `added_tokens.json`, `tokenizer_config.json` are ignored.\n" +
          "Consider converting `tokenizer.model` to `tokenizer.json` by compiling the model " +
          "with MLC again, or see if MLC's huggingface provides this file.",
      );
      const url = new URL("tokenizer.model", baseUrl).href;
      const model = await modelCache.fetchWithCache(url, "arraybuffer");
      return Tokenizer.fromSentencePiece(model);
    }
    throw new UnsupportedTokenizerFilesError(config.tokenizer_files);
  }
}
