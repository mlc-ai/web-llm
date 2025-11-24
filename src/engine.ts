import * as tvmjs from "@mlc-ai/web-runtime";
import log from "loglevel";
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
  ModelType,
} from "./config";
import { LLMChatPipeline } from "./llm_chat";
import {
  // ChatCompletion
  ChatCompletionRequest,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionRequestNonStreaming,
  ChatCompletionRequestStreaming,
  ChatCompletionRequestBase,
  CompletionUsage,
  ChatCompletionMessageToolCall,
  // Completion
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  CompletionCreateParamsBase,
  CompletionCreateParams,
  Completion,
  CompletionChoice,
  EmbeddingCreateParams,
  CreateEmbeddingResponse,
  Embedding,
} from "./openai_api_protocols/index";
import * as API from "./openai_api_protocols/index";
import {
  InitProgressCallback,
  MLCEngineInterface,
  LogitProcessor,
  LogLevel,
  LatencyBreakdown,
} from "./types";
import {
  compareConversationObject,
  getConversation,
  getConversationFromChatCompletionRequest,
} from "./conversation";
import {
  cleanModelUrl,
  CustomLock,
  findModelRecord,
  getModelIdToUse,
  getToolCallFromOutputMessage,
} from "./support";
import {
  ConfigurationNotInitializedError,
  DeviceLostError,
  EmbeddingUnsupportedModelError,
  FeatureSupportError,
  MissingModelWasmError,
  ShaderF16SupportError,
  WebGPUNotAvailableError,
  ReloadArgumentSizeUnmatchedError,
  IncorrectPipelineLoadedError,
  ReloadModelIdNotUniqueError,
  SpecifiedModelNotFoundError,
  ModelNotLoadedError,
} from "./error";
import { asyncLoadTokenizer } from "./cache_util";
import { EmbeddingPipeline } from "./embedding";

/**
 * Creates `MLCEngine`, and loads `modelId` onto WebGPU.
 *
 * Equivalent to `new webllm.MLCEngine().reload(...)`.
 *
 * @param modelId model_id of the model to load, either string or string[]. When multiple models
 *   are provided, we load all models sequentially. Each modelId needs to either be in
 *   `webllm.prebuiltAppConfig`, or in `engineCOnfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig`.
 * @param chatOpts Extra options to optionally override the `mlc-chat-config.json` of `modelId`.
 *   The size of which needs to match that of `modelId`; chatOpts[i] will be used for modelId[i].
 * @returns An initialized `WebLLM.MLCEngine` with `modelId` loaded.
 * @throws Throws error when device lost (mostly due to OOM); users should re-call `CreateMLCEngine()`,
 *   potentially with a smaller model or smaller context window size.
 */
export async function CreateMLCEngine(
  modelId: string | string[],
  engineConfig?: MLCEngineConfig,
  chatOpts?: ChatOptions | ChatOptions[],
): Promise<MLCEngine> {
  const engine = new MLCEngine(engineConfig);
  await engine.reload(modelId, chatOpts);
  return engine;
}

/**
 * The main interface of MLCEngine, which loads a model and performs tasks.
 *
 * You can either initialize one with `webllm.CreateMLCEngine(modelId)`, or
 * `webllm.MLCEngine().reload(modelId)`.
 */
export class MLCEngine implements MLCEngineInterface {
  // APIs
  /** For chat.completions.create() */
  public chat: API.Chat;
  /** For completions.create() */
  public completions: API.Completions;
  /** For embeddings.create() */
  public embeddings: API.Embeddings;

  // Maps to maintain states of loaded model(s)
  /** Maps each loaded model's modelId to its pipeline */
  private loadedModelIdToPipeline: Map<
    string,
    LLMChatPipeline | EmbeddingPipeline
  >;
  /** Maps each loaded model's modelId to its chatConfig */
  private loadedModelIdToChatConfig: Map<string, ChatConfig>;
  /** Maps each loaded model's modelId to its modelType */
  private loadedModelIdToModelType: Map<string, ModelType>;
  /** Maps each loaded model's modelId to a lock. Ensures
   * each model only processes one request at at time.
   */
  private loadedModelIdToLock: Map<string, CustomLock>;

  // Others
  private logger: (msg: string) => void = log.info;
  private logitProcessorRegistry?: Map<string, LogitProcessor>;
  private initProgressCallback?: InitProgressCallback;
  private appConfig: AppConfig;

  // Signals and flags
  private interruptSignal = false;
  private deviceLostIsError = true; // whether device.lost is due to actual error or model reload
  private reloadController: AbortController | undefined;

  constructor(engineConfig?: MLCEngineConfig) {
    this.loadedModelIdToPipeline = new Map<
      string,
      LLMChatPipeline | EmbeddingPipeline
    >();
    this.loadedModelIdToChatConfig = new Map<string, ChatConfig>();
    this.loadedModelIdToModelType = new Map<string, ModelType>();
    this.loadedModelIdToLock = new Map<string, CustomLock>();
    this.appConfig = engineConfig?.appConfig || prebuiltAppConfig;
    this.setLogLevel(engineConfig?.logLevel || DefaultLogLevel);
    this.setInitProgressCallback(engineConfig?.initProgressCallback);
    this.setLogitProcessorRegistry(engineConfig?.logitProcessorRegistry);

    this.chat = new API.Chat(this);
    this.completions = new API.Completions(this);
    this.embeddings = new API.Embeddings(this);
  }

  //-----------------------
  // 0. Setters and getters
  //-----------------------

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
   * Set MLCEngine logging output level
   *
   * @param logLevel The new log level
   */
  setLogLevel(logLevel: LogLevel) {
    log.setLevel(logLevel);
  }

  //----------------------------------------
  // 1. Model/pipeline loading and unloading
  //----------------------------------------

  async reload(
    modelId: string | string[],
    chatOpts?: ChatOptions | ChatOptions[],
  ): Promise<void> {
    // 0. Unload all loaded models
    await this.unload();
    // 1. Convert inputs to arrays
    if (!Array.isArray(modelId)) {
      modelId = [modelId];
    }
    if (chatOpts !== undefined && !Array.isArray(chatOpts)) {
      chatOpts = [chatOpts];
    }
    // 2. Check whether size matches
    if (chatOpts !== undefined && modelId.length !== chatOpts.length) {
      throw new ReloadArgumentSizeUnmatchedError(
        modelId.length,
        chatOpts.length,
      );
    }
    // 3. Make sure each model in modelId is unique
    if (new Set(modelId).size < modelId.length) {
      throw new ReloadModelIdNotUniqueError(modelId);
    }
    // 4. Sequentially load each model
    // Single abort should stop all to-be-loaded models
    this.reloadController = new AbortController();
    try {
      for (let i = 0; i < modelId.length; i++) {
        await this.reloadInternal(
          modelId[i],
          chatOpts ? chatOpts[i] : undefined,
        );
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        log.warn("Reload() is aborted.", error.message);
        return;
      }
      throw error;
    } finally {
      this.reloadController = undefined;
    }
  }

  private async reloadInternal(
    modelId: string,
    chatOpts?: ChatOptions,
  ): Promise<void> {
    const logitProcessor = this.logitProcessorRegistry?.get(modelId);
    const tstart = performance.now();

    // look up and parse model record, record model type
    const modelRecord = findModelRecord(modelId, this.appConfig);
    const baseUrl =
      typeof document !== "undefined"
        ? document.URL
        : globalThis.location.origin;
    let modelUrl = cleanModelUrl(modelRecord.model);
    if (!modelUrl.startsWith("http")) {
      modelUrl = new URL(modelUrl, baseUrl).href;
    }
    const modelType =
      modelRecord.model_type === undefined || modelRecord.model_type === null
        ? ModelType.LLM
        : modelRecord.model_type;
    this.loadedModelIdToModelType.set(modelId, modelType);

    // instantiate cache
    let configCache: tvmjs.ArtifactCacheTemplate;
    if (this.appConfig.useIndexedDBCache) {
      configCache = new tvmjs.ArtifactIndexedDBCache("webllm/config");
    } else {
      configCache = new tvmjs.ArtifactCache("webllm/config");
    }

    // load config
    const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
    const curModelConfig = {
      ...(await configCache.fetchWithCache(
        configUrl,
        "json",
        this.reloadController?.signal,
      )),
      ...modelRecord.overrides,
      ...chatOpts,
    } as ChatConfig;
    this.loadedModelIdToChatConfig.set(modelId, curModelConfig);

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
        return await wasmCache.fetchWithCache(
          wasmUrl,
          "arraybuffer",
          this.reloadController?.signal,
        );
      }
    };
    const wasmSource = await fetchWasmSource();

    const wasm = new Uint8Array(wasmSource);
    const tvm = await tvmjs.instantiate(
      wasm.buffer,
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
          `Device was lost. This can happen due to insufficient memory or other GPU constraints. ` +
            `Detailed error: ${info}. Please try to reload WebLLM with a less resource-intensive model.`,
        );
        this.unload();
        deviceLostInReload = true;
      }
    });
    tvm.initWebGPU(gpuDetectOutput.device);

    const tokenizer = await asyncLoadTokenizer(
      modelUrl,
      curModelConfig,
      this.appConfig,
      this.logger,
    );
    const cacheType = this.appConfig.useIndexedDBCache ? "indexeddb" : "cache";
    await tvm.fetchTensorCache(
      modelUrl,
      tvm.webgpu(),
      "webllm/model",
      cacheType,
      this.reloadController?.signal,
    );

    // Instantiate pipeline
    // TODO: would be good to somehow check for error when LLMChatPipeline is loaded for an
    // embedding model, and prompt user to use ModelRecord.model_type
    let newPipeline: LLMChatPipeline | EmbeddingPipeline;
    if (modelRecord.model_type === ModelType.embedding) {
      newPipeline = new EmbeddingPipeline(tvm, tokenizer, curModelConfig);
    } else {
      newPipeline = new LLMChatPipeline(
        tvm,
        tokenizer,
        curModelConfig,
        logitProcessor,
      );
    }
    await newPipeline.asyncLoadWebGPUPipelines();
    this.loadedModelIdToPipeline.set(modelId, newPipeline);
    this.loadedModelIdToLock.set(modelId, new CustomLock());

    // Clean up
    const tend = performance.now();
    if (this.initProgressCallback !== undefined) {
      const text = "Finish loading on " + gpuLabel;
      this.initProgressCallback({
        progress: 1,
        timeElapsed: (tend - tstart) / 1e3,
        text: text,
      });
    }
    if (deviceLostInReload) {
      throw new DeviceLostError();
    }
  }

  async unload() {
    this.deviceLostIsError = false; // so that unload() does not trigger device.lost error
    // TODO: can optimize by calling dispose() to all pipelines in parallel. However, need to wait
    // for all sync() to finish before proceeding (e.g. naive forEach does not work)
    for (const entry of Array.from(this.loadedModelIdToPipeline.entries())) {
      const pipeline = entry[1];
      pipeline.dispose();
      // Wait until device is actually destroyed so we can safely set deviceLostIsError back to true
      await pipeline.sync();
    }
    this.loadedModelIdToPipeline.clear();
    this.loadedModelIdToChatConfig.clear();
    this.loadedModelIdToModelType.clear();
    this.loadedModelIdToLock.clear();
    this.deviceLostIsError = true;
    if (this.reloadController) {
      this.reloadController.abort("Engine.unload() is called.");
      this.reloadController = undefined;
    }
  }

  //---------------------------------------------------
  // 2. Underlying auto-regressive generation functions
  //---------------------------------------------------

  private async _generate(
    input:
      | ChatCompletionRequestNonStreaming
      | CompletionCreateParamsNonStreaming,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
  ): Promise<string> {
    this.interruptSignal = false;
    if (genConfig !== undefined) {
      postInitAndCheckGenerationConfigValues(genConfig);
    }
    await this.prefill(input, pipeline, chatConfig, genConfig);

    while (!pipeline.stopped()) {
      if (this.interruptSignal) {
        pipeline.triggerStop();
        break;
      }
      await this.decode(pipeline, genConfig);
    }
    return pipeline.getMessage();
  }

  /**
   * Similar to `_generate()`; but instead of using callback, we use an async iterable.
   */
  asyncGenerate(
    request: ChatCompletionRequestStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
  ): AsyncGenerator<ChatCompletionChunk, void, void>;
  asyncGenerate(
    request: CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
  ): AsyncGenerator<Completion, void, void>;
  async *asyncGenerate(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
  ): AsyncGenerator<ChatCompletionChunk | Completion, void, void> {
    // Since it is an async generator, we need to do fine-grained try-catch to ensure lock is
    // released only when errors occur. Then release at the very end when no error occurs.
    // TODO: This makes code less readable, is there a better way to do this?
    const lock = this.loadedModelIdToLock.get(model)!;

    // 0. Pre-processing
    const isChatCompletion = "messages" in request;
    const isFunctionCalling =
      "tools" in request &&
      request.tools !== undefined &&
      request.tools !== null;
    try {
      if (isFunctionCalling && !isChatCompletion) {
        throw new Error(
          "Expect `chat.completions` with tools, not `completions`.",
        );
      }
      postInitAndCheckGenerationConfigValues(genConfig);
      if (request.seed !== null && request.seed !== undefined) {
        pipeline.setSeed(request.seed);
      }
    } catch (err) {
      await lock.release();
      throw err;
    }

    // 1. Helper function that generates the chunk
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
      selectedPipeline: LLMChatPipeline,
    ): Promise<ChatCompletionChunk | Completion | undefined> {
      // Remove the replacement character (U+FFFD) from the response to handle emojis.
      // Each emoji is made up of multiples of 4 tokens; when truncated, it is displayed as �, so
      // we skip this delta until a full emoji is rendered
      // TODO(Charlie): This does not consider cases of � not being emoji, need to fix with Streamer
      const curMessage = selectedPipeline.getMessage();
      const numTrailingReplacementChar =
        _countTrailingReplacementChar(curMessage);
      if (numTrailingReplacementChar % 4 !== 0) {
        return undefined;
      }

      const deltaMessage = curMessage.slice(prevMessageLength);
      prevMessageLength = curMessage.length;
      const logprobs = request.logprobs
        ? ({
            content: selectedPipeline.getTokenLogprobArray().slice(-1), // always the last entry
          } as ChatCompletionChunk.Choice.Logprobs)
        : null;
      if (isChatCompletion) {
        const chunk: ChatCompletionChunk = {
          id: id,
          choices: [
            {
              delta: { content: deltaMessage, role: "assistant" },
              finish_reason: null, // not finished yet
              index: 0,
              logprobs: logprobs,
            },
          ],
          model: model,
          object: "chat.completion.chunk",
          created: created,
        };
        return chunk;
      } else {
        const chunk: Completion = {
          id: id,
          choices: [
            {
              text: deltaMessage,
              finish_reason: null, // not finished yet
              index: 0,
              logprobs: logprobs,
            },
          ],
          model: model,
          object: "text_completion",
          created: created,
        };
        return chunk;
      }
    }

    // 2. Auto-regressive loop
    let curChunk;
    try {
      await this.prefill(request, pipeline, chatConfig, genConfig);
      curChunk = await _getChunk(pipeline); // prefill produces a chunk
    } catch (err) {
      await lock.release();
      throw err;
    }
    if (curChunk) {
      yield curChunk;
    }

    while (!pipeline.stopped()) {
      if (this.interruptSignal) {
        // TODO: should we directly release lock here and return the async
        // generator? Though no issue observed as of now with interruptGenerate()
        pipeline.triggerStop();
        break;
      }
      try {
        await this.decode(pipeline, genConfig);
        curChunk = await _getChunk(pipeline);
      } catch (err) {
        await lock.release();
        throw err;
      }
      if (curChunk) {
        yield curChunk;
      }
    }

    // Reset seed -- we do not want this seed to affect future requests
    if (request.seed !== null && request.seed !== undefined) {
      pipeline.setSeed(Date.now());
    }

    // 3. Last chunk empty marking the end
    // If function calling, use the last chunk to return tool_calls
    let finish_reason = pipeline.getFinishReason()!;
    let tool_calls:
      | Array<ChatCompletionChunk.Choice.Delta.ToolCall>
      | undefined;
    try {
      if (pipeline.getFinishReason() === "stop" && isFunctionCalling) {
        // If stopped due to length or abort, cannot output return tool_calls field
        finish_reason = "tool_calls";
        const outputMessage = pipeline.getMessage();
        tool_calls = getToolCallFromOutputMessage(
          outputMessage,
          /*isStreaming=*/ true,
        ) as Array<ChatCompletionChunk.Choice.Delta.ToolCall>;
      }
    } catch (err) {
      await lock.release();
      throw err;
    }

    if (isChatCompletion) {
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
    } else {
      const lastChunk: Completion = {
        id: id,
        choices: [
          {
            text: "",
            finish_reason: finish_reason,
            index: 0,
          },
        ],
        model: model,
        object: "text_completion",
        created: created,
      };
      yield lastChunk;
    }

    // 4. Usage chunk
    if (request.stream_options?.include_usage) {
      const usedGrammar =
        "response_format" in request &&
        (request.response_format?.type === "grammar" ||
          request.response_format?.type === "json_object");
      const completion_tokens = pipeline.getCurRoundDecodingTotalTokens();
      const prompt_tokens = pipeline.getCurRoundPrefillTotalTokens();
      const prefill_tokens_per_s = pipeline.getCurRoundPrefillTokensPerSec();
      const decode_tokens_per_s = pipeline.getCurRoundDecodingTokensPerSec();
      const grammar_init_s = pipeline.getCurRoundGrammarInitTotalTime();
      const prefill_time = pipeline.getCurRoundPrefillTotalTime();
      const decode_time = pipeline.getCurRoundDecodingTotalTime();
      const grammar_per_token_s =
        pipeline.getCurRoundGrammarPerTokenTotalTime();
      const latencyBreakdown: LatencyBreakdown =
        pipeline.getCurRoundLatencyBreakdown();

      const defaultExtra = {
        e2e_latency_s: (Date.now() - timeReceived) / 1000,
        prefill_tokens_per_s: prefill_tokens_per_s,
        decode_tokens_per_s: decode_tokens_per_s,
        time_to_first_token_s: prefill_time,
        time_per_output_token_s: decode_time / completion_tokens,
        latencyBreakdown: request.extra_body?.enable_latency_breakdown
          ? latencyBreakdown
          : undefined,
      };
      const usage: CompletionUsage = {
        completion_tokens: completion_tokens,
        prompt_tokens: prompt_tokens,
        total_tokens: completion_tokens + prompt_tokens,
        extra: usedGrammar
          ? {
              ...defaultExtra,
              ...{
                grammar_init_s: grammar_init_s,
                grammar_per_token_s: grammar_per_token_s / completion_tokens,
              },
            }
          : defaultExtra,
      };
      if (isChatCompletion) {
        const usageChunk: ChatCompletionChunk = {
          id: id,
          choices: [],
          usage: usage,
          model: model,
          object: "chat.completion.chunk",
          created: created,
        };
        yield usageChunk;
      } else {
        const usageChunk: Completion = {
          id: id,
          choices: [],
          usage: usage,
          model: model,
          object: "text_completion",
          created: created,
        };
        yield usageChunk;
      }
    }

    await lock.release();
  }

  async interruptGenerate() {
    this.interruptSignal = true;
  }

  //------------------------------
  // 3. High-level generation APIs
  //------------------------------

  /**
   * Completes a single ChatCompletionRequest.
   *
   * @param request A OpenAI-style ChatCompletion request.
   *
   * @note For each choice (i.e. `n`), a request is defined by a single `prefill()` and multiple
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
    const timeReceived = Date.now();
    // 0. Check model loaded and preprocess inputs
    const [selectedModelId, selectedPipeline, selectedChatConfig] =
      this.getLLMStates("ChatCompletionRequest", request.model);
    const selectedModelType =
      this.loadedModelIdToModelType.get(selectedModelId);
    API.postInitAndCheckFieldsChatCompletion(
      request,
      selectedModelId,
      selectedModelType!,
    );
    const genConfig: GenerationConfig = {
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      repetition_penalty: request.repetition_penalty,
      max_tokens: request.max_tokens,
      stop: request.stop,
      top_p: request.top_p,
      temperature: request.temperature,
      logit_bias: request.logit_bias,
      logprobs: request.logprobs,
      top_logprobs: request.top_logprobs,
      response_format: request.response_format,
      ignore_eos: request.ignore_eos,
      enable_thinking: request.extra_body?.enable_thinking,
      enable_latency_breakdown: request.extra_body?.enable_latency_breakdown,
    };

    // 0.5 Block wait until this pipeline finishes all previous requests
    const lock = this.loadedModelIdToLock.get(selectedModelId)!;
    await lock.acquire();

    // 1. If request is streaming, return an AsyncIterable (an iterable version of `_generate()`)
    if (request.stream) {
      return this.asyncGenerate(
        request,
        selectedModelId,
        selectedPipeline,
        selectedChatConfig,
        genConfig,
        timeReceived,
      );
    }

    // Big try-finally to release lock in case of errors
    try {
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(request.seed);
      }

      // 2. If request is non-streaming, directly reuse `_generate()`
      const n = request.n ? request.n : 1;
      const choices: Array<ChatCompletion.Choice> = [];
      let completion_tokens = 0;
      let prompt_tokens = 0;
      let prefill_time = 0;
      let decode_time = 0;
      let grammar_init_s = 0;
      let grammar_per_token_s = 0;
      for (let i = 0; i < n; i++) {
        let outputMessage: string;
        if (this.interruptSignal) {
          // A single interrupt signal should stop all choices' generations
          selectedPipeline.triggerStop();
          outputMessage = "";
        } else {
          outputMessage = await this._generate(
            request,
            selectedPipeline,
            selectedChatConfig,
            genConfig,
          );
        }
        let finish_reason = selectedPipeline.getFinishReason()!;

        // 3. Post processing for function calling
        const isFunctionCalling =
          request.tools !== undefined && request.tools !== null;
        let tool_calls: Array<ChatCompletionMessageToolCall> | undefined;
        if (
          selectedPipeline.getFinishReason() === "stop" &&
          isFunctionCalling
        ) {
          // If stopped due to length or abort, cannot output return tool_calls field
          finish_reason = "tool_calls";
          tool_calls = getToolCallFromOutputMessage(
            outputMessage,
            /*isStreaming=*/ false,
          );
        }

        choices.push({
          finish_reason: finish_reason,
          index: i,
          logprobs: request.logprobs
            ? ({
                content: selectedPipeline.getTokenLogprobArray(),
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
        completion_tokens += selectedPipeline.getCurRoundDecodingTotalTokens();
        prompt_tokens += selectedPipeline.getCurRoundPrefillTotalTokens();
        prefill_time += selectedPipeline.getCurRoundPrefillTotalTime();
        decode_time += selectedPipeline.getCurRoundDecodingTotalTime();
        grammar_init_s += selectedPipeline.getCurRoundGrammarInitTotalTime();
        grammar_per_token_s +=
          selectedPipeline.getCurRoundGrammarPerTokenTotalTime();
      }
      const usedGrammar =
        "response_format" in request &&
        (request.response_format?.type === "grammar" ||
          request.response_format?.type === "json_object");

      const latencyBreakdown: LatencyBreakdown =
        selectedPipeline.getCurRoundLatencyBreakdown();

      const defaultExtra = {
        e2e_latency_s: (Date.now() - timeReceived) / 1000,
        prefill_tokens_per_s: prompt_tokens / prefill_time,
        decode_tokens_per_s: completion_tokens / decode_time,
        time_to_first_token_s: prefill_time,
        time_per_output_token_s: decode_time / completion_tokens,
        latencyBreakdown: request.extra_body?.enable_latency_breakdown
          ? latencyBreakdown
          : undefined,
      };
      const response: ChatCompletion = {
        id: crypto.randomUUID(),
        choices: choices,
        model: selectedModelId,
        object: "chat.completion",
        created: Date.now(),
        usage: {
          completion_tokens: completion_tokens,
          prompt_tokens: prompt_tokens,
          total_tokens: completion_tokens + prompt_tokens,
          extra: usedGrammar
            ? {
                ...defaultExtra,
                ...{
                  grammar_init_s: grammar_init_s,
                  grammar_per_token_s: grammar_per_token_s / completion_tokens,
                },
              }
            : defaultExtra,
        } as CompletionUsage,
      };

      // Reset seed -- we do not want this seed to affect future requests
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(Date.now());
      }
      return response;
    } finally {
      await lock.release();
    }
  }

  /**
   * Completes a single CompletionCreateParams, a text completion with no chat template.
   *
   * @param request A OpenAI-style Completion request.
   *
   * @note For each choice (i.e. `n`), a request is defined by a single `prefill()` and multiple
   * `decode()`. This is important as it determines the behavior of various fields including `seed`.
   */
  async completion(
    request: CompletionCreateParamsNonStreaming,
  ): Promise<Completion>;
  async completion(
    request: CompletionCreateParamsStreaming,
  ): Promise<AsyncIterable<Completion>>;
  async completion(
    request: CompletionCreateParamsBase,
  ): Promise<AsyncIterable<Completion> | Completion>;
  async completion(
    request: CompletionCreateParams,
  ): Promise<AsyncIterable<Completion> | Completion> {
    const timeReceived = Date.now();

    // 0. Check model loaded and preprocess inputs
    const [selectedModelId, selectedPipeline, selectedChatConfig] =
      this.getLLMStates("CompletionCreateParams", request.model);
    API.postInitAndCheckFieldsCompletion(request, selectedModelId);
    const genConfig: GenerationConfig = {
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      repetition_penalty: request.repetition_penalty,
      max_tokens: request.max_tokens,
      stop: request.stop,
      top_p: request.top_p,
      temperature: request.temperature,
      logit_bias: request.logit_bias,
      logprobs: request.logprobs,
      top_logprobs: request.top_logprobs,
      ignore_eos: request.ignore_eos,
    };

    // 0.5 Block wait until this pipeline finishes all previous requests
    const lock = this.loadedModelIdToLock.get(selectedModelId)!;
    await lock.acquire();

    // 1. If request is streaming, return an AsyncIterable (an iterable version of `_generate()`)
    if (request.stream) {
      return this.asyncGenerate(
        request,
        selectedModelId,
        selectedPipeline,
        selectedChatConfig,
        genConfig,
        timeReceived,
      );
    }

    // Big try-finally to release lock in case of errors
    try {
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(request.seed);
      }

      // 2. If request is non-streaming, directly reuse `_generate()`
      const n = request.n ? request.n : 1;
      const choices: Array<CompletionChoice> = [];
      let completion_tokens = 0;
      let prompt_tokens = 0;
      let prefill_time = 0;
      let decode_time = 0;
      for (let i = 0; i < n; i++) {
        let outputMessage: string;
        if (this.interruptSignal) {
          // A single interrupt signal should stop all choices' generations
          selectedPipeline.triggerStop();
          outputMessage = "";
        } else {
          outputMessage = await this._generate(
            request,
            selectedPipeline,
            selectedChatConfig,
            genConfig,
          );
        }
        const finish_reason = selectedPipeline.getFinishReason()!;

        choices.push({
          finish_reason: finish_reason,
          index: i,
          logprobs: request.logprobs
            ? ({
                content: selectedPipeline.getTokenLogprobArray(),
              } as ChatCompletion.Choice.Logprobs)
            : null,
          text: request.echo ? request.prompt + outputMessage : outputMessage,
        });
        completion_tokens += selectedPipeline.getCurRoundDecodingTotalTokens();
        prompt_tokens += selectedPipeline.getCurRoundPrefillTotalTokens();
        prefill_time += selectedPipeline.getCurRoundPrefillTotalTime();
        decode_time += selectedPipeline.getCurRoundDecodingTotalTime();
      }

      const latencyBreakdown: LatencyBreakdown =
        selectedPipeline.getCurRoundLatencyBreakdown();

      const response: Completion = {
        id: crypto.randomUUID(),
        choices: choices,
        model: selectedModelId,
        object: "text_completion",
        created: Date.now(),
        usage: {
          completion_tokens: completion_tokens,
          prompt_tokens: prompt_tokens,
          total_tokens: completion_tokens + prompt_tokens,
          extra: {
            e2e_latency_s: (Date.now() - timeReceived) / 1000,
            prefill_tokens_per_s: prompt_tokens / prefill_time,
            decode_tokens_per_s: completion_tokens / decode_time,
            time_to_first_token_s: prefill_time,
            time_per_output_token_s: decode_time / completion_tokens,
            latencyBreakdown: request.extra_body?.enable_latency_breakdown
              ? latencyBreakdown
              : undefined,
          },
        } as CompletionUsage,
      };

      // Reset seed -- we do not want this seed to affect future requests
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(Date.now());
      }
      return response;
    } finally {
      await lock.release();
    }
  }

  async embedding(
    request: EmbeddingCreateParams,
  ): Promise<CreateEmbeddingResponse> {
    // 0. Preprocess inputs
    const [selectedModelId, selectedPipeline] = this.getEmbeddingStates(
      "EmbeddingCreateParams",
      request.model,
    );
    API.postInitAndCheckFieldsEmbedding(request, selectedModelId);

    // 0.5 Block wait until this pipeline finishes all previous requests
    const lock = this.loadedModelIdToLock.get(selectedModelId)!;
    await lock.acquire();

    try {
      // 1. Call EmbeddingPipeline to get embeddings
      const embedResult: Array<Array<number>> =
        await selectedPipeline.embedStep(request.input);

      // 2. Prepare response
      const batchSize = embedResult.length;
      const data: Array<Embedding> = [];
      for (let i = 0; i < batchSize; i++) {
        const curEmbedding: Embedding = {
          embedding: embedResult[i],
          index: i,
          object: "embedding",
        };
        data.push(curEmbedding);
      }
      return {
        data: data,
        model: selectedModelId,
        object: "list",
        usage: {
          prompt_tokens: selectedPipeline.getCurRoundEmbedTotalTokens(),
          total_tokens: selectedPipeline.getCurRoundEmbedTotalTokens(),
          extra: {
            prefill_tokens_per_s:
              selectedPipeline.getCurRoundEmbedTokensPerSec(),
          },
        },
      };
    } finally {
      await lock.release();
    }
  }

  //-----------------------------
  // 4. WebGPU info-querying helpers
  //-----------------------------

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

  //---------------------------------------------------------------
  // 5. Helper for querying currently loaded model/pipeline/config.
  // Needed due to possibly multiple loaded models.
  //---------------------------------------------------------------

  private getLLMStates(
    requestName: string,
    modelId?: string | null,
  ): [string, LLMChatPipeline, ChatConfig] {
    return this.getModelStates(requestName, ModelType.LLM, modelId) as [
      string,
      LLMChatPipeline,
      ChatConfig,
    ];
  }

  private getEmbeddingStates(
    requestName: string,
    modelId?: string | null,
  ): [string, EmbeddingPipeline, ChatConfig] {
    return this.getModelStates(requestName, ModelType.embedding, modelId) as [
      string,
      EmbeddingPipeline,
      ChatConfig,
    ];
  }

  /**
   * Return the model, its LLMChatPipeline, and ChatConfig to use. Throws error when unclear which
   * model to load. Ensure all loadedModelIdToXXX maps contain entry for the selected modelId.
   * @param requestName The type of request or API to load the model for. Needed for error throwing.
   * @param modelType The typ of model, determining what type of pipeline to expect.
   * @param modelId Model the user specified to load via the request. Required when multiple
   *   models are loaded
   */
  private getModelStates(
    requestName: string,
    modelType: ModelType,
    modelId?: string | null,
  ): [string, LLMChatPipeline | EmbeddingPipeline, ChatConfig] {
    // 0. Select model based on request.model and loadedModelIds
    const loadedModelIds: string[] = Array.from(
      this.loadedModelIdToPipeline.keys(),
    );
    const selectedModelId: string = getModelIdToUse(
      loadedModelIds,
      modelId,
      requestName,
    );

    // 1. Retrieve pipeline
    const selectedPipeline = this.loadedModelIdToPipeline.get(selectedModelId);
    if (modelType === ModelType.LLM) {
      if (!(selectedPipeline instanceof LLMChatPipeline)) {
        throw new IncorrectPipelineLoadedError(
          selectedModelId,
          "LLMChatPipeline",
          requestName,
        );
      }
    } else {
      // ModelType.Embedding
      if (!(selectedPipeline instanceof EmbeddingPipeline)) {
        throw new IncorrectPipelineLoadedError(
          selectedModelId,
          "EmbeddingPipeline",
          requestName,
        );
      }
      if (
        findModelRecord(selectedModelId, this.appConfig).model_type !==
        ModelType.embedding
      ) {
        throw new EmbeddingUnsupportedModelError(selectedModelId);
      }
    }

    // 2. Retrieve chat config
    const selectedChatConfig =
      this.loadedModelIdToChatConfig.get(selectedModelId);
    if (selectedChatConfig === undefined) {
      throw new Error(
        `InternalError: chat config not registered for ${selectedModelId}.`,
      );
    }

    // 3. Make sure lock is initialized
    if (!this.loadedModelIdToLock.has(selectedModelId)) {
      throw new Error(
        `InternalError: loadedModelIdToLock does not contain ${selectedModelId}`,
      );
    }
    return [selectedModelId, selectedPipeline, selectedChatConfig];
  }

  //--------------------------------------------------------------------
  // 6. External low-level APIs that directly interacts with a pipeline.
  //--------------------------------------------------------------------

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
    modelId?: string,
  ): Promise<number> {
    const [, selectedPipeline] = this.getLLMStates(
      "forwardTokensAndSample",
      modelId,
    );
    return selectedPipeline.forwardTokensAndSample(inputIds, isPrefill);
  }

  /**
   * Get the current generated response.
   *
   * @returns The current output message.
   */
  async getMessage(modelId?: string): Promise<string> {
    const [, selectedPipeline] = this.getLLMStates("getMessage", modelId);
    return selectedPipeline.getMessage();
  }

  async runtimeStatsText(modelId?: string): Promise<string> {
    log.warn(
      "WARNING: `runtimeStatsText()` will soon be deprecated. " +
        "Please use `ChatCompletion.usage` for non-streaming requests, or " +
        "`ChatCompletionChunk.usage` for streaming requests, enabled by `stream_options`. " +
        "The only flow that expects to use `runtimeStatsText()` as of now is `forwardTokensAndSample()`.",
    );
    const [, selectedPipeline] = this.getLLMStates("runtimeStatsText", modelId);
    return selectedPipeline.runtimeStatsText();
  }

  async resetChat(keepStats = false, modelId?: string) {
    try {
      const [, selectedPipeline] = this.getLLMStates("resetChat", modelId);
      selectedPipeline.resetChat(keepStats);
    } catch (error) {
      if (
        error instanceof ModelNotLoadedError ||
        error instanceof SpecifiedModelNotFoundError
      ) {
        // Only allow calling resetChat before pipeline instantiated.
        log.debug(
          "Caught an expected error in resetChat, treating it as no-op. Error: ",
          error,
        );
      } else {
        throw error;
      }
    }
  }

  //-----------------------------------------------
  // 7. Prefill and decode given an LLMChatPipeline
  //-----------------------------------------------

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
   * @param input The OpenAI-style prompt to prefill.
   * @param pipeline The loaded pipeline, hence model, to carry out this prefill.
   * @param chatConfig The chat config to use for this model.
   * @param genConfig Generation config.
   */
  async prefill(
    input: ChatCompletionRequest | CompletionCreateParams,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
  ) {
    // TODO: SPECIFY MODEL TO PERFORM PREFILL, HENCE RETRIEVE CONFIG
    if (chatConfig === undefined) {
      throw new ConfigurationNotInitializedError();
    }
    let input_str: string;
    let input_role_str: string | undefined;
    let lastMsgRole = Role.user;
    if ("messages" in input) {
      // For ChatCompletionRequest, we prepare input using `messages`
      // 1. Get new conversation based on request, determine if we are in multiround chatting
      const oldConv = pipeline.getConversationObject();
      const newConv = getConversationFromChatCompletionRequest(
        input,
        chatConfig,
      );
      if (!compareConversationObject(oldConv, newConv)) {
        // Not the same conversation, so not multiround chatting, reset everything (KV cache, etc.)
        pipeline.resetChat();
        pipeline.setConversation(newConv);
      } else if (newConv.messages.length === 0) {
        // Empty oldConv, and no chat history in newConv, so reset and setConversation
        pipeline.resetChat();
        pipeline.setConversation(newConv);
      } else {
        log.info("Multiround chatting, reuse KVCache.");
      }

      // 2. Treat the last message as the usual input
      const last_msg = input.messages[
        input.messages.length - 1
      ] as ChatCompletionMessageParam;
      input_str = last_msg.content as string;
      input_role_str =
        last_msg.role === "user" && last_msg.name ? last_msg.name : undefined;
      lastMsgRole = last_msg.role === "tool" ? Role.tool : Role.user;
    } else {
      // For CompletionCreateParams, the input is just the prompt
      input_str = input.prompt;
      pipeline.resetChat();
      const newConv = getConversation(
        chatConfig.conv_template,
        chatConfig.conv_config,
        true,
      );
      pipeline.setConversation(newConv);
    }
    return pipeline.prefillStep(
      input_str,
      lastMsgRole,
      input_role_str,
      genConfig,
    );
  }

  /**
   * Run a decode step to decode the next token.
   */
  async decode(pipeline: LLMChatPipeline, genConfig?: GenerationConfig) {
    return pipeline.decodeStep(genConfig);
  }
}
