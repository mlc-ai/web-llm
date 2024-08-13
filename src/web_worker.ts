import { AppConfig, ChatOptions, MLCEngineConfig } from "./config";
import {
  MLCEngineInterface,
  InitProgressCallback,
  InitProgressReport,
  LogLevel,
  LogitProcessor,
} from "./types";
import {
  ChatCompletionRequest,
  ChatCompletionRequestBase,
  ChatCompletionRequestStreaming,
  ChatCompletionRequestNonStreaming,
  ChatCompletion,
  ChatCompletionChunk,
  Completion,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  CompletionCreateParamsBase,
  CompletionCreateParams,
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from "./openai_api_protocols/index";
import * as API from "./openai_api_protocols/index";
import {
  MessageContent,
  ReloadParams,
  ForwardTokensAndSampleParams,
  ChatCompletionNonStreamingParams,
  ChatCompletionStreamInitParams,
  ResetChatParams,
  WorkerResponse,
  WorkerRequest,
  CompletionNonStreamingParams,
  EmbeddingParams,
  CompletionStreamInitParams,
  GetMessageParams,
  RuntimeStatsTextParams,
  CompletionStreamNextChunkParams,
} from "./message";
import log from "loglevel";
import { MLCEngine } from "./engine";
import {
  UnknownMessageKindError,
  WorkerEngineModelNotLoadedError,
} from "./error";
import { areArraysEqual } from "./utils";
import { getModelIdToUse } from "./support";

/**
 * Worker handler that can be used in a WebWorker
 *
 * @example
 *
 * // setup a chat worker handler that routes
 * // requests to the chat
 * const engine = new MLCEngine();
 * cont handler = new WebWorkerMLCEngineHandler(engine);
 * onmessage = handler.onmessage;
 */
export class WebWorkerMLCEngineHandler {
  /**
   * The modelId and chatOpts that the underlying engine (backend) is currently loaded with.
   * An engine can be loaded with multiple models, so modelId and chatOpts are lists.
   *
   * TODO(webllm-team): This is always in-sync with `this.engine` unless device is lost due to
   * unexpected reason. Therefore, we should get it from `this.engine` directly and make handler
   * stateless. Besides, consider if we should add appConfig, or use engine's API to find the
   * corresponding model record rather than relying on just the modelId.
   */
  modelId?: string[];
  chatOpts?: ChatOptions[];

  public engine: MLCEngine;
  /** ChatCompletion and Completion share the same chunk generator. Each loaded model has its own. */
  protected loadedModelIdToAsyncGenerator: Map<
    string,
    AsyncGenerator<ChatCompletionChunk | Completion, void, void>
  >;

  /**
   * @param engine A concrete implementation of MLCEngineInterface
   */
  constructor() {
    this.engine = new MLCEngine();
    this.loadedModelIdToAsyncGenerator = new Map<
      string,
      AsyncGenerator<ChatCompletionChunk | Completion, void, void>
    >();
    this.engine.setInitProgressCallback((report: InitProgressReport) => {
      const msg: WorkerResponse = {
        kind: "initProgressCallback",
        uuid: "",
        content: report,
      };
      this.postMessage(msg);
    });
  }

  postMessage(msg: any) {
    // Use Web Worker DOM Message API
    postMessage(msg);
  }

  setLogitProcessorRegistry(
    logitProcessorRegistry?: Map<string, LogitProcessor>,
  ) {
    this.engine.setLogitProcessorRegistry(logitProcessorRegistry);
  }

  async handleTask<T extends MessageContent>(
    uuid: string,
    task: () => Promise<T>,
  ) {
    try {
      const res = await task();
      const msg: WorkerResponse = {
        kind: "return",
        uuid: uuid,
        content: res,
      };
      this.postMessage(msg);
    } catch (err) {
      const errStr = (err as object).toString();
      const msg: WorkerResponse = {
        kind: "throw",
        uuid: uuid,
        content: errStr,
      };
      this.postMessage(msg);
    }
  }

  onmessage(
    event: any,
    onComplete?: (value: any) => void,
    onError?: () => void,
  ) {
    let msg: WorkerRequest;
    if (event instanceof MessageEvent) {
      msg = event.data as WorkerRequest;
    } else {
      msg = event as WorkerRequest;
    }
    switch (msg.kind) {
      case "reload": {
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as ReloadParams;
          await this.engine.reload(params.modelId, params.chatOpts);
          this.modelId = params.modelId;
          this.chatOpts = params.chatOpts;
          onComplete?.(null);
          return null;
        });
        return;
      }
      case "forwardTokensAndSample": {
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as ForwardTokensAndSampleParams;
          const res = await this.engine.forwardTokensAndSample(
            params.inputIds,
            params.isPrefill,
            params.modelId,
          );
          onComplete?.(res);
          return res;
        });
        return;
      }
      // For engine.chat.completions.create()
      case "chatCompletionNonStreaming": {
        // Directly return the ChatCompletion response
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as ChatCompletionNonStreamingParams;
          await this.reloadIfUnmatched(params.modelId, params.chatOpts);
          const res = await this.engine.chatCompletion(params.request);
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "chatCompletionStreamInit": {
        // One-time set up that instantiates the chunk generator in worker
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as ChatCompletionStreamInitParams;
          // Also ensures params.selectedModelId will match what this.engine selects
          await this.reloadIfUnmatched(params.modelId, params.chatOpts);
          // Register new async generator for this new request of the model
          const curGenerator = (await this.engine.chatCompletion(
            params.request,
          )) as AsyncGenerator<ChatCompletionChunk, void, void>;
          this.loadedModelIdToAsyncGenerator.set(
            params.selectedModelId,
            curGenerator,
          );
          onComplete?.(null);
          return null;
        });
        return;
      }
      // For engine.completions.create()
      case "completionNonStreaming": {
        // Directly return the ChatCompletion response
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as CompletionNonStreamingParams;
          await this.reloadIfUnmatched(params.modelId, params.chatOpts);
          const res = await this.engine.completion(params.request);
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "completionStreamInit": {
        // One-time set up that instantiates the chunk generator in worker
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as CompletionStreamInitParams;
          // Also ensures params.selectedModelId will match what this.engine selects
          await this.reloadIfUnmatched(params.modelId, params.chatOpts);
          // Register new async generator for this new request of the model
          const curGenerator = (await this.engine.completion(
            params.request,
          )) as AsyncGenerator<Completion, void, void>;
          this.loadedModelIdToAsyncGenerator.set(
            params.selectedModelId,
            curGenerator,
          );
          onComplete?.(null);
          return null;
        });
        return;
      }
      // Shared by engine.chat.completions.create() and engine.completions.create()
      case "completionStreamNextChunk": {
        // Note: ChatCompletion and Completion share the same chunk generator.
        // For any subsequent request, we return whatever `next()` yields
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as CompletionStreamNextChunkParams;
          const curGenerator = this.loadedModelIdToAsyncGenerator.get(
            params.selectedModelId,
          );
          if (curGenerator === undefined) {
            throw Error(
              "InternalError: Chunk generator in worker should be instantiated by now.",
            );
          }
          // Yield the next chunk
          const { value } = await curGenerator.next();
          onComplete?.(value);
          return value;
        });
        return;
      }
      // For engine.embeddings.create()
      case "embedding": {
        // Directly return the Embeddings response
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as EmbeddingParams;
          await this.reloadIfUnmatched(params.modelId, params.chatOpts);
          const res = await this.engine.embedding(params.request);
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "runtimeStatsText": {
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as RuntimeStatsTextParams;
          const res = await this.engine.runtimeStatsText(params.modelId);
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "interruptGenerate": {
        this.handleTask(msg.uuid, async () => {
          this.engine.interruptGenerate();
          onComplete?.(null);
          return null;
        });
        return;
      }
      case "unload": {
        // Unset modelId and chatOpts since backend unloads the model
        this.handleTask(msg.uuid, async () => {
          await this.engine.unload();
          this.modelId = undefined;
          this.chatOpts = undefined;
          // This may not be cleaned properly when one asyncGenerator finishes.
          // We only clear at unload(), which may not be called upon reload().
          // However, service_worker may skip reload(). Will leave as is for now.
          this.loadedModelIdToAsyncGenerator.clear();
          onComplete?.(null);
          return null;
        });
        return;
      }
      case "resetChat": {
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as ResetChatParams;
          await this.engine.resetChat(params.keepStats, params.modelId);
          onComplete?.(null);
          return null;
        });
        return;
      }
      case "getMaxStorageBufferBindingSize": {
        this.handleTask(msg.uuid, async () => {
          const res = await this.engine.getMaxStorageBufferBindingSize();
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "getGPUVendor": {
        this.handleTask(msg.uuid, async () => {
          const res = await this.engine.getGPUVendor();
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "getMessage": {
        this.handleTask(msg.uuid, async () => {
          const params = msg.content as GetMessageParams;
          const res = await this.engine.getMessage(params.modelId);
          onComplete?.(res);
          return res;
        });
        return;
      }
      case "setLogLevel": {
        const logLevel = msg.content as LogLevel;
        this.engine.setLogLevel(logLevel);
        log.setLevel(logLevel);
        onComplete?.(null);
        return;
      }
      case "setAppConfig": {
        const appConfig = msg.content as AppConfig;
        this.engine.setAppConfig(appConfig);
        onComplete?.(null);
        return;
      }
      case "customRequest": {
        onComplete?.(null);
        return;
      }
      default: {
        if (msg.kind && msg.content) {
          onError?.();
          throw new UnknownMessageKindError(msg.kind, msg.content);
        } else {
          // Ignore irrelavent events
          onComplete?.(null);
        }
      }
    }
  }

  /** Check whether frontend expectation matches with backend (modelId and chatOpts). If not (due
   * to possibly killed service worker), we reload here.
   * For more, see https://github.com/mlc-ai/web-llm/pull/533
   */
  async reloadIfUnmatched(
    expectedModelId: string[],
    expectedChatOpts?: ChatOptions[],
  ) {
    // TODO: should we also check expectedChatOpts here?
    if (!areArraysEqual(this.modelId, expectedModelId)) {
      log.warn(
        "WebWorkerMLCEngine expects model is loaded in WebWorkerMLCEngineHandler, " +
          "but it is not. This may due to web/service worker is unexpectedly killed.\n" +
          "Reloading engine in WebWorkerMLCEngineHandler.",
      );
      await this.engine.reload(expectedModelId, expectedChatOpts);
    }
  }
}

export interface ChatWorker {
  onmessage: any;
  postMessage: (message: any) => void;
}

/**
 * Creates `WebWorkerMLCEngine`, a client that holds the same interface as `MLCEngine`.
 *
 * Equivalent to `new webllm.WebWorkerMLCEngine(worker).reload(...)`.
 *
 * @param worker The worker that holds the actual MLCEngine, initialized with `new Worker()`.
 * @param modelId model_id of the model to load, either string or string[]. When multiple models
 *   are provided, we load all models sequentially. Each modelId needs to either be in
 *   `webllm.prebuiltAppConfig`, or in `engineCOnfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig` for more.
 * @param chatOpts Extra options to optionally override the `mlc-chat-config.json` of `modelId`.
 *   The size of which needs to match that of `modelId`; chatOpts[i] will be used for modelId[i].
 * @returns An initialized `WebLLM.WebWorkerMLCEngine` with `modelId` loaded.
 *
 * @note engineConfig.logitProcessorRegistry is ignored for `CreateWebWorkMLCEngine()`.
 */
export async function CreateWebWorkerMLCEngine(
  worker: any,
  modelId: string | string[],
  engineConfig?: MLCEngineConfig,
  chatOpts?: ChatOptions | ChatOptions[],
): Promise<WebWorkerMLCEngine> {
  const webWorkerMLCEngine = new WebWorkerMLCEngine(worker, engineConfig);
  await webWorkerMLCEngine.reload(modelId, chatOpts);
  return webWorkerMLCEngine;
}

/**
 * A client of MLCEngine that exposes the same interface
 *
 * @example
 *
 * const chat = new webllm.WebWorkerMLCEngine(new Worker(
 *   new URL('./worker.ts', import.meta.url),
 *   {type: 'module'}
 * ));
 */
export class WebWorkerMLCEngine implements MLCEngineInterface {
  public worker: ChatWorker;
  /** For chat.completions.create() */
  public chat: API.Chat;
  /** For completions.create() */
  public completions: API.Completions;
  /** For embeddings.create() */
  public embeddings: API.Embeddings;

  /**
   * The modelId and chatOpts that the frontend expects the backend engine is currently loaded
   * with. Needed for service worker. It is the backend and handler's job to match up with the
   * expectation despite the web/service worker possibly being killed.
   * Since an engine can load multiple models, both modelId and chatOpts are lists.
   */
  modelId?: string[];
  chatOpts?: ChatOptions[];

  private initProgressCallback?: InitProgressCallback;
  private pendingPromise = new Map<string, (msg: WorkerResponse) => void>();

  constructor(worker: ChatWorker, engineConfig?: MLCEngineConfig) {
    this.worker = worker;
    worker.onmessage = (event: any) => {
      this.onmessage.bind(this)(event);
    };

    if (engineConfig?.appConfig) {
      this.setAppConfig(engineConfig?.appConfig);
    }
    if (engineConfig?.logLevel) {
      this.setLogLevel(engineConfig?.logLevel);
    }
    this.setInitProgressCallback(engineConfig?.initProgressCallback);
    if (engineConfig?.logitProcessorRegistry) {
      if (engineConfig?.logitProcessorRegistry) {
        log.warn(
          "Warning: The `logitProcessorRegistry` property in `engineConfig` will be ignored when using the WebWorkerMLCEngine constructor. To set `logitProcessorRegistry`, use the engine constructor within the worker script instead.",
        );
      }
    }

    this.chat = new API.Chat(this);
    this.completions = new API.Completions(this);
    this.embeddings = new API.Embeddings(this);
  }

  setInitProgressCallback(initProgressCallback?: InitProgressCallback) {
    this.initProgressCallback = initProgressCallback;
  }

  getInitProgressCallback(): InitProgressCallback | undefined {
    return this.initProgressCallback;
  }

  setAppConfig(appConfig: AppConfig) {
    const msg: WorkerRequest = {
      kind: "setAppConfig",
      uuid: crypto.randomUUID(),
      content: appConfig,
    };
    this.worker.postMessage(msg);
  }

  setLogLevel(logLevel: LogLevel) {
    log.setLevel(logLevel);
    const msg: WorkerRequest = {
      kind: "setLogLevel",
      uuid: crypto.randomUUID(),
      content: logLevel,
    };
    this.worker.postMessage(msg);
  }

  protected getPromise<T extends MessageContent>(
    msg: WorkerRequest,
  ): Promise<T> {
    const uuid = msg.uuid;
    const executor = (
      resolve: (arg: T) => void,
      reject: (arg: any) => void,
    ) => {
      const cb = (msg: WorkerResponse) => {
        if (msg.kind == "return") {
          resolve(msg.content as T);
        } else {
          if (msg.kind != "throw") {
            reject("Uknown msg kind " + msg.kind);
          } else {
            reject(msg.content);
          }
        }
      };
      this.pendingPromise.set(uuid, cb);
    };
    const promise = new Promise<T>(executor);
    this.worker.postMessage(msg);
    return promise;
  }

  async reload(
    modelId: string | string[],
    chatOpts?: ChatOptions | ChatOptions[],
  ): Promise<void> {
    // Always convert modelId and chatOpts to lists internally for ease of manipulation
    if (!Array.isArray(modelId)) {
      modelId = [modelId];
    }
    if (chatOpts !== undefined && !Array.isArray(chatOpts)) {
      chatOpts = [chatOpts];
    }

    const msg: WorkerRequest = {
      kind: "reload",
      uuid: crypto.randomUUID(),
      content: {
        modelId: modelId,
        chatOpts: chatOpts,
      },
    };
    await this.getPromise<null>(msg);
    this.modelId = modelId;
    this.chatOpts = chatOpts;
  }

  async getMaxStorageBufferBindingSize(): Promise<number> {
    const msg: WorkerRequest = {
      kind: "getMaxStorageBufferBindingSize",
      uuid: crypto.randomUUID(),
      content: null,
    };
    return await this.getPromise<number>(msg);
  }

  async getGPUVendor(): Promise<string> {
    const msg: WorkerRequest = {
      kind: "getGPUVendor",
      uuid: crypto.randomUUID(),
      content: null,
    };
    return await this.getPromise<string>(msg);
  }

  async getMessage(modelId?: string): Promise<string> {
    const msg: WorkerRequest = {
      kind: "getMessage",
      uuid: crypto.randomUUID(),
      content: {
        modelId: modelId,
      },
    };
    return await this.getPromise<string>(msg);
  }

  async runtimeStatsText(modelId?: string): Promise<string> {
    const msg: WorkerRequest = {
      kind: "runtimeStatsText",
      uuid: crypto.randomUUID(),
      content: {
        modelId: modelId,
      },
    };
    return await this.getPromise<string>(msg);
  }

  interruptGenerate(): void {
    const msg: WorkerRequest = {
      kind: "interruptGenerate",
      uuid: crypto.randomUUID(),
      content: null,
    };
    this.getPromise<null>(msg);
  }

  async unload(): Promise<void> {
    const msg: WorkerRequest = {
      kind: "unload",
      uuid: crypto.randomUUID(),
      content: null,
    };
    await this.getPromise<null>(msg);
    this.modelId = undefined;
    this.chatOpts = undefined;
  }

  async resetChat(keepStats = false, modelId?: string): Promise<void> {
    const msg: WorkerRequest = {
      kind: "resetChat",
      uuid: crypto.randomUUID(),
      content: {
        keepStats: keepStats,
        modelId: modelId,
      },
    };
    await this.getPromise<null>(msg);
  }

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
    modelId?: string,
  ): Promise<number> {
    const msg: WorkerRequest = {
      kind: "forwardTokensAndSample",
      uuid: crypto.randomUUID(),
      content: {
        inputIds: inputIds,
        isPrefill: isPrefill,
        modelId: modelId,
      },
    };
    return await this.getPromise<number>(msg);
  }

  /**
   * Every time the generator is called, we post a message to the worker asking it to
   * decode one step, and we expect to receive a message of `ChatCompletionChunk` from
   * the worker which we yield. The last message is `void`, meaning the generator has nothing
   * to yield anymore.
   *
   * @param selectedModelId: The model of whose async generator to call next() to get next chunk.
   *   Needed because an engine can load multiple models.
   *
   * @note ChatCompletion and Completion share the same chunk generator.
   */
  async *asyncGenerate(
    selectedModelId: string,
  ): AsyncGenerator<ChatCompletionChunk | Completion, void, void> {
    // Every time it gets called, sends message to worker, asking for the next chunk
    while (true) {
      const msg: WorkerRequest = {
        kind: "completionStreamNextChunk",
        uuid: crypto.randomUUID(),
        content: {
          selectedModelId: selectedModelId,
        } as CompletionStreamNextChunkParams,
      };
      const ret = await this.getPromise<ChatCompletionChunk>(msg);
      // If the worker's generator reached the end, it would return a `void`
      if (typeof ret !== "object") {
        break;
      }
      yield ret;
    }
  }

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
    if (this.modelId === undefined) {
      throw new WorkerEngineModelNotLoadedError(this.constructor.name);
    }
    // Needed for the streaming case. Consolidate model id to specify
    // which model's asyncGenerator to instantiate or call next() on.
    // Since handler can maintain multiple generators concurrently
    const selectedModelId = getModelIdToUse(
      this.modelId ? this.modelId : [],
      request.model,
      "ChatCompletionRequest",
    );

    if (request.stream) {
      // First let worker instantiate a generator
      const msg: WorkerRequest = {
        kind: "chatCompletionStreamInit",
        uuid: crypto.randomUUID(),
        content: {
          request: request,
          selectedModelId: selectedModelId,
          modelId: this.modelId,
          chatOpts: this.chatOpts,
        },
      };
      await this.getPromise<null>(msg);

      // Then return an async chunk generator that resides on the client side
      return this.asyncGenerate(selectedModelId) as AsyncGenerator<
        ChatCompletionChunk,
        void,
        void
      >;
    }

    // Non streaming case is more straightforward
    const msg: WorkerRequest = {
      kind: "chatCompletionNonStreaming",
      uuid: crypto.randomUUID(),
      content: {
        request: request,
        modelId: this.modelId,
        chatOpts: this.chatOpts,
      },
    };
    return await this.getPromise<ChatCompletion>(msg);
  }

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
    if (this.modelId === undefined) {
      throw new WorkerEngineModelNotLoadedError(this.constructor.name);
    }
    // Needed for the streaming case. Consolidate model id to specify
    // which model's asyncGenerator to instantiate or call next() on.
    // Since handler can maintain multiple generators concurrently
    const selectedModelId = getModelIdToUse(
      this.modelId ? this.modelId : [],
      request.model,
      "CompletionCreateParams",
    );

    if (request.stream) {
      // First let worker instantiate a generator
      const msg: WorkerRequest = {
        kind: "completionStreamInit",
        uuid: crypto.randomUUID(),
        content: {
          request: request,
          selectedModelId: selectedModelId,
          modelId: this.modelId,
          chatOpts: this.chatOpts,
        },
      };
      await this.getPromise<null>(msg);

      // Then return an async chunk generator that resides on the client side
      return this.asyncGenerate(selectedModelId) as AsyncGenerator<
        Completion,
        void,
        void
      >;
    }

    // Non streaming case is more straightforward
    const msg: WorkerRequest = {
      kind: "completionNonStreaming",
      uuid: crypto.randomUUID(),
      content: {
        request: request,
        modelId: this.modelId,
        chatOpts: this.chatOpts,
      },
    };
    return await this.getPromise<Completion>(msg);
  }

  async embedding(
    request: EmbeddingCreateParams,
  ): Promise<CreateEmbeddingResponse> {
    if (this.modelId === undefined) {
      throw new WorkerEngineModelNotLoadedError(this.constructor.name);
    }
    const msg: WorkerRequest = {
      kind: "embedding",
      uuid: crypto.randomUUID(),
      content: {
        request: request,
        modelId: this.modelId,
        chatOpts: this.chatOpts,
      },
    };
    return await this.getPromise<CreateEmbeddingResponse>(msg);
  }

  onmessage(event: any) {
    let msg: WorkerResponse;
    if (event instanceof MessageEvent) {
      msg = event.data as WorkerResponse;
    } else {
      msg = event as WorkerResponse;
    }
    switch (msg.kind) {
      case "initProgressCallback": {
        if (this.initProgressCallback !== undefined) {
          this.initProgressCallback(msg.content as InitProgressReport);
        }
        return;
      }
      case "return": {
        const cb = this.pendingPromise.get(msg.uuid);
        if (cb === undefined) {
          throw Error("return from a unknown uuid msg=" + msg.uuid);
        }
        this.pendingPromise.delete(msg.uuid);
        cb(msg);
        return;
      }
      case "throw": {
        const cb = this.pendingPromise.get(msg.uuid);
        if (cb === undefined) {
          throw Error("return from a unknown uuid, msg=" + msg);
        }
        this.pendingPromise.delete(msg.uuid);
        cb(msg);
        return;
      }
      default: {
        const unknownMsg = msg as any;
        throw new UnknownMessageKindError(unknownMsg.kind, unknownMsg.content);
      }
    }
  }
}
