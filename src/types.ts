import { AppConfig, ChatOptions } from "./config";
import {
  ChatCompletionRequest,
  ChatCompletionRequestBase,
  ChatCompletionRequestStreaming,
  ChatCompletionRequestNonStreaming,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionCreateParams,
  Completion,
  CompletionCreateParamsBase,
  CompletionCreateParamsStreaming,
  CompletionCreateParamsNonStreaming,
  EmbeddingCreateParams,
  CreateEmbeddingResponse,
} from "./openai_api_protocols/index";
import * as API from "./openai_api_protocols/index";

/**
 * Report during intialization.
 */
export interface InitProgressReport {
  progress: number;
  timeElapsed: number;
  text: string;
}

/**
 * Callbacks used to report initialization process.
 */
export type InitProgressCallback = (report: InitProgressReport) => void;

/**
 * A stateful logitProcessor used to post-process logits after forwarding the input and before
 * sampling the next token. If used with `GenerationConfig.logit_bias`, logit_bias is applied after
 * `processLogits()` is called.
 */
export interface LogitProcessor {
  /**
   * Process logits after forward() and before sampling implicitly, happens on the CPU.
   * @param logits The logits right after forward().
   * Returns the processed logits.
   */
  processLogits: (logits: Float32Array) => Float32Array;

  /**
   * Use the sampled token to update the LogitProcessor's internal state. Called implicitly
   * right after the next token is sampled/committed.
   * @param token Token sampled from the processed logits.
   */
  processSampledToken: (token: number) => void;

  /**
   * Called when in `MLCEngine.resetChat()`. Can clear internal states.
   */
  resetState: () => void;
}

/**
 * Common interface of MLCEngine that UI can interact with
 */
export interface MLCEngineInterface {
  /**
   * An object that exposes chat-related APIs.
   */
  chat: API.Chat;

  /**
   * An object that exposes text completion APIs.
   */
  completions: API.Completions;

  /**
   * An object that exposes embeddings APIs.
   */
  embeddings: API.Embeddings;

  /**
   * Set an initialization progress callback function
   * which reports the progress of model loading.
   *
   * This function can be useful to implement an UI that
   * update as we loading the model.
   *
   * @param initProgressCallback The callback function
   */
  setInitProgressCallback: (initProgressCallback: InitProgressCallback) => void;

  /**
   * @returns The current initialization progress callback function.
   */
  getInitProgressCallback: () => InitProgressCallback | undefined;

  /**
   * Setter for the engine's appConfig.
   */
  setAppConfig: (appConfig: AppConfig) => void;

  /**
   * Reload the chat with a new model.
   *
   * @param modelId model_id of the model to load, either string or string[]. When multiple models
   *   are provided, we load all models sequentially. Each modelId needs to either be in
   *   `webllm.prebuiltAppConfig`, or in `engineConfig.appConfig`.
   * @param chatOpts Extra options to optionally override the `mlc-chat-config.json` of `modelId`.
   *   The size of which needs to match that of `modelId`; chatOpts[i] will be used for modelId[i].
   * @returns A promise when reload finishes.
   * @throws Throws error when device lost (mostly due to OOM); users should re-call reload(),
   *   potentially with a smaller model or smaller context window size.
   * @note This is an async function.
   */
  reload: (
    modelId: string | string[],
    chatOpts?: ChatOptions | ChatOptions[],
  ) => Promise<void>;

  /**
   * OpenAI-style API. Generate a chat completion response for the given conversation and
   * configuration. Use `engine.chat.completions.create()` to invoke this API.
   *
   * @param request A OpenAI-style ChatCompletion request.
   *
   * @note The API is completely functional in behavior. That is, a previous request would not
   * affect the current request's result. Thus, for multi-round chatting, users are responsible for
   * maintaining the chat history. With that being said, as an implicit internal optimization, if we
   * detect that the user is performing multi-round chatting, we will preserve the KV cache and only
   * prefill the new tokens.
   * @note For requests sent to the same modelId, will block until all previous requests finish.
   * @note For more, see https://platform.openai.com/docs/api-reference/chat
   */
  chatCompletion(
    request: ChatCompletionRequestNonStreaming,
  ): Promise<ChatCompletion>;
  chatCompletion(
    request: ChatCompletionRequestStreaming,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  chatCompletion(
    request: ChatCompletionRequestBase,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;
  chatCompletion(
    request: ChatCompletionRequest,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;

  /**
   * OpenAI-style API. Completes a CompletionCreateParams, a text completion with no chat template.
   * Use `engine.completions.create()` to invoke this API.
   *
   * @param request An OpenAI-style Completion request.
   *
   * @note For requests sent to the same modelId, will block until all previous requests finish.
   * @note For more, see https://platform.openai.com/docs/api-reference/completions
   */
  completion(request: CompletionCreateParamsNonStreaming): Promise<Completion>;
  completion(
    request: CompletionCreateParamsStreaming,
  ): Promise<AsyncIterable<Completion>>;
  completion(
    request: CompletionCreateParamsBase,
  ): Promise<AsyncIterable<Completion> | Completion>;
  completion(
    request: CompletionCreateParams,
  ): Promise<AsyncIterable<Completion> | Completion>;

  /**
   * OpenAI-style API. Creates an embedding vector representing the input text.
   * Use `engine.embeddings.create()` to invoke this API.
   *
   * @param request An OpenAI-style Embeddings request.
   *
   * @note For requests sent to the same modelId, will block until all previous requests finish.
   * @note For more, see https://platform.openai.com/docs/api-reference/embeddings/create
   */
  embedding(request: EmbeddingCreateParams): Promise<CreateEmbeddingResponse>;

  /**
   * @returns A text summarizing the runtime stats.
   * @param modelId Only required when multiple models are loaded.
   * @note This is an async function
   */
  runtimeStatsText: (modelId?: string) => Promise<string>;

  /**
   * Interrupt the generate process if it is already running.
   */
  interruptGenerate: () => void;

  /**
   * Explicitly unload the currently loaded model(s) and release the related resources. Waits until
   * the webgpu device finishes all submitted work and destroys itself.
   * @note This is an asynchronous function.
   */
  unload: () => Promise<void>;

  /**
   * Reset the current chat session by clear all memories.
   * @param keepStats: If True, do not reset the statistics.
   * @param modelId Only required when multiple models are loaded.
   */
  resetChat: (keepStats?: boolean, modelId?: string) => Promise<void>;

  /**
   * Get the current generated response.
   * @param modelId Only required when multiple models are loaded.
   * @returns The current output message.
   */
  getMessage: (modelId?: string) => Promise<string>;

  /**
   * Returns the device's maxStorageBufferBindingSize, can be used to guess whether the device
   * has limited resources like an Android phone.
   */
  getMaxStorageBufferBindingSize(): Promise<number>;

  /**
   * Returns the device's gpu vendor (e.g. arm, qualcomm, apple) if available. Otherwise return
   * an empty string.
   */
  getGPUVendor(): Promise<string>;

  /**
   * Forward the given input tokens to the model, then sample the next token.
   *
   * This function has side effects as the model will update its KV cache.
   *
   * @param inputIds The input tokens.
   * @param isPrefill True if prefill, false if decode; only used for statistics.
   * @param modelId Only required when multiple models are loaded.
   * @returns Next token sampled.
   * @note This is an async function.
   */
  forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
    modelId?: string,
  ): Promise<number>;

  /**
   * Set MLCEngine logging output level
   *
   * @param logLevel The new log level
   */
  setLogLevel(logLevel: LogLevel): void;
}

export const LOG_LEVELS = {
  TRACE: 0,
  DEBUG: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
  SILENT: 5,
};
export type LogLevel = keyof typeof LOG_LEVELS;

export type LatencyBreakdown = {
  logitProcessorTime: number[];
  logitBiasTime: number[];
  penaltyTime: number[];
  sampleTime: number[];
  totalTime: number[];
  grammarBitmaskTime: number[];
};
