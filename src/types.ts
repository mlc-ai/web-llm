import { AppConfig, ChatOptions, GenerationConfig } from "./config";
import {
  ChatCompletionRequest,
  ChatCompletionRequestBase,
  ChatCompletionRequestStreaming,
  ChatCompletionRequestNonStreaming,
  ChatCompletion,
  ChatCompletionChunk,
} from "./openai_api_protocols/index";
import * as API from "./openai_api_protocols/apis";

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
 * Callbacks used to report initialization process.
 */
export type GenerateProgressCallback = (
  step: number,
  currentMessage: string,
) => void;

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
   * Called when in `ChatModule.resetChat()`. Can clear internal states.
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
   * @param modelId model_id of the model to load.
   * @param chatOpts Extra options to override chat behavior.
   * @returns A promise when reload finishes.
   * @note This is an async function.
   */
  reload: (modelId: string, chatOpts?: ChatOptions) => Promise<void>;

  /**
   * Generate a response for a given input.
   *
   * @param input The input prompt or a non-streaming ChatCompletionRequest.
   * @param progressCallback Callback that is being called to stream intermediate results.
   * @param streamInterval callback interval to call progresscallback
   * @param genConfig Configuration for this single generation that overrides pre-existing configs.
   * @returns The final result.
   *
   * @note This will be deprecated soon. Please use `engine.chat.completions.create()` instead.
   * For multi-round chatting, see `examples/multi-round-chat` on how to use
   * `engine.chat.completions.create()` to achieve the same effect.
   */
  generate: (
    input: string | ChatCompletionRequestNonStreaming,
    progressCallback?: GenerateProgressCallback,
    streamInterval?: number,
    genConfig?: GenerationConfig,
  ) => Promise<string>;

  /**
   * OpenAI-style API. Generate a chat completion response for the given conversation and configuration.
   *
   * The API is completely functional in behavior. That is, a previous request would not affect
   * the current request's result. Thus, for multi-round chatting, users are responsible for
   * maintaining the chat history. With that being said, as an implicit internal optimization, if we
   * detect that the user is performing multiround chatting, we will preserve the KV cache and only
   * prefill the new tokens.
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
   * @returns A text summarizing the runtime stats.
   * @note This is an async function
   */
  runtimeStatsText: () => Promise<string>;

  /**
   * Interrupt the generate process if it is already running.
   */
  interruptGenerate: () => void;

  /**
   * Explicitly unload the current model and release the related resources.
   */
  unload: () => Promise<void>;

  /**
   * Reset the current chat session by clear all memories.
   * @param keepStats: If True, do not reset the statistics.
   */
  resetChat: (keepStats?: boolean) => Promise<void>;

  /**
   * Get the current generated response.
   *
   * @returns The current output message.
   */
  getMessage: () => Promise<string>;

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
   * @returns Next token sampled.
   * @note This is an async function.
   */
  forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
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
