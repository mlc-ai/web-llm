import { AppConfig, ChatOptions, GenerationConfig } from "./config";
import { InitProgressReport, LogLevel } from "./types";
import {
  ChatCompletionRequestStreaming,
  ChatCompletionRequestNonStreaming,
  ChatCompletion,
  ChatCompletionChunk,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  Completion,
  EmbeddingCreateParams,
  CreateEmbeddingResponse,
} from "./openai_api_protocols/index";

/**
 * Message kind used by worker
 */
type RequestKind =
  | "reload"
  | "runtimeStatsText"
  | "interruptGenerate"
  | "unload"
  | "resetChat"
  | "getMaxStorageBufferBindingSize"
  | "getGPUVendor"
  | "forwardTokensAndSample"
  | "chatCompletionNonStreaming"
  | "completionNonStreaming"
  | "embedding"
  | "getMessage"
  | "chatCompletionStreamInit"
  | "completionStreamInit"
  | "completionStreamNextChunk"
  | "customRequest"
  | "keepAlive"
  | "setLogLevel"
  | "setAppConfig";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
type ResponseKind = "return" | "throw" | "initProgressCallback";

export interface ReloadParams {
  modelId: string;
  chatOpts?: ChatOptions;
}
export interface ResetChatParams {
  keepStats: boolean;
}
export interface ForwardTokensAndSampleParams {
  inputIds: Array<number>;
  isPrefill: boolean;
}
export interface ChatCompletionNonStreamingParams {
  request: ChatCompletionRequestNonStreaming;
  // The model and chatOpts that the frontend engine expects the backend to be loaded with.
  // If not loaded due to service worker unexpectedly killed, handler will call reload().
  // TODO(webllm-team): should add appConfig here as well.
  modelId: string;
  chatOpts: ChatOptions;
}
export interface ChatCompletionStreamInitParams {
  request: ChatCompletionRequestStreaming;
  // The model and chatOpts that the frontend engine expects the backend to be loaded with.
  // If not loaded due to service worker unexpectedly killed, handler will call reload().
  // TODO(webllm-team): should add appConfig here as well.
  modelId: string;
  chatOpts: ChatOptions;
}
export interface CompletionNonStreamingParams {
  request: CompletionCreateParamsNonStreaming;
  // The model and chatOpts that the frontend engine expects the backend to be loaded with.
  // If not loaded due to service worker unexpectedly killed, handler will call reload().
  // TODO(webllm-team): should add appConfig here as well.
  modelId: string;
  chatOpts: ChatOptions;
}
export interface CompletionStreamInitParams {
  request: CompletionCreateParamsStreaming;
  // The model and chatOpts that the frontend engine expects the backend to be loaded with.
  // If not loaded due to service worker unexpectedly killed, handler will call reload().
  // TODO(webllm-team): should add appConfig here as well.
  modelId: string;
  chatOpts: ChatOptions;
}
export interface EmbeddingParams {
  request: EmbeddingCreateParams;
  // The model and chatOpts that the frontend engine expects the backend to be loaded with.
  // If not loaded due to service worker unexpectedly killed, handler will call reload().
  // TODO(webllm-team): should add appConfig here as well.
  modelId: string;
  chatOpts: ChatOptions;
}

export interface CustomRequestParams {
  requestName: string;
  requestMessage: string;
}
export type MessageContent =
  | ReloadParams
  | ResetChatParams
  | ForwardTokensAndSampleParams
  | ChatCompletionNonStreamingParams
  | ChatCompletionStreamInitParams
  | CompletionNonStreamingParams
  | CompletionStreamInitParams
  | EmbeddingParams
  | CustomRequestParams
  | InitProgressReport
  | LogLevel
  | string
  | null
  | number
  | ChatCompletion
  | ChatCompletionChunk
  | CreateEmbeddingResponse
  | Completion
  | AppConfig
  | void;
/**
 * The message used in exchange between worker
 * and the main thread.
 */

export type WorkerRequest = {
  kind: RequestKind;
  uuid: string;
  content: MessageContent;
};

type HeartbeatWorkerResponse = {
  kind: "heartbeat";
  uuid: string;
};

type OneTimeWorkerResponse = {
  kind: "return" | "throw";
  uuid: string;
  content: MessageContent;
};

type InitProgressWorkerResponse = {
  kind: "initProgressCallback";
  uuid: string;
  content: InitProgressReport;
};

export type WorkerResponse =
  | OneTimeWorkerResponse
  | InitProgressWorkerResponse
  | HeartbeatWorkerResponse;
