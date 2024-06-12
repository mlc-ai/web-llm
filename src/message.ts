import { AppConfig, ChatOptions, GenerationConfig } from "./config";
import { InitProgressReport, LogLevel } from "./types";
import {
  ChatCompletionRequestStreaming,
  ChatCompletionRequestNonStreaming,
  ChatCompletion,
  ChatCompletionChunk,
} from "./openai_api_protocols/index";

/**
 * Message kind used by worker
 */
type RequestKind =
  | "reload"
  | "generate"
  | "runtimeStatsText"
  | "interruptGenerate"
  | "unload"
  | "resetChat"
  | "getMaxStorageBufferBindingSize"
  | "getGPUVendor"
  | "forwardTokensAndSample"
  | "chatCompletionNonStreaming"
  | "getMessage"
  | "chatCompletionStreamInit"
  | "chatCompletionStreamNextChunk"
  | "customRequest"
  | "keepAlive"
  | "setLogLevel"
  | "setAppConfig";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
type ResponseKind =
  | "return"
  | "throw"
  | "initProgressCallback"
  | "generateProgressCallback";

export interface ReloadParams {
  modelId: string;
  chatOpts?: ChatOptions;
}
export interface GenerateParams {
  input: string | ChatCompletionRequestNonStreaming;
  streamInterval?: number;
  genConfig?: GenerationConfig;
}
export interface ResetChatParams {
  keepStats: boolean;
}
export interface GenerateProgressCallbackParams {
  step: number;
  currentMessage: string;
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

export interface CustomRequestParams {
  requestName: string;
  requestMessage: string;
}
export type MessageContent =
  | GenerateProgressCallbackParams
  | ReloadParams
  | GenerateParams
  | ResetChatParams
  | ForwardTokensAndSampleParams
  | ChatCompletionNonStreamingParams
  | ChatCompletionStreamInitParams
  | CustomRequestParams
  | InitProgressReport
  | LogLevel
  | string
  | null
  | number
  | ChatCompletion
  | ChatCompletionChunk
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

type GenerateProgressWorkerResponse = {
  kind: "generateProgressCallback";
  uuid: string;
  content: {
    step: number;
    currentMessage: string;
  };
};

export type WorkerResponse =
  | OneTimeWorkerResponse
  | InitProgressWorkerResponse
  | GenerateProgressWorkerResponse
  | HeartbeatWorkerResponse;
