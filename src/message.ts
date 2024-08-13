import { AppConfig, ChatOptions } from "./config";
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
  modelId: string[];
  chatOpts?: ChatOptions[];
}
export interface ResetChatParams {
  keepStats: boolean;
  modelId?: string;
}
export interface GetMessageParams {
  modelId?: string;
}
export interface RuntimeStatsTextParams {
  modelId?: string;
}
export interface ForwardTokensAndSampleParams {
  inputIds: Array<number>;
  isPrefill: boolean;
  modelId?: string;
}

// Notes on the following Params with modelId and chatOpts:
// These fields are the model and chatOpts that the frontend engine expects the backend
// to be loaded with. If not loaded due to web/service worker unexpectedly killed,
// handler will call reload(). An engine can load multiple models, hence both are list.
// TODO(webllm-team): should add appConfig here as well if rigorous.
// Fore more, see https://github.com/mlc-ai/web-llm/pull/471

// Note on the messages with selectedModelId:
// This is the modelId this request uses. It is needed to identify which async generator
// to instantiate / use, since an engine can load multiple models, thus the handler
// needs to maintain multiple generators.
export interface ChatCompletionNonStreamingParams {
  request: ChatCompletionRequestNonStreaming;
  modelId: string[];
  chatOpts?: ChatOptions[];
}
export interface ChatCompletionStreamInitParams {
  request: ChatCompletionRequestStreaming;
  selectedModelId: string;
  modelId: string[];
  chatOpts?: ChatOptions[];
}
export interface CompletionNonStreamingParams {
  request: CompletionCreateParamsNonStreaming;
  modelId: string[];
  chatOpts?: ChatOptions[];
}
export interface CompletionStreamInitParams {
  request: CompletionCreateParamsStreaming;
  selectedModelId: string;
  modelId: string[];
  chatOpts?: ChatOptions[];
}
export interface EmbeddingParams {
  request: EmbeddingCreateParams;
  modelId: string[];
  chatOpts?: ChatOptions[];
}
export interface CompletionStreamNextChunkParams {
  selectedModelId: string;
}

export interface CustomRequestParams {
  requestName: string;
  requestMessage: string;
}
export type MessageContent =
  | ReloadParams
  | ResetChatParams
  | GetMessageParams
  | RuntimeStatsTextParams
  | ForwardTokensAndSampleParams
  | ChatCompletionNonStreamingParams
  | ChatCompletionStreamInitParams
  | CompletionNonStreamingParams
  | CompletionStreamInitParams
  | EmbeddingParams
  | CompletionStreamNextChunkParams
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
