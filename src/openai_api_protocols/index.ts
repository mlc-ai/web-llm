/**
 * The input to OpenAI API, directly adopted from openai-node with small tweaks:
 * https://github.com/openai/openai-node/blob/master/src/resources/chat/completions.ts
 *
 * Copyright 2024 OpenAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *      http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export {
  Chat,
  ChatCompletionRequestBase,
  ChatCompletionRequestNonStreaming,
  ChatCompletionRequestStreaming,
  ChatCompletionRequest,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionRequestUnsupportedFields,
  postInitAndCheckFields as postInitAndCheckFieldsChatCompletion,
  ChatCompletionContentPart,
  ChatCompletionContentPartText,
  ChatCompletionContentPartImage,
  ChatCompletionMessageToolCall,
  ChatCompletionRole,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionMessageParam,
  FunctionParameters,
  FunctionDefinition,
  ChatCompletionTool,
  ChatCompletionNamedToolChoice,
  ChatCompletionToolChoiceOption,
  TopLogprob,
  ChatCompletionTokenLogprob,
  ChatCompletionMessage,
  CompletionUsage,
  ResponseFormat,
  ChatCompletionFinishReason,
} from "./chat_completion";

export {
  Completions,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  CompletionCreateParamsBase,
  CompletionCreateParams,
  Completion,
  CompletionChoice,
  postInitAndCheckFields as postInitAndCheckFieldsCompletion,
} from "./completion";

export {
  Embeddings,
  Embedding,
  EmbeddingCreateParams,
  CreateEmbeddingResponse,
  postInitAndCheckFields as postInitAndCheckFieldsEmbedding,
} from "./embedding";
