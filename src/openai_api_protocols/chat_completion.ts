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

import { MLCEngineInterface, LatencyBreakdown } from "../types";
import {
  functionCallingModelIds,
  MessagePlaceholders,
  ModelType,
} from "../config";
import {
  officialHermes2FunctionCallSchemaArray,
  hermes2FunctionCallingSystemPrompt,
} from "../support";
import {
  CustomResponseFormatError,
  CustomSystemPromptError,
  InvalidResponseFormatError,
  InvalidResponseFormatGrammarError,
  InvalidStreamOptionsError,
  MessageOrderError,
  MultipleTextContentError,
  SeedTypeError,
  StreamingCountError,
  SystemMessageOrderError,
  UnsupportedDetailError,
  UnsupportedFieldsError,
  UnsupportedImageURLError,
  UnsupportedModelIdError,
  UserMessageContentErrorForNonVLM,
} from "../error";

/* eslint-disable @typescript-eslint/no-namespace */

export class Chat {
  private engine: MLCEngineInterface;
  completions: Completions;

  constructor(engine: MLCEngineInterface) {
    this.engine = engine;
    this.completions = new Completions(this.engine);
  }
}

export class Completions {
  private engine: MLCEngineInterface;

  constructor(engine: MLCEngineInterface) {
    this.engine = engine;
  }

  create(request: ChatCompletionRequestNonStreaming): Promise<ChatCompletion>;
  create(
    request: ChatCompletionRequestStreaming,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  create(
    request: ChatCompletionRequestBase,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;
  create(
    request: ChatCompletionRequest,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    return this.engine.chatCompletion(request);
  }
}

//////////////////////////////// 0. HIGH-LEVEL INTERFACES ////////////////////////////////

/**
 * OpenAI chat completion request protocol.
 *
 * API reference: https://platform.openai.com/docs/api-reference/chat/create
 * Followed: https://github.com/openai/openai-node/blob/master/src/resources/chat/completions.ts
 *
 * @note `model` is excluded. Instead, call `CreateMLCEngine(model)` or `engine.reload(model)` explicitly before calling this API.
 */
export interface ChatCompletionRequestBase {
  /**
   * A list of messages comprising the conversation so far.
   */
  messages: Array<ChatCompletionMessageParam>;

  /**
   * If set, partial message deltas will be sent. It will be terminated by an empty chunk.
   */
  stream?: boolean | null;

  /**
   * Options for streaming response. Only set this when you set `stream: true`.
   */
  stream_options?: ChatCompletionStreamOptions | null;

  /**
   * How many chat completion choices to generate for each input message.
   */
  n?: number | null;

  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on their
   * existing frequency in the text so far, decreasing the model's likelihood to
   * repeat the same line verbatim.
   *
   * [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/text-generation/parameter-details)
   */
  frequency_penalty?: number | null;

  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on
   * whether they appear in the text so far, increasing the model's likelihood to
   * talk about new topics.
   *
   * [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/text-generation/parameter-details)
   */
  presence_penalty?: number | null;

  /**
   * Penalizes new tokens based on whether they appear in the prompt and the
   * generated text so far. Values greater than 1.0 encourage the model to use new
   * tokens, while values less than 1.0 encourage the model to repeat tokens.
   */
  repetition_penalty?: number | null;

  /**
   * The maximum number of [tokens](/tokenizer) that can be generated in the chat
   * completion.
   *
   * The total length of input tokens and generated tokens is limited by the model's
   * context length.
   */
  max_tokens?: number | null;

  /**
   * Sequences where the API will stop generating further tokens.
   */
  stop?: string | null | Array<string>;

  /**
   * What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
   * make the output more random, while lower values like 0.2 will make it more
   * focused and deterministic.
   */
  temperature?: number | null;

  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the
   * model considers the results of the tokens with top_p probability mass. So 0.1
   * means only the tokens comprising the top 10% probability mass are considered.
   */
  top_p?: number | null;

  /**
   * Modify the likelihood of specified tokens appearing in the completion.
   *
   * Accepts a JSON object that maps tokens (specified by their token ID, which varies per model)
   * to an associated bias value from -100 to 100. Typically, you can see `tokenizer.json` of the
   * model to see which token ID maps to what string. Mathematically, the bias is added to the
   * logits generated by the model prior to sampling. The exact effect will vary per model, but
   * values between -1 and 1 should decrease or increase likelihood of selection; values like -100
   * or 100 should result in a ban or exclusive selection of the relevant token.
   *
   * As an example, you can pass `{"16230": -100}` to prevent the `Hello` token from being
   * generated in Mistral-7B-Instruct-v0.2, according to the mapping in
   * https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/raw/main/tokenizer.json.
   *
   * @note For stateful and customizable / flexible logit processing, see `webllm.LogitProcessor`.
   * @note If used in combination with `webllm.LogitProcessor`, `logit_bias` is applied after
   * `LogitProcessor.processLogits()` is called.
   */
  logit_bias?: Record<string, number> | null;

  /**
   * Whether to return log probabilities of the output tokens or not.
   *
   * If true, returns the log probabilities of each output token returned in the `content` of
   * `message`.
   */
  logprobs?: boolean | null;

  /**
   * An integer between 0 and 5 specifying the number of most likely tokens to return
   * at each token position, each with an associated log probability. `logprobs` must
   * be set to `true` if this parameter is used.
   */
  top_logprobs?: number | null;

  /**
   * If specified, our system will make a best effort to sample deterministically, such that
   * repeated requests with the same `seed` and parameters should return the same result.
   *
   * @note Seeding is done on a request-level rather than choice-level. That is, if `n > 1`, you
   * would still get different content for each `Choice`. But if two requests with `n = 2` are
   * processed with the same seed, the two results should be the same (two choices are different).
   */
  seed?: number | null;

  /**
   * Controls which (if any) function is called by the model. `none` means the model
   * will not call a function and instead generates a message. `auto` means the model
   * can pick between generating a message or calling a function. Specifying a
   * particular function via
   * `{"type": "function", "function": {"name": "my_function"}}` forces the model to
   * call that function.
   *
   * `none` is the default when no functions are present. `auto` is the default if
   * functions are present.
   */
  tool_choice?: ChatCompletionToolChoiceOption;

  /**
   * A list of tools the model may call. Currently, only functions are supported as a
   * tool. Use this to provide a list of functions the model may generate JSON inputs
   * for.
   *
   * The corresponding reply would populate the `tool_calls` field. If used with streaming,
   * the last chunk would contain the `tool_calls` field, while the intermediate chunks would
   * contain the raw string.
   *
   * If the generation terminates due to FinishReason other than "stop" (i.e. "length" or "abort"),
   * then no `tool_calls` will be returned. User can still get the raw string output.
   */
  tools?: Array<ChatCompletionTool>;

  /**
   * An object specifying the format that the model must output.
   *
   * Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the
   * message the model generates is valid JSON.
   *
   * **Important:** when using JSON mode, you **must** also instruct the model to
   * produce JSON yourself via a system or user message. Without this, the model may
   * generate an unending stream of whitespace until the generation reaches the token
   * limit, resulting in a long-running and seemingly "stuck" request. Also note that
   * the message content may be partially cut off if `finish_reason="length"`, which
   * indicates the generation exceeded `max_tokens` or the conversation exceeded the
   * max context length.
   */
  response_format?: ResponseFormat;

  /**
   * If true, will ignore stop string and stop token and generate until max_tokens hit.
   * If unset, will treat as false.
   */
  ignore_eos?: boolean;

  /**
   * ID of the model to use. This equals to `ModelRecord.model_id`, which needs to either be in
   * `webllm.prebuiltAppConfig` or in `engineConfig.appConfig`.
   *
   * @note Call `CreateMLCEngine(model)` or `engine.reload(model)` ahead of time.
   * @note If only one model is loaded in the engine, this field is optional. If multiple models
   *   are loaded, this is required.
   */
  model?: string | null;

  /**
   * Fields specific to WebLLM, not present in OpenAI.
   */
  extra_body?: {
    /**
     * If set to false, prepend a "<think>\n\n</think>\n\n" to the response, preventing the
     * model from generating thinking tokens. If set to true or undefined, does nothing.
     *
     * @note Currently only allowed to be used for Qwen3 models, though not explicitly checked.
     */
    enable_thinking?: boolean | null;

    /**
     * If set to true, the response will include a breakdown of the time spent in various
     * stages of token sampling.
     */
    enable_latency_breakdown?: boolean | null;
  };
}

export interface ChatCompletionRequestNonStreaming
  extends ChatCompletionRequestBase {
  /**
   * If set, partial message deltas will be sent. It will be terminated by an empty chunk.
   */
  stream?: false | null;
}

export interface ChatCompletionRequestStreaming
  extends ChatCompletionRequestBase {
  /**
   * If set, partial message deltas will be sent. It will be terminated by an empty chunk.
   */
  stream: true;
}

export type ChatCompletionRequest =
  | ChatCompletionRequestNonStreaming
  | ChatCompletionRequestStreaming;

/**
 * Represents a chat completion response returned by model, based on the provided input.
 */
export interface ChatCompletion {
  /**
   * A unique identifier for the chat completion.
   */
  id: string;

  /**
   * A list of chat completion choices. Can be more than one if `n` is greater than 1.
   */
  choices: Array<ChatCompletion.Choice>;

  /**
   * The model used for the chat completion.
   */
  model: string;

  /**
   * The object type, which is always `chat.completion`.
   */
  object: "chat.completion";

  /**
   * The Unix timestamp (in seconds) of when the chat completion was created.
   *
   */
  created: number;

  /**
   * Usage statistics for the completion request.
   *
   * @note If we detect user is performing multi-round chatting, only the new portion of the
   * prompt is counted for prompt_tokens. If `n > 1`, all choices' generation usages combined.
   */
  usage?: CompletionUsage;

  /**
   * This fingerprint represents the backend configuration that the model runs with.
   *
   * Can be used in conjunction with the `seed` request parameter to understand when
   * backend changes have been made that might impact determinism.
   *
   * @note Not supported yet.
   */
  system_fingerprint?: string;
}

/**
 * Represents a streamed chunk of a chat completion response returned by model,
 * based on the provided input.
 */
export interface ChatCompletionChunk {
  /**
   * A unique identifier for the chat completion. Each chunk has the same ID.
   */
  id: string;

  /**
   * A list of chat completion choices. Can contain more than one elements if `n` is
   * greater than 1. Can also be empty for the last chunk if you set
   * `stream_options: {"include_usage": true}`.
   */
  choices: Array<ChatCompletionChunk.Choice>;

  /**
   * The Unix timestamp (in seconds) of when the chat completion was created. Each
   * chunk has the same timestamp.
   */
  created: number;

  /**
   * The model to generate the completion.
   */
  model: string;

  /**
   * The object type, which is always `chat.completion.chunk`.
   */
  object: "chat.completion.chunk";

  /**
   * This fingerprint represents the backend configuration that the model runs with.
   * Can be used in conjunction with the `seed` request parameter to understand when
   * backend changes have been made that might impact determinism.
   *
   * @note Not supported yet.
   */
  system_fingerprint?: string;

  /**
   * An optional field that will only be present when you set
   * `stream_options: {"include_usage": true}` in your request. When present, it
   * contains a null value except for the last chunk which contains the token usage
   * statistics for the entire request.
   */
  usage?: CompletionUsage;
}

export const ChatCompletionRequestUnsupportedFields: Array<string> = []; // all supported as of now

/**
 * Post init and verify whether the input of the request is valid. Thus, this function can throw
 * error or in-place update request.
 * @param request User's input request.
 * @param currentModelId The current model loaded that will perform this request.
 * @param currentModelType The type of the model loaded, decide what requests can be handled.
 */
export function postInitAndCheckFields(
  request: ChatCompletionRequest,
  currentModelId: string,
  currentModelType: ModelType,
): void {
  // Generation-related checks and post inits are in `postInitAndCheckGenerationConfigValues()`
  // 1. Check unsupported fields in request
  const unsupported: Array<string> = [];
  ChatCompletionRequestUnsupportedFields.forEach((field) => {
    if (field in request) {
      unsupported.push(field);
    }
  });
  if (unsupported.length > 0) {
    throw new UnsupportedFieldsError(unsupported, "ChatCompletionRequest");
  }

  // 2. Check unsupported messages
  request.messages.forEach(
    (message: ChatCompletionMessageParam, index: number) => {
      // Check content array messages (that are not simple string)
      if (message.role === "user" && typeof message.content !== "string") {
        if (currentModelType !== ModelType.VLM) {
          // Only VLM can handle non-string content (i.e. message with image)
          throw new UserMessageContentErrorForNonVLM(
            currentModelId,
            ModelType[currentModelType],
            message.content,
          );
        }
        let numTextContent = 0;
        for (let i = 0; i < message.content.length; i++) {
          const curContent = message.content[i];
          if (curContent.type === "image_url") {
            // Do not support image_url.detail
            const detail = curContent.image_url.detail;
            if (detail !== undefined && detail !== null) {
              throw new UnsupportedDetailError(detail);
            }
            // Either start with http or data:image for base64
            const url = curContent.image_url.url;
            if (!url.startsWith("data:image") && !url.startsWith("http")) {
              throw new UnsupportedImageURLError(url);
            }
          } else {
            numTextContent += 1;
          }
        }
        if (numTextContent > 1) {
          // Only one text contentPart per message
          // TODO(Charlie): is it always the case that an input can only have one
          // textPart? Or it is only for phi3vision?
          throw new MultipleTextContentError();
        }
      }
      if (message.role === "system" && index !== 0) {
        throw new SystemMessageOrderError();
      }
    },
  );

  // 3. Last message has to be from user or tool
  const lastId = request.messages.length - 1;
  if (
    request.messages[lastId].role !== "user" &&
    request.messages[lastId].role !== "tool"
  ) {
    throw new MessageOrderError(
      "Last message should be from either `user` or `tool`.",
    );
  }

  // 4. If streaming, n cannot be > 1, since we cannot manage multiple sequences at once
  if (request.stream && request.n && request.n > 1) {
    throw new StreamingCountError();
  }

  // 5. Seed should be an integer
  if (request.seed !== undefined && request.seed !== null) {
    if (!Number.isInteger(request.seed)) {
      throw new SeedTypeError(request.seed);
    }
  }

  // 6. Schema can only be specified when type is `json_object`.
  if (
    request.response_format?.schema !== undefined &&
    request.response_format?.schema !== null
  ) {
    if (request.response_format?.type !== "json_object") {
      throw new InvalidResponseFormatError();
    }
  }

  // 6.1 When grammar is specified, the type needs to be grammar
  if (
    request.response_format?.grammar !== undefined &&
    request.response_format?.grammar !== null
  ) {
    if (request.response_format?.type !== "grammar") {
      throw new InvalidResponseFormatGrammarError();
    }
  }

  // 6.2 When type is grammar, the grammar field needs to be specified.
  if (request.response_format?.type === "grammar") {
    if (
      request.response_format?.grammar === undefined ||
      request.response_format?.grammar === null
    ) {
      throw new InvalidResponseFormatGrammarError();
    }
  }

  // 7. Function calling hardcoded handlings
  if (request.tools !== undefined && request.tools !== null) {
    // 7.1 Check if model supports function calling
    if (!functionCallingModelIds.includes(currentModelId)) {
      throw new UnsupportedModelIdError(
        currentModelId,
        functionCallingModelIds,
      );
    }

    // 7.2 Hard coded support for Hermes2Pro following
    // https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-function-calling
    if (currentModelId.startsWith("Hermes-2-Pro-")) {
      // 7.2.1 Update response format for Hermes2Pro function calling to use json schema
      if (
        request.response_format !== undefined &&
        request.response_format !== null
      ) {
        throw new CustomResponseFormatError(request.response_format);
      }
      request.response_format = {
        type: "json_object",
        schema: officialHermes2FunctionCallSchemaArray,
      } as ResponseFormat;

      // 7.2.2 Modify system prompt to provide tools usage
      const hermes2SystemMessage = hermes2FunctionCallingSystemPrompt.replace(
        MessagePlaceholders.hermes_tools,
        JSON.stringify(request.tools),
      );
      // Make sure user did not provide system message already
      for (let i = 0; i < request.messages.length; i++) {
        const message: ChatCompletionMessageParam = request.messages[i];
        if (message.role === "system") {
          throw new CustomSystemPromptError();
        }
      }
      // Prepend a message for hardcoded system prompt
      request.messages.unshift({
        role: "system",
        content: hermes2SystemMessage,
      } as ChatCompletionSystemMessageParam);
    }
  }

  // 8. Only set stream_options when streaming
  if (request.stream_options !== undefined && request.stream_options !== null) {
    if (!request.stream) {
      throw new InvalidStreamOptionsError();
    }
  }
}

//////////////// BELOW ARE INTERFACES THAT SUPPORT THE ONES ABOVE ////////////////

//////////////////////////////// 1. MESSAGES ////////////////////////////////

//////////////////////////////// 1.1. CHAT COMPLETION CONTENT ////////////////////////////////

export type ChatCompletionContentPart =
  | ChatCompletionContentPartText
  | ChatCompletionContentPartImage;

export interface ChatCompletionContentPartText {
  /**
   * The text content.
   */
  text: string;

  /**
   * The type of the content part.
   */
  type: "text";
}

export namespace ChatCompletionContentPartImage {
  export interface ImageURL {
    /**
     * Either a URL of the image or the base64 encoded image data.
     */
    url: string;

    /**
     * Specifies the detail level of the image.
     */
    detail?: "auto" | "low" | "high";
  }
}

export interface ChatCompletionContentPartImage {
  image_url: ChatCompletionContentPartImage.ImageURL;
  /**
   * The type of the content part.
   */
  type: "image_url";
}

//////////////////////////////// 1.2. MESSAGE TOOL CALL ////////////////////////////////

export interface ChatCompletionMessageToolCall {
  /**
   * The ID of the tool call. In WebLLM, it is used as the index of the tool call among all
   * the tools calls in this request generation.
   */
  id: string;

  /**
   * The function that the model called.
   */
  function: ChatCompletionMessageToolCall.Function;

  /**
   * The type of the tool. Currently, only `function` is supported.
   */
  type: "function";
}

export namespace ChatCompletionMessageToolCall {
  /**
   * The function that the model called.
   */
  export interface Function {
    /**
     * The arguments to call the function with, as generated by the model in JSON
     * format.
     */
    arguments: string;

    /**
     * The name of the function to call.
     */
    name: string;
  }
}

//////////////////////////////// 1.3. MESSAGE PARAM ////////////////////////////////

/**
 * The role of the author of a message
 */
export type ChatCompletionRole =
  | "system"
  | "user"
  | "assistant"
  | "tool"
  | "function";

/**
 * Options for streaming response. Only set this when you set `stream: true`.
 */
export interface ChatCompletionStreamOptions {
  /**
   * If set, an additional chunk will be streamed after the last empty chunk.
   * The `usage` field on this chunk shows the token usage statistics for the entire
   * request, and the `choices` field will always be an empty array. All other chunks
   * will also include a `usage` field, but with a null value.
   */
  include_usage?: boolean;
}

export interface ChatCompletionSystemMessageParam {
  /**
   * The contents of the system message.
   */
  content: string;

  /**
   * The role of the messages author, in this case `system`.
   */
  role: "system";
}

export interface ChatCompletionUserMessageParam {
  /**
   * The contents of the user message.
   */
  content: string | Array<ChatCompletionContentPart>;

  /**
   * The role of the messages author, in this case `user`.
   */
  role: "user";

  /**
   * An optional name for the participant. Provides the model information to
   * differentiate between participants of the same role.
   *
   * @note This is experimental, as models typically have predefined names for the user.
   */
  name?: string;
}

export interface ChatCompletionAssistantMessageParam {
  /**
   * The role of the messages author, in this case `assistant`.
   */
  role: "assistant";

  /**
   * The contents of the assistant message. Required unless `tool_calls` is specified.
   */
  content?: string | null;

  /**
   * An optional name for the participant. Provides the model information to
   * differentiate between participants of the same role.
   *
   * @note This is experimental, as models typically have predefined names for the user.
   */
  name?: string;

  /**
   * The tool calls generated by the model, such as function calls.
   */
  tool_calls?: Array<ChatCompletionMessageToolCall>;
}

export interface ChatCompletionToolMessageParam {
  /**
   * The contents of the tool message.
   */
  content: string;

  /**
   * The role of the messages author, in this case `tool`.
   */
  role: "tool";

  /**
   * Tool call that this message is responding to.
   */
  tool_call_id: string;
}

export type ChatCompletionMessageParam =
  | ChatCompletionSystemMessageParam
  | ChatCompletionUserMessageParam
  | ChatCompletionAssistantMessageParam
  | ChatCompletionToolMessageParam;

//////////////////////////////// 2. TOOL USING ////////////////////////////////

/**
 * The parameters the functions accepts, described as a JSON Schema object. See the
 * [guide](https://platform.openai.com/docs/guides/text-generation/function-calling)
 * for examples, and the
 * [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
 * documentation about the format.
 *
 * Omitting `parameters` defines a function with an empty parameter list.
 */
export type FunctionParameters = Record<string, unknown>;

export interface FunctionDefinition {
  /**
   * The name of the function to be called. Must be a-z, A-Z, 0-9, or contain
   * underscores and dashes, with a maximum length of 64.
   */
  name: string;

  /**
   * A description of what the function does, used by the model to choose when and
   * how to call the function.
   */
  description?: string;

  /**
   * The parameters the functions accepts, described as a JSON Schema object. See the
   * [guide](https://platform.openai.com/docs/guides/text-generation/function-calling)
   * for examples, and the
   * [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
   * documentation about the format.
   *
   * Omitting `parameters` defines a function with an empty parameter list.
   */
  parameters?: FunctionParameters;
}

export interface ChatCompletionTool {
  function: FunctionDefinition;

  /**
   * The type of the tool. Currently, only `function` is supported.
   */
  type: "function";
}

/**
 * Specifies a tool the model should use. Use to force the model to call a specific
 * function.
 */
export interface ChatCompletionNamedToolChoice {
  function: ChatCompletionNamedToolChoice.Function;

  /**
   * The type of the tool. Currently, only `function` is supported.
   */
  type: "function";
}

export namespace ChatCompletionNamedToolChoice {
  export interface Function {
    /**
     * The name of the function to call.
     */
    name: string;
  }
}

/**
 * Controls which (if any) function is called by the model. `none` means the model
 * will not call a function and instead generates a message. `auto` means the model
 * can pick between generating a message or calling a function. Specifying a
 * particular function via
 * `{"type": "function", "function": {"name": "my_function"}}` forces the model to
 * call that function.
 *
 * `none` is the default when no functions are present. `auto` is the default if
 * functions are present.
 */
export type ChatCompletionToolChoiceOption =
  | "none"
  | "auto"
  | ChatCompletionNamedToolChoice;

//////////////////////////////// 3. OTHERS ////////////////////////////////

//////////////////////////////// 3.1. LOG PROBS ////////////////////////////////
export interface TopLogprob {
  /**
   * The token.
   */
  token: string;

  /**
   * A list of integers representing the UTF-8 bytes representation of the token.
   * Useful in instances where characters are represented by multiple tokens and
   * their byte representations must be combined to generate the correct text
   * representation. Can be `null` if there is no bytes representation for the token.
   *
   * @note Encoded with `TextEncoder.encode()` and can be decoded with `TextDecoder.decode()`.
   * For details, see https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/encode.
   */
  bytes: Array<number> | null;

  /**
   * The log probability of this token.
   */
  logprob: number;
}

export interface ChatCompletionTokenLogprob {
  /**
   * The token.
   */
  token: string;

  /**
   * A list of integers representing the UTF-8 bytes representation of the token.
   * Useful in instances where characters are represented by multiple tokens and
   * their byte representations must be combined to generate the correct text
   * representation. Can be `null` if there is no bytes representation for the token.
   *
   * @note Encoded with `TextEncoder.encode()` and can be decoded with `TextDecoder.decode()`.
   * For details, see https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/encode.
   */
  bytes: Array<number> | null;

  /**
   * The log probability of this token.
   */
  logprob: number;

  /**
   * List of the most likely tokens and their log probability, at this token
   * position. In rare cases, there may be fewer than the number of requested
   * `top_logprobs` returned.
   */
  top_logprobs: Array<TopLogprob>;
}

//////////////////////////////// 3.2. OTHERS ////////////////////////////////
/**
 * A chat completion message generated by the model.
 */
export interface ChatCompletionMessage {
  /**
   * The contents of the message.
   */
  content: string | null;

  /**
   * The role of the author of this message.
   */
  role: "assistant";

  /**
   * The tool calls generated by the model, such as function calls.
   */
  tool_calls?: Array<ChatCompletionMessageToolCall>;
}

/**
 * Usage statistics for the completion request.
 */
export interface CompletionUsage {
  /**
   * Number of tokens in the generated completion.
   */
  completion_tokens: number;

  /**
   * Number of tokens in the prompt.
   *
   * @note If we detect user is performing multi-round chatting, only the new portion of the
   * prompt is counted for prompt_tokens.
   */
  prompt_tokens: number;

  /**
   * Total number of tokens used in the request (prompt + completion).
   */
  total_tokens: number;

  /**
   * Fields specific to WebLLM, not present in OpenAI.
   */
  extra: {
    /**
     * Total seconds spent on this request, from receiving the request, to generating the response.
     */
    e2e_latency_s: number;

    /**
     * Number of tokens per second for prefilling.
     */
    prefill_tokens_per_s: number;

    /**
     * Number of tokens per second for autoregressive decoding.
     */
    decode_tokens_per_s: number;

    /**
     * Seconds spent to generate the first token since receiving the request. Mainly contains
     * prefilling overhead. If n > 1, it is the sum over all choices.
     */
    time_to_first_token_s: number;

    /**
     * Seconds in between generated tokens. Mainly contains decoding overhead. If n > 1, it
     * is the average over all choices.
     */
    time_per_output_token_s: number;

    /**
     * Seconds spent on initializing grammar matcher for structured output. If n > 1, it
     * is the sum over all choices.
     */
    grammar_init_s?: number;

    /**
     * Seconds per-token that grammar matcher spent on creating bitmask and accepting token for
     * structured output. If n > 1, it is the average over all choices.
     */
    grammar_per_token_s?: number;

    /**
     * If `enable_latency_breakdown` is set to true in the request, this field will be
     * present and contain a breakdown of the time spent in various stages of token sampling.
     */
    latencyBreakdown?: LatencyBreakdown;
  };
}

/**
 * The reason the model stopped generating tokens. This will be `stop` if the model
 * hit a natural stop point or a provided stop sequence, `length` if the maximum
 * number of tokens specified in the request was reached or the context_window_size will
 * be exceeded, `tool_calls` if the model called a tool, or `abort` if user manually stops the
 * generation.
 */
export type ChatCompletionFinishReason =
  | "stop"
  | "length"
  | "tool_calls"
  | "abort";

export namespace ChatCompletion {
  export interface Choice {
    /**
     * The reason the model stopped generating tokens. This will be `stop` if the model
     * hit a natural stop point or a provided stop sequence, `length` if the maximum
     * number of tokens specified in the request was reached, `tool_calls` if the
     * model called a tool, or `abort` if user manually stops the generation.
     */
    finish_reason: ChatCompletionFinishReason;

    /**
     * The index of the choice in the list of choices.
     */
    index: number;

    /**
     * Log probability information for the choice.
     */
    logprobs: Choice.Logprobs | null;

    /**
     * A chat completion message generated by the model.
     */
    message: ChatCompletionMessage;
  }

  export namespace Choice {
    /**
     * Log probability information for the choice.
     */
    export interface Logprobs {
      /**
       * A list of message content tokens with log probability information.
       */
      content: Array<ChatCompletionTokenLogprob> | null;
    }
  }
}

export namespace ChatCompletionChunk {
  export interface Choice {
    /**
     * A chat completion delta generated by streamed model responses.
     */
    delta: Choice.Delta;

    /**
     * The reason the model stopped generating tokens. This will be `stop` if the model
     * hit a natural stop point or a provided stop sequence, `length` if the maximum
     * number of tokens specified in the request was reached, `tool_calls` if the
     * model called a tool, or `abort` if user manually stops the generation.
     */
    finish_reason: ChatCompletionFinishReason | null;

    /**
     * The index of the choice in the list of choices.
     */
    index: number;

    /**
     * Log probability information for the choice.
     */
    logprobs?: Choice.Logprobs | null;
  }

  export namespace Choice {
    /**
     * A chat completion delta generated by streamed model responses.
     */
    export interface Delta {
      /**
       * The contents of the chunk message.
       */
      content?: string | null;

      /**
       * The role of the author of this message.
       */
      role?: "system" | "user" | "assistant" | "tool";

      tool_calls?: Array<Delta.ToolCall>;
    }

    export namespace Delta {
      export interface ToolCall {
        /**
         * The index of the tool call among all the tools calls in this request generation.
         */
        index: number;

        /**
         * The ID of the tool call. Not used in WebLLM.
         */
        id?: string;

        function?: ToolCall.Function;

        /**
         * The type of the tool. Currently, only `function` is supported.
         */
        type?: "function";
      }

      export namespace ToolCall {
        export interface Function {
          /**
           * The arguments to call the function with, as generated by the model in JSON
           * format. Note that the model does not always generate valid JSON, and may
           * hallucinate parameters not defined by your function schema. Validate the
           * arguments in your code before calling your function.
           */
          arguments?: string;

          /**
           * The name of the function to call.
           */
          name?: string;
        }
      }
    }

    /**
     * Log probability information for the choice.
     */
    export interface Logprobs {
      /**
       * A list of message content tokens with log probability information.
       */
      content: Array<ChatCompletionTokenLogprob> | null;
    }
  }
}

/**
 * An object specifying the format that the model must output.
 *
 * Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the
 * message the model generates is valid JSON.
 *
 * Setting to `{ "type": "grammar" }` requires you to also specify the `grammar` field, which
 * is a BNFGrammar string.
 *
 * Setting `schema` specifies the output format of the json object such as properties to include.
 *
 * **Important:** when using JSON mode, you **must** also instruct the model to produce JSON
 * following the schema (if specified) yourself via a system or user message. Without this,
 * the model may generate an unending stream of whitespace until the generation reaches the token
 * limit, resulting in a long-running and seemingly "stuck" request. Also note that
 * the message content may be partially cut off if `finish_reason="length"`, which
 * indicates the generation exceeded `max_tokens` or the conversation exceeded the
 * max context length.
 */
export interface ResponseFormat {
  /**
   * Must be one of `text`, `json_object`, or `grammar`.
   */
  type?: "text" | "json_object" | "grammar";
  /**
   * A schema string in the format of the schema of a JSON file. `type` needs to be `json_object`.
   */
  schema?: string;
  /**
   * An EBNF-formatted string. Needs to be specified when, and only specified when,
   * `type` is `grammar`. The grammar will be normalized (simplified) by default.
   * EBNF grammar: see https://www.w3.org/TR/xml/#sec-notation. Note:
      1. Use # as the comment mark
      2. Use C-style unicode escape sequence \u01AB, \U000001AB, \xAB
      3. A-B (match A and not match B) is not supported yet
      4. Lookahead assertion can be added at the end of a rule to speed up matching. E.g.
      ```
      main ::= "ab" a [a-z]
      a ::= "cd" (=[a-z])
      ```
      The assertion (=[a-z]) means a must be followed by [a-z].
   */
  grammar?: string;
}
