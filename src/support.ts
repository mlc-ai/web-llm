/** Util methods. */
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { AppConfig, MessagePlaceholders, ModelRecord } from "./config";
import {
  ChatCompletionChunk,
  ChatCompletionMessageToolCall,
} from "./openai_api_protocols/index";
import {
  ModelNotFoundError,
  ModelNotLoadedError,
  SpecifiedModelNotFoundError,
  ToolCallOutputInvalidTypeError,
  ToolCallOutputMissingFieldsError,
  ToolCallOutputParseError,
  UnclearModelToUseError,
} from "./error";

/**
 * Based on `p_prob` of size (vocabSize,) which becomes a distribution after calling
 * `applySoftmaxWithTemperature()`, sample `top_logprobs` top-probable tokens.
 *
 * @param num_top_probs: `top_logprobs` from ChatCompletionRequest
 * @param p_prob: `logitsOnCPUArray`, being a distribution after `applySoftmaxWithTemperature()`.
 *
 * Followed implementation of `ComputeTopProbsImpl()` from [https://github.com/mlc-ai/mlc-llm/blob/
 * 5b8c529e9704abd09b0432da6dcb4b013fdf43b1/cpp/serve/sampler/cpu_sampler.cc].
 *
 * @returns Arrays of (tokenID, prob) pairs, ranked from highest prob to least.
 */
export function getTopProbs(
  num_top_probs: number,
  p_prob: Float32Array,
): Array<[number, number]> {
  if (num_top_probs == 0) return [];
  // Initialize to dummy values
  const top_probs: Array<[number, number]> = [];
  const ndata = p_prob.length;
  for (let i = 0; i < num_top_probs; i++) {
    top_probs.push([-1, -1.0]);
  }

  let sum_prob = 0.0;
  // Selection argsort.
  for (let p = 0; p < ndata; p++) {
    let i = num_top_probs - 1;
    for (; i >= 0; --i) {
      if (p_prob[p] > top_probs[i][1]) {
        if (i !== num_top_probs - 1) {
          top_probs[i + 1] = top_probs[i];
        }
      } else {
        break;
      }
    }
    if (i !== num_top_probs - 1) {
      top_probs[i + 1] = [p, p_prob[p]];
    }

    // Early exit
    sum_prob += p_prob[p];
    if (1 - sum_prob <= top_probs[num_top_probs - 1][1]) {
      break;
    }
  }
  return top_probs;
}

/**
 * Get the token table in the form of a string list of tokens, ordered by their token id.
 * @param tokenizer A loaded tokenizer.
 * @note The size of the table (i.e. tokenizer.getVocabSize()) may be smaller than the `vocab_size`
 * in config.json (length of logits), see https://github.com/QwenLM/Qwen2/issues/147 and
 * https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/47.
 */
export function getTokenTableFromTokenizer(tokenizer: Tokenizer): string[] {
  const tokenTable: string[] = [];
  const vocabSize = tokenizer.getVocabSize();
  for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
    tokenTable.push(tokenizer.idToToken(tokenId));
  }
  return tokenTable;
}

/**
 * Postprocess the suffix of ModelRecord.model to be "/resolve/main/" if it is not specified otherwise.
 * e.g. https://huggingface.co/mlc-ai/OpenHermes-2.5-Mistral-7B-q4f16_1-MLC/resolve/main/
 * @return the href of the final URL.
 */
export function cleanModelUrl(modelUrl: string): string {
  // https://huggingface.co/USER/MODEL -> https://huggingface.co/USER/MODEL/
  modelUrl += modelUrl.endsWith("/") ? "" : "/";
  if (!modelUrl.match(/.+\/resolve\/.+\//)) modelUrl += "resolve/main/";
  // https://huggingface.co/USER/MODEL/ -> https://huggingface.co/USER/MODEL/resolve/main/
  return new URL(modelUrl).href;
}

// Constants for Hermes-2-Pro models function calling
// Follows https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-function-calling

/**
 * Json schema used to prompt the model for function calling; directly copied from the official guide.
 * This represents to a single function call.
 */
export const officialHermes2FunctionCallSchema = `{"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"}`;

/**
 * A list of such function calls. Used to specify response format, since the output is expected to
 * be a list of such function calls.
 */
export const officialHermes2FunctionCallSchemaArray = `{"type":"array","items":${officialHermes2FunctionCallSchema}}`;

/**
 * Full system prompt for Hermes-2-Pro function calling.
 */
export const hermes2FunctionCallingSystemPrompt = `You are a function calling AI model. You are 
provided with function signatures within <tools></tools> XML tags. You may call one or more functions 
to assist with the user query. Don't make assumptions about what values to plug into functions. Here 
are the available tools: <tools> ${MessagePlaceholders.hermes_tools}  </tools>. 
Use the following pydantic model json schema for each tool call you will make: 
${officialHermes2FunctionCallSchema} For each function call return a json object.`;

/**
 * Given a string outputMessage, parse it as a JSON object and return an array of tool calls.
 *
 * Expect outputMessage to be a valid JSON string, and expect it to be an array of Function with
 * fields `arguments` and `name`.
 */
export function getToolCallFromOutputMessage(
  outputMessage: string,
  isStreaming: false,
): Array<ChatCompletionMessageToolCall>;
export function getToolCallFromOutputMessage(
  outputMessage: string,
  isStreaming: true,
): Array<ChatCompletionChunk.Choice.Delta.ToolCall>;
export function getToolCallFromOutputMessage(
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
    if (curToolCall.name === undefined || curToolCall.arguments === undefined) {
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

export function findModelRecord(
  modelId: string,
  appConfig: AppConfig,
): ModelRecord {
  const matchedItem = appConfig.model_list.find(
    (item) => item.model_id == modelId,
  );
  if (matchedItem !== undefined) return matchedItem;
  throw new ModelNotFoundError(modelId);
}

/**
 * Return the model to use given the loaded modelIds and requestModel. Throws error when unclear
 * which model to load.
 * @param loadedModelIds Models currently loaded in the engine.
 * @param requestModel Model the user specified to load via the request. Required when multiple
 *   models are loaded
 * @param requestName The type of request or API to load the model for. Needed for error throwing.
 */
export function getModelIdToUse(
  loadedModelIds: string[],
  requestModel: string | undefined | null,
  requestName: string,
): string {
  let selectedModelId: string;
  if (loadedModelIds.length === 0) {
    throw new ModelNotLoadedError(requestName);
  }
  if (requestModel) {
    // If specified model
    if (loadedModelIds.indexOf(requestModel) === -1) {
      throw new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestModel,
        requestName,
      );
    } else {
      selectedModelId = requestModel;
    }
  } else {
    // If not specified
    if (loadedModelIds.length > 1) {
      throw new UnclearModelToUseError(loadedModelIds, requestName);
    } else {
      selectedModelId = loadedModelIds[0];
    }
  }
  return selectedModelId;
}

type Cont = () => void;

/**
 * A lock implemented using Promise.
 *
 * Referred to:
 * - https://jackpordi.com/posts/locks-in-js-because-why-not
 * - https://www.linkedin.com/pulse/asynchronous-locking-using-promises-javascript-abdul-ahad-o7smf/
 */
export class CustomLock {
  private acquired = false;
  private readonly queue: Cont[] = [];

  public async acquire(): Promise<void> {
    if (!this.acquired) {
      // If lock is free, directly return
      this.acquired = true;
    } else {
      // Otherwise, push the request to the queue, and
      // a future release() will resolve it
      return new Promise<void>((resolve) => {
        this.queue.push(resolve);
      });
    }
  }

  public async release(): Promise<void> {
    if (!this.acquired) {
      throw Error("InternalError: expect lock is acquired upon release()");
    }
    if (this.queue.length === 0) {
      // No one is waiting for the lock, so we free it
      this.acquired = false;
      return;
    }

    // Otherwise, hand the execution to the next in queue, and
    // the lock is still acquired
    const cont = this.queue.shift();
    return new Promise((res: Cont) => {
      cont!();
      res();
    });
  }
}
