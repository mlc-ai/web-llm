/** Util methods. */
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { AppConfig, MessagePlaceholders, ModelRecord } from "./config";
import {
  ChatCompletionChunk,
  ChatCompletionContentPartImage,
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

/**
 * TODO: Consider if this is the best strategy (though aligned with mlc-llm). We currently greedily
 * try to fill up prefillChunkSize. Consider the example with 2048 prefill chunk size:
 * const inputData = [
    image1,  // 1921
    rangeArr(0, 2048),
    image2,
  ];
 * Current approach results in chunks: 
   [image1, rangeArr(0, 127)],
   [rangeArr(127, 2048)],
   [image2],
 * This means 4 embedding kernels and 3 prefill kernels.
 * While the optimal chunking may be:
   [image1],
   [rangeArr(0, 2048)],
   [image2],
 * This results in 3 embedding kernels and 3 prefill kernels.
 * However, greedy strategy is more intuitive and probably more generalizable.
 */

/**
 * Chunk the inputData such that each chunk's total input length is smaller than prefill
 * chunk size.
 * @returns [the data chunks, the input length of each chunk]
 * @note precondition: if inputData has image in it, then prefillChunkSize >= IMAGE_EMBED_SIZE.
 */
export function getChunkedPrefillInputData(
  inputData: Array<Array<number> | ImageURL>,
  prefillChunkSize: number,
): [Array<Array<number> | ImageURL>[], Array<number>] {
  const chunks: Array<Array<number> | ImageURL>[] = [];
  const chunkLens: Array<number> = [];
  let curChunk: Array<Array<number> | ImageURL> = [];
  let curChunkLen = 0;
  for (let i = 0; i < inputData.length; i++) {
    let curData: Array<number> | ImageURL = inputData[i];
    const curDataLen = Array.isArray(curData)
      ? curData.length
      : IMAGE_EMBED_SIZE;
    // 1. curData can fit into this chunk
    if (curChunkLen + curDataLen <= prefillChunkSize) {
      curChunk.push(curData);
      curChunkLen += curDataLen;
      if (curChunkLen === prefillChunkSize) {
        chunks.push([...curChunk]);
        chunkLens.push(curChunkLen);
        curChunk = [];
        curChunkLen = 0;
      }
      continue;
    }

    // 2. Otherwise, depends on whether it is token data or image data
    if (Array.isArray(curData)) {
      // 2.1. Token data, which itself needs to be chunked. Keep
      // chunking and finalizing until finished
      while (curData.length > 0) {
        const curDataToChunkLen = Math.min(
          curData.length,
          prefillChunkSize - curChunkLen,
        );
        curChunk.push(curData.slice(0, curDataToChunkLen));
        curChunkLen += curDataToChunkLen;
        curData = curData.slice(curDataToChunkLen);
        if (curChunkLen === prefillChunkSize) {
          // curChunk is now full, so finalize to chunks
          chunks.push([...curChunk]);
          chunkLens.push(curChunkLen);
          curChunk = [];
          curChunkLen = 0;
        }
      }
    } else {
      // 2.2. Image data, which itself cannot be chunked, so cannot fit in current chunk.
      // 2.2.1. Finalize curChunk
      if (curChunk.length === 0) {
        throw new Error(
          "InternalError: do not expect curChunk to be empty when an image does not fit.",
        );
      }
      chunks.push([...curChunk]);
      chunkLens.push(curChunkLen);
      // 2.2.2. Then push image to the new chunk
      curChunk = [curData];
      curChunkLen = IMAGE_EMBED_SIZE;
      if (curChunkLen === prefillChunkSize) {
        chunks.push([...curChunk]);
        chunkLens.push(curChunkLen);
        curChunk = [];
        curChunkLen = 0;
      }
    }
  }
  // Last chunk
  if (curChunk.length > 0) {
    chunks.push([...curChunk]);
    chunkLens.push(curChunkLen);
  }

  return [chunks, chunkLens];
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

// Image related
type ImageURL = ChatCompletionContentPartImage.ImageURL;

// TODO(Charlie): currently hardcoded for phi3.5-vision num_crops 16
export const IMAGE_EMBED_SIZE = 1921;

/**
 * Given a url, get the image data. The url can either start with `http` or `data:image`.
 */
export async function getImageDataFromURL(url: string): Promise<ImageData> {
  const response = await fetch(url, { mode: "cors" });
  const img = await createImageBitmap(await response.blob());
  const canvas = new OffscreenCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not get 2d context");
  }
  ctx.drawImage(img, 0, 0);

  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  return imageData;
}

/**
 * Given an ImageData, return the RGB array in Uint8ClampedArray. Note the ImageData.data
 * is RGBA, so we skip every fourth element of the data. The order goes by rows from the
 * top-left pixel to the bottom-right, in RGB order.
 */
export function getRGBArrayFromImageData(
  imageData: ImageData,
): Uint8ClampedArray {
  const newData = new Uint8ClampedArray(imageData.width * imageData.height * 3);
  for (let i = 0, offset = 0; i < imageData.data.length; i += 4) {
    newData[offset++] = imageData.data[i];
    newData[offset++] = imageData.data[i + 1];
    newData[offset++] = imageData.data[i + 2];
  }
  return newData;
}
