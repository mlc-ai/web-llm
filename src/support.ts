/** Util methods. */
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { MessagePlaceholders } from "./config";

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
