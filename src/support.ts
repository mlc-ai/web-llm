/** Util methods. */
import { Tokenizer } from "@mlc-ai/web-tokenizers";

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
 * Post-process a raw token (which may be a raw byte or contain lower one eights block) to the
 * actual token. We do this in order to conform with the tokenizers' setup.
 *
 * Follow implementation of [https://github.com/mlc-ai/mlc-llm/blob/
 * bcb9b6a33a672a70d760c9a8b03234124aab50c4/cpp/tokenizers.cc#L99]
 */
export function postProcessToken(token: string): string {
  // 1. The token represents a byte.
  const charCode0 = "0".charCodeAt(0);
  const charCode9 = "9".charCodeAt(0);
  const charCodeA = "A".charCodeAt(0);
  if (
    token.length == 6 &&
    token.substring(0, 3) === "<0x" &&
    token.slice(-1) === ">"
  ) {
    let byte = 0;
    for (let i = 0; i < 2; i++) {
      byte *= 16;
      const curCharCode = token.charCodeAt(3 + i);
      if (curCharCode >= charCode0 && curCharCode <= charCode9) {
        byte += curCharCode - charCode0;
      } else {
        byte += curCharCode - charCodeA + 10;
      }
    }
    if (byte < 0 || byte >= 256) {
      throw Error("Expect byte to be in range [0, 256).");
    }
    return String.fromCharCode(byte);
  }

  // 2. The token contains lower one eight block which means space, e.g. `▁response` in Llama-2.
  // https://www.compart.com/en/unicode/U+2581
  const lowerOneEighthBlock = "\u2581";
  token = token.split(lowerOneEighthBlock).join(" ");

  return token;
}

/**
 * Get the token table in the form of a string list of tokens, ordered by their token id.
 * @param tokenizer A loaded tokenizer.
 */
export function getTokenTableFromTokenizer(tokenizer: Tokenizer): string[] {
  const tokenTable: string[] = [];
  const vocabSize = tokenizer.getVocabSize();
  for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
    const token = tokenizer.idToToken(tokenId);
    tokenTable.push(postProcessToken(token));
  }
  return tokenTable;
}
