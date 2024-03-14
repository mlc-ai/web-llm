/** Util methods. */

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
    num_top_probs: number, p_prob: Float32Array
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
