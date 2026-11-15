/** Qwen2.5 Omni codec sampling with caller-controlled categorical draws. */
export interface TalkerSamplingOptions {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  random: () => number;
}

/** Apply HF processor order, then sample the filtered distribution by inverse CDF. */
export function sampleTalkerToken(
  logits: Float32Array,
  history: readonly number[],
  options: TalkerSamplingOptions,
): number {
  const temperature = Math.fround(options.temperature);
  const penalty = Math.fround(options.repetitionPenalty);
  if (
    logits.length !== 8448 ||
    !logits.every(Number.isFinite) ||
    !Number.isFinite(temperature) ||
    temperature <= 0 ||
    !Number.isFinite(penalty) ||
    penalty <= 0 ||
    !Number.isSafeInteger(options.topK) ||
    options.topK < 0 ||
    options.topK > logits.length ||
    !Number.isFinite(options.topP) ||
    options.topP <= 0 ||
    options.topP > 1 ||
    typeof options.random !== "function"
  )
    throw new Error("Invalid Talker sampling inputs");

  const scores = new Float32Array(logits);
  for (const token of new Set(history)) {
    if (!Number.isSafeInteger(token) || token < 0 || token >= scores.length)
      throw new Error("Invalid Talker sampling history");
    scores[token] =
      scores[token] < 0 ? scores[token] * penalty : scores[token] / penalty;
  }
  // Suppress BOS before temperature scaling, as in HF generation.
  scores[8293] = -Infinity;
  for (let i = 0; i < scores.length; i++) {
    if (i === 8293) continue;
    scores[i] /= temperature;
    if (!Number.isFinite(scores[i]))
      throw new Error("Talker sampling score overflow");
  }
  const sorted = Array.from(scores.keys()).sort(
    (a, b) => scores[a] - scores[b] || a - b,
  );
  if (options.topK > 0) {
    const cutoff = scores[sorted[sorted.length - options.topK]];
    for (const token of sorted)
      if (scores[token] < cutoff) scores[token] = -Infinity;
  }
  const probabilities = () => {
    const maximum = scores[sorted[sorted.length - 1]];
    const weights = Float64Array.from(scores, (score) =>
      Math.exp(score - maximum),
    );
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return Float32Array.from(weights, (weight) => weight / total);
  };
  if (options.topP < 1) {
    const weights = probabilities();
    const threshold = Math.fround(1 - options.topP);
    let cumulative = 0;
    // Keep at least the highest-scoring token. Equal scores use token-ID order.
    for (let i = 0; i < sorted.length - 1; i++) {
      const token = sorted[i];
      cumulative += weights[token];
      if (Math.fround(cumulative) <= threshold) scores[token] = -Infinity;
    }
  }
  const weights = probabilities();
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const uniform = options.random();
  if (!Number.isFinite(uniform) || uniform < 0 || uniform >= 1)
    throw new Error("Talker random draw must be in [0, 1)");
  const draw = uniform * total;
  let cumulative = 0,
    last = -1;
  for (let token = 0; token < weights.length; token++) {
    if (weights[token] > 0) last = token;
    cumulative += weights[token];
    if (draw < cumulative) return token;
  }
  if (last < 0) throw new Error("Empty Talker sampling distribution");
  return last;
}
