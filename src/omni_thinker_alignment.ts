/** HF Qwen2.5-Omni text/audio alignment for the verified FP32 Thinker profile. */
import type { GraphTensor } from "./artifact_graph_executor";

export interface ThinkerSpeechTrace {
  inputIds: Int32Array;
  generatedIds: Int32Array;
  inputEmbeddings: readonly Float32Array[];
  hiddenStates: readonly Float32Array[];
  /** Speaker BOS, text EOS and text PAD embeddings, in that order. */
  specialEmbeddings: Float32Array;
  width: number;
  vocabSize: number;
  audioFeaturesProvided: boolean;
  eosIds: readonly number[];
}

/** The final sampled ID is retained even though it has no forwarded hidden state. */
export function alignThinkerSpeechTrace(trace: ThinkerSpeechTrace): {
  prefix: GraphTensor;
  reply: GraphTensor;
} {
  const {
    inputIds,
    generatedIds,
    inputEmbeddings,
    hiddenStates,
    specialEmbeddings,
    width,
    vocabSize,
    audioFeaturesProvided,
    eosIds,
  } = trace;
  if (
    !Number.isSafeInteger(width) ||
    width < 1 ||
    !Number.isSafeInteger(vocabSize) ||
    vocabSize < 1 ||
    !(inputIds instanceof Int32Array) ||
    inputIds.length < 1 ||
    !(generatedIds instanceof Int32Array) ||
    generatedIds.length < 2 ||
    inputEmbeddings.length !== generatedIds.length ||
    hiddenStates.length !== generatedIds.length ||
    typeof audioFeaturesProvided !== "boolean" ||
    [...inputIds, ...generatedIds, ...eosIds].some(
      (id) => !Number.isSafeInteger(id) || id < 0 || id >= vocabSize,
    ) ||
    inputIds.some((id) => [151652, 151653, 151654, 151655, 151656].includes(id))
  )
    throw new Error("Invalid Thinker speech trace cardinality or token IDs");
  const finite = (value: Float32Array, length: number) =>
    value instanceof Float32Array &&
    value.length === length &&
    value.every(Number.isFinite);
  if (!finite(specialEmbeddings, 3 * width))
    throw new Error("Invalid speaker/EOS/PAD embeddings");
  const n = generatedIds.length,
    prompt = inputIds.length;
  const sums = hiddenStates.map((hidden, index) => {
    const length = (index === 0 ? prompt : 1) * width;
    const embedded = inputEmbeddings[index];
    if (!finite(hidden, length) || !finite(embedded, length))
      throw new Error("Invalid Thinker hidden state or embedding");
    const value = Float32Array.from(
      hidden,
      (h, j) =>
        h +
        (index === 0 &&
        audioFeaturesProvided &&
        inputIds[Math.floor(j / width)] === 151646
          ? 0
          : embedded[j]),
    );
    if (!value.every(Number.isFinite))
      throw new Error("Thinker alignment overflow");
    return value;
  });
  const prefix = new Float32Array((prompt + 2) * width);
  prefix.set(sums[0]);
  prefix.set(specialEmbeddings.subarray(0, width), prompt * width);
  prefix.set(sums[1], (prompt + 1) * width);
  const reply = new Float32Array(n * width);
  for (let index = 2; index < n; index++)
    reply.set(sums[index], (index - 2) * width);
  reply.set(specialEmbeddings.subarray(width, 2 * width), (n - 2) * width);
  reply.set(specialEmbeddings.subarray(2 * width), (n - 1) * width);
  const firstEOS = generatedIds.findIndex((id) => eosIds.includes(id));
  const cut = Math.max(1, Math.min(firstEOS < 0 ? n : firstEOS, n - 1));
  reply.set(specialEmbeddings.subarray(width, 2 * width), (cut - 1) * width);
  for (let index = cut; index < n; index++)
    reply.set(specialEmbeddings.subarray(2 * width), index * width);
  return {
    prefix: { shape: [prompt + 2, width], data: prefix },
    reply: { shape: [n, width], data: reply },
  };
}
