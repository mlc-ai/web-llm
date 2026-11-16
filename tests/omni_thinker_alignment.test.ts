import { readFileSync } from "fs";
import { join } from "path";
const fixture = JSON.parse(
  readFileSync(
    join(__dirname, "fixtures/omni_alignment/hf_complete.json"),
    "utf8",
  ),
);
import {
  alignThinkerSpeechTrace,
  ThinkerSpeechTrace,
} from "../src/omni_thinker_alignment";

function realTrace(): ThinkerSpeechTrace {
  return {
    ...fixture,
    inputIds: new Int32Array(fixture.inputIds),
    generatedIds: new Int32Array(fixture.generatedIds),
    inputEmbeddings: fixture.inputEmbeddings.map(
      (x: number[]) => new Float32Array(x),
    ),
    hiddenStates: fixture.hiddenStates.map(
      (x: number[]) => new Float32Array(x),
    ),
    specialEmbeddings: new Float32Array(fixture.specialEmbeddings),
  };
}
function tinyTrace(): ThinkerSpeechTrace {
  return {
    inputIds: new Int32Array([151646, 3]),
    generatedIds: new Int32Array([5, 6, 151645]),
    inputEmbeddings: [
      new Float32Array([1, 2]),
      new Float32Array([3]),
      new Float32Array([4]),
    ],
    hiddenStates: [
      new Float32Array([10, 20]),
      new Float32Array([30]),
      new Float32Array([40]),
    ],
    specialEmbeddings: new Float32Array([100, 200, 300]),
    width: 1,
    vocabSize: 151936,
    audioFeaturesProvided: false,
    eosIds: [151643, 151645],
  };
}
test("matches pinned HF completed-answer alignment including final sampled EOS", () => {
  const result = alignThinkerSpeechTrace(realTrace());
  expect(Array.from(result.prefix.data)).toEqual(fixture.expectedPrefix);
  expect(Array.from(result.reply.data)).toEqual(fixture.expectedReply);
  expect(result.prefix.shape).toEqual([fixture.inputIds.length + 2, 4]);
  expect(result.reply.shape).toEqual([fixture.generatedIds.length, 4]);
});
test("zeroes audio placeholder embeddings only when audio features were supplied", () => {
  const trace = tinyTrace();
  expect(Array.from(alignThinkerSpeechTrace(trace).prefix.data)).toEqual([
    11, 22, 100, 33,
  ]);
  trace.audioFeaturesProvided = true;
  expect(Array.from(alignThinkerSpeechTrace(trace).prefix.data)).toEqual([
    10, 22, 100, 33,
  ]);
});
test("uses EOS cut and repeats PAD after the cut", () => {
  const trace = tinyTrace();
  trace.generatedIds = new Int32Array([5, 151645, 6]);
  expect(Array.from(alignThinkerSpeechTrace(trace).reply.data)).toEqual([
    200, 300, 300,
  ]);
});
test("bounded non-EOS trace still appends EOS and PAD conditioning", () => {
  const trace = tinyTrace();
  trace.generatedIds = new Int32Array([5, 6, 7]);
  expect(Array.from(alignThinkerSpeechTrace(trace).reply.data)).toEqual([
    44, 200, 300,
  ]);
});
test("adds in float32 without modifying caller arrays", () => {
  const trace = tinyTrace();
  trace.hiddenStates[1][0] = 16777216;
  trace.inputEmbeddings[1][0] = 1;
  expect(alignThinkerSpeechTrace(trace).prefix.data[3]).toBe(16777216);
  expect(trace.hiddenStates[1][0]).toBe(16777216);
  expect(trace.inputEmbeddings[1][0]).toBe(1);
});
test.each([
  "missingState",
  "nonfinite",
  "visual",
  "badToken",
  "overflow",
  "singleToken",
])("rejects %s trace", (kind) => {
  const trace = tinyTrace();
  if (kind === "missingState") trace.hiddenStates = trace.hiddenStates.slice(1);
  if (kind === "nonfinite") trace.hiddenStates[0][0] = NaN;
  if (kind === "visual") trace.inputIds[0] = 151655;
  if (kind === "badToken") trace.generatedIds[0] = -1;
  if (kind === "overflow") {
    trace.hiddenStates[1][0] = 3.4e38;
    trace.inputEmbeddings[1][0] = 3.4e38;
  }
  if (kind === "singleToken") trace.generatedIds = new Int32Array([5]);
  expect(() => alignThinkerSpeechTrace(trace)).toThrow();
});
