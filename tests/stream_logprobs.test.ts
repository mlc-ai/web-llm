import { MLCEngine } from "../src/engine";
import { LLMChatPipeline } from "../src/llm_chat";
import { ResumableGenerationCoordinator } from "../src/resumable/coordinator";
import { jest, test, expect } from "@jest/globals";

function entry(token: string) {
  return { token, bytes: [], logprob: -1, top_logprobs: [] };
}

function fixture(logprobs = true) {
  const engine = Object.create(MLCEngine.prototype) as any;
  const pipeline = Object.create(LLMChatPipeline.prototype) as any;
  pipeline.outputMessage = "";
  pipeline.tokenLogprobArray = [];
  const state = {
    id: "stream",
    created: 0,
    prevMessageLength: 0,
    prevLogprobLength: 0,
  };
  const request = { messages: [], stream: true, logprobs };
  const emit = (message: string, token: string, skipEmptyDelta = false) => {
    pipeline.outputMessage = message;
    pipeline.tokenLogprobArray.push(entry(token));
    return engine.makeStreamDeltaChunk(
      request,
      "model",
      pipeline,
      state,
      skipEmptyDelta,
    );
  };
  return { engine, pipeline, state, request, emit };
}

test("completed UTF-8 chunks include buffered token logprobs once", () => {
  const { emit, state } = fixture();
  expect(emit("\uFFFD", "first byte token")).toBeUndefined();
  expect(state.prevLogprobLength).toBe(0);
  const combined = emit("€", "remaining bytes token");
  expect(combined.choices[0].delta.content).toBe("€");
  expect(
    combined.choices[0].logprobs.content.map((item: any) => item.token),
  ).toEqual(["first byte token", "remaining bytes token"]);
  const next = emit("€!", "!");
  expect(next.choices[0].delta.content).toBe("!");
  expect(next.choices[0].logprobs.content).toEqual([entry("!")]);
});

test("ASCII streaming keeps one logprob per emitted token", () => {
  const { emit } = fixture();
  for (const [message, token] of [
    ["a", "a"],
    ["ab", "b"],
  ]) {
    expect(emit(message, token).choices[0].logprobs.content).toEqual([
      entry(token),
    ]);
  }
});

test("disabled logprobs remain null", () => {
  const { emit } = fixture(false);
  expect(emit("a", "a").choices[0].logprobs).toBeNull();
});

test("skipped empty deltas do not consume pending logprobs", () => {
  const { emit, state } = fixture();
  expect(emit("", "empty", true)).toBeUndefined();
  expect(state.prevLogprobLength).toBe(0);
  expect(emit("a", "a", true).choices[0].logprobs.content).toEqual([
    entry("empty"),
    entry("a"),
  ]);
});

test.each([
  [0, false],
  [1, false],
  [0, true],
  [1, true],
] as const)(
  "resumed streams retain %s new token (historical entries present: %s)",
  async (extraEmittedTokens, historicalEntries) => {
    const { engine, pipeline, request } = fixture();
    pipeline.outputMessage = "ab";
    // Real replay does not reconstruct historical logprobs. Also exercise
    // an already-populated array to protect the cursor against duplicate sends.
    pipeline.tokenLogprobArray = historicalEntries
      ? [entry("a"), entry("b")]
      : extraEmittedTokens
        ? [entry("b")]
        : [];
    const coordinator = Object.create(
      ResumableGenerationCoordinator.prototype,
    ) as any;
    coordinator.recordPendingResumeToken = jest.fn(async () => undefined);
    coordinator.closeStreamJournal = jest.fn(async () => undefined);
    coordinator.options = {
      host: {
        resetInterrupt: jest.fn(),
        async *streamCurrentGeneration(
          req: any,
          model: string,
          pipe: any,
          _config: any,
          _received: number,
          state: any,
        ) {
          if (extraEmittedTokens === 0) {
            pipe.outputMessage += "c";
            pipe.tokenLogprobArray.push(entry("c"));
          }
          yield engine.makeStreamDeltaChunk(req, model, pipe, state);
        },
      },
    };
    const stream = coordinator.streamContinuation(
      request,
      "model",
      pipeline,
      {},
      {},
      {},
      {
        extraEmittedTokens,
        streamDeltaStart: extraEmittedTokens ? 1 : undefined,
      },
      {},
      {},
      0,
      async () => undefined,
    );
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    expect(chunks[0].choices[0].logprobs.content).toEqual([
      entry(extraEmittedTokens ? "b" : "c"),
    ]);
  },
);
