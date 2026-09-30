import {
  MODEL_ID,
  EMBED_MODEL_ID,
  FIXED_CREATED_DATE,
  FIXED_CREATED_SECONDS,
  createEngineWithPipeline,
  createEngineWithMultiplePipelines,
  createEngineWithEmbeddingPipeline,
} from "./helpers/engine_fixture";
import {
  ChatCompletion,
  ChatCompletionRequest,
  Completion,
  CompletionCreateParams,
  EmbeddingCreateParams,
  ChatCompletionChunk,
} from "../src/openai_api_protocols";
import { MLCEngine } from "../src/engine";
import { UnclearModelToUseError } from "../src/error";
import { jest, test, expect, describe, afterEach } from "@jest/globals";
afterEach(() => {
  jest.useRealTimers();
});
describe("MLCEngine deterministic integration", () => {
  test("chatCompletion aggregates usage without WebGPU", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine, pipeline } = createEngineWithPipeline(3);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "system", content: "Stay concise." },
        { role: "user", content: "What is new?" },
      ],
      n: 2,
    };
    const response = (await engine.chatCompletion(request)) as ChatCompletion;

    expect(response.choices).toHaveLength(2);
    response.choices.forEach((choice) => {
      expect(choice.message?.content).toContain("What is new?");
    });
    expect(response.created).toBe(FIXED_CREATED_SECONDS);
    expect(response.usage?.completion_tokens).toBe(8);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
    expect((pipeline as any).prefillCallCount).toBe(2);
  });

  test("usage counts the token sampled after prefill", async () => {
    const { engine } = createEngineWithPipeline(2);
    const response = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Count" }],
    })) as ChatCompletion;

    // One token comes from prefill and two from decode steps.
    expect(response.usage?.completion_tokens).toBe(3);
    expect(response.usage?.total_tokens).toBe(
      3 + (response.usage?.prompt_tokens ?? 0),
    );
    expect(response.usage?.extra?.time_per_output_token_s).toBeGreaterThan(0);
    expect(Number.isFinite(response.usage?.extra?.decode_tokens_per_s)).toBe(
      true,
    );
  });

  test("a prior interrupt does not abort the next non-streaming chat request", async () => {
    const { engine, pipeline } = createEngineWithPipeline(2);
    await engine.interruptGenerate();

    const response = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Continue" }],
    })) as ChatCompletion;

    expect(response.choices[0].message.content).toContain("Continue");
    expect(response.choices[0].finish_reason).not.toBe("abort");
    expect((pipeline as any).prefillCallCount).toBe(1);
  });

  test("a prior interrupt does not abort the next non-streaming text request", async () => {
    const { engine, pipeline } = createEngineWithPipeline(2);
    await engine.interruptGenerate();

    const response = (await engine.completion({
      model: MODEL_ID,
      prompt: "Continue",
    })) as Completion;

    expect(response.choices[0].text).toContain("Continue");
    expect(response.choices[0].finish_reason).not.toBe("abort");
    expect((pipeline as any).prefillCallCount).toBe(1);
  });

  test("completion echoes prompt when requested", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine } = createEngineWithPipeline(1);
    const request: CompletionCreateParams = {
      model: MODEL_ID,
      prompt: "Alpha ",
      n: 1,
      echo: true,
    };
    const response = (await engine.completion(request)) as Completion;

    expect(response.choices).toHaveLength(1);
    expect(response.choices[0].text.startsWith("Alpha ")).toBe(true);
    expect(response.created).toBe(FIXED_CREATED_SECONDS);
    expect(response.usage?.completion_tokens).toBe(2);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  test("forwardTokensAndSample and runtimeStatsText use mock pipeline", async () => {
    const { engine } = createEngineWithPipeline();
    await expect(
      engine.forwardTokensAndSample([9, 4, 2], true, MODEL_ID),
    ).resolves.toBe(9);
    await expect(engine.runtimeStatsText(MODEL_ID)).resolves.toContain(
      "prefills=",
    );
  });

  test("chatCompletion streaming yields chunks, final delta, and usage data", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine } = createEngineWithPipeline(2);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "system", content: "rules" },
        { role: "user", content: "Stream please" },
      ],
      stream: true,
      stream_options: { include_usage: true },
    };
    const iterable = (await engine.chatCompletion(
      request,
    )) as AsyncIterable<ChatCompletionChunk>;
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of iterable) {
      chunks.push(chunk);
    }
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    expect(chunks[0].choices[0].delta?.content).toContain("Stream please");
    expect(
      chunks.every((chunk) => chunk.created === FIXED_CREATED_SECONDS),
    ).toBe(true);
    const finalChunk = chunks[chunks.length - 2];
    expect(finalChunk.choices[0].finish_reason).toEqual("stop");
    const usageChunk = chunks[chunks.length - 1];
    expect(usageChunk.usage?.completion_tokens).toBeGreaterThan(0);
    expect(usageChunk.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  test("chatCompletion without specifying model when multiple loaded throws error", async () => {
    const engine = createEngineWithMultiplePipelines();
    await expect(
      engine.chatCompletion({
        // purposely omit model to trigger ambiguity
        model: undefined as any,
        messages: [{ role: "user", content: "Hello" }],
      }),
    ).rejects.toBeInstanceOf(UnclearModelToUseError);
  });

  test("embedding API uses mock pipeline and returns usage", async () => {
    const { engine } = createEngineWithEmbeddingPipeline();
    const request: EmbeddingCreateParams = {
      model: EMBED_MODEL_ID,
      input: "abc",
    };
    const response = await engine.embedding(request);
    expect(response.data).toHaveLength(1);
    expect(response.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
    expect(response.usage?.extra?.prefill_tokens_per_s).toBeGreaterThan(0);
  });
});

describe("ordinary stream lifecycle", () => {
  for (const endpoint of ["chat", "completion"] as const) {
    const start = async (engine: MLCEngine, seed?: number) => {
      const stream =
        endpoint === "chat"
          ? await engine.chatCompletion({
              model: MODEL_ID,
              messages: [{ role: "user", content: "Stream" }],
              stream: true,
              seed,
            })
          : await engine.completion({
              model: MODEL_ID,
              prompt: "Stream",
              stream: true,
              seed,
            });
      return stream[Symbol.asyncIterator]();
    };

    test(`${endpoint}: unconsumed stream owns no lock, including return before next`, async () => {
      const { engine } = createEngineWithPipeline(4);
      const lock = (engine as any).loadedModelIdToLock.get(MODEL_ID);
      const acquire = jest.spyOn(lock, "acquire");
      const release = jest.spyOn(lock, "release");
      const iterator = await start(engine);
      expect(acquire).not.toHaveBeenCalled();
      await iterator.return!();
      expect(acquire).not.toHaveBeenCalled();
      expect(release).not.toHaveBeenCalled();
      expect(await iterator.next()).toMatchObject({ done: true });
    });

    test(`${endpoint}: return releases a started stream exactly once`, async () => {
      const { engine } = createEngineWithPipeline(4);
      const lock = (engine as any).loadedModelIdToLock.get(MODEL_ID);
      const release = jest.spyOn(lock, "release");
      const iterator = await start(engine);
      await iterator.next();
      await iterator.return!();
      await iterator.return!();
      expect(release).toHaveBeenCalledTimes(1);
      const next = await start(engine);
      expect(await next.next()).toMatchObject({ done: false });
      await next.return!();
    });

    test(`${endpoint}: closing an unused seeded stream leaves pipeline state untouched`, async () => {
      const { engine, pipeline } = createEngineWithPipeline(4);
      const seed = jest.spyOn(pipeline, "setSeed");
      const stop = jest.spyOn(pipeline, "triggerStop");
      const iterator = await start(engine, 17);
      await iterator.return!();
      expect(seed).not.toHaveBeenCalled();
      expect(stop).not.toHaveBeenCalled();
    });

    test(`${endpoint}: cancellation stops the reply and resets its seed before releasing the lock`, async () => {
      jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
      const { engine, pipeline } = createEngineWithPipeline(4);
      const seed = jest.spyOn(pipeline, "setSeed");
      const stop = jest.spyOn(pipeline, "triggerStop");
      const lock = (engine as any).loadedModelIdToLock.get(MODEL_ID);
      const release = jest.spyOn(lock, "release");
      const iterator = await start(engine, 17);
      await iterator.next();
      await iterator.return!();
      await iterator.return!();
      expect(pipeline.stopped()).toBe(true);
      expect(stop).toHaveBeenCalledTimes(1);
      expect(seed.mock.calls).toEqual([[17], [FIXED_CREATED_DATE.getTime()]]);
      expect(seed.mock.invocationCallOrder[1]).toBeLessThan(
        release.mock.invocationCallOrder[0],
      );
      expect(stop.mock.invocationCallOrder[0]).toBeLessThan(
        release.mock.invocationCallOrder[0],
      );
      const next = await start(engine);
      await next.next();
      await next.return!();
      expect(seed).toHaveBeenCalledTimes(2);
    });

    test(`${endpoint}: normal completion resets the seed without aborting the reply`, async () => {
      jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
      const { engine, pipeline } = createEngineWithPipeline(2);
      const seed = jest.spyOn(pipeline, "setSeed");
      const stop = jest.spyOn(pipeline, "triggerStop");
      const iterator = await start(engine, 17);
      while (!(await iterator.next()).done) {
        // Consume through the final chunk so normal cleanup runs.
      }
      expect(seed.mock.calls).toEqual([[17], [FIXED_CREATED_DATE.getTime()]]);
      expect(stop).not.toHaveBeenCalled();
    });

    for (const stage of ["prefill", "decode"] as const) {
      test(`${endpoint}: ${stage} failure releases the model lock`, async () => {
        const { engine, pipeline } = createEngineWithPipeline(4);
        const seed = jest.spyOn(pipeline, "setSeed");
        const stop = jest.spyOn(pipeline, "triggerStop");
        const lock = (engine as any).loadedModelIdToLock.get(MODEL_ID);
        const release = jest.spyOn(lock, "release");
        const error = new Error("inference failed");
        const fail = jest
          .spyOn(engine as any, stage)
          .mockRejectedValueOnce(error);
        const iterator = await start(engine, 17);
        if (stage === "decode") await iterator.next();
        await expect(iterator.next()).rejects.toBe(error);
        expect(release).toHaveBeenCalledTimes(1);
        expect(seed).toHaveBeenCalledTimes(2);
        expect(seed.mock.calls[0]).toEqual([17]);
        expect(seed.mock.calls[1][0]).not.toBe(17);
        expect(stop).toHaveBeenCalledTimes(stage === "decode" ? 1 : 0);
        fail.mockRestore();
        const next = await start(engine);
        expect(await next.next()).toMatchObject({ done: false });
        await next.return!();
      });
    }

    test(`${endpoint}: concurrent iterators wait for the active stream`, async () => {
      const { engine, pipeline } = createEngineWithPipeline(4);
      const first = await start(engine);
      const second = await start(engine);
      await first.next();
      let secondStarted = false;
      const pending = second.next().then((value) => {
        secondStarted = true;
        return value;
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(secondStarted).toBe(false);
      expect(pipeline.prefillCallCount).toBe(1);
      await first.return!();
      expect(await pending).toMatchObject({ done: false });
      expect(pipeline.prefillCallCount).toBe(2);
      await second.return!();
    });
  }
});
