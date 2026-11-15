import { test, expect, loadModel, modelId } from "./webgpu.mjs";

test("real WebGPU ordinary legacy completions stream and cancel without locking out later requests", async ({
  page,
}) => {
  await loadModel(page);
  const result = await page.evaluate(async () => {
    const request = {
      prompt: "List the numbers from 1 through 10.",
      max_tokens: 8,
      ignore_eos: true,
      seed: 17,
      temperature: 0.7,
    };
    const engine = globalThis.gpuEngine;
    const baseline = (await engine.completion(request)).choices[0].text;
    let streamed = "";
    for await (const chunk of await engine.completion({
      ...request,
      stream: true,
    })) {
      streamed += chunk.choices[0]?.text ?? "";
    }
    const cancelled = (await engine.completion({ ...request, stream: true }))[
      Symbol.asyncIterator
    ]();
    await cancelled.next();
    await cancelled.return();
    const unused = (await engine.completion({ ...request, stream: true }))[
      Symbol.asyncIterator
    ]();
    await unused.return();
    const afterCancel = (await engine.completion(request)).choices[0].text;
    return { baseline, streamed, afterCancel };
  });
  expect(result.baseline.length).toBeGreaterThan(0);
  expect(result.streamed).toBe(result.baseline);
  expect(result.afterCancel).toBe(result.baseline);
});

for (const endpoint of ["chatCompletion", "completion"]) {
  test(`real WebGPU ${endpoint} cancellation clears the request seed before the next request`, async ({
    page,
  }) => {
    await loadModel(page);
    const result = await page.evaluate(
      async ({ endpoint, modelId }) => {
        const engine = globalThis.gpuEngine;
        const pipeline = engine.loadedModelIdToPipeline.get(modelId);
        const assignedSeeds = [];
        const setSeed = pipeline.setSeed.bind(pipeline);
        pipeline.setSeed = (seed) => {
          assignedSeeds.push(seed);
          return setSeed(seed);
        };
        const request = {
          max_tokens: 8,
          ignore_eos: true,
          temperature: 0.7,
          ...(endpoint === "chatCompletion"
            ? { messages: [{ role: "user", content: "Name three animals." }] }
            : { prompt: "Name three animals." }),
        };
        for await (const chunk of await engine[endpoint]({
          ...request,
          seed: 17,
          stream: true,
        })) {
          void chunk;
          break;
        }
        const seedsAfterCancel = [...assignedSeeds];
        const following = await engine[endpoint](request);
        return { seedsAfterCancel, assignedSeeds, following };
      },
      { endpoint, modelId },
    );
    expect(result.seedsAfterCancel).toHaveLength(2);
    expect(result.seedsAfterCancel[0]).toBe(17);
    expect(result.seedsAfterCancel[1]).not.toBe(17);
    expect(result.assignedSeeds).toEqual(result.seedsAfterCancel);
    expect(result.following.choices).toHaveLength(1);
  });
}

test("real WebGPU chat cancellation preserves prefix reuse like explicit interruption", async ({
  page,
}) => {
  await loadModel(page);
  const results = await page.evaluate(async (modelId) => {
    const engine = globalThis.gpuEngine;
    const pipeline = engine.loadedModelIdToPipeline.get(modelId);
    const request = {
      messages: [
        { role: "user", content: "Name several colorful imaginary animals." },
      ],
      max_tokens: 8,
      ignore_eos: true,
      seed: 17,
      temperature: 0.7,
    };
    let resets = 0;
    const resetChat = pipeline.resetChat.bind(pipeline);
    pipeline.resetChat = (...args) => {
      resets++;
      return resetChat(...args);
    };
    const results = [];
    for (const mode of ["return", "interrupt", "complete"]) {
      let reply = "";
      let first = true;
      for await (const chunk of await engine.chatCompletion({
        ...request,
        stream: true,
      })) {
        reply += chunk.choices[0]?.delta?.content ?? "";
        if (first) {
          first = false;
          if (mode === "return") break;
          if (mode === "interrupt") await engine.interruptGenerate();
        }
      }
      const stopped = pipeline.stopped();
      const finishReason = pipeline.getFinishReason();
      const lastReply = pipeline.getConversationObject().messages.at(-1)[2];
      const before = resets;
      await engine.chatCompletion({
        ...request,
        messages: [
          ...request.messages,
          { role: "assistant", content: reply },
          { role: "user", content: "Name one more." },
        ],
      });
      results.push({
        mode,
        stopped,
        finishReason,
        replyFinalized: lastReply === reply,
        followupResets: resets - before,
      });
    }
    return results;
  }, modelId);
  for (const result of results) {
    expect(result.stopped).toBe(true);
    expect(result.replyFinalized).toBe(true);
    expect(result.followupResets).toBe(0);
    expect(result.finishReason).toBe(
      result.mode === "complete" ? "length" : "abort",
    );
  }
});
