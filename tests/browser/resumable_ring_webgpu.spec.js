import {
  expect,
  test,
  loadModel,
  makeRequest,
  baselineText,
} from "./webgpu.mjs";

// Resumable requests decode on the readback ring like any other request.
// Checkpoints are taken at committed positions while the ring has speculative
// tokens in flight, and the recovered continuation runs on the ring again and
// stops inside its speculative window.
for (const [sampling, overrides] of [
  ["greedy", { temperature: 0 }],
  ["seeded", { temperature: 0.7, seed: 17 }],
]) {
  for (const recoveryMode of ["kv", "token_replay"]) {
    test(`real WebGPU ring recovery is exact with a stop in the speculative window (${sampling}, ${recoveryMode})`, async ({
      page,
    }) => {
      await loadModel(page);
      const unstopped = makeRequest({
        ...overrides,
        max_tokens: 200,
        ignore_eos: false,
        extra_body: { sample_readback_max_speculative_tokens: 8 },
      });
      const full = await baselineText(page, unstopped);
      // A stop string whose first occurrence is late in the reply, after the
      // crash point below.
      let stop;
      for (let at = Math.floor(full.length * 0.7); at < full.length - 4; at++) {
        const candidate = full.slice(at, at + 4);
        if (full.indexOf(candidate) >= full.length * 0.6) {
          stop = candidate;
          break;
        }
      }
      expect(stop).toBeDefined();
      const request = { ...unstopped, stop: [stop] };
      const expected = await baselineText(page, request);
      expect(full.startsWith(expected)).toBe(true);
      expect(expected.length).toBeLessThan(full.length);

      const sessionId = `ring-${sampling}-${recoveryMode}`;
      const kv = recoveryMode === "kv";
      const crashed = await page.evaluate(
        async ({ request, sessionId, kv }) => {
          const { setResumableFaultHook } = globalThis.webllmBrowserHarness;
          let commits = 0;
          setResumableFaultHook((point) => {
            if (point === "checkpoint.after_commit") commits++;
          });
          const stream = await globalThis.gpuEngine.chatCompletion({
            ...request,
            stream: true,
            extra_body: {
              ...request.extra_body,
              resumable: {
                enabled: true,
                sessionId,
                checkpointPrompt: kv,
                checkpointIntervalTokens: kv ? 16 : 4096,
                strictPersistence: true,
              },
            },
          });
          globalThis.gpuStream = stream[Symbol.asyncIterator]();
          let text = "";
          for (let chunks = 0; kv ? commits < 2 : chunks < 30; chunks++) {
            const { done, value } = await globalThis.gpuStream.next();
            if (done) throw new Error("Generation finished before the crash");
            text += value.choices[0]?.delta.content ?? "";
          }
          const pipeline = globalThis.gpuEngine.loadedModelIdToPipeline.get(
            request.model,
          );
          return {
            text,
            mode: pipeline.sampleReadbackPathSelection.activeMode,
          };
        },
        { request, sessionId, kv },
      );
      expect(crashed.mode).toBe("ring_vector");
      expect(expected.startsWith(crashed.text)).toBe(true);

      // Navigating destroys the engine mid-generation.
      await loadModel(page);
      const resumed = await page.evaluate(
        (sessionId) =>
          globalThis.gpuEngine.resumeChatCompletion(sessionId, {
            continueGeneration: true,
          }),
        sessionId,
      );
      expect(resumed).toMatchObject({ recoveredText: expected, recoveryMode });
    });
  }
}
