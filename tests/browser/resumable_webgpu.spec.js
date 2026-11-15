import { expect, test, loadModel, modelId } from "./webgpu.mjs";

// Explicitly opt in: downloads model weights and needs a checkpoint-capable
// model library, a WebGPU adapter, and shader-f16 support. No mocked inference.
for (const [durabilityMode, strictPersistence] of [
  ["exact", true],
  ["exact", false],
  ["relaxed", true],
  ["relaxed", false],
]) {
  for (const checkpointPrompt of [false, true]) {
    test(`real WebGPU survives repeated reloads with ${checkpointPrompt ? "KV" : "token"} recovery (${durabilityMode}, strict=${strictPersistence})`, async ({
      page,
    }) => {
      await loadModel(page);
      const sessionId = `browser-gpu-${checkpointPrompt}-${durabilityMode}`;
      const baseline = await page.evaluate(
        async ({
          modelId,
          checkpointPrompt,
          sessionId,
          durabilityMode,
          strictPersistence,
        }) => {
          const request = {
            model: modelId,
            messages: [
              {
                role: "user",
                content:
                  "What is 2 + 2? " +
                  "Remember this conversation for later. ".repeat(24),
              },
              { role: "assistant", content: "4." },
              { role: "user", content: "What is 3 + 3?" },
              { role: "assistant", content: "6." },
              { role: "user", content: "List the numbers from 1 through 20." },
            ],
            seed: 17,
            temperature: 0.7,
            max_tokens: 24,
            ignore_eos: true,
          };
          const response = await Promise.race([
            globalThis.gpuEngine.chatCompletion(request),
            globalThis.gpuFailure,
          ]);
          const stream = await globalThis.gpuEngine.chatCompletion({
            ...request,
            stream: true,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId,
                checkpointPrompt,
                checkpointIntervalTokens: 512,
                durabilityMode,
                strictPersistence,
              },
            },
          });
          globalThis.gpuStream = stream[Symbol.asyncIterator]();
          await globalThis.gpuStream.next();
          await globalThis.gpuStream.next();
          return {
            request,
            text: response.choices[0].message.content,
            promptTokens: response.usage.prompt_tokens,
          };
        },
        {
          modelId,
          checkpointPrompt,
          sessionId,
          durabilityMode,
          strictPersistence,
        },
      );
      expect(baseline.promptTokens).toBeGreaterThan(128);

      // Navigate without return()/interruptGenerate(): this destroys the engine
      // while generation is unfinished, releasing browser-owned Web Locks.
      await loadModel(page);
      const mode = await page.evaluate(async (sessionId) => {
        const sessions = await globalThis.gpuEngine.listResumableSessions();
        return sessions.find((session) => session.sessionId === sessionId)
          ?.recoveryMode;
      }, sessionId);
      expect(mode).toBe(checkpointPrompt ? "kv" : "token_replay");
      await page.evaluate(async (sessionId) => {
        const stream = await globalThis.gpuEngine.resumeChatCompletion(
          sessionId,
          { continueGeneration: true, stream: true },
        );
        globalThis.gpuStream = stream[Symbol.asyncIterator]();
        await globalThis.gpuStream.next();
        await globalThis.gpuStream.next();
      }, sessionId);

      await loadModel(page);
      const result = await page.evaluate(
        async ({ sessionId, request }) => {
          const resumed = await globalThis.gpuEngine.resumeChatCompletion(
            sessionId,
            { continueGeneration: true },
          );
          const metrics = globalThis.gpuEngine.lastResumableMetrics;
          const finished = (
            await globalThis.gpuEngine.listResumableSessions()
          ).find((session) => session.sessionId === sessionId);
          const saved =
            await globalThis.gpuEngine.resumeChatCompletion(sessionId);
          await globalThis.gpuEngine.deleteResumableSession(sessionId);
          const followup = {
            ...request,
            max_tokens: 4,
            messages: [
              ...request.messages,
              { role: "assistant", content: resumed.recoveredText },
              { role: "user", content: "Continue." },
            ],
          };
          const warm = await globalThis.gpuEngine.chatCompletion(followup);
          const newSessionId = `${sessionId}-next`;
          const fresh = await globalThis.gpuEngine.chatCompletion({
            ...followup,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId: newSessionId,
                checkpointPrompt: false,
                strictPersistence: true,
              },
            },
          });
          await globalThis.gpuEngine.deleteResumableSession(newSessionId);
          await globalThis.gpuEngine.unload();
          return {
            resumed,
            metrics,
            finished,
            saved,
            warmPromptTokens: warm.usage.prompt_tokens,
            freshPromptTokens: fresh.usage.prompt_tokens,
          };
        },
        { sessionId, request: baseline.request },
      );
      expect(result.resumed.recoveredText).toBe(baseline.text);
      expect(result.resumed.recoveryMode).toBe(
        checkpointPrompt ? "kv" : "token_replay",
      );
      expect(result.finished).toMatchObject({
        resumable: false,
        recoveryMode: "none",
      });
      expect(result.saved).toMatchObject({
        recoveredText: baseline.text,
        recoveryMode: "text_only",
      });
      expect(result.warmPromptTokens).toBeLessThan(result.freshPromptTokens);
      expect(result.freshPromptTokens).toBeGreaterThan(baseline.promptTokens);
    });
  }
}
