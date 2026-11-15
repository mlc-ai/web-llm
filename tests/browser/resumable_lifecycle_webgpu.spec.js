import {
  expect,
  test,
  loadModel,
  makeRequest,
  baselineText,
  inspectSession,
  watchGPUErrors,
} from "./webgpu.mjs";

for (const checkpointPrompt of [false, true]) {
  for (const durabilityMode of ["exact", "relaxed"]) {
    test(`real WebGPU browser-process crash, KV=${checkpointPrompt}, ${durabilityMode}`, async ({
      page,
      context,
      launchTestContext,
    }) => {
      await loadModel(page);
      const request = makeRequest({
        frequency_penalty: 0.2,
        presence_penalty: 0.1,
        repetition_penalty: 1.1,
        extra_body: { enable_thinking: false },
      });
      const baseline = await baselineText(page, request);
      const sessionId = "browser-process-crash";
      await page.evaluate(
        async ({ request, sessionId, checkpointPrompt, durabilityMode }) => {
          const stream = await globalThis.gpuEngine.chatCompletion({
            ...request,
            stream: true,
            extra_body: {
              ...request.extra_body,
              resumable: {
                enabled: true,
                sessionId,
                checkpointPrompt,
                durabilityMode,
                strictPersistence: true,
              },
            },
          });
          globalThis.gpuStream = stream[Symbol.asyncIterator]();
          for (let i = 0; i < 10; i++) await globalThis.gpuStream.next();
        },
        { request, sessionId, checkpointPrompt, durabilityMode },
      );
      const before = await inspectSession(page, sessionId);
      expect(
        before.records.filter((record) => record.type === 5).length,
      ).toBeGreaterThan(0);
      const cdp = await context.newCDPSession(page);
      // Crash the dedicated test browser, not just a page or GPU process. No
      // iterator.return(), engine.unload(), or graceful browser close precedes it.
      await expect(cdp.send("Browser.crash")).rejects.toThrow();
      await context.close();
      const restarted = await launchTestContext();
      const recoveredPage = await restarted.newPage();
      const checkErrors = await watchGPUErrors(recoveredPage);
      await loadModel(recoveredPage);
      const after = await inspectSession(recoveredPage, sessionId);
      const savedTokens = after.records.filter(
        (record) => record.type === 5,
      ).length;
      if (durabilityMode === "exact") {
        expect(savedTokens).toBe(
          before.records.filter((record) => record.type === 5).length,
        );
      } else {
        expect(savedTokens).toBeGreaterThanOrEqual(8);
      }
      const resumed = await recoveredPage.evaluate(
        (sessionId) =>
          globalThis.gpuEngine.resumeChatCompletion(sessionId, {
            continueGeneration: true,
          }),
        sessionId,
      );
      expect(resumed).toMatchObject({
        recoveredText: baseline,
        recoveryMode: checkpointPrompt ? "kv" : "token_replay",
      });
      expect(
        (await inspectSession(recoveredPage, sessionId)).directories,
      ).toEqual([]);
      checkErrors();
    });
  }
}

for (const cancellation of ["return", "interrupt"]) {
  test(`real WebGPU session isolation and ${cancellation} release locks`, async ({
    page,
    context,
  }) => {
    await loadModel(page);
    const request = makeRequest();
    const baseline = await baselineText(page, request);
    const sessionId = "session-a";
    const otherPage = await context.newPage();
    await otherPage.goto("/");
    await otherPage.waitForFunction(
      () => globalThis.webllmBrowserHarness !== undefined,
    );
    await otherPage.evaluate(() => {
      globalThis.reader = new globalThis.webllmBrowserHarness.MLCEngine();
    });
    // Creating (and closing) a never-started stream must not block an ordinary
    // request or create a session. The following real inference would hang if it did.
    const unstarted = await page.evaluate(async (request) => {
      const stream = await globalThis.gpuEngine.chatCompletion({
        ...request,
        stream: true,
        extra_body: {
          resumable: { enabled: true, sessionId: "never-started" },
        },
      });
      const normal = await globalThis.gpuEngine.chatCompletion(request);
      await stream[Symbol.asyncIterator]().return();
      return {
        text: normal.choices[0].message.content,
        sessions: await globalThis.gpuEngine.listResumableSessions(),
      };
    }, request);
    expect(unstarted).toEqual({ text: baseline, sessions: [] });

    await page.evaluate(
      async ({ request, sessionId }) => {
        const stream = await globalThis.gpuEngine.chatCompletion({
          ...request,
          stream: true,
          extra_body: {
            resumable: {
              enabled: true,
              sessionId,
              checkpointPrompt: false,
              strictPersistence: true,
            },
          },
        });
        globalThis.gpuStream = stream[Symbol.asyncIterator]();
        await globalThis.gpuStream.next();
        await globalThis.gpuStream.next();
        globalThis.otherSettled = false;
        globalThis.otherRequest = globalThis.gpuEngine
          .chatCompletion({
            ...request,
            messages: [{ role: "user", content: "Say hello." }],
            max_tokens: 4,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId: "session-b",
                checkpointPrompt: false,
                strictPersistence: true,
              },
            },
          })
          .then((result) => {
            globalThis.otherSettled = true;
            return result;
          });
      },
      { request, sessionId },
    );
    const active = await otherPage.evaluate(async (sessionId) => {
      const saved = await globalThis.reader.resumeChatCompletion(sessionId);
      const errors = [];
      for (const operation of [
        () => globalThis.reader.deleteResumableSession(sessionId),
        () =>
          globalThis.reader.resumeChatCompletion(sessionId, {
            continueGeneration: true,
          }),
      ]) {
        try {
          await operation();
        } catch (err) {
          errors.push(err.message);
        }
      }
      return { saved, errors };
    }, sessionId);
    expect(active.saved.emittedTokens).toBeGreaterThan(0);
    expect(active.errors).toHaveLength(2);
    for (const message of active.errors)
      expect(message).toContain("already active");
    expect(await page.evaluate(() => globalThis.otherSettled)).toBe(false);
    const cancelled = await page.evaluate(async (cancellation) => {
      if (cancellation === "return") {
        await globalThis.gpuStream.return();
      } else {
        globalThis.gpuEngine.interruptGenerate();
        while (!(await globalThis.gpuStream.next()).done) {
          /* drain abort */
        }
      }
      return (await globalThis.otherRequest).choices[0].message.content;
    }, cancellation);
    // Upstream's interrupt flag also aborts a queued non-streaming request
    // before prefill; no resumable session is created for that skipped request.
    if (cancellation === "interrupt") expect(cancelled).toBe("");
    else expect(cancelled.length).toBeGreaterThan(0);
    const storedA = await inspectSession(page, sessionId);
    expect(storedA.records.some((record) => record.type === 8)).toBe(true);
    // Model-less reading works after release; continuation stays text-only.
    const modelLess = await otherPage.evaluate(
      (sessionId) =>
        globalThis.reader.resumeChatCompletion(sessionId, {
          continueGeneration: true,
        }),
      sessionId,
    );
    expect(modelLess).toMatchObject({
      recoveredText: active.saved.recoveredText,
      recoveryMode: "text_only",
    });
    const resumed = await page.evaluate(
      (sessionId) =>
        globalThis.gpuEngine.resumeChatCompletion(sessionId, {
          continueGeneration: true,
        }),
      sessionId,
    );
    expect(resumed).toMatchObject({
      recoveredText: baseline,
      recoveryMode: "token_replay",
    });
    const reuse = await page.evaluate(
      async ({ request, sessionId }) => {
        const newRequest = {
          ...request,
          max_tokens: 4,
          extra_body: {
            resumable: {
              enabled: true,
              sessionId,
              checkpointPrompt: false,
              strictPersistence: true,
            },
          },
        };
        let duplicateError;
        try {
          await globalThis.gpuEngine.chatCompletion(newRequest);
        } catch (err) {
          duplicateError = err.message;
        }
        await globalThis.gpuEngine.deleteResumableSession(sessionId);
        const fresh = await globalThis.gpuEngine.chatCompletion(newRequest);
        return {
          duplicateError,
          completionTokens: fresh.usage.completion_tokens,
          sessions: await globalThis.gpuEngine.listResumableSessions(),
        };
      },
      { request, sessionId },
    );
    expect(reuse.duplicateError).toContain("already exists");
    // The usage counter counts decode steps (not the prefill-sampled token).
    expect(reuse.completionTokens).toBe(3);
    expect(
      (await inspectSession(page, sessionId)).records.filter(
        (record) => record.type === 5,
      ),
    ).toHaveLength(4);
    expect(reuse.sessions.map((session) => session.sessionId).sort()).toEqual([
      "session-a",
      ...(cancellation === "return" ? ["session-b"] : []),
    ]);
    await otherPage.close();
  });
}

test("real WebGPU low reported quota skips KV but retains strict token recovery", async ({
  page,
}) => {
  await loadModel(page);
  const request = makeRequest();
  const baseline = await baselineText(page, request);
  // Modern Chromium pads estimates independently of the enforced quota. Inject
  // only the low estimate to exercise this advisory preflight branch.
  await page.evaluate(() => {
    globalThis.navigator.storage.estimate = async () => ({
      usage: 0,
      quota: 256 * 1024 * 1024,
    });
  });
  const free = await page.evaluate(async () => {
    const { quota, usage } = await globalThis.navigator.storage.estimate();
    return quota - usage;
  });
  expect(free).toBeLessThan(512 * 1024 * 1024);
  const sessionId = "low-quota";
  await page.evaluate(
    async ({ request, sessionId }) => {
      const stream = await globalThis.gpuEngine.chatCompletion({
        ...request,
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId,
            checkpointPrompt: true,
            strictPersistence: true,
          },
        },
      });
      globalThis.gpuStream = stream[Symbol.asyncIterator]();
      await globalThis.gpuStream.next();
      await globalThis.gpuStream.next();
    },
    { request, sessionId },
  );
  expect((await inspectSession(page, sessionId)).checkpoints).toEqual([]);
  await loadModel(page);
  const resumed = await page.evaluate(
    (sessionId) =>
      globalThis.gpuEngine.resumeChatCompletion(sessionId, {
        continueGeneration: true,
      }),
    sessionId,
  );
  expect(resumed).toMatchObject({
    recoveredText: baseline,
    recoveryMode: "token_replay",
  });
});

test("real WebGPU ordinary interrupt also aborts a queued non-streaming request", async ({
  page,
}) => {
  await loadModel(page);
  const result = await page.evaluate(async (request) => {
    const stream = await globalThis.gpuEngine.chatCompletion({
      ...request,
      stream: true,
    });
    const iterator = stream[Symbol.asyncIterator]();
    await iterator.next();
    const queued = globalThis.gpuEngine.chatCompletion(request);
    globalThis.gpuEngine.interruptGenerate();
    while (!(await iterator.next()).done) {
      /* drain abort */
    }
    return {
      response: await queued,
      sessions: await globalThis.gpuEngine.listResumableSessions(),
    };
  }, makeRequest());
  expect(result.response.choices[0].message.content).toBe("");
  expect(result.sessions).toEqual([]);
});

for (const strictPersistence of [false, true]) {
  test(`real WebGPU enforced quota failure, strict=${strictPersistence}`, async ({
    page,
    context,
  }) => {
    await loadModel(page);
    const request = makeRequest();
    const baseline = await baselineText(page, request);
    const cdp = await context.newCDPSession(page);
    const origin = "http://127.0.0.1:4178";
    const actual = await cdp.send("Storage.getUsageAndQuota", { origin });
    await cdp.send("Storage.overrideQuotaForOrigin", {
      origin,
      quotaSize: actual.usage + 1024 * 1024,
    });
    const limited = await cdp.send("Storage.getUsageAndQuota", { origin });
    expect(limited.overrideActive).toBe(true);
    expect(limited.quota - limited.usage).toBeLessThan(2 * 1024 * 1024);
    const reported = await page.evaluate(async () => {
      const { quota, usage } = await globalThis.navigator.storage.estimate();
      return quota - usage;
    });
    // Confirms the case where preflight cannot predict the actual write failure.
    expect(reported).toBeGreaterThan(512 * 1024 * 1024);
    const sessionId = "enforced-quota";
    const result = await page.evaluate(
      async ({ request, sessionId, strictPersistence }) => {
        const prototype = globalThis.FileSystemWritableFileStream.prototype;
        const write = prototype.write;
        const writeErrors = [];
        // Observe the browser's actual quota rejection without changing it.
        prototype.write = async function (...args) {
          try {
            return await write.apply(this, args);
          } catch (err) {
            writeErrors.push({ name: err.name, message: err.message });
            throw err;
          }
        };
        try {
          const response = await globalThis.gpuEngine.chatCompletion({
            ...request,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId,
                checkpointPrompt: true,
                strictPersistence,
              },
            },
          });
          return { text: response.choices[0].message.content, writeErrors };
        } catch (err) {
          return { error: err.message, errorName: err.name, writeErrors };
        } finally {
          prototype.write = write;
        }
      },
      { request, sessionId, strictPersistence },
    );
    const stored = await inspectSession(page, sessionId);
    expect(stored.checkpoints).toEqual([]);
    expect(result.writeErrors).toContainEqual(
      expect.objectContaining({ name: "QuotaExceededError" }),
    );
    if (strictPersistence) {
      expect(result.errorName).toBe("QuotaExceededError");
      expect(stored.records.some((record) => record.type === 7)).toBe(false);
    } else {
      expect(result.text).toBe(baseline);
      expect(result.error).toBeUndefined();
      expect(stored.records.some((record) => record.type === 7)).toBe(true);
    }
    await cdp.send("Storage.overrideQuotaForOrigin", { origin });
    await loadModel(page);
    const recovered = await page.evaluate(
      (sessionId) =>
        globalThis.gpuEngine.resumeChatCompletion(sessionId, {
          continueGeneration: true,
        }),
      sessionId,
    );
    expect(recovered).toMatchObject({
      recoveredText: baseline,
      recoveryMode: strictPersistence ? "token_replay" : "text_only",
    });
  });
}

test("real WebGPU legacy completions and unsupported resumable inputs reject without sessions", async ({
  page,
}) => {
  await loadModel(page);
  const errors = await page.evaluate(async (request) => {
    const resumable = { enabled: true, sessionId: "rejected" };
    const requests = [
      () =>
        globalThis.gpuEngine.completion({
          model: request.model,
          prompt: "Hello",
          extra_body: { resumable },
        }),
      () =>
        globalThis.gpuEngine.chatCompletion({
          ...request,
          n: 2,
          extra_body: { resumable },
        }),
      () =>
        globalThis.gpuEngine.chatCompletion({
          ...request,
          response_format: { type: "json_object" },
          extra_body: { resumable },
        }),
      () =>
        globalThis.gpuEngine.chatCompletion({
          ...request,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: { url: "https://example.invalid/image.png" },
                },
              ],
            },
          ],
          extra_body: { resumable },
        }),
    ];
    const errors = [];
    for (const run of requests) {
      try {
        await run();
        errors.push(undefined);
      } catch (err) {
        errors.push(err.message);
      }
    }
    return {
      errors,
      sessions: await globalThis.gpuEngine.listResumableSessions(),
    };
  }, makeRequest());
  expect(errors.sessions).toEqual([]);
  expect(errors.errors[0]).toContain("extra_body.resumable");
  for (const error of errors.errors) expect(typeof error).toBe("string");
  // Rejection must leave the model usable for ordinary requests.
  expect((await baselineText(page, makeRequest())).length).toBeGreaterThan(0);
});
