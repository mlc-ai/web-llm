import {
  expect,
  test,
  loadModel,
  makeRequest,
  baselineText,
  inspectSession,
} from "./webgpu.mjs";

// Pause production writes at a known boundary, then destroy the page without
// unwinding the generator. Inference and all writes before the pause are real.
for (const [point, storeCheckpointLogits, recoveryMode] of [
  ["journal.before_append", true, "token_replay"],
  ["checkpoint.after_page_group", true, "token_replay"],
  ["checkpoint.after_next_logits", true, "token_replay"],
  ["checkpoint.after_meta", true, "token_replay"],
  ["checkpoint.after_complete", true, "token_replay"],
  ["checkpoint.before_commit", true, "token_replay"],
  ["checkpoint.after_commit", true, "kv"],
  ["checkpoint.after_commit", false, "token_replay"],
]) {
  test(`real WebGPU zero-token recovery at ${point}, logits=${storeCheckpointLogits}`, async ({
    page,
  }) => {
    await loadModel(page);
    const request = makeRequest();
    const baseline = await baselineText(page, request);
    const sessionId = "zero-token";
    await page.evaluate(
      ({ request, sessionId, point, storeCheckpointLogits }) => {
        const { setResumableFaultHook, JournalRecordType } =
          globalThis.webllmBrowserHarness;
        setResumableFaultHook(async (current, context) => {
          if (
            current === point &&
            (point !== "journal.before_append" ||
              context.recordType === JournalRecordType.GeneratedToken)
          ) {
            globalThis.faultReached = true;
            await new Promise(() => undefined);
          }
        });
        globalThis.pendingGeneration = globalThis.gpuEngine.chatCompletion({
          ...request,
          extra_body: {
            resumable: {
              enabled: true,
              sessionId,
              checkpointPrompt: point !== "journal.before_append",
              storeCheckpointLogits,
              strictPersistence: true,
            },
          },
        });
        void globalThis.pendingGeneration.catch((err) => {
          globalThis.generationError = err.message;
        });
      },
      { request, sessionId, point, storeCheckpointLogits },
    );
    await page.waitForFunction(
      () => globalThis.faultReached || globalThis.generationError,
    );
    expect(
      await page.evaluate(() => globalThis.generationError),
    ).toBeUndefined();
    const before = await inspectSession(page, sessionId);
    expect(before.records.filter((record) => record.type === 5)).toHaveLength(
      0,
    );
    expect(before.records.some((record) => record.type === 4)).toBe(true);

    await loadModel(page);
    const resumed = await page.evaluate(
      (sessionId) =>
        globalThis.gpuEngine.resumeChatCompletion(sessionId, {
          continueGeneration: true,
        }),
      sessionId,
    );
    expect(resumed).toMatchObject({ recoveredText: baseline, recoveryMode });
    const after = await inspectSession(page, sessionId);
    expect(after.directories).toEqual([]);
    expect(after.records.filter((record) => record.type === 5)).toHaveLength(
      request.max_tokens,
    );
    expect(after.records.filter((record) => record.type === 7)).toHaveLength(1);
  });
}

for (const corruption of ["none", "newest", "all"]) {
  test(`real WebGPU decode retention and torn-tail repair, corrupt=${corruption}`, async ({
    page,
  }) => {
    await loadModel(page);
    const request = makeRequest({ max_tokens: 80 });
    const baseline = await baselineText(page, request);
    const sessionId = "decode-retention";
    const commits = await page.evaluate(
      async ({ request, sessionId }) => {
        const { setResumableFaultHook } = globalThis.webllmBrowserHarness;
        const commits = [];
        const restore = setResumableFaultHook((point, context) => {
          if (point === "checkpoint.after_commit") commits.push(context);
        });
        const stream = await globalThis.gpuEngine.chatCompletion({
          ...request,
          stream: true,
          extra_body: {
            resumable: {
              enabled: true,
              sessionId,
              checkpointPrompt: false,
              checkpointIntervalTokens: 16,
              storeCheckpointLogits: false,
              strictPersistence: true,
            },
          },
        });
        globalThis.gpuStream = stream[Symbol.asyncIterator]();
        while (commits.length < 4) {
          if ((await globalThis.gpuStream.next()).done) {
            throw new Error(
              "Generation finished before four decode checkpoints",
            );
          }
        }
        await globalThis.gpuStream.next();
        await globalThis.gpuStream.next();
        restore();
        return commits;
      },
      { request, sessionId },
    );
    const before = await inspectSession(page, sessionId);
    expect(before.checkpoints.map((ref) => ref.checkpointId)).toEqual(
      commits.slice(-2).map((commit) => commit.checkpointId),
    );
    expect(before.directories).toHaveLength(2);
    const chosen = commits.at(corruption === "newest" ? -2 : -1);
    const expectedTail = before.records.filter(
      (record) =>
        record.type === 5 &&
        record.payload.globalTokenPos >= chosen.processedSeqLen,
    ).length;
    expect(expectedTail).toBeGreaterThan(0);

    // Add unfinished/uncommitted directories and a physically torn journal.
    // Flip payload bytes without updating their CRC to simulate corruption.
    await page.evaluate(
      async ({ sessionId, corruption, refs }) => {
        const { BrowserOPFSFileStore, ResumableSessionStore } =
          globalThis.webllmBrowserHarness;
        const files = new BrowserOPFSFileStore();
        const sessions = new ResumableSessionStore(files);
        const paths = sessions.getSessionPaths(sessionId);
        await files.mkdir(`${paths.kvDir}/unfinished`);
        const uncommitted = `${paths.kvDir}/uncommitted`;
        await files.mkdir(uncommitted);
        for (const name of await files.list(refs.at(-1).path)) {
          let data = await files.read(`${refs.at(-1).path}/${name}`);
          if (name === "meta.json") {
            const meta = JSON.parse(new globalThis.TextDecoder().decode(data));
            meta.checkpointId = "uncommitted";
            data = new globalThis.TextEncoder().encode(JSON.stringify(meta));
          }
          await files.write(`${uncommitted}/${name}`, data);
        }
        const targets =
          corruption === "all"
            ? refs
            : corruption === "newest"
              ? refs.slice(-1)
              : [];
        for (const ref of targets) {
          const name = (await files.list(ref.path)).find((name) =>
            name.endsWith(".wkv"),
          );
          if (!name) throw new Error("Checkpoint has no page-group payload");
          const data = new Uint8Array(await files.read(`${ref.path}/${name}`));
          data[0] ^= 1;
          await files.write(`${ref.path}/${name}`, data);
        }
        await files.append(paths.journalPath, new Uint8Array([1, 2, 3]));
      },
      { sessionId, corruption, refs: before.checkpoints },
    );
    expect((await inspectSession(page, sessionId)).stoppedReason).toBe(
      "partial_record",
    );

    await loadModel(page);
    const probes = await page.evaluate(() =>
      globalThis.gpuEngine.listResumableSessions(),
    );
    const recoveryMode = corruption === "all" ? "token_replay" : "kv";
    expect(probes).toContainEqual(
      expect.objectContaining({ sessionId, recoveryMode }),
    );
    const repaired = await inspectSession(page, sessionId);
    expect(repaired.stoppedReason).toBeUndefined();
    expect(repaired.directories).toHaveLength(
      corruption === "all" ? 0 : corruption === "newest" ? 1 : 2,
    );
    const resumed = await page.evaluate(
      (sessionId) =>
        globalThis.gpuEngine.resumeChatCompletion(sessionId, {
          continueGeneration: true,
        }),
      sessionId,
    );
    expect(resumed).toMatchObject({ recoveredText: baseline, recoveryMode });
    if (recoveryMode === "kv")
      expect(resumed.replayedTokens).toBe(expectedTail);
    expect((await inspectSession(page, sessionId)).directories).toEqual([]);
  });
}

for (const durabilityMode of ["exact", "relaxed"]) {
  for (const strictPersistence of [false, true]) {
    test(`real WebGPU token-write failure (${durabilityMode}, strict=${strictPersistence})`, async ({
      page,
    }) => {
      await loadModel(page);
      const request = makeRequest();
      const baseline = await baselineText(page, request);
      const sessionId = "write-failure";
      const outcome = await page.evaluate(
        async ({ request, sessionId, durabilityMode, strictPersistence }) => {
          const { setResumableFaultHook, JournalRecordType } =
            globalThis.webllmBrowserHarness;
          let tokenWrites = 0;
          const restore = setResumableFaultHook((point, context) => {
            if (
              point === "journal.before_append" &&
              context.recordType === JournalRecordType.GeneratedToken &&
              ++tokenWrites === 3
            ) {
              throw new Error("injected token-write failure");
            }
          });
          let error;
          let text = "";
          try {
            const stream = await globalThis.gpuEngine.chatCompletion({
              ...request,
              stream: true,
              extra_body: {
                resumable: {
                  enabled: true,
                  sessionId,
                  checkpointPrompt: false,
                  durabilityMode,
                  strictPersistence,
                },
              },
            });
            for await (const chunk of stream)
              text += chunk.choices[0]?.delta.content ?? "";
          } catch (err) {
            error = err.message;
          } finally {
            restore();
          }
          return { text, error, tokenWrites };
        },
        { request, sessionId, durabilityMode, strictPersistence },
      );
      expect(outcome.tokenWrites).toBe(3);
      if (strictPersistence) {
        expect(outcome.error).toBe("injected token-write failure");
        expect(outcome.text).not.toBe(baseline);
      } else {
        expect(outcome.error).toBeUndefined();
        expect(outcome.text).toBe(baseline);
      }
      const saved = await inspectSession(page, sessionId);
      expect(saved.records.filter((record) => record.type === 5)).toHaveLength(
        2,
      );
      expect(saved.records.some((record) => record.type === 7)).toBe(false);
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
  }
}
