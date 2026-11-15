import { expect, test } from "./fixtures.mjs";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(
    () => globalThis.webllmBrowserHarness !== undefined,
  );
});

test("OPFS journal repair and cross-context locking use browser primitives", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    globalThis.webllmBrowserHarness.runOPFSRegression(),
  );

  expect(result).toEqual({
    text: "one-two",
    webLocksAvailable: true,
    acquiredWhileHeld: false,
    acquiredAfterRelease: true,
    stoppedBeforeRepair: "partial_record",
    stoppedAfterRepair: undefined,
    repairedRecordCount: 1,
  });
});

test("LLMChatPipeline samples the first token from a zero-token replay", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    globalThis.webllmBrowserHarness.runFirstTokenReplayRegression(),
  );

  expect(result).toEqual({
    forwardedPrompt: [1, 2, 3],
    outputIds: [9],
    promptLogitsDisposed: true,
    replayedTokens: 0,
    sampledTokenId: 17,
    sampledTokenPosition: 3,
    committedText: "first",
  });
});

test("LLMChatPipeline retains only final logits while forwarding replay tokens", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    globalThis.webllmBrowserHarness.runKnownTokenForwardingRegression(),
  );

  expect(result).toEqual({
    forwardedTokens: 10,
    chunkCount: 3,
    detached: [3],
    finalLogitsId: 3,
    endedScopes: 1,
  });
});

for (const failure of ["sample", "grammar"]) {
  test(`prefill releases all tensors after ${failure} failure without unhandled rejections`, async ({
    page,
  }) => {
    const errors = [];
    page.on("pageerror", (err) => errors.push(err.message));
    const result = await page.evaluate(
      (failure) =>
        globalThis.webllmBrowserHarness.runPrefillFailureRegression(failure),
      failure,
    );
    expect(result.message).toContain(
      failure === "grammar"
        ? "Failed to initialize the grammar matcher"
        : "sampling failed",
    );
    expect(result).toMatchObject({ allocated: 5, live: 0, scopes: 0 });
    expect(errors).toEqual([]);
  });
}

test("unused resumable storage observes asynchronous OPFS rejection", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));
  const message = await page.evaluate(async () => {
    const { BrowserOPFSFileStore } = globalThis.webllmBrowserHarness;
    const store = new BrowserOPFSFileStore(
      Promise.reject(new Error("OPFS denied")),
    );
    await new Promise((resolve) => globalThis.setTimeout(resolve, 10));
    try {
      await store.read("journal.bin");
    } catch (err) {
      return err.message;
    }
  });
  expect(message).toBe("OPFS denied");
  expect(errors).toEqual([]);
});
