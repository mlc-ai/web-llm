import { test, expect } from "./fixtures.mjs";

test("browser worker read-ahead completes cleanly after the final chunk", async ({
  page,
}) => {
  await page.goto("/");
  const results = await page.evaluate(async () => {
    const moduleURL = new globalThis.URL(
      "/harness.js",
      globalThis.location.href,
    ).href;
    const { WebWorkerMLCEngine } = await import(moduleURL);
    const workerURL = globalThis.URL.createObjectURL(
      new globalThis.Blob(
        [
          `import { WebWorkerMLCEngineHandler } from ${JSON.stringify(moduleURL)};
       const handler = new WebWorkerMLCEngineHandler();
       handler.streamIdToAsyncGenerator.set('read-ahead', (async function* () {
         yield { object: 'chat.completion.chunk', choices: [] };
       })());
       onmessage = event => handler.onmessage(event);`,
        ],
        { type: "text/javascript" },
      ),
    );
    const worker = new globalThis.Worker(workerURL, { type: "module" });
    try {
      const client = new WebWorkerMLCEngine(worker);
      const iterator = client.asyncGenerate("read-ahead");
      return await Promise.all([
        iterator.next(),
        iterator.next(),
        iterator.next(),
      ]);
    } finally {
      worker.terminate();
      globalThis.URL.revokeObjectURL(workerURL);
    }
  });
  expect(results.map((result) => result.done)).toEqual([false, true, true]);
  expect(results[0].value.object).toBe("chat.completion.chunk");
});
