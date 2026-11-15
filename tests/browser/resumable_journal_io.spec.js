import { test, expect } from "./fixtures.mjs";

for (const syncAccess of [false, true]) {
  test(`journal snapshots exclude cross-worker append and deletion, syncAccess=${syncAccess}`, async ({
    page,
  }) => {
    await page.goto("/");
    const result = await page.evaluate(async (syncAccess) => {
      const source = `
        import { BrowserOPFSFileStore, ResumableSessionStore, appendJournalRecord, readJournalRecords } from "/harness.js";
        ${syncAccess ? 'Object.defineProperty(navigator, "locks", { value: undefined });' : ""}
        const files = new BrowserOPFSFileStore();
        const sessions = new ResumableSessionStore(files);
        const id = "journal-concurrency";
        const path = sessions.getSessionPaths(id).journalPath;
        let releaseRead;
        onmessage = async ({ data: { id: requestId, operation } }) => {
          try {
            let result;
            if (operation === "create") {
              await sessions.createNewSession(id);
              await appendJournalRecord(files, path, { type: 1, seqNo: 1, createdAtMs: 1, payload: { sessionId: id } });
            } else if (operation === "read") {
              const read = files.read.bind(files);
              files.read = async (path) => {
                const data = await read(path);
                await new Promise(resolve => {
                  releaseRead = resolve;
                  postMessage({ reading: true });
                });
                return data;
              };
              result = (await readJournalRecords(files, path)).records.length;
              files.read = read;
            } else if (operation === "release") {
              releaseRead();
            } else if (operation === "append") {
              await appendJournalRecord(files, path, { type: 2, seqNo: 2, createdAtMs: 2, payload: { tokenIds: [1] } });
              result = (await readJournalRecords(files, path)).records.length;
            } else if (operation === "delete") {
              await sessions.deleteSession(id);
              result = (await sessions.openSession(id)) === undefined;
            }
            postMessage({ id: requestId, result });
          } catch (err) {
            postMessage({ id: requestId, error: err.name + ": " + err.message });
          }
        };
      `;
      // Blob modules require an absolute import URL.
      const url = globalThis.URL.createObjectURL(
        new globalThis.Blob(
          [
            source.replace(
              '"/harness.js"',
              JSON.stringify(
                new globalThis.URL("/harness.js", globalThis.location.href)
                  .href,
              ),
            ),
          ],
          { type: "text/javascript" },
        ),
      );
      const workers = [];
      const makeWorker = () => {
        const worker = new globalThis.Worker(url, { type: "module" });
        workers.push(worker);
        const pending = new Map();
        let nextId = 0;
        let notifyRead;
        worker.onmessage = ({ data }) => {
          if (data.reading) return notifyRead();
          const { resolve, reject } = pending.get(data.id);
          pending.delete(data.id);
          if (data.error) reject(new Error(data.error));
          else resolve(data.result);
        };
        worker.onerror = (event) => {
          for (const { reject } of pending.values())
            reject(new Error(event.message));
        };
        return {
          call: (operation) =>
            new Promise((resolve, reject) => {
              const id = nextId++;
              pending.set(id, { resolve, reject });
              worker.postMessage({ id, operation });
            }),
          reading: () =>
            new Promise((resolve) => {
              notifyRead = resolve;
            }),
        };
      };
      try {
        const writer = makeWorker();
        const reader = makeWorker();
        await writer.call("create");
        const outcomes = [];
        for (const operation of ["append", "delete"]) {
          const reading = reader.reading();
          const snapshot = reader.call("read");
          await reading;
          let mutationFinished = false;
          const mutation = writer.call(operation).then((result) => {
            mutationFinished = true;
            return result;
          });
          await new Promise((resolve) => globalThis.setTimeout(resolve, 50));
          const blocked = !mutationFinished;
          await reader.call("release");
          outcomes.push({
            blocked,
            snapshot: await snapshot,
            result: await mutation,
          });
        }
        // Deletion must leave no visible session, even with physical sidecars.
        await writer.call("create");
        const reading = reader.reading();
        void reader.call("read");
        await reading;
        // A crashed reader must not leave a permanent lock behind.
        workers[1].terminate();
        const afterCrash = await writer.call("append");
        await writer.call("delete");
        return { outcomes, afterCrash };
      } finally {
        for (const worker of workers) worker.terminate();
        globalThis.URL.revokeObjectURL(url);
      }
    }, syncAccess);
    expect(result).toEqual({
      outcomes: [
        { blocked: true, snapshot: 1, result: 2 },
        { blocked: true, snapshot: 2, result: true },
      ],
      afterCrash: 2,
    });
  });
}
