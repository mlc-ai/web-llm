import { MemoryFileStore } from "./helpers/memory_file_store";
import {
  JournalRecordType,
  appendJournalRecord,
} from "../src/resumable/journal";
import { ResumableSessionStore } from "../src/resumable/session_store";
import { ResumableCheckpointWriter } from "../src/resumable/checkpoint_writer";
import { test, expect, jest } from "@jest/globals";

const encoder = new TextEncoder();

function makeStore(): {
  files: MemoryFileStore;
  sessions: ResumableSessionStore;
} {
  const files = new MemoryFileStore();
  return {
    files,
    sessions: new ResumableSessionStore(files, {
      rootPath: "resume-root",
      now: () => 1000,
    }),
  };
}

test("session store creates, opens, and lists deterministic session dirs", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a", {
    modelId: "model-a",
  });

  expect(session.paths).toEqual({
    sessionDir: "resume-root/sessions/session-a",
    manifestPath: "resume-root/sessions/session-a/manifest.json",
    journalPath: "resume-root/sessions/session-a/journal.bin",
    lockPath: "resume-root/sessions/session-a/lock",
    kvDir: "resume-root/sessions/session-a/kv",
  });
  expect(await files.list("resume-root/sessions")).toEqual(["session-a"]);
  expect((await sessions.openSession("session-a"))?.manifest?.modelId).toBe(
    "model-a",
  );
  expect((await sessions.listSessions()).map((item) => item.sessionId)).toEqual(
    ["session-a"],
  );
  expect(await sessions.openSession("missing")).toBeUndefined();
});

test("deleteSession removes journal and kv directories", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  const checkpoint = sessions.getCheckpointRef("session-a", "checkpoint_000");

  await files.write(session.paths.journalPath, encoder.encode("journal"));
  await files.write(checkpoint.completePath, encoder.encode(""));
  await sessions.deleteSession("session-a");

  expect(await files.read(session.paths.journalPath)).toBeUndefined();
  expect(await files.list(session.paths.kvDir)).toEqual([]);
  expect(await sessions.openSession("session-a")).toBeUndefined();
});

test("journal and generation lock-only directories are invisible and reusable", async () => {
  const { files, sessions } = makeStore();
  const paths = sessions.getSessionPaths("lock-only");
  await files.write(paths.lockPath, new Uint8Array());
  await files.write(`${paths.journalPath}.lock`, new Uint8Array());
  expect(await sessions.openSession("lock-only")).toBeUndefined();
  expect(await sessions.listSessions()).toEqual([]);
  await expect(sessions.createNewSession("lock-only")).resolves.toMatchObject({
    sessionId: "lock-only",
  });
});

test("manifest helpers read valid manifests and ignore corrupt JSON", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a", {
    modelId: "model-a",
  });

  expect(
    await sessions.tryWriteManifest({
      ...session.manifest!,
      updatedAtMs: 1200,
    }),
  ).toBe(true);
  expect(await sessions.readManifest("session-a")).toMatchObject({
    modelId: "model-a",
    updatedAtMs: 1200,
  });

  await files.write(session.paths.manifestPath, encoder.encode("{bad"));
  expect(await sessions.readManifest("session-a")).toBeUndefined();
});

test("rebuild inputs use journal presence and committed checkpoints", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  const committed = sessions.getCheckpointRef("session-a", "checkpoint_000");
  const incomplete = sessions.getCheckpointRef("session-a", "checkpoint_001");

  await files.write(committed.completePath, encoder.encode(""));
  await files.write(`${incomplete.path}/meta.json`, encoder.encode("{}"));
  await appendJournalRecord(files, session.paths.journalPath, {
    type: JournalRecordType.CheckpointCommit,
    seqNo: 1,
    createdAtMs: 1000,
    payload: {
      checkpointId: "checkpoint_000",
      processedSeqLen: 32,
      path: committed.path,
    },
  });

  const inputs = await sessions.getManifestRebuildInputs("session-a");
  expect(inputs?.hasJournal).toBe(true);
  expect(inputs?.journalPath).toBe(session.paths.journalPath);
  expect(inputs?.checkpoints.map((item) => item.checkpointId)).toEqual([
    "checkpoint_000",
  ]);
});

test("startup cleanup removes incomplete and complete-uncommitted checkpoints", async () => {
  const { files, sessions } = makeStore();
  await sessions.createSession("session-a");
  await sessions.createSession("session-b");
  const keep = sessions.getCheckpointRef("session-a", "checkpoint_keep");
  const dropA = sessions.getCheckpointRef("session-a", "checkpoint_drop");
  const dropB = sessions.getCheckpointRef("session-b", "checkpoint_drop");

  await files.write(keep.completePath, encoder.encode(""));
  await files.write(`${dropA.path}/meta.json`, encoder.encode("{}"));
  await files.write(`${dropB.path}/meta.json`, encoder.encode("{}"));

  const removed = await sessions.cleanupIncompleteCheckpoints();

  expect(removed.map((item) => `${item.checkpointId}:${item.path}`)).toEqual([
    "checkpoint_drop:resume-root/sessions/session-a/kv/checkpoint_drop",
    "checkpoint_keep:resume-root/sessions/session-a/kv/checkpoint_keep",
    "checkpoint_drop:resume-root/sessions/session-b/kv/checkpoint_drop",
  ]);
  expect(await files.list("resume-root/sessions/session-a/kv")).toEqual([]);
  expect(await files.list("resume-root/sessions/session-b/kv")).toEqual([]);
});

test("checkpoint maintenance retains two commits and cleans orphans with one index scan", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  for (let index = 1; index <= 3; index++) {
    const checkpointId = `checkpoint_${index}`;
    const ref = sessions.getCheckpointRef("session-a", checkpointId);
    await new ResumableCheckpointWriter(files, sessions).writeCheckpoint({
      sessionId: "session-a",
      checkpointId,
      processedSeqLen: index,
      pageGroups: [],
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.CheckpointCommit,
      seqNo: index,
      createdAtMs: 1000 + index,
      payload: {
        checkpointId,
        processedSeqLen: index,
        path: ref.path,
      },
    });
  }

  const incomplete = sessions.getCheckpointRef("session-a", "incomplete");
  await files.write(`${incomplete.path}/meta.json`, encoder.encode("{}"));
  await new ResumableCheckpointWriter(files, sessions).writeCheckpoint({
    sessionId: "session-a",
    checkpointId: "uncommitted",
    processedSeqLen: 4,
    pageGroups: [],
  });
  const read = jest.spyOn(files, "read");
  const list = jest.spyOn(files, "list");
  const removed = await sessions.pruneCommittedCheckpoints("session-a", 2);

  expect(
    read.mock.calls.filter(([path]) => path === session.paths.journalPath),
  ).toHaveLength(1);
  expect(
    list.mock.calls.filter(([path]) => path === session.paths.kvDir),
  ).toHaveLength(1);
  expect(removed.map((ref) => ref.checkpointId)).toEqual([
    "checkpoint_1",
    "incomplete",
    "uncommitted",
  ]);
  expect(await files.list(session.paths.kvDir)).toEqual([
    "checkpoint_2",
    "checkpoint_3",
  ]);
});

test("checkpoint pruning does not count a missing committed directory toward retention", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  for (let index = 1; index <= 3; index++) {
    const checkpointId = `checkpoint_${index}`;
    const ref = sessions.getCheckpointRef("session-a", checkpointId);
    if (index < 3) {
      await new ResumableCheckpointWriter(files, sessions).writeCheckpoint({
        sessionId: "session-a",
        checkpointId,
        processedSeqLen: index,
        pageGroups: [],
      });
    }
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.CheckpointCommit,
      seqNo: index,
      createdAtMs: 1000 + index,
      payload: {
        checkpointId,
        processedSeqLen: index,
        path: ref.path,
      },
    });
  }

  await sessions.pruneCommittedCheckpoints("session-a", 2);

  expect(await files.list(session.paths.kvDir)).toEqual([
    "checkpoint_1",
    "checkpoint_2",
  ]);
});

test.each(["crc", "metadata", "missing", "sequence", "io"])(
  "pruning keeps valid fallbacks when the newest checkpoint has %s damage",
  async (damage) => {
    const { files, sessions } = makeStore();
    const session = await sessions.createSession("session-a");
    for (let index = 1; index <= 3; index++) {
      const checkpointId = `checkpoint_${index}`;
      await new ResumableCheckpointWriter(files, sessions).writeCheckpoint({
        sessionId: "session-a",
        checkpointId,
        processedSeqLen: index,
        pageGroups: [
          {
            groupId: 0,
            layerStart: 0,
            layerEnd: 1,
            fileName: "pages.wkv",
            data: new Uint8Array([1, 2, 3]),
          },
        ],
      });
      await appendJournalRecord(files, session.paths.journalPath, {
        type: JournalRecordType.CheckpointCommit,
        seqNo: index,
        createdAtMs: index,
        payload: { checkpointId, processedSeqLen: index },
      });
    }
    const newest = sessions.getCheckpointRef("session-a", "checkpoint_3");
    if (damage === "crc")
      await files.write(`${newest.path}/pages.wkv`, new Uint8Array([3, 2, 1]));
    if (damage === "metadata")
      await files.write(`${newest.path}/meta.json`, encoder.encode("{"));
    if (damage === "missing") await files.remove(`${newest.path}/pages.wkv`);
    if (damage === "sequence") {
      const meta = JSON.parse(
        new TextDecoder().decode(await files.read(`${newest.path}/meta.json`)),
      );
      meta.processedSeqLen = 99;
      await files.write(
        `${newest.path}/meta.json`,
        encoder.encode(JSON.stringify(meta)),
      );
    }
    if (damage === "io") {
      await files.mkdir(`${session.paths.kvDir}/orphan`);
      const read = files.read.bind(files);
      jest.spyOn(files, "read").mockImplementation(async (path) => {
        if (path === `${newest.path}/pages.wkv`)
          throw new Error("storage unavailable");
        return read(path);
      });
      await expect(
        sessions.pruneCommittedCheckpoints("session-a"),
      ).rejects.toThrow("storage unavailable");
      expect(await files.list(session.paths.kvDir)).toHaveLength(4);
    } else {
      await sessions.pruneCommittedCheckpoints("session-a");
      expect(await files.list(session.paths.kvDir)).toEqual([
        "checkpoint_1",
        "checkpoint_2",
      ]);
    }
  },
);

test("session deletion refuses an active session and removes it under lock", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  const release = await files.tryLock(session.paths.lockPath);

  await expect(sessions.deleteSession("session-a")).rejects.toThrow(
    "Resumable session is already active: session-a",
  );
  release!();
  await sessions.deleteSession("session-a");
  await expect(sessions.openSession("session-a")).resolves.toBeUndefined();
});
