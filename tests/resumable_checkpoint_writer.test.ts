import { MemoryFileStore } from "./helpers/memory_file_store";
import {
  ResumableCheckpointWriter,
  readResumableCheckpointPayload,
} from "../src/resumable/checkpoint_writer";
import {
  JournalRecordType,
  appendJournalRecord,
  readJournalRecords,
} from "../src/resumable/journal";
import { ResumableSessionStore } from "../src/resumable/session_store";
import { test, expect } from "@jest/globals";

const decoder = new TextDecoder();
const encoder = new TextEncoder();

function makeStore(): {
  files: MemoryFileStore;
  sessions: ResumableSessionStore;
  writer: ResumableCheckpointWriter;
} {
  const files = new MemoryFileStore();
  const sessions = new ResumableSessionStore(files, {
    rootPath: "resume-root",
    now: () => 1000,
  });
  return {
    files,
    sessions,
    writer: new ResumableCheckpointWriter(files, sessions, () => 1200),
  };
}

test("checkpoint writer stages complete files before a journal commit", async () => {
  const { files, sessions, writer } = makeStore();

  const ref = await writer.writeCheckpoint({
    sessionId: "session-a",
    checkpointId: "checkpoint_00000000_00000004",
    processedSeqLen: 4,
    layoutHash: "layout-a",
    metadata: { source: "prompt" },
    pageGroups: [
      {
        groupId: 0,
        layerStart: 0,
        layerEnd: 1,
        data: new Uint8Array([1, 2, 3, 4]),
      },
    ],
    nextLogits: new Uint8Array([5, 6]),
  });

  expect(await files.list(ref.path)).toEqual([
    "complete",
    "group_000_layers_000_001_pages.wkv",
    "meta.json",
    "next_logits.f16",
  ]);
  const meta = JSON.parse(
    decoder.decode((await files.read(`${ref.path}/meta.json`))!),
  );
  expect(meta).toMatchObject({
    checkpointId: "checkpoint_00000000_00000004",
    processedSeqLen: 4,
    layoutHash: "layout-a",
    createdAtMs: 1200,
    metadata: { source: "prompt" },
    pageGroups: [
      {
        path: "group_000_layers_000_001_pages.wkv",
        bytes: 4,
        crc32c: expect.stringMatching(/^[0-9a-f]{8}$/),
      },
    ],
    nextLogits: {
      path: "next_logits.f16",
      bytes: 2,
      crc32c: expect.stringMatching(/^[0-9a-f]{8}$/),
    },
  });

  expect(await sessions.listCommittedCheckpoints("session-a")).toEqual([]);
  const session = (await sessions.openSession("session-a"))!;
  await appendJournalRecord(files, session.paths.journalPath, {
    type: JournalRecordType.CheckpointCommit,
    seqNo: 1,
    createdAtMs: 1201,
    payload: {
      checkpointId: ref.checkpointId,
      processedSeqLen: 4,
      path: ref.path,
      layoutHash: "layout-a",
    },
  });
  expect((await sessions.listCommittedCheckpoints("session-a"))[0]).toEqual(
    ref,
  );
  const payload = await readResumableCheckpointPayload(files, ref);
  expect(payload?.pageGroups[0].data).toEqual(new Uint8Array([1, 2, 3, 4]));
  expect(payload?.nextLogits?.data).toEqual(new Uint8Array([5, 6]));
  expect(
    (await sessions.getManifestRebuildInputs("session-a"))?.checkpoints,
  ).toEqual([ref]);

  const journal = await readJournalRecords(
    files,
    "resume-root/sessions/session-a/journal.bin",
  );
  expect(journal.records).toEqual([
    expect.objectContaining({
      type: JournalRecordType.CheckpointCommit,
      seqNo: 1,
      payload: expect.objectContaining({
        checkpointId: "checkpoint_00000000_00000004",
        processedSeqLen: 4,
        layoutHash: "layout-a",
      }),
    }),
  ]);
});

test("checkpoint without complete marker is ignored", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  const ref = sessions.getCheckpointRef("session-a", "checkpoint_incomplete");
  await files.write(`${ref.path}/meta.json`, encoder.encode("{}"));
  await appendJournalRecord(files, session.paths.journalPath, {
    type: JournalRecordType.CheckpointCommit,
    seqNo: 1,
    createdAtMs: 1000,
    payload: {
      checkpointId: ref.checkpointId,
      processedSeqLen: 4,
      path: ref.path,
    },
  });

  expect(await sessions.listCommittedCheckpoints("session-a")).toEqual([]);
  expect(await sessions.cleanupIncompleteCheckpoints("session-a")).toEqual([
    ref,
  ]);
});

test("checkpoint with complete marker but no journal commit is ignored", async () => {
  const { files, sessions } = makeStore();
  await sessions.createSession("session-a");
  const ref = sessions.getCheckpointRef("session-a", "checkpoint_complete");
  await files.write(`${ref.path}/complete`, encoder.encode(""));

  expect(await sessions.listCommittedCheckpoints("session-a")).toEqual([]);
  expect(
    (await sessions.getManifestRebuildInputs("session-a"))?.checkpoints,
  ).toEqual([]);
});
