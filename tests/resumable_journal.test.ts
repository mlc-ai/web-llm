import { CrossContextLockUnavailableError } from "../src/resumable/opfs_file_store";
import {
  JournalRecord,
  JournalRecordType,
  journalLockPath,
  appendJournalRecord,
  decodeJournalRecordAt,
  encodeJournalRecord,
  readJournalRecords,
  repairJournalTail,
  scanJournalRecords,
} from "../src/resumable/journal";
import {
  ResumableGenerationJournal,
  normalizeResumableGenerationConfig,
} from "../src/resumable/generation";
import {
  applyGeneratedTokenText,
  readResumableReplayState,
} from "../src/resumable/replay";
import { ResumableSessionStore } from "../src/resumable/session_store";
import {
  probeReplayState,
  probeResumableSession,
} from "../src/resumable/session_probe";
import { test, expect, jest } from "@jest/globals";
import { MemoryFileStore, bytes } from "./helpers/memory_file_store";

const HEADER_SIZE = 30;

function concat(...parts: ArrayBuffer[]): ArrayBuffer {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(new Uint8Array(part), offset);
    offset += part.byteLength;
  }
  return out.buffer;
}

function record<T extends JournalRecord>(record: T): T {
  return record;
}

const records: JournalRecord[] = [
  record({
    type: JournalRecordType.SessionBegin,
    seqNo: 1,
    createdAtMs: 100,
    payload: { sessionId: "s1", modelId: "m1", request: { temperature: 0.7 } },
  }),
  record({
    type: JournalRecordType.PromptTokens,
    seqNo: 2,
    createdAtMs: 101,
    payload: { tokenIds: [1, 2, 3], text: "prompt" },
  }),
  record({
    type: JournalRecordType.AssistantPrefixTokens,
    seqNo: 3,
    createdAtMs: 102,
    payload: { tokenIds: [4, 5], text: "<think></think>" },
  }),
  record({
    type: JournalRecordType.GenerationConfig,
    seqNo: 4,
    createdAtMs: 103,
    payload: { config: { max_tokens: 8, temperature: 0 } },
  }),
  record({
    type: JournalRecordType.GeneratedToken,
    seqNo: 5,
    createdAtMs: 104,
    payload: {
      globalTokenPos: 9,
      tokenId: 42,
      textDelta: "hi",
      rngState: 7,
    },
  }),
  record({
    type: JournalRecordType.CheckpointCommit,
    seqNo: 6,
    createdAtMs: 105,
    payload: {
      checkpointId: "checkpoint_000",
      processedSeqLen: 512,
      layoutHash: "abc",
    },
  }),
  record({
    type: JournalRecordType.GenerationFinished,
    seqNo: 7,
    createdAtMs: 106,
    payload: { finishReason: "stop", emittedTokens: 2 },
  }),
  record({
    type: JournalRecordType.GenerationAborted,
    seqNo: 8,
    createdAtMs: 107,
    payload: { reason: "user" },
  }),
  record({
    type: JournalRecordType.EngineError,
    seqNo: 9,
    createdAtMs: 108,
    payload: { message: "boom", name: "Error" },
  }),
];

test("journal records round-trip through binary codec", () => {
  for (const original of records) {
    const encoded = encodeJournalRecord(original);
    const decoded = decodeJournalRecordAt(encoded);
    expect(decoded.record).toEqual(original);
    expect(decoded.nextOffset).toBe(encoded.byteLength);
  }
});

test("journal scanner returns all valid records", () => {
  const data = concat(...records.map(encodeJournalRecord));
  const result = scanJournalRecords(data);

  expect(result.stoppedReason).toBeUndefined();
  expect(result.validBytes).toBe(data.byteLength);
  expect(result.records).toEqual(records);
});

test("journal scanner ignores partial trailing record", () => {
  const first = encodeJournalRecord(records[0]);
  const partialSecond = encodeJournalRecord(records[1]).slice(
    0,
    HEADER_SIZE + 3,
  );
  const result = scanJournalRecords(concat(first, partialSecond));

  expect(result.records).toEqual([records[0]]);
  expect(result.validBytes).toBe(first.byteLength);
  expect(result.stoppedReason).toBe("partial_record");
});

test("journal scanner stops at CRC mismatch", () => {
  const first = encodeJournalRecord(records[0]);
  const second = encodeJournalRecord(records[1]);
  const data = new Uint8Array(concat(first, second));
  data[first.byteLength + HEADER_SIZE] ^= 1;

  const result = scanJournalRecords(data.buffer);

  expect(result.records).toEqual([records[0]]);
  expect(result.validBytes).toBe(first.byteLength);
  expect(result.stoppedReason).toBe("crc_mismatch");
});

test("journal append and read helpers use OPFS file store", async () => {
  const store = new MemoryFileStore();

  await appendJournalRecord(store, "journal.bin", records[0]);
  await appendJournalRecord(store, "journal.bin", records[1]);

  const result = await readJournalRecords(store, "journal.bin");
  expect(result.records).toEqual([records[0], records[1]]);
  expect(await readJournalRecords(store, "missing.bin")).toEqual({
    records: [],
    validBytes: 0,
  });
});

test("journal tail repair truncates bytes after the valid record prefix", async () => {
  const store = new MemoryFileStore();
  const valid = encodeJournalRecord(records[0]);
  await store.write(
    "journal.bin",
    concat(valid, new Uint8Array([1, 2, 3]).buffer),
  );

  const repaired = await repairJournalTail(store, "journal.bin");

  expect(repaired.records).toEqual([records[0]]);
  expect(repaired.stoppedReason).toBeUndefined();
  expect((await store.read("journal.bin"))?.byteLength).toBe(valid.byteLength);
  expect((await readJournalRecords(store, "journal.bin")).records).toEqual([
    records[0],
  ]);
});

test("journal inspection remains available when cross-context locking is unsupported", async () => {
  const store = new MemoryFileStore();
  await store.write("journal.bin", encodeJournalRecord(records[0]));
  jest
    .spyOn(store, "lock")
    .mockRejectedValue(
      new CrossContextLockUnavailableError("journal.bin.lock"),
    );
  expect((await readJournalRecords(store, "journal.bin")).records).toEqual([
    records[0],
  ]);
  await expect(
    appendJournalRecord(store, "journal.bin", records[1]),
  ).rejects.toBeInstanceOf(CrossContextLockUnavailableError);
  await expect(repairJournalTail(store, "journal.bin")).rejects.toBeInstanceOf(
    CrossContextLockUnavailableError,
  );
});

test("journal reads hold exclusion until snapshot bytes are materialized", async () => {
  const store = new MemoryFileStore();
  const path = "journal.bin";
  await appendJournalRecord(store, path, records[0]);
  const read = store.read.bind(store);
  let finishRead!: () => void;
  let notifyRead!: () => void;
  const reading = new Promise<void>((resolve) => {
    notifyRead = resolve;
  });
  store.read = async (path) => {
    const data = await read(path);
    await new Promise<void>((resolve) => {
      finishRead = resolve;
      notifyRead();
    });
    return data;
  };
  const snapshot = readJournalRecords(store, path);
  await reading;
  expect(await store.tryLock(journalLockPath(path))).toBeUndefined();
  const append = jest.spyOn(store, "append");
  const writing = appendJournalRecord(store, path, records[1]);
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(append).not.toHaveBeenCalled();
  finishRead();
  expect((await snapshot).records).toEqual([records[0]]);
  await writing;
  store.read = read;
  expect((await readJournalRecords(store, path)).records).toEqual(
    records.slice(0, 2),
  );
});

test.each(["read", "append", "repair"] as const)(
  "journal %s releases its I/O lock after failure",
  async (operation) => {
    const store = new MemoryFileStore();
    const path = "journal.bin";
    const error = new Error("storage failure");
    await store.write(path, new Uint8Array([1]));
    jest
      .spyOn(store, operation === "repair" ? "write" : operation)
      .mockRejectedValueOnce(error);
    const result =
      operation === "append"
        ? appendJournalRecord(store, path, records[0])
        : operation === "repair"
          ? repairJournalTail(store, path)
          : readJournalRecords(store, path);
    await expect(result).rejects.toBe(error);
    const release = await store.tryLock(journalLockPath(path));
    expect(release).toBeDefined();
    release!();
  },
);

test("relaxed token writes serialize before checkpoint commits", async () => {
  const store = new MemoryFileStore();
  const originalAppend = store.append.bind(store);
  let releaseAppend!: () => void;
  let notifyAppend!: () => void;
  const appendStarted = new Promise<void>((resolve) => {
    notifyAppend = resolve;
  });
  let pauseGeneratedToken = false;
  store.append = async (path, data) => {
    const record = decodeJournalRecordAt(bytes(data).buffer).record;
    if (
      pauseGeneratedToken &&
      record.type === JournalRecordType.GeneratedToken
    ) {
      await new Promise<void>((resolve) => {
        releaseAppend = resolve;
        notifyAppend();
      });
    }
    await originalAppend(path, data);
  };
  const sessions = new ResumableSessionStore(store, {
    rootPath: "resume-root",
  });
  const journal = new ResumableGenerationJournal(
    store,
    sessions,
    normalizeResumableGenerationConfig({
      enabled: true,
      sessionId: "session-relaxed-order",
      durabilityMode: "relaxed",
    })!,
  );
  await journal.begin({
    modelId: "model-a",
    request: { messages: [] },
    promptTokenIds: [1, 2],
    assistantPrefixTokenIds: [],
    generationConfig: {},
  });
  pauseGeneratedToken = true;
  await journal.recordGeneratedToken({
    globalTokenPos: 2,
    tokenId: 3,
    textDelta: "x",
    textPrefixLength: 0,
    rngState: 4,
  });

  let committed = false;
  const commit = journal
    .recordCheckpointCommit({
      checkpointId: "checkpoint_2",
      processedSeqLen: 2,
      path: "kv/checkpoint_2",
    })
    .then(() => {
      committed = true;
    });
  await appendStarted;
  expect(committed).toBe(false);
  releaseAppend();
  await commit;

  const scan = await readJournalRecords(
    store,
    "resume-root/sessions/session-relaxed-order/journal.bin",
  );
  expect(scan.records.slice(-2).map((record) => record.type)).toEqual([
    JournalRecordType.GeneratedToken,
    JournalRecordType.CheckpointCommit,
  ]);
  expect(scan.records.slice(-2).map((record) => record.seqNo)).toEqual([4, 5]);
  await journal.close();
});

test("relaxed persistence never writes records after an earlier append failure", async () => {
  const store = new MemoryFileStore();
  const originalAppend = store.append.bind(store);
  let failed = false;
  store.append = async (path, data) => {
    const journalRecord = decodeJournalRecordAt(bytes(data).buffer).record;
    if (!failed && journalRecord.type === JournalRecordType.GeneratedToken) {
      failed = true;
      throw new Error("token append failed");
    }
    await originalAppend(path, data);
  };
  const sessions = new ResumableSessionStore(store, {
    rootPath: "resume-root",
  });
  const journal = new ResumableGenerationJournal(
    store,
    sessions,
    normalizeResumableGenerationConfig({
      enabled: true,
      sessionId: "session-relaxed-failure",
      durabilityMode: "relaxed",
      strictPersistence: false,
    })!,
  );
  await journal.begin({
    modelId: "model-a",
    request: { messages: [] },
    promptTokenIds: [1, 2],
    assistantPrefixTokenIds: [],
    generationConfig: {},
  });

  for (let index = 0; index < 8; index++) {
    await journal.recordGeneratedToken({
      globalTokenPos: 2 + index,
      tokenId: 10 + index,
      textDelta: String(index),
      textPrefixLength: index,
      rngState: index,
    });
  }
  await journal.close();

  const scan = await readJournalRecords(
    store,
    "resume-root/sessions/session-relaxed-failure/journal.bin",
  );
  expect(failed).toBe(true);
  expect(
    scan.records.filter(
      (journalRecord) =>
        journalRecord.type === JournalRecordType.GeneratedToken,
    ),
  ).toHaveLength(0);
});

test("text replay applies reversible patches and legacy deltas", () => {
  let text = applyGeneratedTokenText("", "caf\ufffd", 0);
  text = applyGeneratedTokenText(text, "\u00e9", 3);
  text = applyGeneratedTokenText(text, "!", text.length);
  text = applyGeneratedTokenText(text, " legacy");

  expect(text).toBe("caf\u00e9! legacy");
  expect(() => applyGeneratedTokenText("short", "bad", 6)).toThrow(
    "Invalid resumable text prefix length 6 for text of length 5.",
  );
});

test("resumable generation journal rejects active writers and session reuse", async () => {
  const store = new MemoryFileStore();
  const sessions = new ResumableSessionStore(store, {
    rootPath: "resume-root",
  });
  const config = normalizeResumableGenerationConfig({
    enabled: true,
    sessionId: "session-a",
  })!;
  const init = {
    modelId: "model-a",
    request: { messages: [] },
    promptTokenIds: [1, 2, 3],
    assistantPrefixTokenIds: [],
    generationConfig: {},
  };
  const first = new ResumableGenerationJournal(store, sessions, config);
  const second = new ResumableGenerationJournal(store, sessions, config);

  await first.begin(init);
  await expect(second.begin(init)).rejects.toThrow(
    "Resumable session is already active: session-a",
  );

  await first.close();
  await expect(second.begin(init)).rejects.toThrow(
    "Resumable session already exists: session-a",
  );
});

test("the journal model id takes precedence over a stale manifest", async () => {
  const store = new MemoryFileStore();
  const sessions = new ResumableSessionStore(store);
  const session = await sessions.createSession("session-a", {
    modelId: "stale-model",
  });
  await appendJournalRecord(store, session.paths.journalPath, {
    type: JournalRecordType.SessionBegin,
    seqNo: 1,
    createdAtMs: 100,
    payload: { sessionId: "session-a", modelId: "model-a" },
  });
  expect((await readResumableReplayState(store, session)).modelId).toBe(
    "model-a",
  );
  expect((await probeResumableSession(store, session)).modelId).toBe("model-a");
});

test("replay rejects journals containing multiple session beginnings", async () => {
  const store = new MemoryFileStore();
  const sessions = new ResumableSessionStore(store, {
    rootPath: "resume-root",
  });
  const session = await sessions.createSession("session-a");
  const sessionBegin = record({
    type: JournalRecordType.SessionBegin,
    seqNo: 1,
    createdAtMs: 100,
    payload: { sessionId: "session-a", modelId: "model-a" },
  });
  await appendJournalRecord(store, session.paths.journalPath, sessionBegin);
  await appendJournalRecord(store, session.paths.journalPath, {
    ...sessionBegin,
    seqNo: 2,
  });

  await expect(readResumableReplayState(store, session)).rejects.toThrow(
    "Resumable session session-a contains multiple session-begin records.",
  );
  await expect(probeResumableSession(store, session)).rejects.toThrow(
    "Resumable session session-a contains multiple session-begin records.",
  );
});

test.each([
  ["empty", false, "none", "missing journal records"],
  ["prompt", true, "token_replay", "generation incomplete"],
  ["token", true, "token_replay", "generation incomplete"],
  ["abort", true, "token_replay", "generation aborted"],
  ["error", true, "token_replay", "engine error: decode failed"],
  ["finished", false, "none", "generation finished"],
  [
    "missing-rng",
    false,
    "text_only",
    "missing RNG state; token replay unavailable",
  ],
] as const)(
  "probe and replay share the %s journal snapshot",
  async (stage, resumable, recoveryMode, reason) => {
    const files = new MemoryFileStore();
    const sessions = new ResumableSessionStore(files);
    const config = normalizeResumableGenerationConfig({
      enabled: true,
      sessionId: "snapshot",
    })!;
    if (stage === "empty") {
      await sessions.createSession(config.sessionId);
    } else {
      const journal = new ResumableGenerationJournal(files, sessions, config);
      await journal.begin({
        modelId: "model",
        request: {},
        promptTokenIds: [1, 2],
        assistantPrefixTokenIds: [],
        generationConfig: {},
      });
      if (stage !== "prompt") {
        await journal.recordGeneratedToken({
          globalTokenPos: 2,
          tokenId: 3,
          textDelta: "hi",
          textPrefixLength: 0,
          rngState: stage === "missing-rng" ? undefined : 42,
        });
      }
      if (stage === "abort" || stage === "finished")
        await journal.recordGenerationEnd({
          finishReason: stage === "abort" ? "abort" : "stop",
          emittedTokens: 1,
        });
      if (stage === "error")
        await journal.recordEngineError({ err: new Error("decode failed") });
      await journal.close();
    }
    const session = (await sessions.openSession(config.sessionId))!;
    const state = await readResumableReplayState(files, session);
    expect(await probeResumableSession(files, session)).toEqual(
      probeReplayState(state),
    );
    expect(probeReplayState(state)).toMatchObject({
      resumable,
      recoveryMode,
      reason,
      emittedTokens: state.generatedTokens.length,
      processedSeqLen: state.processedSeqLen,
    });
    expect(state.recoveredText).toBe(
      stage === "empty" || stage === "prompt" ? "" : "hi",
    );
  },
);
