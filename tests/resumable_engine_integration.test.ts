import {
  MODEL_ID,
  mockChatConfig,
  createEngineWithPipeline,
} from "./helpers/engine_fixture";
import {
  ChatCompletion,
  ChatCompletionRequest,
  CompletionCreateParams,
  ChatCompletionChunk,
} from "../src/openai_api_protocols";
import { MLCEngine } from "../src/engine";
import { ModelType } from "../src/config";
import { MemoryFileStore, bytes } from "./helpers/memory_file_store";
import {
  readJournalRecords,
  appendJournalRecord,
  decodeJournalRecordAt,
  JournalRecordType,
} from "../src/resumable/journal";
import {
  ResumableInjectedFault,
  setResumableFaultHook,
} from "../src/resumable/fault_injection";
import type {
  ResumableFaultContext,
  ResumableFaultPoint,
} from "../src/resumable/fault_injection";
import { ResumableSessionStore } from "../src/resumable/session_store";
import { jest, test, expect, describe, afterEach } from "@jest/globals";
import log from "loglevel";

test("MLCEngine resumable helpers tolerate unavailable OPFS for list/delete", async () => {
  const engine = new MLCEngine();
  await expect(engine.listResumableSessions()).resolves.toEqual([]);
  await expect(
    engine.deleteResumableSession("session-a"),
  ).resolves.toBeUndefined();
  await expect(engine.resumeChatCompletion("session-a")).rejects.toThrow(
    "OPFS is unavailable in this environment",
  );
});

function attachResumableStore(engine: MLCEngine, files: MemoryFileStore): void {
  const internal = engine as any;
  internal.resumableFileStore = files;
  internal.resumableSessionStore = new ResumableSessionStore(files, {
    rootPath: "resume-root",
  });
}

async function readSessionJournal(files: MemoryFileStore, sessionId: string) {
  return readJournalRecords(
    files,
    `resume-root/sessions/${sessionId}/journal.bin`,
  );
}

let restoreResumableFaultHook: (() => void) | undefined;

function clearResumableFaultHook(): void {
  restoreResumableFaultHook?.();
  restoreResumableFaultHook = undefined;
}

function injectResumableFaultOnce(
  point: ResumableFaultPoint,
  matches: (context: ResumableFaultContext) => boolean = () => true,
): void {
  clearResumableFaultHook();
  let fired = false;
  restoreResumableFaultHook = setResumableFaultHook((actual, context) => {
    if (!fired && actual === point && matches(context)) {
      fired = true;
      throw new ResumableInjectedFault(actual, context);
    }
  });
}

function setNavigatorStorageEstimate(estimate: {
  quota?: number;
  usage?: number;
}): () => void {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      storage: {
        estimate: jest.fn(async () => estimate),
      },
    },
  });
  return () => {
    if (previous === undefined) {
      delete (globalThis as any).navigator;
    } else {
      Object.defineProperty(globalThis, "navigator", previous);
    }
  };
}

afterEach(() => {
  clearResumableFaultHook();
  jest.useRealTimers();
});

describe("MLCEngine resumable integration", () => {
  test.each([
    [false, "missing-rng"],
    [true, "missing-rng"],
    [false, "custom-processor"],
    [true, "custom-processor"],
  ] as const)(
    "recovery eligibility agrees with KV=%s for %s",
    async (kv, blocker) => {
      const { engine, pipeline } = createEngineWithPipeline(3);
      pipeline.enablePromptCheckpoint = kv;
      if (blocker === "missing-rng")
        jest.spyOn(pipeline, "getRNGState").mockReturnValue(undefined);
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const stream = await engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Interrupted" }],
        stream: true,
        extra_body: { resumable: { enabled: true, sessionId: "eligibility" } },
      });
      const iterator = stream[Symbol.asyncIterator]();
      await iterator.next();
      await iterator.return!();
      if (blocker === "custom-processor")
        engine.setLogitProcessorRegistry(
          new Map([
            [
              MODEL_ID,
              {
                processLogits: (logits: Float32Array) => logits,
                processSampledToken: () => undefined,
                resetState: () => undefined,
              },
            ],
          ]),
        );
      const restoreKV = jest.spyOn(pipeline, "replayFromPromptCheckpoint");
      const replayTokens = jest.spyOn(pipeline, "replayGenerationTokens");
      const recoveredText = await engine.getMessage(MODEL_ID);
      await expect(
        engine.resumeChatCompletion("eligibility", {
          continueGeneration: true,
        }),
      ).resolves.toMatchObject({
        recoveryMode: "text_only",
        recoveredText,
        replayedTokens: 0,
      });
      expect(restoreKV).not.toHaveBeenCalled();
      expect(replayTokens).not.toHaveBeenCalled();
    },
  );

  test.each([false, true])(
    "shared startup journals prompt/checkpoint/token in order (stream=%s)",
    async (stream) => {
      const { engine, pipeline } = createEngineWithPipeline(2);
      pipeline.enablePromptCheckpoint = true;
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const result = await engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Shared startup" }],
        stream,
        seed: 17,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "shared-start",
            strictPersistence: true,
          },
        },
      });
      if (Symbol.asyncIterator in result)
        for await (const chunk of result) expect(chunk.model).toBe(MODEL_ID);
      const { records } = await readSessionJournal(files, "shared-start");
      const kinds = records.map((record) => record.type);
      expect(kinds.indexOf(JournalRecordType.PromptTokens)).toBeLessThan(
        kinds.indexOf(JournalRecordType.CheckpointCommit),
      );
      expect(kinds.indexOf(JournalRecordType.CheckpointCommit)).toBeLessThan(
        kinds.indexOf(JournalRecordType.GeneratedToken),
      );
      expect(
        kinds.filter((kind) => kind === JournalRecordType.SessionBegin),
      ).toHaveLength(1);
      expect(
        kinds.filter((kind) => kind === JournalRecordType.GenerationFinished),
      ).toHaveLength(1);
      expect(pipeline.prefillCallCount).toBe(1);
      await expect(
        engine.chatCompletion({
          model: MODEL_ID,
          messages: [{ role: "user", content: "Next request" }],
        }),
      ).resolves.toBeDefined();
    },
  );

  test.each([false, true])(
    "first checkpoint failure releases session/model ownership (stream=%s)",
    async (stream) => {
      const { engine, pipeline } = createEngineWithPipeline(2);
      pipeline.enablePromptCheckpoint = true;
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const write = files.write.bind(files);
      jest.spyOn(files, "write").mockImplementation(async (path, data) => {
        if (path.includes("/kv/")) throw new Error("checkpoint write failed");
        await write(path, data);
      });
      const generate = async () => {
        const result = await engine.chatCompletion({
          model: MODEL_ID,
          messages: [{ role: "user", content: "Fail checkpoint" }],
          stream,
          extra_body: {
            resumable: {
              enabled: true,
              sessionId: "failed-start",
              strictPersistence: true,
            },
          },
        });
        if (Symbol.asyncIterator in result)
          for await (const chunk of result) expect(chunk.model).toBe(MODEL_ID);
      };
      await expect(generate()).rejects.toThrow("checkpoint write failed");
      const { records } = await readSessionJournal(files, "failed-start");
      expect(
        records.some((record) => record.type === JournalRecordType.EngineError),
      ).toBe(true);
      expect(
        records.some(
          (record) => record.type === JournalRecordType.GeneratedToken,
        ),
      ).toBe(false);
      await expect(
        engine.deleteResumableSession("failed-start"),
      ).resolves.toBeUndefined();
      await expect(
        engine.chatCompletion({
          model: MODEL_ID,
          messages: [{ role: "user", content: "After failure" }],
        }),
      ).resolves.toBeDefined();
    },
  );

  test("completion rejects resumable extra_body without creating a session", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);

    await expect(
      engine.completion({
        model: MODEL_ID,
        prompt: "Do not persist this",
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "completion-session",
          },
        },
      } as unknown as CompletionCreateParams),
    ).rejects.toThrow("extra_body.resumable");
    await expect(files.list("resume-root/sessions")).resolves.toEqual([]);
  });

  test("resumable chatCompletion journals visible text and token ids", async () => {
    const { engine } = createEngineWithPipeline(2);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const response = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Persist this" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-a",
        },
      },
    })) as ChatCompletion;

    const journal = await readSessionJournal(files, "session-a");
    const generated = journal.records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102,
    ]);
    expect(generated.map((record) => record.payload.textDelta).join("")).toBe(
      response.choices[0].message.content,
    );
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.GenerationFinished,
      ),
    ).toBe(true);
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-a",
        resumable: false,
        reason: "generation finished",
        modelId: MODEL_ID,
        emittedTokens: 3,
        recoveryMode: "none",
      }),
    ]);
  });

  test("resumable prefill does not reuse a compatible conversation KV cache", () => {
    const { engine, pipeline } = createEngineWithPipeline(1);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "user", content: "First" },
        { role: "assistant", content: "Answer" },
        { role: "user", content: "Second" },
      ],
    };
    const internal = engine as any;

    internal.preparePrefillInput(request, pipeline, mockChatConfig, true);
    pipeline.resetCount = 0;
    internal.preparePrefillInput(request, pipeline, mockChatConfig, true);
    expect(pipeline.resetCount).toBe(0);

    internal.preparePrefillInput(request, pipeline, mockChatConfig, false);
    expect(pipeline.resetCount).toBe(1);
  });

  test("a resumable session persists the complete multi-turn request", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const messages: ChatCompletionRequest["messages"] = [
      { role: "user", content: "user1" },
      { role: "assistant", content: "assistant1" },
      { role: "user", content: "user2" },
      { role: "assistant", content: "assistant2" },
      { role: "user", content: "user3" },
    ];

    await engine.chatCompletion({
      model: MODEL_ID,
      messages,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-multi-turn",
        },
      },
    });

    const journal = await readSessionJournal(files, "session-multi-turn");
    expect(journal.records).toContainEqual(
      expect.objectContaining({
        type: JournalRecordType.SessionBegin,
        payload: expect.objectContaining({
          request: expect.objectContaining({ messages }),
        }),
      }),
    );
  });

  test("exact-mode resumable streaming waits for generated token journal append", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    let generatedTokenAppendCount = 0;
    files.pauseAppendWhen = (_path, data) => {
      const record = decodeJournalRecordAt(data.buffer).record;
      if (record.type !== JournalRecordType.GeneratedToken) {
        return false;
      }
      generatedTokenAppendCount++;
      return true;
    };
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Stream persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-stream",
          durabilityMode: "exact",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();
    let resolved = false;
    const firstChunk = iterator.next().then((value) => {
      resolved = true;
      return value;
    });

    for (let i = 0; i < 10 && generatedTokenAppendCount === 0; i++) {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    }
    expect(generatedTokenAppendCount).toBe(1);
    expect(resolved).toBe(false);
    files.pauseAppendWhen = undefined;
    files.releaseAllAppends();
    const result = await firstChunk;
    expect(result.done).toBe(false);
    expect(result.value.choices[0].delta?.content).toContain("Stream persist");
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate releases the model lock
    }
  });

  test("strictPersistence false falls back after asynchronous storage initialization failure", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    files.tryLock = jest.fn(async () => {
      throw new Error("asynchronous OPFS initialization failed");
    });
    attachResumableStore(engine, files);

    const response = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Storage fallback" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-storage-fallback",
          strictPersistence: false,
        },
      },
    })) as ChatCompletion;

    expect(response.choices[0].message.content).toBe(
      "user:Storage fallback|token1|",
    );
  });

  test("strictPersistence false keeps generating after an asynchronous token write failure", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    const sampleDecode = jest.spyOn(pipeline, "sampleDecodeStep");
    const files = new MemoryFileStore();
    const originalAppend = files.append.bind(files);
    let failedGeneratedTokenWrite = false;
    files.append = async (path, data) => {
      const record = decodeJournalRecordAt(bytes(data).buffer).record;
      if (
        !failedGeneratedTokenWrite &&
        record.type === JournalRecordType.GeneratedToken
      ) {
        failedGeneratedTokenWrite = true;
        throw new Error("asynchronous token write failed");
      }
      await originalAppend(path, data);
    };
    attachResumableStore(engine, files);

    const response = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Write fallback" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-write-fallback",
          strictPersistence: false,
          checkpointIntervalTokens: 1,
        },
      },
    })) as ChatCompletion;

    expect(failedGeneratedTokenWrite).toBe(true);
    expect(response.choices[0].message.content).toBe(
      "user:Write fallback|token1||token2||token3|",
    );
    expect(sampleDecode).toHaveBeenCalled();
    for (const [, options] of sampleDecode.mock.calls) {
      expect(options).toMatchObject({ captureCheckpoint: false });
    }
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-write-fallback",
        emittedTokens: 0,
        recoveryMode: "token_replay",
      }),
    ]);
  });

  test.each([false, true])(
    "finished KV cleanup failure does not discard the response (strict: %s)",
    async (strictPersistence) => {
      const { engine, pipeline } = createEngineWithPipeline(1);
      pipeline.enablePromptCheckpoint = true;
      const files = new MemoryFileStore();
      const remove = files.remove.bind(files);
      files.remove = async (path, options) => {
        if (path.endsWith("/kv")) throw new Error("cleanup failed");
        return remove(path, options);
      };
      attachResumableStore(engine, files);
      const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);
      try {
        const response = await engine.chatCompletion({
          model: MODEL_ID,
          messages: [{ role: "user", content: "Persist before cleanup" }],
          extra_body: {
            resumable: {
              enabled: true,
              sessionId: "session-cleanup-failure",
              strictPersistence,
            },
          },
        });
        expect(response.choices[0].message.content).toBe(
          "user:Persist before cleanup|token1|",
        );
        const scan = await readSessionJournal(files, "session-cleanup-failure");
        expect(scan.records.at(-1)?.type).toBe(
          JournalRecordType.GenerationFinished,
        );
        files.remove = remove;
        await engine.listResumableSessions();
        expect(
          await files.list("resume-root/sessions/session-cleanup-failure/kv"),
        ).toEqual([]);
      } finally {
        warn.mockRestore();
      }
    },
  );

  test("strictPersistence true surfaces asynchronous storage initialization failure", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    files.tryLock = jest.fn(async () => {
      throw new Error("asynchronous OPFS initialization failed");
    });
    attachResumableStore(engine, files);

    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Storage strict" }],
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-storage-strict",
            strictPersistence: true,
          },
        },
      }),
    ).rejects.toThrow("asynchronous OPFS initialization failed");
  });

  test("strictPersistence true surfaces an asynchronous token write failure", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    const originalAppend = files.append.bind(files);
    files.append = async (path, data) => {
      const record = decodeJournalRecordAt(bytes(data).buffer).record;
      if (record.type === JournalRecordType.GeneratedToken) {
        throw new Error("asynchronous token write failed");
      }
      await originalAppend(path, data);
    };
    attachResumableStore(engine, files);

    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Write strict" }],
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-write-strict",
            strictPersistence: true,
          },
        },
      }),
    ).rejects.toThrow("asynchronous token write failed");
  });

  test("stream return records an abort and releases model and session locks", async () => {
    const { engine } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Cancel stream" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-cancel-stream",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await expect(iterator.next()).resolves.toMatchObject({ done: false });
    await iterator.return!();

    const journal = await readSessionJournal(files, "session-cancel-stream");
    expect(journal.records.at(-1)?.type).toBe(
      JournalRecordType.GenerationAborted,
    );
    const release = await files.tryLock(
      "resume-root/sessions/session-cancel-stream/lock",
    );
    expect(release).toBeDefined();
    release!();
    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "After cancel" }],
      }),
    ).resolves.toBeDefined();
  });

  test.each([false, true])(
    "a never-started resumed stream acquires no locks (torn journal: %s)",
    async (torn) => {
      const { engine } = createEngineWithPipeline(3);
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const initial = (await engine.chatCompletion({
        model: MODEL_ID,
        seed: 7,
        messages: [{ role: "user", content: "Lazy resume" }],
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-lazy-resume",
          },
        },
      })) as AsyncIterable<ChatCompletionChunk>;
      const initialIterator = initial[Symbol.asyncIterator]();
      await initialIterator.next();
      await initialIterator.return!();

      if (torn) {
        await files.append(
          "resume-root/sessions/session-lazy-resume/journal.bin",
          new Uint8Array([1, 2, 3]),
        );
      }
      const resumed = (await engine.resumeChatCompletion(
        "session-lazy-resume",
        {
          continueGeneration: true,
          stream: true,
        },
      )) as AsyncIterable<ChatCompletionChunk>;

      const neverStartedRelease = await files.tryLock(
        "resume-root/sessions/session-lazy-resume/lock",
      );
      expect(neverStartedRelease).toBeDefined();
      neverStartedRelease!();

      await resumed[Symbol.asyncIterator]().return!();
      await expect(
        engine.resumeChatCompletion("session-lazy-resume", {
          continueGeneration: true,
        }),
      ).resolves.toMatchObject({ recoveryMode: "token_replay" });
    },
  );

  test.each([
    [false, false],
    [false, true],
    [true, false],
    [true, true],
  ])(
    "stream cancellation honors persistence strictness (resumed: %s, strict: %s)",
    async (resumed, strictPersistence) => {
      const { engine } = createEngineWithPipeline(5);
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const sessionId = "session-cancel-persist-failure";
      let stream = await engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Cancel" }],
        seed: 7,
        stream: true,
        extra_body: {
          resumable: { enabled: true, sessionId, strictPersistence },
        },
      });
      let iterator = stream[Symbol.asyncIterator]();
      await iterator.next();
      if (resumed) {
        await iterator.return!();
        stream = (await engine.resumeChatCompletion(sessionId, {
          continueGeneration: true,
          stream: true,
        })) as AsyncIterable<ChatCompletionChunk>;
        iterator = stream[Symbol.asyncIterator]();
        await iterator.next();
      }
      const append = files.append.bind(files);
      files.append = async (path, data) => {
        if (
          decodeJournalRecordAt(bytes(data).buffer).record.type ===
          JournalRecordType.GenerationAborted
        )
          throw new Error("abort write failed");
        return append(path, data);
      };
      if (strictPersistence)
        await expect(iterator.return!()).rejects.toThrow("abort write failed");
      else
        await expect(iterator.return!()).resolves.toMatchObject({ done: true });
      const release = await files.tryLock(
        `resume-root/sessions/${sessionId}/lock`,
      );
      expect(release).toBeDefined();
      release!();
      await expect(
        engine.chatCompletion({
          model: MODEL_ID,
          messages: [{ role: "user", content: "Next request" }],
        }),
      ).resolves.toBeDefined();
    },
  );

  test("returning a started resumed stream records an abort and releases its locks", async () => {
    const { engine, pipeline } = createEngineWithPipeline(5);
    const seed = jest.spyOn(pipeline, "setSeed");
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const initial = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 7,
      messages: [{ role: "user", content: "Cancel resumed stream" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-cancel-resume",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const initialIterator = initial[Symbol.asyncIterator]();
    await initialIterator.next();
    await initialIterator.return!();
    expect(seed.mock.calls).toEqual([[7], [expect.any(Number)]]);
    expect(seed.mock.calls[1][0]).not.toBe(7);

    const resumed = (await engine.resumeChatCompletion(
      "session-cancel-resume",
      { continueGeneration: true, stream: true },
    )) as AsyncIterable<ChatCompletionChunk>;
    const resumedIterator = resumed[Symbol.asyncIterator]();
    await expect(resumedIterator.next()).resolves.toMatchObject({
      done: false,
    });
    const seedsBeforeReturn = seed.mock.calls.length;
    await resumedIterator.return!();
    expect(seed).toHaveBeenCalledTimes(seedsBeforeReturn + 1);
    expect(seed.mock.calls.at(-1)![0]).not.toBe(7);

    const journal = await readSessionJournal(files, "session-cancel-resume");
    expect(journal.records.at(-1)?.type).toBe(
      JournalRecordType.GenerationAborted,
    );
    const release = await files.tryLock(
      "resume-root/sessions/session-cancel-resume/lock",
    );
    expect(release).toBeDefined();
    release!();
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "An unseeded follow-up" }],
    });
    expect(seed).toHaveBeenCalledTimes(seedsBeforeReturn + 1);
  });

  test("token replay recovers a crash before the first generated token", async () => {
    const { engine } = createEngineWithPipeline(2);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    injectResumableFaultOnce(
      "journal.after_append",
      (context) => context.recordType === JournalRecordType.GenerationConfig,
    );

    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        seed: 11,
        messages: [{ role: "user", content: "Crash before token" }],
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-zero-token-replay",
            checkpointPrompt: false,
            strictPersistence: true,
          },
        },
      }),
    ).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    expect(
      (await readSessionJournal(files, "session-zero-token-replay")).records,
    ).not.toContainEqual(
      expect.objectContaining({ type: JournalRecordType.GeneratedToken }),
    );
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-zero-token-replay",
        resumable: true,
        emittedTokens: 0,
        recoveryMode: "token_replay",
      }),
    ]);
    await expect(
      engine.resumeChatCompletion("session-zero-token-replay", {
        continueGeneration: true,
      }),
    ).resolves.toMatchObject({
      recoveryMode: "token_replay",
      replayedTokens: 0,
      recoveredText: "first|token1||token2|",
      emittedTokens: 3,
    });
  });

  test("streamed KV recovery emits the token sampled from checkpoint logits", async () => {
    const { engine, pipeline } = createEngineWithPipeline(2);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    injectResumableFaultOnce("checkpoint.after_commit");
    const initial = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 13,
      messages: [{ role: "user", content: "Checkpoint first token" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-checkpoint-first-token",
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    await expect(initial[Symbol.asyncIterator]().next()).rejects.toThrow(
      "Injected resumable fault",
    );
    clearResumableFaultHook();

    const resumed = (await engine.resumeChatCompletion(
      "session-checkpoint-first-token",
      { continueGeneration: true, stream: true },
    )) as AsyncIterable<ChatCompletionChunk>;
    const iterator = resumed[Symbol.asyncIterator]();
    const first = await iterator.next();
    expect(first.value.choices[0].delta?.content).toBe("first");
    while (!(await iterator.next()).done) {
      // drain to verify normal completion and cleanup
    }
    const generated = (
      await readSessionJournal(files, "session-checkpoint-first-token")
    ).records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102,
    ]);
  });

  test("resumable multimodal input is rejected before inference", async () => {
    const { engine, pipeline } = createEngineWithPipeline(1);
    (engine as any).loadedModelIdToModelType.set(MODEL_ID, ModelType.VLM);
    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "describe" },
              {
                type: "image_url",
                image_url: { url: "data:image/png;base64,AA==" },
              },
            ],
          },
        ],
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-multimodal",
          },
        },
      } as any),
    ).rejects.toThrow("Resumable generation supports text-only prompts.");
    expect(pipeline.prefillCallCount).toBe(0);
  });

  test("listResumableSessions reports interrupted token replay candidates", async () => {
    const { engine } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Interrupt persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-interrupted",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    const first = await iterator.next();
    expect(first.done).toBe(false);
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const sessions = await engine.listResumableSessions();
    expect(sessions).toEqual([
      expect.objectContaining({
        sessionId: "session-interrupted",
        resumable: true,
        reason: "generation aborted",
        modelId: MODEL_ID,
        emittedTokens: 1,
        recoveryMode: "token_replay",
      }),
    ]);
    expect(sessions[0].processedSeqLen).toBeGreaterThan(0);

    await engine.deleteResumableSession("session-interrupted");
    await expect(engine.listResumableSessions()).resolves.toEqual([]);
    await expect(files.list("resume-root/sessions")).resolves.toEqual([]);
  });

  test("resumeChatCompletion continues interrupted generation by token replay", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    const first = await iterator.next();
    expect(first.value.choices[0].delta?.content).toBe("user:Resume persist");
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const result = (await engine.resumeChatCompletion("session-resume", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-resume",
      recoveryMode: "token_replay",
      replayedTokens: 1,
      emittedTokens: 4,
      recoveredText: "user:Resume persist|token1||token2||token3|",
    });
    const journal = await readSessionJournal(files, "session-resume");
    const generated = journal.records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102, 103,
    ]);
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.GenerationFinished,
      ),
    ).toBe(true);

    const resetCountAfterResume = pipeline.resetCount;
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [
        { role: "user", content: "Resume persist" },
        { role: "assistant", content: result.recoveredText },
        { role: "user", content: "Follow up" },
      ],
    });
    expect(pipeline.resetCount).toBe(resetCountAfterResume);
  });

  test("resumeChatCompletion can stream continuation after token replay", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume stream" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-stream",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const result = await engine.resumeChatCompletion("session-resume-stream", {
      continueGeneration: true,
      stream: true,
    });
    expect(Symbol.asyncIterator in (result as object)).toBe(true);
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }
    expect(
      chunks.map((chunk) => chunk.choices[0]?.delta?.content ?? "").join(""),
    ).toBe("|token1||token2||token3|");
    expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe("stop");
    const journal = await readSessionJournal(files, "session-resume-stream");
    const generated = journal.records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102, 103,
    ]);
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.GenerationFinished,
      ),
    ).toBe(true);
  });

  test("resumeChatCompletion can resume again after resumed journal append crash", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume twice" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-twice",
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    injectResumableFaultOnce(
      "journal.after_append",
      (context) => context.recordType === JournalRecordType.GeneratedToken,
    );
    await expect(
      engine.resumeChatCompletion("session-resume-twice", {
        continueGeneration: true,
      }),
    ).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    const crashedJournal = await readSessionJournal(
      files,
      "session-resume-twice",
    );
    expect(
      crashedJournal.records.filter(
        (record) => record.type === JournalRecordType.GeneratedToken,
      ),
    ).toHaveLength(2);
    expect(
      crashedJournal.records.some(
        (record) => record.type === JournalRecordType.EngineError,
      ),
    ).toBe(false);

    const result = (await engine.resumeChatCompletion("session-resume-twice", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-resume-twice",
      recoveryMode: "token_replay",
      replayedTokens: 2,
      emittedTokens: 4,
      recoveredText: "user:Resume twice|token1||token2||token3|",
    });
  });

  test("resumeChatCompletion continueGeneration rejects missing resumable config", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const sessions = new ResumableSessionStore(files, {
      rootPath: "resume-root",
    });
    const session = await sessions.createSession("session-missing-resumable", {
      modelId: MODEL_ID,
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.SessionBegin,
      seqNo: 1,
      createdAtMs: Date.now(),
      payload: {
        sessionId: "session-missing-resumable",
        modelId: MODEL_ID,
        request: {
          model: MODEL_ID,
          messages: [{ role: "user", content: "Missing config" }],
        },
      },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.PromptTokens,
      seqNo: 2,
      createdAtMs: Date.now(),
      payload: { tokenIds: [1, 2, 3] },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.GenerationConfig,
      seqNo: 3,
      createdAtMs: Date.now(),
      payload: { config: { max_tokens: 3 } },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.GeneratedToken,
      seqNo: 4,
      createdAtMs: Date.now(),
      payload: {
        globalTokenPos: 3,
        tokenId: 100,
        textDelta: "partial",
        rngState: 1,
      },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.GenerationAborted,
      seqNo: 5,
      createdAtMs: Date.now(),
      payload: { reason: "abort" },
    });

    await expect(
      engine.resumeChatCompletion("session-missing-resumable"),
    ).resolves.toMatchObject({
      sessionId: "session-missing-resumable",
      recoveredText: "partial",
      recoveryMode: "text_only",
    });
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-missing-resumable",
        resumable: false,
        reason: "missing resumable generation config; continuation unavailable",
        recoveryMode: "text_only",
      }),
    ]);
    await expect(
      engine.resumeChatCompletion("session-missing-resumable", {
        continueGeneration: true,
      }),
    ).rejects.toThrow(
      "Resumable session session-missing-resumable is missing or has malformed resumable generation config.",
    );
  });

  test("resumeChatCompletion continueGeneration fails while session is active", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume lock" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-lock",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const release = await files.tryLock(
      "resume-root/sessions/session-resume-lock/lock",
    );
    expect(release).toBeDefined();
    await expect(
      engine.resumeChatCompletion("session-resume-lock", {
        continueGeneration: true,
      }),
    ).rejects.toThrow(
      "Resumable session is already active: session-resume-lock",
    );
    await expect(
      engine.resumeChatCompletion("session-resume-lock"),
    ).resolves.toMatchObject({
      sessionId: "session-resume-lock",
      recoveryMode: "text_only",
      recoveredText: "user:Resume lock",
    });

    release!();
    await expect(
      engine.resumeChatCompletion("session-resume-lock", {
        continueGeneration: true,
      }),
    ).resolves.toMatchObject({
      sessionId: "session-resume-lock",
      recoveryMode: "token_replay",
    });
  });

  test("resumeChatCompletion restores from prompt checkpoint when available", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "KV resume" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-kv-resume",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const checkpoints = await files.list(
      "resume-root/sessions/session-kv-resume/kv",
    );
    expect(checkpoints).toEqual(["checkpoint_00000000_00000009"]);
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-kv-resume",
        recoveryMode: "kv",
      }),
    ]);

    const result = (await engine.resumeChatCompletion("session-kv-resume", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-kv-resume",
      recoveryMode: "kv",
      replayedTokens: 1,
      recoveredText: "user:KV resume|token1||token2||token3|",
    });
    expect(pipeline.promptCheckpointRestoreCount).toBe(1);

    const resetCountAfterResume = pipeline.resetCount;
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [
        { role: "user", content: "KV resume" },
        { role: "assistant", content: result.recoveredText },
        { role: "user", content: "Follow up" },
      ],
    });
    expect(pipeline.resetCount).toBe(resetCountAfterResume);
  });

  test("resumeChatCompletion falls back to token replay when its only checkpoint is corrupt", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const sessionId = "session-corrupt-prompt-checkpoint";
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Corrupt KV" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const kvRoot = `resume-root/sessions/${sessionId}/kv`;
    const checkpoints = await files.list(kvRoot);
    expect(checkpoints).toHaveLength(1);
    await files.write(
      `${kvRoot}/${checkpoints[0]}/meta.json`,
      new TextEncoder().encode("{"),
    );
    const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);

    const result = (await engine.resumeChatCompletion(sessionId, {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId,
      recoveryMode: "token_replay",
      replayedTokens: 1,
      recoveredText: "user:Corrupt KV|token1||token2||token3|",
    });
    expect(pipeline.promptCheckpointRestoreCount).toBe(0);
    expect(await files.list(kvRoot)).toEqual([]);
    warn.mockRestore();
  });

  test("resumable crash after GENERATED_TOKEN append resumes by token replay", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    injectResumableFaultOnce(
      "journal.after_append",
      (context) => context.recordType === JournalRecordType.GeneratedToken,
    );
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Crash token" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-crash-token",
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await expect(iterator.next()).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    const journal = await readSessionJournal(files, "session-crash-token");
    expect(
      journal.records.filter(
        (record) => record.type === JournalRecordType.GeneratedToken,
      ),
    ).toHaveLength(1);
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.EngineError,
      ),
    ).toBe(false);

    const result = (await engine.resumeChatCompletion("session-crash-token", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-crash-token",
      recoveryMode: "token_replay",
      replayedTokens: 1,
      recoveredText: "user:Crash token|token1||token2||token3|",
    });
  });

  test.each([
    ["checkpoint.after_page_group", "token_replay"],
    ["checkpoint.after_next_logits", "token_replay"],
    ["checkpoint.after_meta", "token_replay"],
    ["checkpoint.after_complete", "token_replay"],
    ["checkpoint.before_commit", "token_replay"],
    ["checkpoint.after_commit", "kv"],
  ] as Array<[ResumableFaultPoint, "token_replay" | "kv"]>)(
    "decode checkpoint crash at %s resumes by %s",
    async (faultPoint, recoveryMode) => {
      const sessionId = `session-${faultPoint.replaceAll(".", "-")}`;
      const { engine, pipeline } = createEngineWithPipeline(4);
      pipeline.enableDecodeCheckpoint = true;
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      injectResumableFaultOnce(faultPoint);
      const iterable = (await engine.chatCompletion({
        model: MODEL_ID,
        seed: 1234,
        messages: [{ role: "user", content: "KV crash" }],
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId,
            checkpointPrompt: false,
            checkpointIntervalTokens: 1,
            strictPersistence: true,
          },
        },
      })) as AsyncIterable<ChatCompletionChunk>;
      const iterator = iterable[Symbol.asyncIterator]();

      await expect(iterator.next()).resolves.toMatchObject({
        done: false,
      });
      await expect(iterator.next()).rejects.toThrow("Injected resumable fault");
      clearResumableFaultHook();

      const journal = await readSessionJournal(files, sessionId);
      expect(
        journal.records.some(
          (record) => record.type === JournalRecordType.EngineError,
        ),
      ).toBe(false);
      await expect(engine.listResumableSessions()).resolves.toEqual([
        expect.objectContaining({
          sessionId,
          recoveryMode,
        }),
      ]);

      const result = (await engine.resumeChatCompletion(sessionId, {
        continueGeneration: true,
      })) as import("../src/types").ResumeResult;
      expect(result).toMatchObject({
        sessionId,
        recoveryMode,
      });
      expect(result.recoveredText).toBe(
        "user:KV crash|token1||token2||token3||token4|",
      );
    },
  );

  test("resumeChatCompletion writes decode checkpoints during resumed continuation", async () => {
    const { engine, pipeline } = createEngineWithPipeline(4);
    pipeline.enableDecodeCheckpoint = true;
    pipeline.checkpointPageSize = 1;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume checkpoint" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-checkpoint",
          checkpointPrompt: false,
          checkpointIntervalTokens: 1,
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    injectResumableFaultOnce("checkpoint.after_commit");
    await expect(
      engine.resumeChatCompletion("session-resume-checkpoint", {
        continueGeneration: true,
      }),
    ).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    const crashedJournal = await readSessionJournal(
      files,
      "session-resume-checkpoint",
    );
    expect(
      crashedJournal.records.some(
        (record) => record.type === JournalRecordType.CheckpointCommit,
      ),
    ).toBe(true);
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-resume-checkpoint",
        recoveryMode: "kv",
      }),
    ]);

    const result = (await engine.resumeChatCompletion(
      "session-resume-checkpoint",
      { continueGeneration: true },
    )) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-resume-checkpoint",
      recoveryMode: "kv",
      recoveredText: "user:Resume checkpoint|token1||token2||token3||token4|",
    });
  });

  test("resumable metrics report journal, checkpoint, and restore timing", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Metrics" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-metrics",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const writeMetrics = (engine as any).lastResumableMetrics;
    expect(writeMetrics.journalAppendMs).toBeGreaterThanOrEqual(0);
    expect(writeMetrics.checkpointWriteMs).toBeGreaterThanOrEqual(0);

    const result = (await engine.resumeChatCompletion("session-metrics", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result.recoveryMode).toBe("kv");
    const resumeMetrics = (engine as any).lastResumableMetrics;
    expect(resumeMetrics.kvRestoreMs).toBeGreaterThanOrEqual(0);
    expect(resumeMetrics.resumeFirstTokenMs).toBeGreaterThanOrEqual(0);
  });

  test("low storage quota skips KV checkpoints but keeps token journal", async () => {
    const restoreNavigator = setNavigatorStorageEstimate({
      quota: 128 * 1024 * 1024,
      usage: 120 * 1024 * 1024,
    });
    try {
      const { engine, pipeline } = createEngineWithPipeline(3);
      pipeline.enablePromptCheckpoint = true;
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const iterable = (await engine.chatCompletion({
        model: MODEL_ID,
        seed: 1234,
        messages: [{ role: "user", content: "Low quota" }],
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-low-quota",
          },
        },
      })) as AsyncIterable<ChatCompletionChunk>;
      const iterator = iterable[Symbol.asyncIterator]();

      await iterator.next();
      await engine.interruptGenerate();
      while (!(await iterator.next()).done) {
        // drain the generator so asyncGenerate records the abort and releases the lock
      }

      const journal = await readSessionJournal(files, "session-low-quota");
      expect(
        journal.records.some(
          (record) => record.type === JournalRecordType.GeneratedToken,
        ),
      ).toBe(true);
      await expect(
        files.list("resume-root/sessions/session-low-quota/kv"),
      ).resolves.toEqual([]);
      await expect(engine.listResumableSessions()).resolves.toEqual([
        expect.objectContaining({
          sessionId: "session-low-quota",
          recoveryMode: "token_replay",
        }),
      ]);
    } finally {
      restoreNavigator();
    }
  });

  test("finished resumable generation removes persisted KV", async () => {
    const { engine, pipeline } = createEngineWithPipeline(1);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);

    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Finish cleanup" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-finish-cleanup",
        },
      },
    });

    await expect(
      files.list("resume-root/sessions/session-finish-cleanup/kv"),
    ).resolves.toEqual([]);
    await expect(
      files.list("resume-root/sessions/session-finish-cleanup"),
    ).resolves.not.toContain("kv");
  });

  test("resumeChatCompletion restores from latest complete decode checkpoint", async () => {
    const { engine, pipeline } = createEngineWithPipeline(5);
    pipeline.enableDecodeCheckpoint = true;
    pipeline.checkpointPageSize = 2;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "KVdecode" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-decode-kv-resume",
          checkpointPrompt: false,
          checkpointIntervalTokens: 2,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await iterator.next();
    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const incompleteCheckpoint = "checkpoint_00000000_00000012";
    const incompletePath = `resume-root/sessions/session-decode-kv-resume/kv/${incompleteCheckpoint}`;
    await files.mkdir(incompletePath);
    await appendJournalRecord(
      files,
      "resume-root/sessions/session-decode-kv-resume/journal.bin",
      {
        type: JournalRecordType.CheckpointCommit,
        seqNo: 999,
        createdAtMs: Date.now(),
        payload: {
          checkpointId: incompleteCheckpoint,
          processedSeqLen: 12,
          path: incompletePath,
          layoutHash: "mock-layout",
        },
      },
    );
    await expect(
      files.list("resume-root/sessions/session-decode-kv-resume/kv"),
    ).resolves.toEqual(["checkpoint_00000000_00000010", incompleteCheckpoint]);

    const result = (await engine.resumeChatCompletion(
      "session-decode-kv-resume",
      { continueGeneration: true },
    )) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-decode-kv-resume",
      recoveryMode: "kv",
      replayedTokens: 1,
      recoveredText: "user:KVdecode|token1||token2||token3||token4||token5|",
    });
    expect(pipeline.restoredCheckpointSeqLen).toBe(10);
  });

  test("resumeChatCompletion skips a corrupt newer checkpoint and restores an older one", async () => {
    const { engine, pipeline } = createEngineWithPipeline(7);
    pipeline.enableDecodeCheckpoint = true;
    pipeline.checkpointPageSize = 2;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const sessionId = "session-corrupt-newest-checkpoint";
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "KVdecode" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId,
          checkpointPrompt: false,
          checkpointIntervalTokens: 2,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    for (let i = 0; i < 5; i++) {
      await iterator.next();
    }
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const kvRoot = `resume-root/sessions/${sessionId}/kv`;
    const checkpoints = await files.list(kvRoot);
    expect(checkpoints).toEqual([
      "checkpoint_00000000_00000010",
      "checkpoint_00000000_00000012",
    ]);
    const newestCheckpoint = checkpoints[1];
    const newestPath = `${kvRoot}/${newestCheckpoint}`;
    const pageGroupFile = (await files.list(newestPath)).find((name) =>
      name.endsWith(".wkv"),
    );
    expect(pageGroupFile).toBeDefined();
    const pageGroupPath = `${newestPath}/${pageGroupFile!}`;
    const pageGroupData = await files.read(pageGroupPath);
    expect(pageGroupData).toBeDefined();
    const corruptPageGroup = new Uint8Array(pageGroupData!);
    corruptPageGroup[0] ^= 1;
    await files.write(pageGroupPath, corruptPageGroup);
    const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);

    const result = (await engine.resumeChatCompletion(sessionId, {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId,
      recoveryMode: "kv",
      replayedTokens: 3,
      recoveredText:
        "user:KVdecode|token1||token2||token3||token4||token5||token6||token7|",
    });
    expect(pipeline.restoredCheckpointSeqLen).toBe(10);
    expect(warn).toHaveBeenCalledWith(
      `Ignoring invalid KV checkpoint ${newestCheckpoint}:`,
      expect.objectContaining({
        message: expect.stringContaining("CRC mismatch"),
      }),
    );
    warn.mockRestore();
  });

  test("resumeChatCompletion falls back to text-only when model is unavailable", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Model mismatch" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-model-mismatch",
        },
      },
    });

    const secondEngine = new MLCEngine();
    attachResumableStore(secondEngine, files);
    const result = (await secondEngine.resumeChatCompletion(
      "session-model-mismatch",
      { continueGeneration: true },
    )) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-model-mismatch",
      recoveryMode: "text_only",
      recoveredText: "user:Model mismatch|token1|",
      replayedTokens: 0,
    });
  });

  test("resumable generation rejects grammar before inference", async () => {
    const { engine, pipeline } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);

    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Grammar persist" }],
        response_format: { type: "json_object" },
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-grammar",
          },
        },
      }),
    ).rejects.toThrow(
      "Resumable generation does not support grammar-constrained response formats.",
    );
    expect(pipeline.prefillCallCount).toBe(0);
    await expect(engine.listResumableSessions()).resolves.toEqual([]);
  });

  test("resumable generation rejects custom logit processors before inference", async () => {
    const { engine, pipeline } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    engine.setLogitProcessorRegistry(
      new Map([
        [
          MODEL_ID,
          {
            processLogits: (logits: Float32Array) => logits,
            processSampledToken: () => undefined,
            resetState: () => undefined,
          },
        ],
      ]),
    );

    await expect(
      engine.chatCompletion({
        model: MODEL_ID,
        messages: [{ role: "user", content: "Processor persist" }],
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-logit-processor",
          },
        },
      }),
    ).rejects.toThrow(
      "Resumable generation does not support a custom LogitProcessor.",
    );
    expect(pipeline.prefillCallCount).toBe(0);
    await expect(engine.listResumableSessions()).resolves.toEqual([]);
  });

  test("listResumableSessions repairs a torn journal tail", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Corrupt persist" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-corrupt",
        },
      },
    });
    await files.append(
      "resume-root/sessions/session-corrupt/journal.bin",
      new Uint8Array([1, 2, 3]),
    );

    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-corrupt",
        resumable: false,
        recoveryMode: "none",
        reason: "generation finished",
      }),
    ]);
    expect(
      (
        await readJournalRecords(
          files,
          "resume-root/sessions/session-corrupt/journal.bin",
        )
      ).stoppedReason,
    ).toBeUndefined();
  });
});
