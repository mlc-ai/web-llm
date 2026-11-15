import { LLMChatPipeline } from "../../src/llm_chat.ts";
import { MLCEngine } from "../../src/engine.ts";
export { LLMChatPipeline };
export * as xgrammar from "@mlc-ai/web-xgrammar";
export {
  WebWorkerMLCEngine,
  WebWorkerMLCEngineHandler,
} from "../../src/web_worker.ts";
import { prebuiltAppConfig } from "../../src/config.ts";
import {
  appendJournalRecord,
  readJournalRecords,
  repairJournalTail,
  JournalRecordType,
} from "../../src/resumable/journal.ts";
import { BrowserOPFSFileStore } from "../../src/resumable/opfs_file_store.ts";
import { ResumableSessionStore } from "../../src/resumable/session_store.ts";
import { setResumableFaultHook } from "../../src/resumable/fault_injection.ts";

export {
  BrowserOPFSFileStore,
  ResumableSessionStore,
  appendJournalRecord,
  readJournalRecords,
};

const encoder = new globalThis.TextEncoder();
const decoder = new globalThis.TextDecoder();

export async function tryAcquireOPFSLock(path) {
  const store = new BrowserOPFSFileStore();
  const release = await store.tryLock(path);
  if (release === undefined) {
    return false;
  }
  release();
  return true;
}

export async function runOPFSRegression() {
  const prefix = `webllm-browser-test-${globalThis.crypto.randomUUID()}`;
  const store = new BrowserOPFSFileStore();
  try {
    await store.write(`${prefix}/data.txt`, encoder.encode("one"));
    await store.append(`${prefix}/data.txt`, encoder.encode("-two"));
    const text = decoder.decode(await store.read(`${prefix}/data.txt`));

    const lockPath = `${prefix}/session.lock`;
    const releaseMain = await store.lock(lockPath);
    const moduleUrl = new globalThis.URL(
      "/harness.js",
      globalThis.location.href,
    ).href;
    const worker = new globalThis.Worker(
      globalThis.URL.createObjectURL(
        new globalThis.Blob(
          [
            `import { tryAcquireOPFSLock } from ${JSON.stringify(moduleUrl)};\n` +
              `onmessage = async (event) => postMessage(await tryAcquireOPFSLock(event.data));`,
          ],
          { type: "text/javascript" },
        ),
      ),
      { type: "module" },
    );
    const askWorker = () =>
      new Promise((resolve, reject) => {
        worker.onmessage = (event) => resolve(event.data);
        worker.onerror = reject;
        worker.postMessage(lockPath);
      });
    const acquiredWhileHeld = await askWorker();
    releaseMain();
    const acquiredAfterRelease = await askWorker();
    worker.terminate();

    const journalPath = `${prefix}/journal.bin`;
    await appendJournalRecord(store, journalPath, {
      type: JournalRecordType.SessionBegin,
      seqNo: 1,
      createdAtMs: 1,
      payload: { sessionId: "browser", modelId: "model" },
    });
    await store.append(journalPath, new Uint8Array([1, 2, 3]));
    const beforeRepair = await readJournalRecords(store, journalPath);
    await repairJournalTail(store, journalPath);
    const afterRepair = await readJournalRecords(store, journalPath);

    return {
      text,
      webLocksAvailable: globalThis.navigator.locks !== undefined,
      acquiredWhileHeld,
      acquiredAfterRelease,
      stoppedBeforeRepair: beforeRepair.stoppedReason,
      stoppedAfterRepair: afterRepair.stoppedReason,
      repairedRecordCount: afterRepair.records.length,
    };
  } finally {
    await store.remove(prefix, { recursive: true });
  }
}

export async function runFirstTokenReplayRegression() {
  const pipeline = Object.create(LLMChatPipeline.prototype);
  let promptLogitsDisposed = false;
  let forwardedPrompt = [];
  pipeline.resetStatsPerPrefill = false;
  pipeline.config = {
    conv_template: {
      system_template: "{system_message}",
      system_message: "",
      roles: { user: "user", assistant: "assistant" },
      seps: ["\n"],
      stop_token_ids: [],
      stop_str: [],
    },
  };
  pipeline.resetChat = () => undefined;
  pipeline.setConversation = () => undefined;
  pipeline.resetGenerationRoundState = () => undefined;
  pipeline.prepareGrammarMatcherForSampling = async () => undefined;
  pipeline.device = { sync: async () => undefined };
  pipeline.outputIds = [];
  pipeline.forwardKnownTokens = async (tokens) => {
    forwardedPrompt = [...tokens];
    return {
      dispose: () => {
        promptLogitsDisposed = true;
      },
    };
  };
  pipeline.sampleFromRawLogits = async () => 17;
  pipeline.commitSampledStep = (sampled) => ({
    ...sampled,
    textDelta: "first",
    textPrefixLength: 0,
    outputMessage: "first",
    stopped: false,
  });

  const replay = await pipeline.replayGenerationTokens([1, 2, 3], [9], [], {
    temperature: 0,
  });
  return {
    forwardedPrompt,
    outputIds: pipeline.outputIds,
    promptLogitsDisposed,
    replayedTokens: replay.replayedTokens,
    sampledTokenId: replay.sampledToken?.tokenId,
    sampledTokenPosition: replay.sampledToken?.globalTokenPos,
    committedText: replay.committedToken?.outputMessage,
  };
}

export async function runKnownTokenForwardingRegression() {
  const pipeline = Object.create(LLMChatPipeline.prototype);
  const detached = [];
  let endedScopes = 0;
  let nextLogitsId = 0;
  pipeline.prefillChunkSize = 4;
  pipeline.filledKVCacheLength = 0;
  pipeline.prefillTotalTime = 0;
  pipeline.prefillTotalTokens = 0;
  pipeline.curRoundPrefillTotalTime = 0;
  pipeline.curRoundPrefillTotalTokens = 0;
  pipeline.tvm = {
    beginScope: () => undefined,
    endScope: () => {
      endedScopes += 1;
    },
    detachFromCurrentScope: (value) => {
      detached.push(value.id);
      return value;
    },
  };
  pipeline.embedAndForward = async (_chunk, chunkLength) => {
    pipeline.filledKVCacheLength += chunkLength;
    nextLogitsId += 1;
    return { id: nextLogitsId };
  };

  const finalLogits = await pipeline.forwardKnownTokens(
    Array.from({ length: 10 }, (_, index) => index),
    true,
  );
  return {
    forwardedTokens: pipeline.filledKVCacheLength,
    chunkCount: nextLogitsId,
    detached,
    finalLogitsId: finalLogits?.id,
    endedScopes,
  };
}

globalThis.webllmBrowserHarness = {
  MLCEngine,
  prebuiltAppConfig,
  BrowserOPFSFileStore,
  ResumableSessionStore,
  readJournalRecords,
  JournalRecordType,
  setResumableFaultHook,
  LLMChatPipeline,
  runOPFSRegression,
  runFirstTokenReplayRegression,
  runKnownTokenForwardingRegression,
};

// Only the tensor/forward boundary is simulated. Run the production prefill
// orchestration and real XGrammar WASM compiler, including asynchronous failure.
globalThis.webllmBrowserHarness.runPrefillFailureRegression = async (
  failure,
) => {
  const pipeline = Object.create(LLMChatPipeline.prototype);
  const scopes = [];
  const live = new Set();
  let allocated = 0;
  Object.assign(pipeline, {
    resetStatsPerPrefill: false,
    appearedTokensFreq: new Map(),
    imageDataCache: new Map(),
    conversation: { isTextCompletion: true },
    prefillChunkSize: 2,
    filledKVCacheLength: 0,
    fullVocabSize: 3,
    stopTokens: [0],
    token_postproc_method: "raw",
    prepend_space_in_encode: false,
    tokenizer: {
      getVocabSize: () => 3,
      idToToken: (id) => ["<eos>", "a", "b"][id],
    },
    device: { sync: async () => undefined },
    tvm: {
      beginScope: () => scopes.push(new Set()),
      detachFromCurrentScope: (tensor) => {
        scopes.at(-1).delete(tensor);
        return tensor;
      },
      endScope: () => {
        for (const tensor of scopes.pop()) tensor.dispose();
      },
    },
    getInputData: async () => [[Array(10).fill(1)], 10, () => 0],
    embedAndForward: async (_chunk, length) => {
      // Let grammar initialization reject while forward work remains pending.
      await new Promise((resolve) => globalThis.setTimeout(resolve, 10));
      pipeline.filledKVCacheLength += length;
      const tensor = { dispose: () => live.delete(tensor) };
      scopes.at(-1).add(tensor);
      live.add(tensor);
      allocated++;
      return tensor;
    },
    sampleFromRawLogits: async () => {
      throw new Error("sampling failed");
    },
  });
  let message;
  try {
    await pipeline.samplePrefillStep(
      "prompt",
      "user",
      undefined,
      failure === "grammar"
        ? { response_format: { type: "grammar", grammar: "not a grammar" } }
        : undefined,
    );
  } catch (err) {
    message = err.message;
  } finally {
    pipeline.grammarCompiler?.dispose();
    pipeline.xgTokenizerInfo?.dispose();
  }
  return { message, allocated, live: live.size, scopes: scopes.length };
};
