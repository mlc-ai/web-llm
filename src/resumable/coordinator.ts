import log from "loglevel";
import {
  ChatConfig,
  GenerationConfig,
  Role,
  postInitAndCheckGenerationConfigValues,
} from "../config";
import { getConversationFromChatCompletionRequest } from "../conversation";
import {
  StreamGenerationState,
  StreamContinuationOptions,
  onceAsync,
  managedAsyncIterable,
  lazyAsyncIterable,
  isAsyncIterable,
} from "../streaming";
import { CustomLock } from "../support";
import {
  CommittedGenerationStep,
  KVCheckpointData,
  KVCheckpointSnapshot,
  LLMChatPipeline,
  ReplayedGenerationToken,
  SampledGenerationStep,
  SamplePrefillOptions,
} from "../llm_chat";
import {
  ChatCompletionRequest,
  ChatCompletionRequestNonStreaming,
  ChatCompletionRequestStreaming,
  CompletionCreateParams,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  ChatCompletionChunk,
  Completion,
} from "../openai_api_protocols";
import {
  ResumeProbeResult,
  ResumeResult,
  ResumeChatCompletionOptions,
} from "../types";
import {
  ResumableCheckpointPayload,
  ResumableCheckpointWriter,
  readResumableCheckpointPayload,
  resolveCommittedCheckpointChains,
} from "./checkpoint_writer";
import { isResumableInjectedFault } from "./fault_injection";
import {
  NormalizedResumableGenerationConfig,
  ResumableGenerationJournal,
  normalizeResumableGenerationConfig,
} from "./generation";
import { OPFSFileStore } from "./opfs_file_store";
import {
  ResumableReplayState,
  hasUnsupportedGrammarReplay,
  readResumableReplayState,
} from "./replay";
import { probeReplayState } from "./session_probe";
import { ResumableSessionStore } from "./session_store";
import { ResumableSessionHandle } from "./types";

const MIN_FREE_SPACE_BEFORE_KV_CHECKPOINT_BYTES = 512 * 1024 * 1024;
const CHECKPOINT_RETENTION_COUNT = 2;

/** A captured checkpoint whose readback and write overlap later decoding. */
interface PendingKVCheckpoint {
  snapshot: KVCheckpointSnapshot;
  abort: AbortController;
  done: Promise<void>;
}

/** The committed checkpoint the next one extends with a delta. */
interface CheckpointChainTip {
  checkpointId: string;
  processedSeqLen: number;
}

interface DecodeCheckpointScheduler {
  intervalTokens: number;
  lastCheckpointSeqLen: number;
  pageSize?: number;
  /** Undefined until a checkpoint of this KV cache is committed or restored. */
  chainTip?: CheckpointChainTip;
  pending?: PendingKVCheckpoint;
  /** A strict-persistence checkpoint failure, rethrown on the decode path. */
  failure?: unknown;
}

interface ResumeContinuationState {
  recoveryMode: "kv" | "token_replay";
  replayedTokens: number;
  extraEmittedTokens: number;
  firstTokenRecorded: boolean;
  lastCheckpointSeqLen: number;
  checkpointPageSize?: number;
  /** The restored checkpoint, which later checkpoints extend. */
  chainTip?: CheckpointChainTip;
  pendingJournaledToken?: CommittedGenerationStep;
  /** Message offset preceding a newly sampled recovery token. */
  streamDeltaStart?: number;
}

export interface ResumableEngineMetrics {
  journalAppendMs: number;
  checkpointWriteMs: number;
  kvRestoreMs: number;
  tokenReplayMs: number;
  resumeFirstTokenMs?: number;
}

interface KVResumeCheckpoint {
  checkpoint: KVCheckpointData;
  chain: KVCheckpointData[];
  checkpointId: string;
  coveredGeneratedTokens: ReplayedGenerationToken[];
  tailGeneratedTokens: ReplayedGenerationToken[];
}

interface LockedResumableSession {
  files: OPFSFileStore;
  sessions: ResumableSessionStore;
  session: ResumableSessionHandle;
  state: ResumableReplayState;
  release: () => void;
}

export interface ResumableGenerationHost {
  getModel(modelId: string):
    | {
        modelId: string;
        pipeline: LLMChatPipeline;
        chatConfig: ChatConfig;
        lock: CustomLock;
      }
    | undefined;
  resetInterrupt(): void;
  isInterrupted(): boolean;
  samplePrefill(
    input: ChatCompletionRequest | CompletionCreateParams,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    options: SamplePrefillOptions & { reuseKVCache?: boolean },
  ): Promise<SampledGenerationStep>;
  decode(
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
  ): Promise<CommittedGenerationStep>;
  streamCurrentGeneration(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
    timeReceived: number,
    state: StreamGenerationState,
    decodeStep: () => Promise<void>,
    options: StreamContinuationOptions,
  ): AsyncGenerator<ChatCompletionChunk | Completion, void, void>;
}

export interface ResumableCoordinatorOptions {
  host: ResumableGenerationHost;
  getFileStore: () => OPFSFileStore;
  getSessionStore: () => ResumableSessionStore;
  hasCustomLogitProcessor: (modelId: string) => boolean;
  onMetricsChanged?: (metrics: ResumableEngineMetrics) => void;
}

function isOPFSUnavailableError(err: unknown): boolean {
  return (
    err instanceof Error &&
    err.message === "OPFS is unavailable in this environment"
  );
}

function promptCheckpointId(processedSeqLen: number): string {
  return `checkpoint_00000000_${processedSeqLen.toString().padStart(8, "0")}`;
}

function requestSeed(request: unknown): number | undefined {
  const seed = (request as { seed?: unknown } | undefined)?.seed;
  return typeof seed === "number" ? seed : undefined;
}

function hasMultimodalInput(request: ChatCompletionRequest): boolean {
  return request.messages.some((message) => {
    if (!Array.isArray(message.content)) {
      return false;
    }
    return message.content.some(
      (part) =>
        typeof part === "object" &&
        part !== null &&
        (part as { type?: unknown }).type !== "text",
    );
  });
}

async function estimateFreeStorageBytes(): Promise<number | undefined> {
  const storage = (
    globalThis.navigator as
      | {
          storage?: {
            estimate?: () => Promise<{ quota?: number; usage?: number }>;
          };
        }
      | undefined
  )?.storage;
  if (storage?.estimate === undefined) {
    return undefined;
  }
  try {
    const estimate = await storage.estimate();
    if (
      typeof estimate.quota !== "number" ||
      typeof estimate.usage !== "number"
    ) {
      return undefined;
    }
    return Math.max(0, estimate.quota - estimate.usage);
  } catch {
    return undefined;
  }
}

async function hasKVCheckpointQuota(
  estimatedCheckpointBytes = 0,
): Promise<boolean> {
  const freeBytes = await estimateFreeStorageBytes();
  if (freeBytes === undefined) {
    return true;
  }
  return (
    freeBytes >=
    Math.max(
      MIN_FREE_SPACE_BEFORE_KV_CHECKPOINT_BYTES,
      2 * estimatedCheckpointBytes,
    )
  );
}

function metadataInteger(
  metadata: Record<string, unknown>,
  field: string,
): number | undefined {
  const value = metadata[field];
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

function checkpointPageSize(
  checkpoint: Pick<KVCheckpointData, "metadata">,
): number | undefined {
  return (
    metadataInteger(checkpoint.metadata, "pageSize") ??
    metadataInteger(checkpoint.metadata, "page_size")
  );
}

function normalizeCheckpointInterval(
  intervalTokens: number,
  pageSize?: number,
): number {
  if (pageSize === undefined) {
    return intervalTokens;
  }
  return Math.ceil(intervalTokens / pageSize) * pageSize;
}

function shouldCaptureDecodeCheckpoint(
  scheduler: DecodeCheckpointScheduler,
  nextProcessedSeqLen: number,
): boolean {
  const intervalTokens = normalizeCheckpointInterval(
    scheduler.intervalTokens,
    scheduler.pageSize,
  );
  if (nextProcessedSeqLen - scheduler.lastCheckpointSeqLen < intervalTokens) {
    return false;
  }
  return (
    scheduler.pageSize === undefined ||
    nextProcessedSeqLen % scheduler.pageSize === 0
  );
}

function noteCommittedCheckpoint(
  scheduler: DecodeCheckpointScheduler,
  checkpointId: string,
  checkpoint: Pick<KVCheckpointData, "processedSeqLen" | "metadata">,
): void {
  scheduler.lastCheckpointSeqLen = checkpoint.processedSeqLen;
  scheduler.pageSize = checkpointPageSize(checkpoint) ?? scheduler.pageSize;
  scheduler.chainTip = {
    checkpointId,
    processedSeqLen: checkpoint.processedSeqLen,
  };
}

// The journal append for a committed token is awaited before the token is
// exposed. On the baseline readback path, starting the next forward pass
// first lets the GPU overlap it; the ring and lookahead paths already have.
function prefetchDecodeForward(
  journal: ResumableGenerationJournal,
  pipeline: LLMChatPipeline,
): void {
  if (journal.active && !pipeline.stopped()) {
    pipeline.prefetchDecodeForward();
  }
}

function createCheckpointScheduler(
  journal: ResumableGenerationJournal,
  prefillStep: SampledGenerationStep,
): DecodeCheckpointScheduler {
  return {
    intervalTokens: journal.checkpointIntervalTokens,
    lastCheckpointSeqLen: prefillStep.globalTokenPos,
  };
}

function throwCheckpointFailure(
  scheduler: DecodeCheckpointScheduler | undefined,
): void {
  if (scheduler?.failure !== undefined) {
    throw scheduler.failure;
  }
}

function replayPromptSeqLen(state: ResumableReplayState): number {
  if (state.emittedTokens === 0) {
    return state.promptTokenIds.length;
  }
  return Math.max(0, state.processedSeqLen - state.emittedTokens);
}

function isChatCompletionReplayRequest(
  request: unknown,
): request is ChatCompletionRequest {
  const messages = (request as { messages?: unknown } | undefined)?.messages;
  return (
    typeof request === "object" &&
    request !== null &&
    Array.isArray(messages) &&
    messages.length > 0
  );
}

export class ResumableGenerationCoordinator {
  private lastMetrics?: ResumableEngineMetrics;

  constructor(private readonly options: ResumableCoordinatorOptions) {}

  /**
   * Decode on the request's readback path and journal the committed token
   * before the caller exposes it. A checkpoint is taken at the committed
   * token's position, never past it: the ring may already have fed
   * speculative tokens that a stop will pop.
   */
  private async decodeGenerationStep(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
  ): Promise<void> {
    throwCheckpointFailure(scheduler);
    const committed = await this.options.host.decode(pipeline, genConfig);
    prefetchDecodeForward(journal, pipeline);
    await this.recordGeneratedToken(journal, pipeline, committed);
    if (
      journal.active &&
      !pipeline.stopped() &&
      scheduler.pending === undefined &&
      (await this.shouldCaptureDecodeCheckpoint(
        scheduler,
        committed.globalTokenPos,
      ))
    ) {
      this.startKVCheckpoint(
        journal,
        scheduler,
        pipeline.captureCheckpoint(
          committed.globalTokenPos,
          scheduler.chainTip?.processedSeqLen,
        ),
      );
    }
  }

  private finalizeResumedConversation(
    state: ResumableReplayState,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
  ): void {
    if (!isChatCompletionReplayRequest(state.request)) {
      return;
    }
    const conversation = getConversationFromChatCompletionRequest(
      state.request,
      chatConfig,
      true,
    );
    conversation.appendReplyHeader(Role.assistant);
    conversation.finishReply(pipeline.getMessage());
    pipeline.setConversation(conversation);
  }

  private async finishContinuation(
    state: ResumableReplayState,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    restored: ResumeContinuationState,
    journal: ResumableGenerationJournal,
    checkpointScheduler: DecodeCheckpointScheduler,
    promptSeqLen: number,
  ): Promise<ResumeResult> {
    this.options.host.resetInterrupt();
    const firstTokenStart = performance.now();
    let recordedFirstToken = restored.firstTokenRecorded;
    try {
      await this.recordPendingResumeToken(journal, pipeline, restored);
      while (!pipeline.stopped()) {
        if (this.options.host.isInterrupted()) {
          pipeline.triggerStop();
          break;
        }
        await this.decodeGenerationStep(
          journal,
          checkpointScheduler,
          pipeline,
          genConfig,
        );
        if (!recordedFirstToken) {
          this.metrics().resumeFirstTokenMs =
            performance.now() - firstTokenStart;
          recordedFirstToken = true;
        }
      }
      await this.finishJournal(
        journal,
        checkpointScheduler,
        pipeline.getFinishReason(),
      );
      this.finalizeResumedConversation(state, pipeline, chatConfig);
      return {
        sessionId: state.sessionId,
        recoveredText: pipeline.getMessage(),
        emittedTokens: journal.emittedTokenCount,
        processedSeqLen: promptSeqLen + journal.emittedTokenCount,
        replayedTokens: restored.replayedTokens,
        recoveryMode: restored.recoveryMode,
      };
    } catch (err) {
      await this.recordEngineError(journal, checkpointScheduler, err);
      throw err;
    } finally {
      await this.closeJournal(journal, checkpointScheduler);
    }
  }

  private async *streamContinuation(
    request: ChatCompletionRequestStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    state: ResumableReplayState,
    restored: ResumeContinuationState,
    journal: ResumableGenerationJournal,
    checkpointScheduler: DecodeCheckpointScheduler,
    promptSeqLen: number,
    release: () => Promise<void>,
  ): AsyncGenerator<ChatCompletionChunk, void, void> {
    this.options.host.resetInterrupt();
    const firstTokenStart = performance.now();
    let recordedFirstToken = restored.firstTokenRecorded;
    let journalEnded = false;
    let failure: unknown;
    const streamState: StreamGenerationState = {
      id: crypto.randomUUID(),
      created: Math.floor(Date.now() / 1000),
      prevMessageLength:
        restored.streamDeltaStart ?? pipeline.getMessage().length,
      prevLogprobLength: Math.max(
        0,
        pipeline.getTokenLogprobArray().length - restored.extraEmittedTokens,
      ),
    };

    try {
      await this.recordPendingResumeToken(journal, pipeline, restored);
      const chunks = this.options.host.streamCurrentGeneration(
        request,
        model,
        pipeline,
        genConfig,
        Date.now(),
        streamState,
        async () => {
          await this.decodeGenerationStep(
            journal,
            checkpointScheduler,
            pipeline,
            genConfig,
          );
          if (!recordedFirstToken) {
            this.metrics().resumeFirstTokenMs =
              performance.now() - firstTokenStart;
            recordedFirstToken = true;
          }
        },
        {
          emitCurrent: restored.extraEmittedTokens > 0,
          skipEmptyDelta: true,
          completionTokenOffset: restored.extraEmittedTokens,
          beforeFinalChunk: async () => {
            await this.finishJournal(
              journal,
              checkpointScheduler,
              pipeline.getFinishReason(),
            );
            journalEnded = true;
            this.finalizeResumedConversation(state, pipeline, chatConfig);
          },
        },
      );
      for await (const chunk of chunks) {
        yield chunk as ChatCompletionChunk;
      }
    } catch (err) {
      failure = err;
      await this.recordEngineError(journal, checkpointScheduler, err);
      throw err;
    } finally {
      try {
        await this.closeStreamJournal(
          journal,
          checkpointScheduler,
          pipeline,
          !journalEnded,
          failure,
        );
      } finally {
        try {
          if (request.seed != null) pipeline.setSeed(Date.now());
        } finally {
          await release();
        }
      }
    }
  }

  // Keep the journal-begin boundary explicit: errors after it belong to this
  // generation, including failures while writing the first checkpoint/token.
  private async prefillGeneration(
    input: ChatCompletionRequest | CompletionCreateParams,
    modelId: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    journal: ResumableGenerationJournal,
  ): Promise<SampledGenerationStep> {
    const prefillStep = await this.options.host.samplePrefill(
      input,
      pipeline,
      chatConfig,
      genConfig,
      {
        capturePromptCheckpoint:
          await this.shouldCapturePromptCheckpoint(journal),
        storeCheckpointLogits: journal.storeCheckpointLogits,
        reuseKVCache: false,
      },
    );
    try {
      if (prefillStep.promptTokenIds === undefined) {
        throw new Error(
          "Resumable generation currently supports text-only prompts.",
        );
      }
      await journal.begin({
        modelId,
        request: input,
        promptTokenIds: prefillStep.promptTokenIds,
        assistantPrefixTokenIds: prefillStep.assistantPrefixTokenIds ?? [],
        generationConfig: genConfig,
      });
    } catch (err) {
      // Free the prompt snapshot's readback buffers now, not at the next reset.
      prefillStep.promptCheckpoint?.discard();
      throw err;
    }
    return prefillStep;
  }

  private async commitPrefill(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
    prefillStep: SampledGenerationStep,
  ): Promise<void> {
    const committedPrefill = pipeline.commitSampledStep(prefillStep, genConfig);
    prefetchDecodeForward(journal, pipeline);
    try {
      await this.recordGeneratedToken(journal, pipeline, committedPrefill);
    } catch (err) {
      prefillStep.promptCheckpoint?.discard();
      throw err;
    }
    // Read the prompt checkpoint back once the first token is durable, so its
    // copies stay out of the time to first token.
    this.startKVCheckpoint(journal, scheduler, prefillStep.promptCheckpoint);
  }

  async *streamGeneration(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
    streamState: StreamGenerationState,
    journal: ResumableGenerationJournal,
  ): AsyncGenerator<ChatCompletionChunk | Completion, void, void> {
    this.resetMetrics();
    let journalStarted = false;
    let journalEnded = false;
    let failure: unknown;
    let scheduler: DecodeCheckpointScheduler | undefined;
    try {
      const prefillStep = await this.prefillGeneration(
        request,
        model,
        pipeline,
        chatConfig,
        genConfig,
        journal,
      );
      journalStarted = journal.active;
      const checkpoints = createCheckpointScheduler(journal, prefillStep);
      scheduler = checkpoints;
      await this.commitPrefill(
        journal,
        checkpoints,
        pipeline,
        genConfig,
        prefillStep,
      );
      yield* this.options.host.streamCurrentGeneration(
        request,
        model,
        pipeline,
        genConfig,
        timeReceived,
        streamState,
        () =>
          this.decodeGenerationStep(journal, checkpoints, pipeline, genConfig),
        {
          emitCurrent: true,
          beforeFinalChunk: async () => {
            await this.finishJournal(
              journal,
              scheduler,
              pipeline.getFinishReason(),
            );
            journalEnded = true;
          },
        },
      );
    } catch (err) {
      failure = err;
      if (journalStarted) await this.recordEngineError(journal, scheduler, err);
      throw err;
    } finally {
      await this.closeStreamJournal(
        journal,
        scheduler,
        pipeline,
        journalStarted && !journalEnded,
        failure,
      );
    }
  }

  async generate(
    input:
      | ChatCompletionRequestNonStreaming
      | CompletionCreateParamsNonStreaming,
    modelId: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    journal: ResumableGenerationJournal,
  ): Promise<string> {
    this.resetMetrics();
    this.options.host.resetInterrupt();
    if (genConfig !== undefined) {
      postInitAndCheckGenerationConfigValues(genConfig);
    }

    let journalStarted = false;
    let scheduler: DecodeCheckpointScheduler | undefined;
    try {
      const prefillStep = await this.prefillGeneration(
        input,
        modelId,
        pipeline,
        chatConfig,
        genConfig,
        journal,
      );
      journalStarted = journal.active;
      scheduler = createCheckpointScheduler(journal, prefillStep);
      await this.commitPrefill(
        journal,
        scheduler,
        pipeline,
        genConfig,
        prefillStep,
      );

      while (!pipeline.stopped()) {
        if (this.options.host.isInterrupted()) {
          pipeline.triggerStop();
          break;
        }
        await this.decodeGenerationStep(
          journal,
          scheduler,
          pipeline,
          genConfig,
        );
      }
      await this.finishJournal(journal, scheduler, pipeline.getFinishReason());
      return pipeline.getMessage();
    } catch (err) {
      if (journalStarted) {
        await this.recordEngineError(journal, scheduler, err);
      }
      throw err;
    } finally {
      await this.closeJournal(journal, scheduler);
    }
  }

  normalizeForRequest(
    request: ChatCompletionRequest,
    modelId: string,
  ): NormalizedResumableGenerationConfig | undefined {
    const config = normalizeResumableGenerationConfig(
      request.extra_body?.resumable,
    );
    if (config === undefined) {
      return undefined;
    }
    if ((request.n ?? 1) > 1) {
      throw new Error("Resumable generation currently requires n <= 1.");
    }
    if (
      hasUnsupportedGrammarReplay({
        response_format: request.response_format,
      })
    ) {
      throw new Error(
        "Resumable generation does not support grammar-constrained response formats.",
      );
    }
    if (this.options.hasCustomLogitProcessor(modelId)) {
      throw new Error(
        "Resumable generation does not support a custom LogitProcessor.",
      );
    }
    if (hasMultimodalInput(request)) {
      throw new Error("Resumable generation supports text-only prompts.");
    }
    return config;
  }

  private resetMetrics(): ResumableEngineMetrics {
    const metrics = {
      journalAppendMs: 0,
      checkpointWriteMs: 0,
      kvRestoreMs: 0,
      tokenReplayMs: 0,
    };
    this.lastMetrics = metrics;
    this.options.onMetricsChanged?.(metrics);
    return metrics;
  }

  private metrics(): ResumableEngineMetrics {
    return this.lastMetrics ?? this.resetMetrics();
  }

  tryCreateJournal(
    config: NormalizedResumableGenerationConfig | undefined,
  ): ResumableGenerationJournal | undefined {
    if (config === undefined) {
      return undefined;
    }
    try {
      return new ResumableGenerationJournal(
        this.options.getFileStore(),
        this.options.getSessionStore(),
        config,
      );
    } catch (err) {
      if (config.strictPersistence) {
        throw err;
      }
      log.warn("Resumable journal disabled:", err);
      return undefined;
    }
  }

  private async recordGeneratedToken(
    journal: ResumableGenerationJournal | undefined,
    pipeline: LLMChatPipeline,
    committed: CommittedGenerationStep,
  ): Promise<void> {
    if (journal === undefined) {
      return;
    }
    const start = performance.now();
    try {
      await journal.recordGeneratedToken({
        globalTokenPos: committed.globalTokenPos,
        tokenId: committed.tokenId,
        textDelta: committed.textDelta,
        textPrefixLength: committed.textPrefixLength,
        rngState: pipeline.getCommittedRNGState(),
        logprob: committed.logprob,
      });
    } finally {
      this.metrics().journalAppendMs += performance.now() - start;
    }
  }

  private async shouldCaptureDecodeCheckpoint(
    scheduler: DecodeCheckpointScheduler,
    nextProcessedSeqLen: number,
  ): Promise<boolean> {
    return (
      shouldCaptureDecodeCheckpoint(scheduler, nextProcessedSeqLen) &&
      (await hasKVCheckpointQuota())
    );
  }

  private async shouldCapturePromptCheckpoint(
    journal: ResumableGenerationJournal,
  ): Promise<boolean> {
    return journal.checkpointPrompt && (await hasKVCheckpointQuota());
  }

  private async finishJournal(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler | undefined,
    finishReason: string | undefined,
  ): Promise<void> {
    await this.settleKVCheckpoint(scheduler, finishReason === "abort");
    if (finishReason !== "abort") throwCheckpointFailure(scheduler);
    await journal.recordGenerationEnd({
      finishReason,
      emittedTokens: journal.emittedTokenCount,
    });
    if (journal.active && finishReason !== "abort") {
      await this.cleanupKVAfterFinish(journal);
    }
  }

  private async recordEngineError(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler | undefined,
    err: unknown,
  ): Promise<void> {
    await this.settleKVCheckpoint(scheduler);
    if (!isResumableInjectedFault(err)) {
      await journal.recordEngineError({ err }).catch(() => undefined);
    }
  }

  private async closeJournal(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler | undefined,
  ): Promise<void> {
    await this.settleKVCheckpoint(scheduler);
    await journal.close();
  }

  private async closeStreamJournal(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler | undefined,
    pipeline: LLMChatPipeline,
    unfinished: boolean,
    failure: unknown,
  ): Promise<void> {
    try {
      try {
        if (unfinished && !isResumableInjectedFault(failure)) {
          pipeline.triggerStop();
          await this.finishJournal(journal, scheduler, "abort");
        }
      } finally {
        await this.closeJournal(journal, scheduler);
      }
    } catch (err) {
      // Preserve an existing generation error. Otherwise, let cancellation
      // report strict persistence failures to its caller.
      if (failure === undefined) throw err;
    }
  }

  private createResumeCheckpointScheduler(
    journal: ResumableGenerationJournal,
    restored: ResumeContinuationState,
  ): DecodeCheckpointScheduler {
    return {
      intervalTokens: journal.checkpointIntervalTokens,
      lastCheckpointSeqLen: restored.lastCheckpointSeqLen,
      pageSize: restored.checkpointPageSize,
      chainTip: restored.chainTip,
    };
  }

  private async createResumeContinuationJournal(
    locked: LockedResumableSession,
  ): Promise<ResumableGenerationJournal> {
    const { files, sessions, session, state } = locked;
    if (state.resumableConfig === undefined) {
      throw new Error(
        `Resumable session ${state.sessionId} is missing journal configuration.`,
      );
    }
    const journal = new ResumableGenerationJournal(
      files,
      sessions,
      state.resumableConfig,
    );
    await journal.attachExistingSession(session, state.emittedTokens);
    return journal;
  }

  private async recordPendingResumeToken(
    journal: ResumableGenerationJournal,
    pipeline: LLMChatPipeline,
    restored: ResumeContinuationState,
  ): Promise<void> {
    if (restored.pendingJournaledToken === undefined) {
      return;
    }
    await this.recordGeneratedToken(
      journal,
      pipeline,
      restored.pendingJournaledToken,
    );
    restored.pendingJournaledToken = undefined;
  }

  async resume(
    sessionId: string,
    options?: ResumeChatCompletionOptions,
  ): Promise<ResumeResult | AsyncIterable<ChatCompletionChunk>> {
    if (options?.continueGeneration !== true) {
      return this.readTextOnlyResult(sessionId);
    }

    if (options.stream === true) {
      const replayState = await this.readReplayState(sessionId);
      if (
        this.canAttemptContinuation(replayState) &&
        isChatCompletionReplayRequest(replayState.request) &&
        replayState.request.stream === true
      ) {
        if (this.options.host.getModel(replayState.modelId) === undefined) {
          return this.makeTextOnlyResumeResult(replayState);
        }
        return lazyAsyncIterable(async () => {
          const resumed = await this.resumeEager(sessionId, options);
          if (!isAsyncIterable<ChatCompletionChunk>(resumed)) {
            throw new Error(
              `Resumable session ${sessionId} became unavailable for continued streaming.`,
            );
          }
          return resumed;
        });
      }
    }

    return this.resumeEager(sessionId, options);
  }

  private async resumeEager(
    sessionId: string,
    options: ResumeChatCompletionOptions | undefined,
  ): Promise<ResumeResult | AsyncIterable<ChatCompletionChunk>> {
    const locked = await this.openLockedSession(sessionId);
    let releaseSessionLockOnExit = true;
    try {
      this.resetMetrics();
      const replayState = locked.state;
      if (replayState.resumableConfig === undefined) {
        throw new Error(
          `Resumable session ${sessionId} is missing or has malformed resumable generation config.`,
        );
      }

      const model = this.options.host.getModel(replayState.modelId);
      if (model === undefined)
        return this.makeTextOnlyResumeResult(replayState);
      const {
        modelId: selectedModelId,
        pipeline: selectedPipeline,
        chatConfig: selectedChatConfig,
        lock,
      } = model;
      await lock.acquire();
      let releaseModelLockOnExit = true;
      try {
        if (replayState.generationConfig === undefined) {
          return this.makeTextOnlyResumeResult(replayState);
        }
        const genConfig = replayState.generationConfig;
        postInitAndCheckGenerationConfigValues(genConfig);
        let restored = await this.tryRestoreKVResumeState(
          locked,
          selectedPipeline,
          genConfig,
        );
        if (restored === undefined) {
          restored = await this.restoreByTokenReplay(
            replayState,
            selectedPipeline,
            genConfig,
          );
        }
        if (restored === undefined) {
          return this.makeTextOnlyResumeResult(replayState);
        }

        const resumeJournal =
          await this.createResumeContinuationJournal(locked);
        const checkpointScheduler = this.createResumeCheckpointScheduler(
          resumeJournal,
          restored,
        );
        const promptSeqLen = replayPromptSeqLen(replayState);

        if (
          options?.stream === true &&
          isChatCompletionReplayRequest(replayState.request) &&
          replayState.request.stream === true
        ) {
          releaseModelLockOnExit = false;
          releaseSessionLockOnExit = false;
          const cleanup = onceAsync(async () => {
            try {
              await resumeJournal.close();
            } finally {
              try {
                await lock.release();
              } finally {
                locked.release();
              }
            }
          });
          const source = this.streamContinuation(
            replayState.request,
            selectedModelId,
            selectedPipeline,
            selectedChatConfig,
            genConfig,
            replayState,
            restored,
            resumeJournal,
            checkpointScheduler,
            promptSeqLen,
            cleanup,
          );
          return managedAsyncIterable(source, cleanup);
        }

        return await this.finishContinuation(
          replayState,
          selectedPipeline,
          selectedChatConfig,
          genConfig,
          restored,
          resumeJournal,
          checkpointScheduler,
          promptSeqLen,
        );
      } finally {
        if (releaseModelLockOnExit) {
          await lock.release();
        }
      }
    } finally {
      if (releaseSessionLockOnExit) {
        locked.release();
      }
    }
  }

  private async openLockedSession(
    sessionId: string,
  ): Promise<LockedResumableSession> {
    const files = this.options.getFileStore();
    const sessions = this.options.getSessionStore();
    const session = await sessions.openSession(sessionId);
    if (session === undefined) {
      throw new Error(`Resumable session not found: ${sessionId}`);
    }
    const release = await files.tryLock(session.paths.lockPath);
    if (release === undefined) {
      throw new Error(`Resumable session is already active: ${sessionId}`);
    }
    try {
      const state = await sessions.repairSession(session);
      return { files, sessions, session, state, release };
    } catch (err) {
      release();
      throw err;
    }
  }

  private async readReplayState(
    sessionId: string,
  ): Promise<ResumableReplayState> {
    const files = this.options.getFileStore();
    const session = await this.options.getSessionStore().openSession(sessionId);
    if (session === undefined) {
      throw new Error(`Resumable session not found: ${sessionId}`);
    }
    return readResumableReplayState(files, session);
  }

  private canAttemptContinuation(state: ResumableReplayState): boolean {
    return (
      state.resumableConfig !== undefined &&
      // The read-only stream probe sees an un-repaired tail. Actual recovery
      // truncates it under the session lock, acquired only on first next().
      this.getContinuationBlockReason({
        ...state,
        scanStoppedReason: undefined,
      }) === undefined
    );
  }

  private async readTextOnlyResult(sessionId: string): Promise<ResumeResult> {
    const sessions = this.options.getSessionStore();
    const session = await sessions.openSession(sessionId);
    if (session === undefined) {
      throw new Error(`Resumable session not found: ${sessionId}`);
    }
    return this.makeTextOnlyResumeResult(
      await sessions.inspectSession(session),
    );
  }

  async listSessions(): Promise<ResumeProbeResult[]> {
    let files: OPFSFileStore;
    let sessions: ResumableSessionStore;
    try {
      files = this.options.getFileStore();
      sessions = this.options.getSessionStore();
    } catch (err) {
      if (isOPFSUnavailableError(err)) {
        return [];
      }
      throw err;
    }

    let handles: ResumableSessionHandle[];
    try {
      handles = await sessions.listSessions();
    } catch (err) {
      if (isOPFSUnavailableError(err)) {
        return [];
      }
      throw err;
    }

    const results: ResumeProbeResult[] = [];
    for (const session of handles) {
      try {
        const state = await sessions.inspectSession(session);
        let result = probeReplayState(state);
        if (
          result.resumable &&
          result.recoveryMode === "token_replay" &&
          (await this.readBestKVCheckpoint(files, sessions, session, state)) !==
            undefined
        ) {
          result = { ...result, recoveryMode: "kv" };
        }
        if (
          result.resumable &&
          this.options.hasCustomLogitProcessor(result.modelId)
        ) {
          result = {
            ...result,
            resumable: false,
            reason: "custom LogitProcessor replay unsupported",
            recoveryMode: result.emittedTokens > 0 ? "text_only" : "none",
          };
        }
        results.push(result);
      } catch (err) {
        results.push({
          sessionId: session.sessionId,
          resumable: false,
          reason: `session inspection failed: ${
            err instanceof Error ? err.message : String(err)
          }`,
          modelId: session.manifest?.modelId ?? "",
          emittedTokens: 0,
          processedSeqLen: 0,
          recoveryMode: "none",
        });
      }
    }
    return results;
  }

  async deleteSession(sessionId: string): Promise<void> {
    try {
      await this.options.getSessionStore().deleteSession(sessionId);
    } catch (err) {
      if (!isOPFSUnavailableError(err)) {
        throw err;
      }
    }
  }

  private async tryRestoreKVResumeState(
    locked: LockedResumableSession,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
  ): Promise<ResumeContinuationState | undefined> {
    const { files, sessions, session, state } = locked;
    if (this.getContinuationBlockReason(state) !== undefined) {
      return undefined;
    }

    try {
      const resume = await this.readBestKVCheckpoint(
        files,
        sessions,
        session,
        state,
      );
      if (resume === undefined) {
        return undefined;
      }
      if (resume.tailGeneratedTokens.length === 0) {
        const seed = requestSeed(state.request);
        const rngState = resume.coveredGeneratedTokens.at(-1)?.rngState;
        if (rngState !== undefined) {
          if (!pipeline.setRNGState(rngState)) {
            return undefined;
          }
        } else if (seed !== undefined) {
          pipeline.setSeed(seed);
        } else if (state.generatedTokens.length > 0) {
          return undefined;
        }
      }
      const restoreStart = performance.now();
      const replay = await pipeline.replayFromPromptCheckpoint(
        resume.chain,
        state.assistantPrefixTokenIds,
        resume.coveredGeneratedTokens,
        resume.tailGeneratedTokens,
        genConfig,
      );
      const restoreElapsed = performance.now() - restoreStart;
      this.metrics().kvRestoreMs += restoreElapsed;
      if (replay.sampledToken !== undefined) {
        this.metrics().resumeFirstTokenMs = restoreElapsed;
      }
      const rngState = resume.tailGeneratedTokens.at(-1)?.rngState;
      if (rngState !== undefined && !pipeline.setRNGState(rngState)) {
        pipeline.resetChat();
        return undefined;
      }

      return {
        recoveryMode: "kv",
        replayedTokens: replay.replayedTokens,
        extraEmittedTokens: replay.sampledToken === undefined ? 0 : 1,
        firstTokenRecorded: replay.sampledToken !== undefined,
        lastCheckpointSeqLen: resume.checkpoint.processedSeqLen,
        checkpointPageSize: checkpointPageSize(resume.checkpoint),
        chainTip: {
          checkpointId: resume.checkpointId,
          processedSeqLen: resume.checkpoint.processedSeqLen,
        },
        streamDeltaStart: replay.committedToken?.textPrefixLength,
        pendingJournaledToken:
          replay.sampledToken !== undefined ? replay.committedToken : undefined,
      };
    } catch (err) {
      log.warn("KV checkpoint restore failed; falling back:", err);
      pipeline.resetChat();
      return undefined;
    }
  }

  private async restoreByTokenReplay(
    state: ResumableReplayState,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
  ): Promise<ResumeContinuationState | undefined> {
    const blockReason = this.getContinuationBlockReason(state);
    if (blockReason !== undefined) {
      log.warn(
        `Resumable session ${state.sessionId} recovered as text-only: ${blockReason}.`,
      );
      return undefined;
    }
    if (state.generatedTokens.length === 0) {
      const seed = requestSeed(state.request);
      if (seed !== undefined) {
        pipeline.setSeed(seed);
      }
    }
    const replayStart = performance.now();
    const replay = (await pipeline.replayGenerationTokens(
      state.promptTokenIds,
      state.assistantPrefixTokenIds,
      state.generatedTokens,
      genConfig,
    )) ?? {
      // Compatibility with pipeline-like integrations compiled against the
      // previous void return; the production pipeline always returns detail.
      replayedTokens: state.generatedTokens.length,
      sampledFromCheckpointLogits: false,
    };
    const replayElapsed = performance.now() - replayStart;
    this.metrics().tokenReplayMs += replayElapsed;
    if (replay.sampledToken !== undefined) {
      this.metrics().resumeFirstTokenMs = replayElapsed;
    }
    if (state.generatedTokens.length > 0) {
      const rngState = state.generatedTokens.at(-1)?.rngState;
      if (!pipeline.setRNGState(rngState)) {
        pipeline.resetChat();
        return undefined;
      }
    }
    return {
      recoveryMode: "token_replay",
      replayedTokens: replay.replayedTokens,
      extraEmittedTokens: replay.sampledToken === undefined ? 0 : 1,
      firstTokenRecorded: replay.sampledToken !== undefined,
      lastCheckpointSeqLen: replayPromptSeqLen(state),
      streamDeltaStart: replay.committedToken?.textPrefixLength,
      pendingJournaledToken:
        replay.sampledToken !== undefined ? replay.committedToken : undefined,
    };
  }

  private makeTextOnlyResumeResult(state: ResumableReplayState): ResumeResult {
    return {
      sessionId: state.sessionId,
      recoveredText: state.recoveredText,
      emittedTokens: state.emittedTokens,
      processedSeqLen: state.processedSeqLen,
      replayedTokens: 0,
      recoveryMode: "text_only",
    };
  }

  /** Read back and write a captured checkpoint while decoding continues. */
  private startKVCheckpoint(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    snapshot: KVCheckpointSnapshot | undefined,
  ): void {
    if (snapshot === undefined) {
      return;
    }
    if (!journal.active || scheduler.pending !== undefined) {
      snapshot.discard();
      return;
    }
    const abort = new AbortController();
    const parent = snapshot.pageBegin > 0 ? scheduler.chainTip : undefined;
    const done = this.writeKVCheckpoint(
      journal,
      scheduler,
      snapshot,
      parent,
      abort.signal,
    ).finally(() => {
      scheduler.pending = undefined;
    });
    scheduler.pending = { snapshot, abort, done };
  }

  /**
   * End an unfinished checkpoint with its generation. An aborted generation
   * stays resumable, so its checkpoint is completed and committed first.
   * Otherwise it is stopped: a finished generation deletes its KV directory,
   * and an error ends the generation at once. A stopped checkpoint's
   * directory was never committed and is removed; earlier commits stay usable.
   */
  private async settleKVCheckpoint(
    scheduler: DecodeCheckpointScheduler | undefined,
    complete = false,
  ): Promise<void> {
    const pending = scheduler?.pending;
    if (pending === undefined) {
      return;
    }
    if (!complete) {
      pending.abort.abort();
      pending.snapshot.discard();
    }
    await pending.done;
  }

  // Never rejects: a strict-persistence failure is stored on the scheduler and
  // rethrown by the next decode step or by the end of the generation.
  private async writeKVCheckpoint(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    snapshot: KVCheckpointSnapshot,
    parent: CheckpointChainTip | undefined,
    signal: AbortSignal,
  ): Promise<void> {
    if (
      !(await hasKVCheckpointQuota(snapshot.byteLength)) ||
      signal.aborted ||
      !journal.active
    ) {
      snapshot.discard();
      return;
    }
    const sessions = this.options.getSessionStore();
    const start = performance.now();
    // Batches are written as the readback delivers them; the readback waits
    // for each write, so a checkpoint is never buffered whole in memory.
    const stream = new ResumableCheckpointWriter(
      this.options.getFileStore(),
      sessions,
    ).beginCheckpoint(
      {
        sessionId: journal.sessionId,
        checkpointId: promptCheckpointId(snapshot.processedSeqLen),
        processedSeqLen: snapshot.processedSeqLen,
        layoutHash: snapshot.layoutHash,
        parentCheckpointId: parent?.checkpointId,
        pageBegin: snapshot.pageBegin,
        pageGroups: snapshot.pageGroups,
      },
      signal,
    );
    stream.catch(() => undefined);
    try {
      snapshot.start(async (group, data) => (await stream).write(group, data));
      const done = await snapshot.done;
      if (done === undefined || signal.aborted || !journal.active) {
        await (await stream).abort();
        await sessions.reconcileCheckpoints(journal.sessionId);
        return;
      }
      const ref = await (
        await stream
      ).finish(done.nextLogits?.data, {
        ...snapshot.metadata,
        nextLogitsShape: done.nextLogits?.shape,
        nextLogitsDtype: done.nextLogits?.dtype,
      });
      await journal.recordCheckpointCommit({
        checkpointId: ref.checkpointId,
        processedSeqLen: snapshot.processedSeqLen,
        path: ref.path,
        layoutHash: snapshot.layoutHash,
      });
      if (!journal.active) {
        await sessions.reconcileCheckpoints(journal.sessionId);
        return;
      }
      await sessions.pruneCommittedCheckpoints(
        journal.sessionId,
        CHECKPOINT_RETENTION_COUNT,
      );
      noteCommittedCheckpoint(scheduler, ref.checkpointId, snapshot);
    } catch (err) {
      await stream.then((open) => open.abort()).catch(() => undefined);
      await sessions.reconcileCheckpoints(journal.sessionId).catch(() => []);
      if (!signal.aborted) {
        this.noteCheckpointFailure(journal, scheduler, err);
      }
    } finally {
      this.metrics().checkpointWriteMs += performance.now() - start;
    }
  }

  private noteCheckpointFailure(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    err: unknown,
  ): void {
    if (journal.strictPersistence) {
      scheduler.failure ??= err;
    } else {
      log.warn("KV checkpoint disabled:", err);
    }
  }

  private async cleanupKVAfterFinish(
    journal: ResumableGenerationJournal,
  ): Promise<void> {
    try {
      await this.options.getSessionStore().deleteKV(journal.sessionId);
    } catch (err) {
      // The finished record is durable already. Housekeeping must not turn a
      // successfully persisted response into a failed, non-resumable request.
      log.warn("KV checkpoint cleanup failed:", err);
    }
  }

  private checkpointPayloadToKVData(
    payload: ResumableCheckpointPayload,
  ): KVCheckpointData {
    const metadata = payload.meta.metadata;
    if (metadata === undefined) {
      throw new Error("KV checkpoint metadata is missing runtime metadata.");
    }
    const nextLogitsShape = metadata.nextLogitsShape;
    const nextLogitsDtype = metadata.nextLogitsDtype;
    return {
      processedSeqLen: payload.meta.processedSeqLen,
      layoutHash: payload.meta.layoutHash,
      metadata,
      pageBegin: payload.meta.pageBegin ?? 0,
      pageGroups: payload.pageGroups.map((group) => ({
        groupId: group.groupId,
        layerStart: group.layerStart,
        layerEnd: group.layerEnd,
        data: group.data,
      })),
      nextLogits:
        payload.nextLogits === undefined
          ? undefined
          : {
              shape: Array.isArray(nextLogitsShape)
                ? (nextLogitsShape as number[])
                : [],
              dtype:
                typeof nextLogitsDtype === "string"
                  ? nextLogitsDtype
                  : "float32",
              data: payload.nextLogits.data,
            },
    };
  }

  private async readBestKVCheckpoint(
    files: OPFSFileStore,
    sessions: ResumableSessionStore,
    session: ResumableSessionHandle,
    state: ResumableReplayState,
  ): Promise<KVResumeCheckpoint | undefined> {
    const promptSeqLen = state.promptTokenIds.length;
    const refs = await sessions.listCommittedCheckpoints(session.sessionId);
    const chains = await resolveCommittedCheckpointChains(files, refs);
    chains.sort(
      (a, b) => b.at(-1)!.meta.processedSeqLen - a.at(-1)!.meta.processedSeqLen,
    );
    for (const chain of chains) {
      const tip = chain.at(-1)!;
      if (
        tip.meta.processedSeqLen < promptSeqLen ||
        tip.meta.processedSeqLen > state.processedSeqLen
      ) {
        continue;
      }
      try {
        const checkpoints: KVCheckpointData[] = [];
        for (const { ref } of chain) {
          const payload = await readResumableCheckpointPayload(files, ref);
          if (payload === undefined) break;
          checkpoints.push(this.checkpointPayloadToKVData(payload));
        }
        if (checkpoints.length === chain.length) {
          const checkpoint = checkpoints.at(-1)!;
          return {
            checkpoint,
            chain: checkpoints,
            checkpointId: tip.ref.checkpointId,
            coveredGeneratedTokens: state.generatedTokens.filter(
              (token) => token.globalTokenPos < checkpoint.processedSeqLen,
            ),
            tailGeneratedTokens: state.generatedTokens.filter(
              (token) => token.globalTokenPos >= checkpoint.processedSeqLen,
            ),
          };
        }
      } catch (err) {
        log.warn(
          `Ignoring invalid KV checkpoint ${tip.ref.checkpointId}:`,
          err,
        );
      }
    }
    return undefined;
  }

  private getContinuationBlockReason(
    state: ResumableReplayState,
  ): string | undefined {
    if (state.finished) {
      return "generation already finished";
    }
    if (state.scanStoppedReason !== undefined) {
      return `journal scan stopped at ${state.scanStoppedReason}`;
    }
    if (state.promptTokenIds.length === 0) {
      return "missing prompt token record";
    }
    if (state.generationConfig === undefined) {
      return "missing generation config record";
    }
    if (hasUnsupportedGrammarReplay(state.generationConfig)) {
      return "unsupported grammar replay";
    }
    if (
      state.generatedTokens.length > 0 &&
      state.generatedTokens.at(-1)?.rngState === undefined
    ) {
      return "missing RNG state";
    }
    if (state.modelId === "") {
      return "missing model id";
    }
    if (this.options.hasCustomLogitProcessor(state.modelId)) {
      return "custom LogitProcessor replay unsupported";
    }
    return undefined;
  }
}
