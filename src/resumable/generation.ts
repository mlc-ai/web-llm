import { GenerationConfig } from "../config";
import { ResumableGenerationConfig } from "../types";
import { OPFSFileStore } from "./opfs_file_store";
import {
  CheckpointCommitPayload,
  GeneratedTokenPayload,
  JournalRecord,
  JournalRecordType,
  appendJournalRecord,
  readJournalRecords,
} from "./journal";
import { triggerResumableFault } from "./fault_injection";
import {
  ResumableSessionExistsError,
  ResumableSessionStore,
} from "./session_store";
import { ResumableSessionHandle } from "./types";

export interface NormalizedResumableGenerationConfig {
  enabled: true;
  sessionId: string;
  checkpointIntervalTokens: number;
  checkpointPrompt: boolean;
  durabilityMode: "exact" | "relaxed";
  strictPersistence: boolean;
  storeCheckpointLogits: boolean;
}

export interface ResumableGenerationJournalInit {
  modelId: string;
  request: unknown;
  promptTokenIds: number[];
  assistantPrefixTokenIds: number[];
  generationConfig: GenerationConfig;
}

export interface ResumableGeneratedTokenInput {
  globalTokenPos: number;
  tokenId: number;
  textDelta: string;
  textPrefixLength: number;
  rngState?: unknown;
  logprob?: unknown;
}

export interface ResumableGenerationEndInput {
  finishReason?: string;
  emittedTokens: number;
}

export interface EngineErrorInput {
  err: unknown;
}

const DEFAULT_CHECKPOINT_INTERVAL_TOKENS = 512;

function defaultSessionId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `session-${Date.now()}`;
}

function nowMs(): number {
  return Date.now();
}

function errorField(
  err: unknown,
  field: "name" | "message" | "stack",
): string | undefined {
  const value = (err as Record<string, unknown>)?.[field];
  return typeof value === "string" ? value : undefined;
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}

export function normalizeResumableGenerationConfig(
  config?: ResumableGenerationConfig | null,
): NormalizedResumableGenerationConfig | undefined {
  if (config?.enabled !== true) {
    return undefined;
  }
  const checkpointIntervalTokens =
    config.checkpointIntervalTokens ?? DEFAULT_CHECKPOINT_INTERVAL_TOKENS;
  if (!Number.isInteger(checkpointIntervalTokens)) {
    throw new Error("resumable.checkpointIntervalTokens must be an integer.");
  }
  if (checkpointIntervalTokens <= 0) {
    throw new Error("resumable.checkpointIntervalTokens must be positive.");
  }

  return {
    enabled: true,
    sessionId: config.sessionId ?? defaultSessionId(),
    checkpointIntervalTokens,
    checkpointPrompt: config.checkpointPrompt ?? true,
    durabilityMode: config.durabilityMode ?? "exact",
    strictPersistence: config.strictPersistence ?? false,
    storeCheckpointLogits: config.storeCheckpointLogits ?? true,
  };
}

export class ResumableGenerationJournal {
  private session?: ResumableSessionHandle;
  private releaseSessionLock?: () => void;
  private seqNo = 0;
  private emittedTokens = 0;
  private disabled = false;
  private pendingError?: unknown;
  private pendingWrite: Promise<void> = Promise.resolve();
  private lastFlushAtMs = nowMs();

  constructor(
    private readonly files: OPFSFileStore,
    private readonly sessions: ResumableSessionStore,
    private readonly config: NormalizedResumableGenerationConfig,
  ) {}

  get sessionId(): string {
    return this.config.sessionId;
  }

  get emittedTokenCount(): number {
    return this.emittedTokens;
  }

  get checkpointPrompt(): boolean {
    return this.config.checkpointPrompt;
  }

  get checkpointIntervalTokens(): number {
    return this.config.checkpointIntervalTokens;
  }

  get storeCheckpointLogits(): boolean {
    return this.config.storeCheckpointLogits;
  }

  get strictPersistence(): boolean {
    return this.config.strictPersistence;
  }

  get active(): boolean {
    return !this.disabled && this.session !== undefined;
  }

  async begin(init: ResumableGenerationJournalInit): Promise<void> {
    const paths = this.sessions.getSessionPaths(this.config.sessionId);
    let releaseSessionLock: (() => void) | undefined;
    try {
      releaseSessionLock = await this.files.tryLock(paths.lockPath);
    } catch (err) {
      if (this.config.strictPersistence) {
        throw err;
      }
      this.disabled = true;
      this.pendingError = err;
      return;
    }
    if (releaseSessionLock === undefined) {
      throw new Error(
        `Resumable session is already active: ${this.config.sessionId}`,
      );
    }
    this.releaseSessionLock = releaseSessionLock;

    try {
      this.session = await this.sessions.createNewSession(
        this.config.sessionId,
        { modelId: init.modelId },
      );
      await this.writeInitialRecords(init);
    } catch (err) {
      this.releaseSessionLock?.();
      this.releaseSessionLock = undefined;
      if (
        !this.config.strictPersistence &&
        !(err instanceof ResumableSessionExistsError)
      ) {
        this.disabled = true;
        this.pendingError = err;
        return;
      }
      throw err;
    }
  }

  async attachExistingSession(
    session: ResumableSessionHandle,
    emittedTokens: number,
  ): Promise<void> {
    if (session.sessionId !== this.config.sessionId) {
      throw new Error(
        `Resumable journal session mismatch: expected ${this.config.sessionId}, got ${session.sessionId}.`,
      );
    }
    if (!Number.isInteger(emittedTokens) || emittedTokens < 0) {
      throw new Error("Resumable emitted token count must be non-negative.");
    }
    this.session = session;
    this.emittedTokens = emittedTokens;
    const scan = await readJournalRecords(
      this.files,
      session.paths.journalPath,
    );
    this.seqNo =
      scan.records.length === 0
        ? 0
        : Math.max(...scan.records.map((record) => record.seqNo));
  }

  async recordGeneratedToken(
    input: ResumableGeneratedTokenInput,
  ): Promise<void> {
    this.emittedTokens++;
    const payload: GeneratedTokenPayload = {
      globalTokenPos: input.globalTokenPos,
      tokenId: input.tokenId,
      textDelta: input.textDelta,
      textPrefixLength: input.textPrefixLength,
    };
    if (input.rngState !== undefined) {
      payload.rngState = input.rngState;
    }
    if (input.logprob !== undefined) {
      payload.logprob = input.logprob;
    }

    const shouldWait =
      this.config.durabilityMode === "exact" ||
      this.emittedTokens % 8 === 0 ||
      nowMs() - this.lastFlushAtMs >= 250;
    await this.append(
      {
        type: JournalRecordType.GeneratedToken,
        seqNo: this.nextSeqNo(),
        createdAtMs: nowMs(),
        payload,
      },
      shouldWait,
    );
  }

  async recordCheckpointCommit(
    payload: CheckpointCommitPayload,
  ): Promise<void> {
    const faultContext = {
      path: payload.path,
      sessionId: this.sessionId,
      checkpointId: payload.checkpointId,
      processedSeqLen: payload.processedSeqLen,
    };
    await triggerResumableFault("checkpoint.before_commit", faultContext);
    await this.append(
      {
        type: JournalRecordType.CheckpointCommit,
        seqNo: this.nextSeqNo(),
        createdAtMs: nowMs(),
        payload,
      },
      true,
    );
    await triggerResumableFault("checkpoint.after_commit", faultContext);
  }

  async recordGenerationEnd(input: ResumableGenerationEndInput): Promise<void> {
    const finishReason = input.finishReason;
    if (finishReason === "abort") {
      await this.append(
        {
          type: JournalRecordType.GenerationAborted,
          seqNo: this.nextSeqNo(),
          createdAtMs: nowMs(),
          payload: { reason: "abort" },
        },
        true,
      );
    } else {
      await this.append(
        {
          type: JournalRecordType.GenerationFinished,
          seqNo: this.nextSeqNo(),
          createdAtMs: nowMs(),
          payload: {
            finishReason,
            emittedTokens: input.emittedTokens,
          },
        },
        true,
      );
    }
  }

  async recordEngineError(input: EngineErrorInput): Promise<void> {
    await this.append(
      {
        type: JournalRecordType.EngineError,
        seqNo: this.nextSeqNo(),
        createdAtMs: nowMs(),
        payload: {
          message: errorMessage(input.err),
          name: errorField(input.err, "name"),
          stack: errorField(input.err, "stack"),
        },
      },
      true,
    );
  }

  async close(): Promise<void> {
    try {
      await this.flush();
    } finally {
      this.releaseSessionLock?.();
      this.releaseSessionLock = undefined;
    }
  }

  async refreshSeqNo(): Promise<void> {
    if (this.session === undefined) {
      throw new Error("Resumable journal session is not initialized.");
    }
    const scan = await readJournalRecords(
      this.files,
      this.session.paths.journalPath,
    );
    this.seqNo =
      scan.records.length === 0
        ? 0
        : Math.max(...scan.records.map((record) => record.seqNo));
  }

  private async writeInitialRecords(
    init: ResumableGenerationJournalInit,
  ): Promise<void> {
    await this.append(
      {
        type: JournalRecordType.SessionBegin,
        seqNo: this.nextSeqNo(),
        createdAtMs: nowMs(),
        payload: {
          sessionId: this.config.sessionId,
          modelId: init.modelId,
          createdAtMs: nowMs(),
          request: init.request,
        },
      },
      true,
    );
    await this.append(
      {
        type: JournalRecordType.PromptTokens,
        seqNo: this.nextSeqNo(),
        createdAtMs: nowMs(),
        payload: { tokenIds: init.promptTokenIds },
      },
      true,
    );
    if (init.assistantPrefixTokenIds.length > 0) {
      await this.append(
        {
          type: JournalRecordType.AssistantPrefixTokens,
          seqNo: this.nextSeqNo(),
          createdAtMs: nowMs(),
          payload: { tokenIds: init.assistantPrefixTokenIds },
        },
        true,
      );
    }
    await this.append(
      {
        type: JournalRecordType.GenerationConfig,
        seqNo: this.nextSeqNo(),
        createdAtMs: nowMs(),
        payload: {
          config: {
            generationConfig: init.generationConfig,
            resumable: this.config,
          },
        },
      },
      true,
    );
  }

  private nextSeqNo(): number {
    this.seqNo += 1;
    return this.seqNo;
  }

  private async append(record: JournalRecord, wait: boolean): Promise<void> {
    if (this.disabled) {
      return;
    }
    const write = this.pendingWrite.then(async () => {
      if (this.disabled || this.pendingError !== undefined) {
        return;
      }
      if (this.session === undefined) {
        throw new Error("Resumable journal session is not initialized.");
      }
      await appendJournalRecord(
        this.files,
        this.session.paths.journalPath,
        record,
      );
    });
    this.pendingWrite = write.catch((err) => {
      this.pendingError ??= err;
    });
    if (!wait) {
      return;
    }
    await this.flush(write);
    this.lastFlushAtMs = nowMs();
  }

  private async flush(write: Promise<void> = this.pendingWrite): Promise<void> {
    try {
      await write;
    } catch (err) {
      this.pendingError = err;
    }
    if (this.pendingError !== undefined) {
      this.disabled = true;
      if (this.config.strictPersistence) {
        throw this.pendingError;
      }
    }
  }
}
