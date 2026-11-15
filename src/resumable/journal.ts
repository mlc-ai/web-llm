import { crc32c } from "./crc32c";
import {
  CrossContextLockUnavailableError,
  OPFSFileStore,
} from "./opfs_file_store";
import { triggerResumableFault } from "./fault_injection";

export const JOURNAL_MAGIC = 0x574c4c4a;

const HEADER_SIZE = 30;
const RECORD_TYPE_OFFSET = 4;
const SEQ_NO_OFFSET = 6;
const CREATED_AT_MS_OFFSET = 14;
const PAYLOAD_LEN_OFFSET = 22;
const PAYLOAD_CRC32C_OFFSET = 26;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export enum JournalRecordType {
  SessionBegin = 1,
  PromptTokens = 2,
  AssistantPrefixTokens = 3,
  GenerationConfig = 4,
  GeneratedToken = 5,
  CheckpointCommit = 6,
  GenerationFinished = 7,
  GenerationAborted = 8,
  EngineError = 9,
}

export interface SessionBeginPayload {
  sessionId: string;
  modelId?: string;
  createdAtMs?: number;
  request?: unknown;
}

export interface TokenIdsPayload {
  tokenIds: number[];
  text?: string;
}

export interface GenerationConfigPayload {
  config: unknown;
}

export interface GeneratedTokenPayload {
  globalTokenPos: number;
  tokenId: number;
  textDelta: string;
  /**
   * Number of UTF-16 code units retained from the previously recovered text.
   * Absent on legacy append-only records.
   */
  textPrefixLength?: number;
  rngState?: unknown;
  logprob?: unknown;
}

export interface CheckpointCommitPayload {
  checkpointId: string;
  processedSeqLen: number;
  path?: string;
  layoutHash?: string;
}

export interface GenerationFinishedPayload {
  finishReason?: string;
  emittedTokens?: number;
}

export interface GenerationAbortedPayload {
  reason?: string;
}

export interface EngineErrorPayload {
  message: string;
  name?: string;
  stack?: string;
}

export type JournalRecord =
  | JournalRecordBase<JournalRecordType.SessionBegin, SessionBeginPayload>
  | JournalRecordBase<JournalRecordType.PromptTokens, TokenIdsPayload>
  | JournalRecordBase<JournalRecordType.AssistantPrefixTokens, TokenIdsPayload>
  | JournalRecordBase<
      JournalRecordType.GenerationConfig,
      GenerationConfigPayload
    >
  | JournalRecordBase<JournalRecordType.GeneratedToken, GeneratedTokenPayload>
  | JournalRecordBase<
      JournalRecordType.CheckpointCommit,
      CheckpointCommitPayload
    >
  | JournalRecordBase<
      JournalRecordType.GenerationFinished,
      GenerationFinishedPayload
    >
  | JournalRecordBase<
      JournalRecordType.GenerationAborted,
      GenerationAbortedPayload
    >
  | JournalRecordBase<JournalRecordType.EngineError, EngineErrorPayload>;

export interface JournalRecordBase<
  T extends JournalRecordType,
  P extends object,
> {
  type: T;
  seqNo: number;
  createdAtMs: number;
  payload: P;
}

export interface DecodedJournalRecord {
  record: JournalRecord;
  nextOffset: number;
}

export interface JournalScanResult {
  records: JournalRecord[];
  validBytes: number;
  stoppedReason?:
    | "partial_record"
    | "bad_magic"
    | "crc_mismatch"
    | "invalid_payload";
}

function bytes(data: string): Uint8Array<ArrayBuffer> {
  const encoded = textEncoder.encode(data);
  const copy = new Uint8Array(encoded.byteLength);
  copy.set(encoded);
  return copy;
}

function payloadBytes(payload: object): Uint8Array<ArrayBuffer> {
  return bytes(JSON.stringify(payload));
}

function copyRecordBytes(
  header: ArrayBuffer,
  payload: Uint8Array,
): ArrayBuffer {
  const out = new Uint8Array(HEADER_SIZE + payload.byteLength);
  out.set(new Uint8Array(header), 0);
  out.set(payload, HEADER_SIZE);
  return out.buffer;
}

function parsePayload(
  type: JournalRecordType,
  payload: Uint8Array,
): JournalRecord["payload"] {
  const parsed = JSON.parse(textDecoder.decode(payload)) as unknown;
  switch (type) {
    case JournalRecordType.SessionBegin:
      return parsed as SessionBeginPayload;
    case JournalRecordType.PromptTokens:
    case JournalRecordType.AssistantPrefixTokens:
      return parsed as TokenIdsPayload;
    case JournalRecordType.GenerationConfig:
      return parsed as GenerationConfigPayload;
    case JournalRecordType.GeneratedToken:
      return parsed as GeneratedTokenPayload;
    case JournalRecordType.CheckpointCommit:
      return parsed as CheckpointCommitPayload;
    case JournalRecordType.GenerationFinished:
      return parsed as GenerationFinishedPayload;
    case JournalRecordType.GenerationAborted:
      return parsed as GenerationAbortedPayload;
    case JournalRecordType.EngineError:
      return parsed as EngineErrorPayload;
    default:
      throw new Error(`Unknown journal record type: ${type}`);
  }
}

export function encodeJournalRecord(record: JournalRecord): ArrayBuffer {
  const payload = payloadBytes(record.payload);
  const header = new ArrayBuffer(HEADER_SIZE);
  const view = new DataView(header);
  view.setUint32(0, JOURNAL_MAGIC, true);
  view.setUint16(RECORD_TYPE_OFFSET, record.type, true);
  view.setBigUint64(SEQ_NO_OFFSET, BigInt(record.seqNo), true);
  view.setBigUint64(CREATED_AT_MS_OFFSET, BigInt(record.createdAtMs), true);
  view.setUint32(PAYLOAD_LEN_OFFSET, payload.byteLength, true);
  view.setUint32(PAYLOAD_CRC32C_OFFSET, crc32c(payload), true);
  return copyRecordBytes(header, payload);
}

export function decodeJournalRecordAt(
  data: ArrayBuffer,
  offset = 0,
): DecodedJournalRecord {
  const view = new DataView(data);
  if (offset + HEADER_SIZE > data.byteLength) {
    throw new Error("Partial journal record header");
  }
  const magic = view.getUint32(offset, true);
  if (magic !== JOURNAL_MAGIC) {
    throw new Error("Bad journal record magic");
  }
  const type = view.getUint16(
    offset + RECORD_TYPE_OFFSET,
    true,
  ) as JournalRecordType;
  const seqNo = Number(view.getBigUint64(offset + SEQ_NO_OFFSET, true));
  const createdAtMs = Number(
    view.getBigUint64(offset + CREATED_AT_MS_OFFSET, true),
  );
  const payloadLen = view.getUint32(offset + PAYLOAD_LEN_OFFSET, true);
  const expectedCrc = view.getUint32(offset + PAYLOAD_CRC32C_OFFSET, true);
  const payloadStart = offset + HEADER_SIZE;
  const nextOffset = payloadStart + payloadLen;
  if (nextOffset > data.byteLength) {
    throw new Error("Partial journal record payload");
  }
  const payload = new Uint8Array(data, payloadStart, payloadLen);
  if (crc32c(payload) !== expectedCrc) {
    throw new Error("Journal payload CRC mismatch");
  }
  return {
    record: {
      type,
      seqNo,
      createdAtMs,
      payload: parsePayload(type, payload),
    } as JournalRecord,
    nextOffset,
  };
}

export function scanJournalRecords(data: ArrayBuffer): JournalScanResult {
  const records: JournalRecord[] = [];
  let offset = 0;
  const view = new DataView(data);
  while (offset < data.byteLength) {
    if (offset + HEADER_SIZE > data.byteLength) {
      return { records, validBytes: offset, stoppedReason: "partial_record" };
    }
    const payloadLen = view.getUint32(offset + PAYLOAD_LEN_OFFSET, true);
    if (offset + HEADER_SIZE + payloadLen > data.byteLength) {
      return { records, validBytes: offset, stoppedReason: "partial_record" };
    }
    const magic = view.getUint32(offset, true);
    if (magic !== JOURNAL_MAGIC) {
      return { records, validBytes: offset, stoppedReason: "bad_magic" };
    }
    const expectedCrc = view.getUint32(offset + PAYLOAD_CRC32C_OFFSET, true);
    const payloadStart = offset + HEADER_SIZE;
    const payload = new Uint8Array(data, payloadStart, payloadLen);
    if (crc32c(payload) !== expectedCrc) {
      return { records, validBytes: offset, stoppedReason: "crc_mismatch" };
    }
    try {
      const decoded = decodeJournalRecordAt(data, offset);
      records.push(decoded.record);
      offset = decoded.nextOffset;
    } catch {
      return { records, validBytes: offset, stoppedReason: "invalid_payload" };
    }
  }
  return { records, validBytes: offset };
}

// Readers wait only for the current I/O operation. They do not acquire the
// session's generation lock. The sidecar also supports sync-access locking
// in workers without Web Locks.
export function journalLockPath(path: string): string {
  return `${path}.lock`;
}

export async function appendJournalRecord(
  store: OPFSFileStore,
  path: string,
  record: JournalRecord,
): Promise<void> {
  const faultContext = {
    path,
    recordType: record.type,
    seqNo: record.seqNo,
  };
  await triggerResumableFault("journal.before_append", faultContext);
  const release = await store.lock(journalLockPath(path));
  try {
    await store.append(path, encodeJournalRecord(record));
  } finally {
    release();
  }
  await triggerResumableFault("journal.after_append", faultContext);
}

export async function readJournalRecords(
  store: OPFSFileStore,
  path: string,
): Promise<JournalScanResult> {
  let release: (() => void) | undefined;
  try {
    release = await store.lock(journalLockPath(path));
  } catch (err) {
    // Preserve best-effort text inspection in environments that cannot lock.
    // Persistence/continuation still require cross-context exclusion.
    if (!(err instanceof CrossContextLockUnavailableError)) {
      throw err;
    }
  }
  let data: ArrayBuffer | undefined;
  try {
    // File snapshots can become unreadable when a concurrent OPFS writable
    // commits. Hold exclusion until all snapshot bytes have been materialized.
    data = await store.read(path);
  } finally {
    release?.();
  }
  if (data === undefined) {
    return { records: [], validBytes: 0 };
  }
  return scanJournalRecords(data);
}

/**
 * Truncate a torn or corrupt journal tail while retaining every validated
 * record before it. Callers must hold the session lock while repairing.
 */
export async function repairJournalTail(
  store: OPFSFileStore,
  path: string,
): Promise<JournalScanResult> {
  const release = await store.lock(journalLockPath(path));
  try {
    const data = await store.read(path);
    if (data === undefined) {
      return { records: [], validBytes: 0 };
    }
    const scan = scanJournalRecords(data);
    if (scan.stoppedReason === undefined) {
      return scan;
    }
    await store.write(path, data.slice(0, scan.validBytes));
    return {
      records: scan.records,
      validBytes: scan.validBytes,
    };
  } finally {
    release();
  }
}
