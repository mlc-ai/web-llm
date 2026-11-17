import { crc32c } from "./crc32c";
import { triggerResumableFault } from "./fault_injection";
import { OPFSFileStore, OPFSWritableFile } from "./opfs_file_store";
import type { ResumableSessionStore } from "./session_store";
import { ResumableCheckpointRef } from "./types";

type CheckpointByteSource = ArrayBufferLike | ArrayBufferView<ArrayBufferLike>;

export interface ResumableCheckpointPageGroup {
  groupId: number;
  layerStart: number;
  layerEnd: number;
  data: CheckpointByteSource;
  fileName?: string;
}

/** A checkpoint's description; its page bytes are written separately. */
export type ResumableCheckpointStreamInput = Omit<
  ResumableCheckpointWriteInput,
  "pageGroups" | "nextLogits"
> & { pageGroups: Array<Omit<ResumableCheckpointPageGroup, "data">> };

/**
 * A checkpoint whose page groups are written as their bytes arrive, without
 * holding a whole group in memory. Files are committed as in writeCheckpoint.
 */
export interface ResumableCheckpointStream {
  /** Append bytes to page group `index` (in input order); groups in order. */
  write(index: number, data: Uint8Array): Promise<void>;
  /**
   * Close the page groups, then write logits, metadata and the marker.
   * `metadata`, if given, replaces the input's.
   */
  finish(
    nextLogits?: CheckpointByteSource,
    metadata?: Record<string, unknown>,
  ): Promise<ResumableCheckpointRef>;
  /** Abandon an open page group file. */
  abort(): Promise<void>;
}

export interface ResumableCheckpointWriteInput {
  sessionId: string;
  checkpointId: string;
  processedSeqLen: number;
  layoutHash?: string;
  /** The checkpoint this one extends; its pages start at pageBegin. */
  parentCheckpointId?: string;
  pageBegin?: number;
  metadata?: Record<string, unknown>;
  pageGroups: ResumableCheckpointPageGroup[];
  nextLogits?: CheckpointByteSource;
}

interface CheckpointFileMeta {
  path: string;
  bytes: number;
  crc32c: string;
}

export type StoredCheckpointFileMeta = CheckpointFileMeta;

export interface StoredCheckpointPageGroup extends StoredCheckpointFileMeta {
  groupId: number;
  layerStart: number;
  layerEnd: number;
}

/**
 * A checkpoint stores KV pages [pageBegin, ceil(processedSeqLen / page_size)).
 * A delta (pageBegin > 0) names the checkpoint holding the pages before it;
 * a restore imports the chain from the full checkpoint up.
 */
export interface StoredResumableCheckpoint {
  checkpointId: string;
  processedSeqLen: number;
  layoutHash?: string;
  parentCheckpointId?: string;
  pageBegin?: number;
  createdAtMs: number;
  pageGroups: StoredCheckpointPageGroup[];
  nextLogits?: StoredCheckpointFileMeta;
  metadata?: Record<string, unknown>;
}

export interface StoredCheckpointChainLink {
  ref: ResumableCheckpointRef;
  meta: StoredResumableCheckpoint;
}

export interface ResumableCheckpointPayload {
  meta: StoredResumableCheckpoint;
  pageGroups: Array<StoredCheckpointPageGroup & { data: Uint8Array }>;
  nextLogits?: StoredCheckpointFileMeta & { data: Uint8Array };
}

/** Invalid stored bytes, as distinct from a transient storage read failure. */
export class InvalidCheckpointError extends Error {}

const META_FILE = "meta.json";
const NEXT_LOGITS_FILE = "next_logits.f16";
const COMPLETE_FILE = "complete";
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

// Copies only bytes that are not already backed by an ArrayBuffer; KV page
// groups are large enough that an extra copy shows up in decode latency.
function bytes(data: CheckpointByteSource): Uint8Array<ArrayBuffer> {
  const view = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  return view.buffer instanceof ArrayBuffer
    ? (view as Uint8Array<ArrayBuffer>)
    : new Uint8Array(view);
}

function crc32cHex(data: Uint8Array): string {
  return crc32c(data).toString(16).padStart(8, "0");
}

function encodeText(data: string): ArrayBuffer {
  const encoded = textEncoder.encode(data);
  const copy = new Uint8Array(encoded.byteLength);
  copy.set(encoded);
  return copy.buffer;
}

function parseCheckpointMeta(
  data: ArrayBuffer | undefined,
): StoredResumableCheckpoint | undefined {
  if (data === undefined) {
    return undefined;
  }
  let parsed: Partial<StoredResumableCheckpoint> | null;
  try {
    parsed = JSON.parse(textDecoder.decode(data));
  } catch {
    return undefined;
  }
  if (parsed === null || typeof parsed !== "object") return undefined;
  const processedSeqLen = parsed.processedSeqLen;
  if (
    typeof parsed.checkpointId !== "string" ||
    typeof processedSeqLen !== "number" ||
    !Number.isInteger(processedSeqLen) ||
    processedSeqLen < 0 ||
    typeof parsed.createdAtMs !== "number" ||
    !Array.isArray(parsed.pageGroups) ||
    !parsed.pageGroups.every(
      (group) =>
        isCheckpointFileMeta(group) &&
        Number.isSafeInteger(group.groupId) &&
        group.groupId >= 0 &&
        Number.isSafeInteger(group.layerStart) &&
        group.layerStart >= 0 &&
        Number.isSafeInteger(group.layerEnd) &&
        group.layerEnd >= group.layerStart,
    ) ||
    (parsed.nextLogits !== undefined &&
      !isCheckpointFileMeta(parsed.nextLogits)) ||
    (parsed.parentCheckpointId !== undefined &&
      (typeof parsed.parentCheckpointId !== "string" ||
        !isSafeFileName(parsed.parentCheckpointId))) ||
    (parsed.pageBegin !== undefined &&
      (!Number.isSafeInteger(parsed.pageBegin) || parsed.pageBegin < 0)) ||
    (parsed.pageBegin ?? 0) > 0 !== (parsed.parentCheckpointId !== undefined)
  ) {
    return undefined;
  }
  return parsed as StoredResumableCheckpoint;
}

function isCheckpointFileMeta(meta: CheckpointFileMeta): boolean {
  return (
    meta !== null &&
    typeof meta === "object" &&
    typeof meta.path === "string" &&
    isSafeFileName(meta.path) &&
    Number.isSafeInteger(meta.bytes) &&
    meta.bytes >= 0 &&
    typeof meta.crc32c === "string" &&
    /^[0-9a-f]{8}$/.test(meta.crc32c)
  );
}

async function readCheckedFile(
  files: OPFSFileStore,
  path: string,
  meta: CheckpointFileMeta,
): Promise<Uint8Array<ArrayBuffer>> {
  const data = await files.read(joinPath(path, meta.path));
  if (data === undefined) {
    throw new InvalidCheckpointError(
      `Checkpoint file is missing: ${meta.path}`,
    );
  }
  const bytes = new Uint8Array(data);
  if (bytes.byteLength !== meta.bytes) {
    throw new InvalidCheckpointError(
      `Checkpoint file ${meta.path} size mismatch: expected ${meta.bytes}, got ${bytes.byteLength}.`,
    );
  }
  const actual = crc32cHex(bytes);
  if (actual !== meta.crc32c) {
    throw new InvalidCheckpointError(
      `Checkpoint file ${meta.path} CRC mismatch: expected ${meta.crc32c}, got ${actual}.`,
    );
  }
  return new Uint8Array(bytes);
}

function joinPath(...parts: string[]): string {
  return parts
    .flatMap((part) => part.split("/"))
    .filter((part) => part !== "")
    .join("/");
}

function isSafeFileName(value: string): boolean {
  return (
    value !== "" &&
    value !== "." &&
    value !== ".." &&
    !value.includes("/") &&
    !value.includes("\0")
  );
}

function assertSafeFileName(value: string): void {
  if (!isSafeFileName(value)) {
    throw new Error(`Invalid checkpoint file name: ${value}`);
  }
}

function pad3(value: number): string {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(
      `Checkpoint numeric segment must be a non-negative integer: ${value}`,
    );
  }
  return value.toString().padStart(3, "0");
}

function pageGroupFileName(
  group: Omit<ResumableCheckpointPageGroup, "data">,
): string {
  if (group.fileName !== undefined) {
    assertSafeFileName(group.fileName);
    return group.fileName;
  }
  return `group_${pad3(group.groupId)}_layers_${pad3(group.layerStart)}_${pad3(
    group.layerEnd,
  )}_pages.wkv`;
}

export class ResumableCheckpointWriter {
  constructor(
    private readonly files: OPFSFileStore,
    private readonly sessions: ResumableSessionStore,
    private readonly now: () => number = Date.now,
  ) {}

  /** An aborted signal stops the write between files and rejects. */
  async writeCheckpoint(
    input: ResumableCheckpointWriteInput,
    signal?: AbortSignal,
  ): Promise<ResumableCheckpointRef> {
    const stream = await this.beginCheckpoint(input, signal);
    try {
      for (const [index, group] of input.pageGroups.entries()) {
        await stream.write(index, bytes(group.data));
      }
      return await stream.finish(input.nextLogits);
    } catch (err) {
      await stream.abort();
      throw err;
    }
  }

  /**
   * Create the checkpoint directory and return a stream for its page bytes.
   * Each page group file is checksummed as it is written and closed when the
   * next group starts or the stream finishes; the metadata and the complete
   * marker follow every payload file, so a partial checkpoint is never
   * complete. An aborted signal stops the stream between writes and rejects.
   */
  async beginCheckpoint(
    input: ResumableCheckpointStreamInput,
    signal?: AbortSignal,
  ): Promise<ResumableCheckpointStream> {
    if (!Number.isInteger(input.processedSeqLen) || input.processedSeqLen < 0) {
      throw new Error(
        "Checkpoint processedSeqLen must be a non-negative integer.",
      );
    }

    await this.sessions.createSession(input.sessionId);
    const ref = this.sessions.getCheckpointRef(
      input.sessionId,
      input.checkpointId,
    );
    await this.files.remove(ref.path, { recursive: true });
    await this.files.mkdir(ref.path);
    const faultContext = {
      path: ref.path,
      sessionId: input.sessionId,
      checkpointId: input.checkpointId,
      processedSeqLen: input.processedSeqLen,
    };

    const pageGroups: StoredCheckpointPageGroup[] = input.pageGroups.map(
      (group) => ({
        groupId: group.groupId,
        layerStart: group.layerStart,
        layerEnd: group.layerEnd,
        path: pageGroupFileName(group),
        bytes: 0,
        crc32c: "",
      }),
    );
    // Groups before `next` are closed; `open` is group next - 1, if any.
    let next = 0;
    let open: OPFSWritableFile | undefined;
    let crc = 0;
    const closeOpen = async () => {
      if (open === undefined) return;
      const group = pageGroups[next - 1];
      await open.close();
      open = undefined;
      group.crc32c = crc.toString(16).padStart(8, "0");
      await triggerResumableFault("checkpoint.after_page_group", {
        ...faultContext,
        path: joinPath(ref.path, group.path),
        groupId: group.groupId,
      });
    };
    // Open groups up to `index`, writing any skipped ones empty.
    const advanceTo = async (index: number) => {
      while (next <= index) {
        await closeOpen();
        signal?.throwIfAborted();
        open = await this.files.openWritable(
          joinPath(ref.path, pageGroups[next].path),
        );
        crc = 0;
        next++;
      }
    };

    return {
      write: async (index, data) => {
        if (index < next - 1 || index >= pageGroups.length) {
          throw new Error(`Checkpoint page group ${index} is out of order.`);
        }
        await advanceTo(index);
        signal?.throwIfAborted();
        await open!.write(bytes(data));
        pageGroups[index].bytes += data.byteLength;
        crc = crc32c(data, crc);
      },
      finish: async (nextLogitsData, metadata = input.metadata) => {
        await advanceTo(pageGroups.length - 1);
        await closeOpen();

        signal?.throwIfAborted();
        let nextLogits: CheckpointFileMeta | undefined;
        if (nextLogitsData !== undefined) {
          const data = bytes(nextLogitsData);
          await this.files.write(joinPath(ref.path, NEXT_LOGITS_FILE), data);
          nextLogits = {
            path: NEXT_LOGITS_FILE,
            bytes: data.byteLength,
            crc32c: crc32cHex(data),
          };
        }
        await triggerResumableFault(
          "checkpoint.after_next_logits",
          faultContext,
        );

        signal?.throwIfAborted();
        const createdAtMs = this.now();
        await this.files.write(
          joinPath(ref.path, META_FILE),
          encodeText(
            `${JSON.stringify({
              checkpointId: input.checkpointId,
              processedSeqLen: input.processedSeqLen,
              layoutHash: input.layoutHash,
              parentCheckpointId: input.parentCheckpointId,
              pageBegin: input.pageBegin,
              createdAtMs,
              pageGroups,
              nextLogits,
              metadata,
            })}\n`,
          ),
        );
        await triggerResumableFault("checkpoint.after_meta", faultContext);
        signal?.throwIfAborted();
        await this.files.write(
          joinPath(ref.path, COMPLETE_FILE),
          encodeText(""),
        );
        await triggerResumableFault("checkpoint.after_complete", faultContext);
        return ref;
      },
      abort: async () => {
        await open?.abort().catch(() => undefined);
        open = undefined;
      },
    };
  }
}

export async function readResumableCheckpointMeta(
  files: OPFSFileStore,
  ref: ResumableCheckpointRef,
): Promise<StoredResumableCheckpoint | undefined> {
  const entries = await files.list(ref.path);
  if (!entries.includes(COMPLETE_FILE)) {
    return undefined;
  }
  const meta = parseCheckpointMeta(
    await files.read(joinPath(ref.path, META_FILE)),
  );
  if (
    meta === undefined ||
    meta.checkpointId !== ref.checkpointId ||
    ![...meta.pageGroups, meta.nextLogits].every(
      (file) => file === undefined || entries.includes(file.path),
    )
  ) {
    return undefined;
  }
  return meta;
}

function checkpointPageSize(
  meta: StoredResumableCheckpoint,
): number | undefined {
  const pageSize = meta.metadata?.page_size;
  return typeof pageSize === "number" &&
    Number.isInteger(pageSize) &&
    pageSize > 0
    ? pageSize
    : undefined;
}

/** Whether `child` can be imported right after `parent`. */
function extendsCheckpoint(
  child: StoredResumableCheckpoint,
  parent: StoredResumableCheckpoint,
): boolean {
  const pageSize = checkpointPageSize(child);
  return (
    pageSize !== undefined &&
    parent.checkpointId === child.parentCheckpointId &&
    parent.layoutHash === child.layoutHash &&
    parent.processedSeqLen < child.processedSeqLen &&
    (child.pageBegin ?? 0) <= Math.ceil(parent.processedSeqLen / pageSize)
  );
}

/**
 * The restorable chain of each given committed checkpoint, in input order,
 * omitting checkpoints whose chain does not resolve. A chain lists the full
 * checkpoint first and the given checkpoint last; every link must be among
 * `refs` and have valid metadata. `readMeta` may return undefined to treat a
 * checkpoint as unusable.
 */
export async function resolveCommittedCheckpointChains(
  files: OPFSFileStore,
  refs: ResumableCheckpointRef[],
  readMeta: (
    ref: ResumableCheckpointRef,
  ) => Promise<StoredResumableCheckpoint | undefined> = (ref) =>
    readResumableCheckpointMeta(files, ref),
): Promise<StoredCheckpointChainLink[][]> {
  const refsById = new Map(refs.map((ref) => [ref.checkpointId, ref]));
  const metas = new Map<
    string,
    Promise<StoredResumableCheckpoint | undefined>
  >();
  const meta = (ref: ResumableCheckpointRef) => {
    let cached = metas.get(ref.checkpointId);
    if (cached === undefined) {
      cached = readMeta(ref);
      metas.set(ref.checkpointId, cached);
    }
    return cached;
  };
  const chains: StoredCheckpointChainLink[][] = [];
  for (const tip of refs) {
    const chain: StoredCheckpointChainLink[] = [];
    let ref: ResumableCheckpointRef | undefined = tip;
    while (ref !== undefined) {
      const link = await meta(ref);
      if (
        link === undefined ||
        (chain.length > 0 && !extendsCheckpoint(chain[0].meta, link))
      ) {
        break;
      }
      chain.unshift({ ref, meta: link });
      if (link.parentCheckpointId === undefined) {
        chains.push(chain);
        break;
      }
      ref = refsById.get(link.parentCheckpointId);
    }
  }
  return chains;
}

/** Payload files a restore reads. */
export function checkpointPayloadFiles(
  meta: StoredResumableCheckpoint,
): number {
  return meta.pageGroups.length + (meta.nextLogits === undefined ? 0 : 1);
}

/** Bytes a restore reads: every payload file named by the metadata. */
export function checkpointPayloadBytes(
  meta: StoredResumableCheckpoint,
): number {
  return meta.pageGroups.reduce(
    (sum, group) => sum + group.bytes,
    meta.nextLogits?.bytes ?? 0,
  );
}

export async function readResumableCheckpointPayload(
  files: OPFSFileStore,
  ref: ResumableCheckpointRef,
): Promise<ResumableCheckpointPayload | undefined> {
  const meta = await readResumableCheckpointMeta(files, ref);
  if (meta === undefined) {
    return undefined;
  }

  const pageGroups = [];
  for (const group of meta.pageGroups) {
    pageGroups.push({
      ...group,
      data: await readCheckedFile(files, ref.path, group),
    });
  }
  const nextLogits =
    meta.nextLogits === undefined
      ? undefined
      : {
          ...meta.nextLogits,
          data: await readCheckedFile(files, ref.path, meta.nextLogits),
        };

  return { meta, pageGroups, nextLogits };
}
