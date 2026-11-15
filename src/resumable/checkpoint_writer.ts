import { crc32c } from "./crc32c";
import { triggerResumableFault } from "./fault_injection";
import { OPFSFileStore } from "./opfs_file_store";
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

export interface ResumableCheckpointWriteInput {
  sessionId: string;
  checkpointId: string;
  processedSeqLen: number;
  layoutHash?: string;
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

export interface StoredResumableCheckpoint {
  checkpointId: string;
  processedSeqLen: number;
  layoutHash?: string;
  createdAtMs: number;
  pageGroups: StoredCheckpointPageGroup[];
  nextLogits?: StoredCheckpointFileMeta;
  metadata?: Record<string, unknown>;
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

function bytes(data: CheckpointByteSource): Uint8Array<ArrayBuffer> {
  const view = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  return new Uint8Array(view);
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
      !isCheckpointFileMeta(parsed.nextLogits))
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

function pageGroupFileName(group: ResumableCheckpointPageGroup): string {
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

  async writeCheckpoint(
    input: ResumableCheckpointWriteInput,
  ): Promise<ResumableCheckpointRef> {
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

    const pageGroups: Array<
      CheckpointFileMeta & {
        groupId: number;
        layerStart: number;
        layerEnd: number;
      }
    > = [];
    for (const group of input.pageGroups) {
      const fileName = pageGroupFileName(group);
      const data = bytes(group.data);
      await this.files.write(joinPath(ref.path, fileName), data);
      await triggerResumableFault("checkpoint.after_page_group", {
        ...faultContext,
        path: joinPath(ref.path, fileName),
        groupId: group.groupId,
      });
      pageGroups.push({
        groupId: group.groupId,
        layerStart: group.layerStart,
        layerEnd: group.layerEnd,
        path: fileName,
        bytes: data.byteLength,
        crc32c: crc32cHex(data),
      });
    }

    let nextLogits: CheckpointFileMeta | undefined;
    if (input.nextLogits !== undefined) {
      const data = bytes(input.nextLogits);
      await this.files.write(joinPath(ref.path, NEXT_LOGITS_FILE), data);
      nextLogits = {
        path: NEXT_LOGITS_FILE,
        bytes: data.byteLength,
        crc32c: crc32cHex(data),
      };
    }
    await triggerResumableFault("checkpoint.after_next_logits", faultContext);

    const createdAtMs = this.now();
    await this.files.write(
      joinPath(ref.path, META_FILE),
      encodeText(
        `${JSON.stringify({
          checkpointId: input.checkpointId,
          processedSeqLen: input.processedSeqLen,
          layoutHash: input.layoutHash,
          createdAtMs,
          pageGroups,
          nextLogits,
          metadata: input.metadata,
        })}\n`,
      ),
    );
    await triggerResumableFault("checkpoint.after_meta", faultContext);
    await this.files.write(joinPath(ref.path, COMPLETE_FILE), encodeText(""));
    await triggerResumableFault("checkpoint.after_complete", faultContext);

    return ref;
  }
}

export async function readResumableCheckpointPayload(
  files: OPFSFileStore,
  ref: ResumableCheckpointRef,
): Promise<ResumableCheckpointPayload | undefined> {
  if ((await files.read(ref.completePath)) === undefined) {
    return undefined;
  }
  const meta = parseCheckpointMeta(
    await files.read(joinPath(ref.path, META_FILE)),
  );
  if (meta === undefined || meta.checkpointId !== ref.checkpointId) {
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
