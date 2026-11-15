import { OPFSFileStore } from "../../src/resumable/opfs_file_store";

function normalize(path: string): string {
  return path
    .split("/")
    .filter((part) => part !== "")
    .join("/");
}

function parentDirs(path: string): string[] {
  const parts = normalize(path).split("/");
  parts.pop();
  const dirs: string[] = [];
  for (let i = 1; i <= parts.length; i++) {
    dirs.push(parts.slice(0, i).join("/"));
  }
  return dirs;
}

export function bytes(data: BufferSource): Uint8Array<ArrayBuffer> {
  const view = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  return new Uint8Array(view);
}

export class MemoryFileStore implements OPFSFileStore {
  private readonly files = new Map<string, Uint8Array<ArrayBuffer>>();
  private readonly dirs = new Set<string>([""]);
  private readonly locks = new Set<string>();
  private readonly lockWaiters = new Map<
    string,
    Array<(release: () => void) => void>
  >();
  public pauseAppends = false;
  public pauseAppendWhen?: (
    path: string,
    data: Uint8Array<ArrayBuffer>,
  ) => boolean;
  private readonly appendResolvers: Array<() => void> = [];

  async read(path: string): Promise<ArrayBuffer | undefined> {
    const data = this.files.get(normalize(path));
    return data === undefined ? undefined : new Uint8Array(data).buffer;
  }

  async write(path: string, data: BufferSource): Promise<void> {
    const normalized = normalize(path);
    for (const dir of parentDirs(normalized)) {
      this.dirs.add(dir);
    }
    this.files.set(normalized, bytes(data));
  }

  async append(path: string, data: BufferSource): Promise<void> {
    const normalized = normalize(path);
    const chunk = bytes(data);
    if (
      this.pauseAppends ||
      (this.pauseAppendWhen?.(normalized, chunk) ?? false)
    ) {
      await new Promise<void>((resolve) => {
        this.appendResolvers.push(resolve);
      });
    }
    const prev = this.files.get(normalized) ?? new Uint8Array();
    const next = new Uint8Array(prev.byteLength + chunk.byteLength);
    next.set(prev);
    next.set(chunk, prev.byteLength);
    await this.write(normalized, next);
  }

  async remove(path: string, opts?: { recursive?: boolean }): Promise<void> {
    const normalized = normalize(path);
    if (this.files.delete(normalized)) {
      return;
    }
    const prefix = `${normalized}/`;
    const childFiles = [...this.files.keys()].filter((key) =>
      key.startsWith(prefix),
    );
    const childDirs = [...this.dirs].filter((dir) => dir.startsWith(prefix));
    if (
      opts?.recursive !== true &&
      (childFiles.length > 0 || childDirs.length > 0)
    ) {
      throw new Error("Directory is not empty");
    }
    this.dirs.delete(normalized);
    childFiles.forEach((key) => this.files.delete(key));
    childDirs.forEach((dir) => this.dirs.delete(dir));
  }

  async list(path: string): Promise<string[]> {
    const normalized = normalize(path);
    const prefix = normalized === "" ? "" : `${normalized}/`;
    if (normalized !== "" && !this.dirs.has(normalized)) {
      return [];
    }
    const names = new Set<string>();
    for (const key of [...this.dirs, ...this.files.keys()]) {
      if (key === normalized || !key.startsWith(prefix)) {
        continue;
      }
      names.add(key.slice(prefix.length).split("/")[0]);
    }
    return [...names].sort();
  }

  async mkdir(path: string): Promise<void> {
    const normalized = normalize(path);
    for (const dir of [...parentDirs(normalized), normalized]) {
      this.dirs.add(dir);
    }
  }

  async lock(path: string): Promise<() => void> {
    const key = normalize(path);
    if (!this.locks.has(key)) {
      this.locks.add(key);
      return this.makeRelease(key);
    }
    return new Promise((resolve) => {
      const waiters = this.lockWaiters.get(key) ?? [];
      waiters.push(resolve);
      this.lockWaiters.set(key, waiters);
    });
  }

  async tryLock(path: string): Promise<(() => void) | undefined> {
    const key = normalize(path);
    if (this.locks.has(key)) return undefined;
    this.locks.add(key);
    return this.makeRelease(key);
  }

  private makeRelease(key: string): () => void {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const waiter = this.lockWaiters.get(key)?.shift();
      if (waiter) waiter(this.makeRelease(key));
      else {
        this.lockWaiters.delete(key);
        this.locks.delete(key);
      }
    };
  }

  releaseOneAppend(): void {
    this.appendResolvers.shift()?.();
  }

  releaseAllAppends(): void {
    while (this.appendResolvers.length > 0) {
      this.releaseOneAppend();
    }
  }
}
