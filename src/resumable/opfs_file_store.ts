export interface OPFSFileStore {
  read(path: string): Promise<ArrayBuffer | undefined>;
  write(path: string, data: BufferSource): Promise<void>;
  append(path: string, data: BufferSource): Promise<void>;
  remove(path: string, opts?: { recursive?: boolean }): Promise<void>;
  list(path: string): Promise<string[]>;
  mkdir(path: string): Promise<void>;
  lock(path: string): Promise<() => void>;
  tryLock(path: string): Promise<(() => void) | undefined>;
}

type DirectoryEntriesHandle = FileSystemDirectoryHandle & {
  keys?: () => AsyncIterable<string>;
  entries?: () => AsyncIterable<[string, FileSystemHandle]>;
};

type WritableFileHandle = FileSystemFileHandle & {
  createWritable?: (options?: {
    keepExistingData?: boolean;
  }) => Promise<FileSystemWritableFileStream>;
};

const LOCK_POLL_INTERVAL_MS = 25; // OPFS exposes no wait primitive for sync access handle locks
const processLocks = new Map<string, Promise<void>>();
const WEB_LOCK_PREFIX = "webllm-resumable:";

export class CrossContextLockUnavailableError extends Error {
  constructor(path: string) {
    super(
      `Cross-context locking is unavailable for resumable storage: ${path}`,
    );
    this.name = "CrossContextLockUnavailableError";
  }
}

type SyncAccessHandleResult =
  | { state: "acquired"; access: FileSystemSyncAccessHandle }
  | { state: "busy" | "unavailable" };

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function splitPath(path: string): string[] {
  if (path.includes("\0")) {
    throw new Error("OPFS paths cannot contain null bytes");
  }
  const parts: string[] = [];
  for (const part of path.split("/")) {
    if (part === "" || part === ".") {
      continue;
    }
    if (part === "..") {
      throw new Error("OPFS paths cannot contain '..'");
    }
    parts.push(part);
  }
  return parts;
}

function filePath(path: string): { dirs: string[]; name: string } {
  const parts = splitPath(path);
  const name = parts.pop();
  if (name === undefined) {
    throw new Error("OPFS file path cannot be the root directory");
  }
  return { dirs: parts, name };
}

function errorName(err: unknown): string | undefined {
  return (err as { name?: string })?.name;
}

function isNotFound(err: unknown): boolean {
  return errorName(err) === "NotFoundError";
}

function isSyncLockBusy(err: unknown): boolean {
  return errorName(err) === "NoModificationAllowedError";
}

function isSyncUnavailable(err: unknown): boolean {
  if (err instanceof TypeError) {
    return true;
  }
  const name = errorName(err);
  return name === "InvalidStateError" || name === "NotSupportedError";
}

async function acquireProcessLock(
  key: string,
  waitForLock: boolean,
): Promise<(() => void) | undefined> {
  const previous = processLocks.get(key);
  if (previous !== undefined && !waitForLock) {
    return undefined;
  }
  let releaseCurrent!: () => void;
  const current = new Promise<void>((resolve) => {
    releaseCurrent = resolve;
  });
  const tail = (previous ?? Promise.resolve()).then(
    () => current,
    () => current,
  );
  processLocks.set(key, tail);
  await previous?.catch(() => undefined);

  let released = false;
  return () => {
    if (released) {
      return;
    }
    released = true;
    releaseCurrent();
    if (processLocks.get(key) === tail) {
      processLocks.delete(key);
    }
  };
}

type WebLockManager = {
  request(
    name: string,
    options: { mode: "exclusive"; ifAvailable?: boolean },
    callback: (lock: unknown | null) => Promise<void> | void,
  ): Promise<void>;
};

function getWebLockManager(): WebLockManager | undefined {
  return (globalThis.navigator as { locks?: WebLockManager } | undefined)
    ?.locks;
}

async function acquireWebLock(
  key: string,
  waitForLock: boolean,
): Promise<(() => void) | undefined> {
  const manager = getWebLockManager();
  if (manager === undefined) {
    return undefined;
  }
  let releaseHold!: () => void;
  const hold = new Promise<void>((resolve) => {
    releaseHold = resolve;
  });
  let settle!: (release: (() => void) | undefined) => void;
  let reject!: (err: unknown) => void;
  let settled = false;
  const acquired = new Promise<(() => void) | undefined>((resolve, rej) => {
    settle = resolve;
    reject = rej;
  });
  void manager
    .request(
      `${WEB_LOCK_PREFIX}${key}`,
      {
        mode: "exclusive",
        ...(waitForLock ? {} : { ifAvailable: true }),
      },
      async (lock) => {
        if (lock === null) {
          settled = true;
          settle(undefined);
          return;
        }
        let released = false;
        settled = true;
        settle(() => {
          if (!released) {
            released = true;
            releaseHold();
          }
        });
        await hold;
      },
    )
    .catch((err) => {
      if (!settled) {
        reject(err);
      }
    });
  return acquired;
}

function getNavigatorOPFSRoot(): Promise<FileSystemDirectoryHandle> {
  const storage = (globalThis.navigator as any)?.storage as
    | { getDirectory?: () => Promise<FileSystemDirectoryHandle> }
    | undefined;
  if (storage?.getDirectory === undefined) {
    throw new Error("OPFS is unavailable in this environment");
  }
  return storage.getDirectory();
}

async function collectNames(dir: FileSystemDirectoryHandle): Promise<string[]> {
  const iterableDir = dir as DirectoryEntriesHandle;
  const names: string[] = [];
  if (iterableDir.keys !== undefined) {
    for await (const name of iterableDir.keys()) {
      names.push(name);
    }
  } else if (iterableDir.entries !== undefined) {
    for await (const [name] of iterableDir.entries()) {
      names.push(name);
    }
  } else {
    throw new Error("OPFS directory listing is unavailable");
  }
  names.sort();
  return names;
}

export class BrowserOPFSFileStore implements OPFSFileStore {
  private readonly root: Promise<FileSystemDirectoryHandle>;

  constructor(
    root?: FileSystemDirectoryHandle | Promise<FileSystemDirectoryHandle>,
  ) {
    this.root =
      root === undefined ? getNavigatorOPFSRoot() : Promise.resolve(root);
    // A streaming request may never be consumed. Observe initialization failure
    // now; storage operations still receive the original rejection when awaited.
    void this.root.catch(() => undefined);
  }

  async read(path: string): Promise<ArrayBuffer | undefined> {
    const file = await this.getFile(path, false);
    if (file === undefined) {
      return undefined;
    }
    return (await file.getFile()).arrayBuffer();
  }

  async write(path: string, data: BufferSource): Promise<void> {
    const file = (await this.getFile(path, true))!;
    const writable = await this.createWritable(file);
    if (writable === undefined) {
      throw new Error("OPFS write requires createWritable");
    }
    try {
      await writable.write(data);
    } catch (err) {
      // Abandon partial writes without masking the original storage failure.
      await writable.abort().catch(() => undefined);
      throw err;
    }
    await writable.close();
  }

  async append(path: string, data: BufferSource): Promise<void> {
    const file = (await this.getFile(path, true))!;
    const writable = await this.createWritable(file, {
      keepExistingData: true,
    });
    if (writable === undefined) {
      throw new Error("OPFS append requires createWritable");
    }
    try {
      const size = (await file.getFile()).size;
      await writable.seek(size);
      await writable.write(data);
    } catch (err) {
      await writable.abort().catch(() => undefined);
      throw err;
    }
    await writable.close();
  }

  async remove(path: string, opts?: { recursive?: boolean }): Promise<void> {
    const parts = splitPath(path);
    if (parts.length === 0) {
      throw new Error("OPFS remove path cannot be the root directory");
    }
    const name = parts.pop()!;
    const dir = await this.getDirectory(parts, false);
    if (dir === undefined) {
      return;
    }
    try {
      await dir.removeEntry(name, { recursive: opts?.recursive === true });
    } catch (err) {
      if (!isNotFound(err)) {
        throw err;
      }
    }
  }

  async list(path: string): Promise<string[]> {
    const dir = await this.getDirectory(splitPath(path), false);
    if (dir === undefined) {
      return [];
    }
    return collectNames(dir);
  }

  async mkdir(path: string): Promise<void> {
    await this.getDirectory(splitPath(path), true);
  }

  async lock(path: string): Promise<() => void> {
    const release = await this.acquireLock(path, true);
    if (release === undefined) {
      throw new Error(`Unable to acquire OPFS lock: ${path}`);
    }
    return release;
  }

  async tryLock(path: string): Promise<(() => void) | undefined> {
    return this.acquireLock(path, false);
  }

  private async acquireLock(
    path: string,
    waitForLock: boolean,
  ): Promise<(() => void) | undefined> {
    const key = splitPath(path).join("/");
    if (key === "") {
      throw new Error("OPFS lock path cannot be the root directory");
    }
    const releaseProcessLock = await acquireProcessLock(key, waitForLock);
    if (releaseProcessLock === undefined) {
      return undefined;
    }
    try {
      if (getWebLockManager() !== undefined) {
        const releaseWebLock = await acquireWebLock(key, waitForLock);
        if (releaseWebLock === undefined) {
          releaseProcessLock();
          return undefined;
        }
        let released = false;
        return () => {
          if (released) {
            return;
          }
          released = true;
          releaseWebLock();
          releaseProcessLock();
        };
      }
      const file = await this.getFile(key, true);
      if (file === undefined) {
        throw new Error(`Unable to create OPFS lock file: ${path}`);
      }
      const accessResult = await this.createSyncAccessHandle(file, waitForLock);
      if (accessResult.state === "busy") {
        releaseProcessLock();
        return undefined;
      }
      if (accessResult.state === "unavailable") {
        releaseProcessLock();
        throw new CrossContextLockUnavailableError(path);
      }
      if (accessResult.state !== "acquired") {
        releaseProcessLock();
        return undefined;
      }
      const access = accessResult.access;
      let released = false;
      return () => {
        if (released) {
          return;
        }
        released = true;
        access?.close();
        releaseProcessLock();
      };
    } catch (err) {
      releaseProcessLock();
      throw err;
    }
  }

  private async getDirectory(
    parts: string[],
    create: boolean,
  ): Promise<FileSystemDirectoryHandle | undefined> {
    let dir = await this.root;
    for (const part of parts) {
      try {
        dir = await dir.getDirectoryHandle(part, { create });
      } catch (err) {
        if (!create && isNotFound(err)) {
          return undefined;
        }
        throw err;
      }
    }
    return dir;
  }

  private async getFile(
    path: string,
    create: boolean,
  ): Promise<FileSystemFileHandle | undefined> {
    const { dirs, name } = filePath(path);
    const dir = await this.getDirectory(dirs, create);
    if (dir === undefined) {
      return undefined;
    }
    try {
      return await dir.getFileHandle(name, { create });
    } catch (err) {
      if (!create && isNotFound(err)) {
        return undefined;
      }
      throw err;
    }
  }

  private async createWritable(
    file: FileSystemFileHandle,
    opts?: { keepExistingData?: boolean },
  ): Promise<FileSystemWritableFileStream | undefined> {
    const createWritable = (file as WritableFileHandle).createWritable;
    if (createWritable === undefined) {
      return undefined;
    }
    return createWritable.call(file, opts);
  }

  private async createSyncAccessHandle(
    file: FileSystemFileHandle,
    waitForLock: boolean,
  ): Promise<SyncAccessHandleResult> {
    const create = file.createSyncAccessHandle;
    while (true) {
      try {
        return { state: "acquired", access: await create.call(file) };
      } catch (err) {
        if (isSyncLockBusy(err)) {
          if (waitForLock) {
            await sleep(LOCK_POLL_INTERVAL_MS);
            continue;
          }
          return { state: "busy" };
        }
        if (isSyncUnavailable(err)) {
          return { state: "unavailable" };
        }
        throw err;
      }
    }
  }
}

export function createOPFSFileStore(
  root?: FileSystemDirectoryHandle | Promise<FileSystemDirectoryHandle>,
): OPFSFileStore {
  return new BrowserOPFSFileStore(root);
}
