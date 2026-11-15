import {
  BrowserOPFSFileStore,
  CrossContextLockUnavailableError,
} from "../src/resumable/opfs_file_store";
import { jest, test, expect } from "@jest/globals";

function domError(name: string): Error {
  const err = new Error(name);
  err.name = name;
  return err;
}

function bytes(data: BufferSource): Uint8Array<ArrayBuffer> {
  const view = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  return new Uint8Array(view);
}

function writeAt(
  target: Uint8Array,
  source: BufferSource,
  at: number,
): Uint8Array<ArrayBuffer> {
  const chunk = bytes(source);
  const next = new Uint8Array(
    Math.max(target.byteLength, at + chunk.byteLength),
  );
  next.set(target);
  next.set(chunk, at);
  return next;
}

class MemoryFile {
  data = new Uint8Array();
  locked = false;
}

class MemorySyncAccessHandle {
  constructor(private readonly file: MemoryFile) {}

  close(): void {
    this.file.locked = false;
  }
}

class MemoryWritableFileStream {
  private cursor = 0;

  constructor(
    private readonly file: MemoryFile,
    opts?: { keepExistingData?: boolean },
  ) {
    if (opts?.keepExistingData !== true) {
      this.file.data = new Uint8Array();
    }
  }

  async seek(position: number): Promise<void> {
    this.cursor = position;
  }

  async write(data: BufferSource): Promise<void> {
    const chunk = bytes(data);
    this.file.data = writeAt(this.file.data, chunk, this.cursor);
    this.cursor += chunk.byteLength;
  }

  async close(): Promise<void> {}
}

type MemoryEntry = MemoryDirectoryHandle | MemoryFileHandle;

class MemoryFileHandle {
  readonly kind = "file";

  constructor(
    private readonly file: MemoryFile,
    private readonly syncAccess: boolean,
  ) {}

  async getFile(): Promise<File> {
    const data = bytes(this.file.data);
    return {
      size: data.byteLength,
      arrayBuffer: async () => data.buffer.slice(0),
    } as File;
  }

  async createWritable(opts?: {
    keepExistingData?: boolean;
  }): Promise<FileSystemWritableFileStream> {
    return new MemoryWritableFileStream(
      this.file,
      opts,
    ) as unknown as FileSystemWritableFileStream;
  }

  async createSyncAccessHandle(): Promise<FileSystemSyncAccessHandle> {
    if (!this.syncAccess) {
      throw new TypeError("sync access handles are unavailable");
    }
    if (this.file.locked) {
      throw domError("NoModificationAllowedError");
    }
    this.file.locked = true;
    return new MemorySyncAccessHandle(
      this.file,
    ) as unknown as FileSystemSyncAccessHandle;
  }
}

class MemoryDirectoryHandle {
  readonly kind = "directory";
  private readonly children = new Map<string, MemoryEntry>();

  constructor(private readonly syncAccess = true) {}

  async getDirectoryHandle(
    name: string,
    opts?: { create?: boolean },
  ): Promise<FileSystemDirectoryHandle> {
    const existing = this.children.get(name);
    if (existing instanceof MemoryDirectoryHandle) {
      return existing as unknown as FileSystemDirectoryHandle;
    }
    if (existing !== undefined) {
      throw domError("TypeMismatchError");
    }
    if (opts?.create === true) {
      const dir = new MemoryDirectoryHandle(this.syncAccess);
      this.children.set(name, dir);
      return dir as unknown as FileSystemDirectoryHandle;
    }
    throw domError("NotFoundError");
  }

  async getFileHandle(
    name: string,
    opts?: { create?: boolean },
  ): Promise<FileSystemFileHandle> {
    const existing = this.children.get(name);
    if (existing instanceof MemoryFileHandle) {
      return existing as unknown as FileSystemFileHandle;
    }
    if (existing !== undefined) {
      throw domError("TypeMismatchError");
    }
    if (opts?.create === true) {
      const file = new MemoryFileHandle(new MemoryFile(), this.syncAccess);
      this.children.set(name, file);
      return file as unknown as FileSystemFileHandle;
    }
    throw domError("NotFoundError");
  }

  async removeEntry(
    name: string,
    opts?: { recursive?: boolean },
  ): Promise<void> {
    const existing = this.children.get(name);
    if (existing === undefined) {
      throw domError("NotFoundError");
    }
    if (
      existing instanceof MemoryDirectoryHandle &&
      opts?.recursive !== true &&
      existing.children.size !== 0
    ) {
      throw domError("InvalidModificationError");
    }
    this.children.delete(name);
  }

  async *keys(): AsyncIterable<string> {
    for (const key of this.children.keys()) {
      yield key;
    }
  }
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function makeStore(syncAccess = true): BrowserOPFSFileStore {
  return new BrowserOPFSFileStore(
    new MemoryDirectoryHandle(
      syncAccess,
    ) as unknown as FileSystemDirectoryHandle,
  );
}

function text(data: ArrayBuffer | undefined): string | undefined {
  return data === undefined ? undefined : decoder.decode(data);
}

function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test("OPFS file store reads and writes path-based files", async () => {
  const store = makeStore();

  expect(await store.read("/missing.bin")).toBeUndefined();
  await store.write("/a/b/file.txt", encoder.encode("hello"));

  expect(text(await store.read("a/b/file.txt"))).toBe("hello");
});

test("OPFS file store appends to existing and new files", async () => {
  const store = makeStore();

  await store.write("log.txt", encoder.encode("hello"));
  await store.append("log.txt", encoder.encode(" world"));
  await store.append("new-log.txt", encoder.encode("first"));

  expect(text(await store.read("log.txt"))).toBe("hello world");
  expect(text(await store.read("new-log.txt"))).toBe("first");
});

test("append aborts its writable when reading the file size fails", async () => {
  const abort = jest.fn(async () => undefined);
  const store = new BrowserOPFSFileStore({
    getFileHandle: async () => ({
      createWritable: async () => ({ abort }),
      getFile: async () => {
        throw new Error("file size unavailable");
      },
    }),
  } as unknown as FileSystemDirectoryHandle);
  await expect(
    store.append("journal.bin", new Uint8Array([1])),
  ).rejects.toThrow("file size unavailable");
  expect(abort).toHaveBeenCalledTimes(1);
});

test.each(["write", "append"] as const)(
  "%s preserves the original write error even if abort fails",
  async (operation) => {
    const error = domError("QuotaExceededError");
    const abort = jest.fn(async () => {
      throw new Error("stream already errored");
    });
    const close = jest.fn(async () => {
      throw new Error("cannot close errored stream");
    });
    const store = new BrowserOPFSFileStore({
      getFileHandle: async () => ({
        getFile: async () => ({ size: 0 }),
        createWritable: async () => ({
          seek: async () => undefined,
          write: async () => {
            throw error;
          },
          abort,
          close,
        }),
      }),
    } as unknown as FileSystemDirectoryHandle);
    await expect(
      store[operation]("journal.bin", new Uint8Array([1])),
    ).rejects.toBe(error);
    expect(abort).toHaveBeenCalledTimes(1);
    expect(close).not.toHaveBeenCalled();
  },
);

test.each(["write", "append"] as const)(
  "%s propagates a commit failure from close",
  async (operation) => {
    const error = domError("QuotaExceededError");
    const store = new BrowserOPFSFileStore({
      getFileHandle: async () => ({
        getFile: async () => ({ size: 0 }),
        createWritable: async () => ({
          seek: async () => undefined,
          write: async () => undefined,
          close: async () => {
            throw error;
          },
        }),
      }),
    } as unknown as FileSystemDirectoryHandle);
    await expect(
      store[operation]("journal.bin", new Uint8Array([1])),
    ).rejects.toBe(error);
  },
);

test("an unused store observes root rejection and still reports it on access", async () => {
  const store = new BrowserOPFSFileStore(
    Promise.reject(new Error("OPFS denied")),
  );
  await tick();
  await expect(store.read("journal.bin")).rejects.toThrow("OPFS denied");
});

test("OPFS file store lists, creates, and removes directories", async () => {
  const store = makeStore();

  await store.mkdir("root/b");
  await store.write("root/c.txt", encoder.encode("c"));
  await store.write("root/a.txt", encoder.encode("a"));

  expect(await store.list("root")).toEqual(["a.txt", "b", "c.txt"]);
  await store.remove("root/a.txt");
  await store.remove("root/b", { recursive: true });
  await store.remove("root/missing");

  expect(await store.list("root")).toEqual(["c.txt"]);
  await store.remove("root", { recursive: true });
  expect(await store.list("root")).toEqual([]);
});

test.each([true, false])(
  "OPFS file store lock serializes same-path contenders with cross-context backend=%s",
  async (syncAccess) => {
    const previousNavigator = Object.getOwnPropertyDescriptor(
      globalThis,
      "navigator",
    );
    const webLockRequest = jest.fn(
      async (
        _name: string,
        _options: unknown,
        callback: (lock: object) => Promise<void>,
      ) => callback({}),
    );
    if (!syncAccess) {
      Object.defineProperty(globalThis, "navigator", {
        configurable: true,
        value: { locks: { request: webLockRequest } },
      });
    }
    const store = makeStore(syncAccess);

    try {
      const releaseFirst = await store.lock("locks/session");
      let secondAcquired = false;
      const secondLock = store.lock("locks/session").then((release) => {
        secondAcquired = true;
        return release;
      });

      await tick();
      expect(secondAcquired).toBe(false);
      expect(await store.tryLock("locks/session")).toBeUndefined();

      releaseFirst();
      const releaseSecond = await secondLock;
      expect(secondAcquired).toBe(true);
      releaseSecond();

      const releaseThird = await store.tryLock("locks/session");
      expect(releaseThird).toBeDefined();
      releaseThird!();
      if (!syncAccess) {
        expect(webLockRequest).toHaveBeenCalled();
      }
    } finally {
      if (previousNavigator === undefined) {
        delete (globalThis as any).navigator;
      } else {
        Object.defineProperty(globalThis, "navigator", previousNavigator);
      }
    }
  },
);

test("OPFS locking rejects process-only fallback", async () => {
  const previousNavigator = Object.getOwnPropertyDescriptor(
    globalThis,
    "navigator",
  );
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {},
  });
  try {
    await expect(
      makeStore(false).tryLock("locks/session"),
    ).rejects.toBeInstanceOf(CrossContextLockUnavailableError);
  } finally {
    if (previousNavigator === undefined) {
      delete (globalThis as any).navigator;
    } else {
      Object.defineProperty(globalThis, "navigator", previousNavigator);
    }
  }
});
