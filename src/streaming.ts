export interface StreamGenerationState {
  id: string;
  created: number;
  prevMessageLength: number;
  prevLogprobLength: number;
}

export interface StreamContinuationOptions {
  emitCurrent?: boolean;
  skipEmptyDelta?: boolean;
  completionTokenOffset?: number;
  beforeFinalChunk?: () => Promise<void>;
}

export function onceAsync(action: () => Promise<void>): () => Promise<void> {
  let promise: Promise<void> | undefined;
  return () => {
    promise ??= action();
    return promise;
  };
}

/** Ensure return() releases resources even before an async generator starts. */
export function managedAsyncIterable<T>(
  source: AsyncGenerator<T, void, void>,
  cleanup: () => Promise<void>,
): AsyncIterable<T> {
  let closed = false;
  const close = onceAsync(async () => {
    closed = true;
    await cleanup();
  });
  const iterator: AsyncIterableIterator<T> = {
    async next(): Promise<IteratorResult<T, void>> {
      if (closed) {
        return { done: true, value: undefined };
      }
      try {
        const result = await source.next();
        if (result.done) {
          await close();
        }
        return result;
      } catch (err) {
        await close();
        throw err;
      }
    },
    async return(): Promise<IteratorResult<T, void>> {
      try {
        await source.return(undefined);
      } finally {
        await close();
      }
      return { done: true, value: undefined };
    },
    async throw(err?: unknown): Promise<IteratorResult<T, void>> {
      try {
        if (source.throw !== undefined) {
          return await source.throw(err);
        }
        throw err;
      } finally {
        await close();
      }
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
  return iterator;
}

/** Defer constructing a resource-owning stream until its first operation. */
export function lazyAsyncIterable<T>(
  factory: () => Promise<AsyncIterable<T>>,
): AsyncIterable<T> {
  let sourcePromise: Promise<AsyncIterator<T>> | undefined;
  let closed = false;
  const getSource = (): Promise<AsyncIterator<T>> => {
    sourcePromise ??= factory().then((source) =>
      source[Symbol.asyncIterator](),
    );
    return sourcePromise;
  };
  const iterator: AsyncIterableIterator<T> = {
    async next(): Promise<IteratorResult<T, void>> {
      if (closed) {
        return { done: true, value: undefined };
      }
      try {
        const result = await (await getSource()).next();
        closed = result.done === true;
        return result;
      } catch (err) {
        closed = true;
        throw err;
      }
    },
    async return(): Promise<IteratorResult<T, void>> {
      closed = true;
      if (sourcePromise !== undefined) {
        const source = await sourcePromise;
        await source.return?.();
      }
      return { done: true, value: undefined };
    },
    async throw(err?: unknown): Promise<IteratorResult<T, void>> {
      closed = true;
      if (sourcePromise !== undefined) {
        const source = await sourcePromise;
        if (source.throw !== undefined) {
          return source.throw(err);
        }
      }
      throw err;
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
  return iterator;
}

export function isAsyncIterable<T>(value: unknown): value is AsyncIterable<T> {
  return (
    typeof value === "object" && value !== null && Symbol.asyncIterator in value
  );
}
