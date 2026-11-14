/** Host-side graph scheduling with injected backends; no WebGPU backend is supplied here. */
import {
  CompiledProgramArtifactV2,
  GraphEndpoint,
  GraphPort,
  ModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  parseModelPackageManifestV2,
  planTaskOutputs,
  validateGraphArtifactPair,
} from "./artifact_manifest_v2";

export interface GraphTensor {
  /** float16 payloads contain IEEE 754 binary16 bits in a Uint16Array. */
  data: Int32Array | Uint16Array | Float32Array | Uint8Array;
  shape: number[];
}
export type GraphInputEvent =
  | { kind: "data"; port: string; tensor: GraphTensor }
  | { kind: "end"; port: string };
export interface GraphComponentOutput {
  port: string;
  tensor: GraphTensor;
}
export interface GraphComponentBackend {
  /** Consume the merged input stream through EOF, then yield any final flush outputs. */
  run(
    inputs: AsyncIterable<GraphInputEvent>,
    signal: AbortSignal,
  ): AsyncIterable<GraphComponentOutput>;
  /** Release all request-local state; called exactly once, including on cancellation. */
  /** Must settle cooperatively; the host cannot forcibly release backend resources. */
  dispose(): Promise<void> | void;
}
export interface GraphComponentFactory {
  /** Create isolated request state. A failed create must clean up its own partial allocation. */
  create(
    id: string,
    component: CompiledProgramArtifactV2["components"][string],
    signal: AbortSignal,
  ): Promise<GraphComponentBackend> | GraphComponentBackend;
}
export interface GraphExecutionOptions {
  model: ModelPackageManifestV2;
  compiled: CompiledProgramArtifactV2;
  task: string;
  inputs: Record<string, AsyncIterable<GraphTensor>>;
  factory: GraphComponentFactory;
  outputs?: string[];
  queueCapacity?: number;
  maxTensorBytes?: number;
  signal?: AbortSignal;
}
export interface ArtifactGraphExecution {
  /** Consume all selected outputs concurrently. Returning early cancels the entire request. */
  outputs: Record<string, AsyncIterable<GraphTensor>>;
  /** Resolves after final output drain and state disposal; rejects on any request failure. */
  done: Promise<void>;
  abort(reason?: Error): void;
}

function abortError(): Error {
  const error = new Error("Artifact graph execution aborted");
  error.name = "AbortError";
  return error;
}
function asError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

/** One consumer, bounded buffered items, FIFO pending producers. */
class Queue<T> implements AsyncIterable<T> {
  private values: T[] = [];
  private readers: {
    resolve: (v: IteratorResult<T>) => void;
    reject: (e: Error) => void;
  }[] = [];
  private writers: {
    value: T;
    resolve: () => void;
    reject: (e: Error) => void;
  }[] = [];
  private closed = false;
  private failure?: Error;
  private claimed = false;
  private resolveDrain!: () => void;
  private rejectDrain!: (e: Error) => void;
  readonly drained: Promise<void>;
  constructor(
    private capacity: number,
    private onReturn: () => void,
  ) {
    this.drained = new Promise((resolve, reject) => {
      this.resolveDrain = resolve;
      this.rejectDrain = reject;
    });
    void this.drained.catch(() => {});
  }
  get complete(): boolean {
    return this.closed && this.values.length === 0 && this.writers.length === 0;
  }
  push(value: T): Promise<void> {
    if (this.failure) return Promise.reject(this.failure);
    if (this.closed)
      return Promise.reject(new Error("Write to closed graph queue"));
    const reader = this.readers.shift();
    if (reader) {
      reader.resolve({ done: false, value });
      return Promise.resolve();
    }
    if (this.values.length < this.capacity) {
      this.values.push(value);
      return Promise.resolve();
    }
    return new Promise((resolve, reject) =>
      this.writers.push({ value, resolve, reject }),
    );
  }
  close(): void {
    this.closed = true;
    this.finishDrain();
  }
  fail(error: Error): void {
    if (this.failure) return;
    this.failure = error;
    this.values = [];
    for (const r of this.readers.splice(0)) r.reject(error);
    for (const w of this.writers.splice(0)) w.reject(error);
    this.rejectDrain(error);
  }
  private finishDrain(): void {
    if (!this.complete || this.failure) return;
    for (const r of this.readers.splice(0))
      r.resolve({ done: true, value: undefined });
    this.resolveDrain();
  }
  private next(): Promise<IteratorResult<T>> {
    if (this.failure) return Promise.reject(this.failure);
    if (this.values.length) {
      const value = this.values.shift()!;
      const writer = this.writers.shift();
      if (writer) {
        this.values.push(writer.value);
        writer.resolve();
      }
      this.finishDrain();
      return Promise.resolve({ done: false, value });
    }
    if (this.closed) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve, reject) =>
      this.readers.push({ resolve, reject }),
    );
  }
  [Symbol.asyncIterator](): AsyncIterator<T> {
    if (this.claimed) throw new Error("Graph streams support one consumer");
    this.claimed = true;
    return {
      next: () => this.next(),
      return: async () => {
        if (!this.complete) this.onReturn();
        return { done: true, value: undefined };
      },
    };
  }
}

function tensorCopy(
  value: GraphTensor,
  spec: GraphPort,
  maxBytes: number,
): GraphTensor {
  const constructors = {
    int32: Int32Array,
    float16: Uint16Array,
    float32: Float32Array,
    uint8: Uint8Array,
  };
  const tag = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(Int32Array.prototype),
    Symbol.toStringTag,
  )!.get!;
  let ordinaryBuffer = false;
  try {
    Object.getOwnPropertyDescriptor(
      ArrayBuffer.prototype,
      "byteLength",
    )!.get!.call(value?.data?.buffer);
    ordinaryBuffer = true;
  } catch {
    /* Shared or non-buffer storage is unsupported. */
  }
  if (
    !value ||
    !ArrayBuffer.isView(value.data) ||
    tag.call(value.data) !== constructors[spec.dtype].name ||
    !ordinaryBuffer
  )
    throw new Error("Graph tensor dtype or buffer mismatch");
  if (!Array.isArray(value.shape) || value.shape.length !== spec.shape.length)
    throw new Error("Graph tensor rank mismatch");
  const symbols = new Map<string, number>();
  let size = 1;
  for (let i = 0; i < value.shape.length; i++) {
    const dim = value.shape[i],
      expected = spec.shape[i];
    if (!Number.isSafeInteger(dim) || dim < 0)
      throw new Error("Invalid graph tensor dimension");
    if (
      typeof expected === "number"
        ? dim !== expected
        : symbols.has(expected) && symbols.get(expected) !== dim
    )
      throw new Error("Graph tensor shape mismatch");
    if (typeof expected === "string") symbols.set(expected, dim);
    size *= dim;
    if (!Number.isSafeInteger(size))
      throw new Error("Graph tensor size overflow");
  }
  if (size !== value.data.length || value.data.byteLength > maxBytes)
    throw new Error("Graph tensor payload size mismatch or limit exceeded");
  return {
    data: new constructors[spec.dtype](value.data),
    shape: [...value.shape],
  };
}

/** Stop waiting even if a foreign iterator ignores cancellation; request its cleanup best-effort. */
async function pump<T>(
  source: AsyncIterable<T>,
  signal: AbortSignal,
  receive: (value: T) => Promise<void>,
): Promise<void> {
  const iterator = source[Symbol.asyncIterator]();
  let ended = false;
  let rejectAbort!: (reason: Error) => void;
  const cancelled = new Promise<never>((_, reject) => {
    rejectAbort = reject;
  });
  void cancelled.catch(() => {});
  const onAbort = () => rejectAbort(asError(signal.reason ?? abortError()));
  signal.addEventListener("abort", onAbort, { once: true });
  if (signal.aborted) onAbort();
  try {
    while (true) {
      const next = await Promise.race([iterator.next(), cancelled]);
      if (next.done) {
        ended = true;
        return;
      }
      await receive(next.value);
    }
  } finally {
    signal.removeEventListener("abort", onAbort);
    if (!ended && iterator.return) {
      try {
        void Promise.resolve(iterator.return()).catch(() => {});
      } catch {
        /* Preserve the request failure. */
      }
    }
  }
}

/** Cancellation can reject setup before a factory settles; dispose any eventual late result. */
async function createBackend(
  create: () => Promise<GraphComponentBackend> | GraphComponentBackend,
  signal: AbortSignal,
): Promise<GraphComponentBackend> {
  if (signal.aborted) throw asError(signal.reason ?? abortError());
  let abandoned = false;
  let late: GraphComponentBackend | undefined;
  const pending = Promise.resolve().then(create);
  const disposeLate = (backend: GraphComponentBackend) => {
    void Promise.resolve()
      .then(() => backend.dispose())
      .catch(() => {});
  };
  void pending.then(
    (backend) => {
      late = backend;
      if (abandoned) disposeLate(backend);
    },
    () => {},
  );
  let rejectAbort!: (error: Error) => void;
  const cancelled = new Promise<never>((_, reject) => {
    rejectAbort = reject;
  });
  const onAbort = () => rejectAbort(asError(signal.reason ?? abortError()));
  signal.addEventListener("abort", onAbort, { once: true });
  if (signal.aborted) onAbort();
  try {
    return await Promise.race([pending, cancelled]);
  } catch (error) {
    abandoned = true;
    if (late) disposeLate(late);
    throw error;
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}

/** Validates a v2 pair, then schedules only the selected output dependency closure. */
export async function createArtifactGraphExecution(
  options: GraphExecutionOptions,
): Promise<ArtifactGraphExecution> {
  const {
    task: taskName,
    factory,
    queueCapacity,
    maxTensorBytes,
    signal,
  } = options;
  const selectedRequest =
    options.outputs === undefined ? undefined : [...options.outputs];
  const suppliedInputs = options.inputs;
  const inputStreams = { ...suppliedInputs };
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const plan = await planTaskOutputs(model, taskName, selectedRequest);
  const task = model.tasks[taskName];
  const selected =
    selectedRequest ??
    Object.keys(task.outputs).filter((id) => !task.outputs[id].optional);
  const capacity = queueCapacity ?? 2,
    maxBytes = maxTensorBytes ?? 64 * 1024 * 1024;
  if (
    !Number.isSafeInteger(capacity) ||
    capacity < 1 ||
    capacity > 1024 ||
    !Number.isSafeInteger(maxBytes) ||
    maxBytes < 1
  )
    throw new Error("Invalid graph queue or tensor limit");
  if (
    !suppliedInputs ||
    Object.keys(inputStreams).length !== plan.inputs.length ||
    plan.inputs.some(
      (id) =>
        !Object.hasOwn(inputStreams, id) ||
        typeof inputStreams[id]?.[Symbol.asyncIterator] !== "function",
    )
  )
    throw new Error("Graph inputs do not match selected task inputs");
  const controller = new AbortController();
  const queues: Queue<unknown>[] = [];
  let failure: Error | undefined;
  let finished = false;
  const stop = (reason: Error = abortError()) => {
    if (failure || finished) return;
    failure = reason;
    controller.abort(reason);
    for (const q of queues) q.fail(reason);
  };
  const externalAbort = () => stop(asError(signal?.reason ?? abortError()));
  signal?.addEventListener("abort", externalAbort, { once: true });
  if (signal?.aborted) externalAbort();
  function queue<T>(cancelOnReturn = false): Queue<T> {
    const q = new Queue<T>(capacity, () => {
      if (cancelOnReturn) stop();
    });
    queues.push(q as Queue<unknown>);
    if (failure) q.fail(failure);
    return q;
  }
  const inboxes = new Map(
    plan.components.map((id) => [id, queue<GraphInputEvent>()]),
  );
  const publicOutputs = new Map(
    selected.map((id) => [id, queue<GraphTensor>(true)]),
  );
  const remaining = new Map(
    plan.components.map((id) => [
      id,
      Object.keys(model.components[id].inputs).length,
    ]),
  );
  for (const id of plan.components)
    if (remaining.get(id) === 0) inboxes.get(id)!.close();
  const backends = new Map<string, GraphComponentBackend>();
  async function dispose(): Promise<void> {
    const results = await Promise.allSettled(
      [...backends.values()].map(async (b) => b.dispose()),
    );
    signal?.removeEventListener("abort", externalAbort);
    const rejected = results.find((r) => r.status === "rejected");
    if (rejected?.status === "rejected" && !failure)
      stop(asError(rejected.reason));
  }
  try {
    for (const id of plan.components) {
      if (failure) throw failure;
      const backend = await createBackend(
        () => factory.create(id, compiled.components[id], controller.signal),
        controller.signal,
      );
      backends.set(id, backend);
    }
    if (failure) throw failure;
  } catch (error) {
    stop(asError(error));
    await dispose();
    throw failure;
  }

  async function send(
    target: GraphEndpoint,
    tensor: GraphTensor,
  ): Promise<void> {
    const inbox = inboxes.get(target.component);
    if (inbox) await inbox.push({ kind: "data", port: target.port, tensor });
  }
  async function end(target: GraphEndpoint): Promise<void> {
    const inbox = inboxes.get(target.component);
    if (!inbox) return;
    await inbox.push({ kind: "end", port: target.port });
    const left = remaining.get(target.component)! - 1;
    remaining.set(target.component, left);
    if (left === 0) inbox.close();
  }
  async function publish(
    source: GraphEndpoint,
    value: GraphTensor,
  ): Promise<void> {
    const spec = model.components[source.component].outputs[source.port];
    const edges = task.edges.filter(
      (e) =>
        e.source.component === source.component &&
        e.source.port === source.port &&
        inboxes.has(e.target.component),
    );
    const outputs = selected.filter(
      (id) =>
        task.outputs[id].source.component === source.component &&
        task.outputs[id].source.port === source.port,
    );
    // Snapshot and clone every fan-out destination before yielding, so buffers are never shared.
    const copies = [
      ...edges.map(() => tensorCopy(value, spec, maxBytes)),
      ...outputs.map(() => tensorCopy(value, spec, maxBytes)),
    ];
    if (!copies.length) tensorCopy(value, spec, maxBytes);
    await Promise.all([
      ...edges.map((e, i) => send(e.target, copies[i])),
      ...outputs.map((id, i) =>
        publicOutputs.get(id)!.push(copies[edges.length + i]),
      ),
    ]);
  }
  const jobs: Promise<void>[] = [];
  for (const id of plan.components)
    jobs.push(
      (async () => {
        const spec = model.components[id],
          counts = new Map<string, number>();
        await pump(
          backends.get(id)!.run(inboxes.get(id)!, controller.signal),
          controller.signal,
          async (output) => {
            if (!output || !Object.hasOwn(spec.outputs, output.port))
              throw new Error(`Unknown graph output port: ${id}`);
            const count = (counts.get(output.port) ?? 0) + 1;
            counts.set(output.port, count);
            if (spec.outputs[output.port].delivery === "value" && count > 1)
              throw new Error("Value output emitted more than once");
            await publish({ component: id, port: output.port }, output.tensor);
          },
        );
        if (!inboxes.get(id)!.complete)
          throw new Error(`Component ended before draining inputs: ${id}`);
        for (const [port, definition] of Object.entries(spec.outputs))
          if (definition.delivery === "value" && counts.get(port) !== 1)
            throw new Error(`Missing value output: ${id}.${port}`);
        await Promise.all(
          task.edges
            .filter((e) => e.source.component === id)
            .map((e) => end(e.target)),
        );
        for (const name of selected)
          if (task.outputs[name].source.component === id)
            publicOutputs.get(name)!.close();
      })(),
    );
  for (const id of plan.inputs)
    jobs.push(
      (async () => {
        const input = task.inputs[id],
          targets = input.targets.filter((e) => inboxes.has(e.component));
        let count = 0;
        await pump(inputStreams[id], controller.signal, async (value) => {
          count++;
          if (input.spec.delivery === "value" && count > 1)
            throw new Error("Value input emitted more than once");
          const copies = targets.map(() =>
            tensorCopy(value, input.spec, maxBytes),
          );
          await Promise.all(targets.map((e, i) => send(e, copies[i])));
        });
        if (input.spec.delivery === "value" && count !== 1)
          throw new Error(`Missing value input: ${id}`);
        await Promise.all(targets.map(end));
      })(),
    );
  // Attach rejection handlers immediately, before returning streams to the caller.
  const done = (async () => {
    try {
      await Promise.all(
        jobs.map((job) =>
          job.catch((e) => {
            stop(asError(e));
            throw e;
          }),
        ),
      );
      await Promise.all([...publicOutputs.values()].map((q) => q.drained));
    } catch (error) {
      stop(asError(error));
    }
    await Promise.allSettled(jobs);
    await dispose();
    finished = true;
    if (failure) throw failure;
  })();
  void done.catch(() => {});
  return { outputs: Object.fromEntries(publicOutputs), done, abort: stop };
}
