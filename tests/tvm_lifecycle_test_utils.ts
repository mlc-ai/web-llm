import type { GraphInputEvent } from "../src/artifact_graph_executor";

/** Exercise an actual factory while its mocked GPU queue is still draining. */
export async function verifyPendingTVMDisposal(
  state: any,
  id: string,
  input: AsyncIterable<GraphInputEvent>,
  failSync = false,
) {
  const controller = new AbortController();
  const component = state.component ?? state.compiled.components[id];
  const backend =
    state.b ?? (await state.factory.create(id, component, controller.signal));
  const resource = state.cache ?? state.vm;
  let enter!: () => void,
    release!: () => void,
    reject!: (reason: Error) => void;
  const started = new Promise<void>((resolve) => {
    enter = resolve;
  });
  const pending = new Promise<void>((resolve, fail) => {
    release = resolve;
    reject = fail;
  });
  state.device.sync.mockImplementation(() => {
    enter();
    return pending;
  });
  const next = backend
    .run(input, controller.signal)
    [Symbol.asyncIterator]()
    .next();
  const rejected = expect(next).rejects.toThrow(
    failSync ? "original sync failure" : /disposed|aborted/,
  );
  await started;
  controller.abort();
  let complete = false;
  const disposal = Promise.resolve(backend.dispose()).then(() => {
    complete = true;
  });
  await Promise.resolve();
  expect(complete).toBe(false);
  expect(resource.dispose).not.toHaveBeenCalled();
  await expect(
    state.factory.create(id, component, new AbortController().signal),
  ).rejects.toThrow(/already in use/);
  if (failSync) reject(Error("original sync failure"));
  else release();
  await rejected;
  await disposal;
  expect(resource.dispose).toHaveBeenCalledTimes(1);
  state.device.sync.mockResolvedValue(undefined);
  const nextBackend = await state.factory.create(
    id,
    component,
    new AbortController().signal,
  );
  await nextBackend.dispose();
}

/** A scope/copy error must not free memory while WebGPU can still write to it. */
export async function verifyExceptionalTVMReadback(
  state: any,
  id: string,
  input: AsyncIterable<GraphInputEvent>,
  kind: "scope" | "copy" | "microtask",
) {
  const controller = new AbortController();
  const signal = controller.signal;
  const component = state.component ?? state.compiled.components[id];
  const backend =
    state.b ?? (await state.factory.create(id, component, signal));
  const resource = state.cache ?? state.vm;
  let cpu: any,
    copied = false,
    enter!: () => void,
    release!: () => void;
  const started = new Promise<void>((resolve) => {
    enter = resolve;
  });
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  state.device.sync.mockImplementation(() => {
    enter();
    return pending;
  });
  const cpuDevice = state.runtime.cpu();
  state.runtime.cpu = () => cpuDevice;
  const empty = state.runtime.empty;
  state.runtime.empty = (...args: any[]) => {
    const value = empty(...args);
    if (args[2] === state.runtime.cpu()) {
      cpu = value;
      if (!jest.isMockFunction(value.dispose))
        value.dispose = jest.fn(value.dispose.bind(value));
      const copy = value.copyFrom.bind(value);
      value.copyFrom = (source: any) => {
        const result = copy(source);
        copied = true;
        if (kind === "copy") throw Error("original copy failure");
        return result;
      };
    }
    return value;
  };
  let disposal: Promise<void> | undefined;
  let complete = false;
  const dispose = () => {
    controller.abort();
    disposal = Promise.resolve(backend.dispose()).then(() => {
      complete = true;
    });
  };
  const end = state.runtime.endScope;
  state.runtime.endScope = () => {
    end();
    if (copied && kind === "scope") throw Error("original scope failure");
    if (copied && kind === "microtask") queueMicrotask(dispose);
  };
  const next = backend.run(input, signal)[Symbol.asyncIterator]().next();
  const rejected = expect(next).rejects.toThrow(
    kind === "microtask" ? /disposed|aborted/ : `original ${kind} failure`,
  );
  await started;
  expect(cpu.dispose).not.toHaveBeenCalled();
  expect(resource.dispose).not.toHaveBeenCalled();
  if (!disposal) dispose();
  await Promise.resolve();
  expect(complete).toBe(false);
  expect(cpu.dispose).not.toHaveBeenCalled();
  release();
  await rejected;
  await disposal;
  expect(cpu.dispose).toHaveBeenCalledTimes(1);
  expect(resource.dispose).toHaveBeenCalledTimes(1);
}
