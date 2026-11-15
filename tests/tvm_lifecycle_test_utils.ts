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
