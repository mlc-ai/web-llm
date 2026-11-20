/** Track submitted GPU work before releasing a request's TVM objects. */
import type * as tvmjs from "@mlc-ai/web-runtime";

export function createTVMRequestDrain(device: tvmjs.DLDevice) {
  let pending: Promise<void> | undefined;
  const operations = new Set<Promise<void>>();
  return {
    run<T>(operation: () => Promise<T>): Promise<T> {
      const work = operation();
      const settled = work.then(
        () => {},
        () => {},
      );
      operations.add(settled);
      void settled.then(() => operations.delete(settled));
      return work;
    },
    sync(): Promise<void> {
      const work = device.sync();
      // Drain observes completion, while the caller still receives the original error.
      pending = work.then(
        () => {},
        () => {},
      );
      return work;
    },
    async wait(): Promise<void> {
      // An error handler may submit a final drain while the first sync settles.
      let observed: Promise<void> | undefined;
      do {
        observed = pending;
        await Promise.all([observed, ...operations]);
      } while (observed !== pending || operations.size !== 0);
    },
  };
}

/** Keep a queued readback destination alive through scope and copy failures. */
export async function readTVMFloat32(
  tvm: tvmjs.Instance,
  drain: ReturnType<typeof createTVMRequestDrain>,
  shape: number[],
  submit: () => tvmjs.Tensor,
  check: () => void,
): Promise<Float32Array> {
  return drain.run(async () => {
    let syncAttempted = false;
    let cpu: tvmjs.Tensor | undefined, data: Float32Array | undefined;
    let failed = false,
      failure: unknown;
    try {
      tvm.beginScope();
      try {
        const value = submit();
        if (
          value.dtype !== "float32" ||
          value.shape.length !== shape.length ||
          !shape.every((v, i) => v === value.shape[i])
        )
          throw new Error("Unexpected TVM readback shape or dtype");
        cpu = tvm.detachFromCurrentScope(
          tvm.empty(shape, "float32", tvm.cpu()),
        );
        cpu.copyFrom(value);
      } catch (error) {
        failed = true;
        failure = error;
      }
      try {
        tvm.endScope();
      } catch (error) {
        if (!failed) {
          failed = true;
          failure = error;
        }
      }
      if (failed) throw failure;
      syncAttempted = true;
      await drain.sync();
      check();
      data = cpu!.toArray() as Float32Array;
    } catch (error) {
      failed = true;
      failure = error;
    }
    if (failed && !syncAttempted) {
      try {
        await drain.sync();
      } catch {
        /* Preserve the operation's first error. */
      }
    }
    try {
      cpu?.dispose();
    } catch (error) {
      if (!failed) {
        failed = true;
        failure = error;
      }
    }
    if (failed) throw failure;
    return data!;
  });
}
