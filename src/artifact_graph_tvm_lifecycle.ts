/** Track submitted GPU work before releasing a request's TVM objects. */
import type * as tvmjs from "@mlc-ai/web-runtime";

export function createTVMRequestDrain(device: tvmjs.DLDevice) {
  let pending: Promise<void> | undefined;
  return {
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
      await pending;
    },
  };
}
