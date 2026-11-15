export type ResumableFaultPoint =
  | "journal.before_append"
  | "journal.after_append"
  | "checkpoint.after_page_group"
  | "checkpoint.after_next_logits"
  | "checkpoint.after_meta"
  | "checkpoint.after_complete"
  | "checkpoint.before_commit"
  | "checkpoint.after_commit";

export interface ResumableFaultContext {
  path?: string;
  sessionId?: string;
  checkpointId?: string;
  processedSeqLen?: number;
  groupId?: number;
  recordType?: number;
  seqNo?: number;
}

export type ResumableFaultHook = (
  point: ResumableFaultPoint,
  context: ResumableFaultContext,
) => void | Promise<void>;

let faultHook: ResumableFaultHook | undefined;

export class ResumableInjectedFault extends Error {
  constructor(
    readonly point: ResumableFaultPoint,
    readonly context: ResumableFaultContext = {},
  ) {
    super(`Injected resumable fault at ${point}`);
    this.name = "ResumableInjectedFault";
  }
}

export function isResumableInjectedFault(err: unknown): boolean {
  return err instanceof ResumableInjectedFault;
}

export function setResumableFaultHook(hook?: ResumableFaultHook): () => void {
  const previous = faultHook;
  faultHook = hook;
  return () => {
    faultHook = previous;
  };
}

export async function triggerResumableFault(
  point: ResumableFaultPoint,
  context: ResumableFaultContext = {},
): Promise<void> {
  await faultHook?.(point, context);
}
