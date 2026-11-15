import { ResumeProbeResult } from "../types";
import { OPFSFileStore } from "./opfs_file_store";
import { ResumableReplayState, readResumableReplayState } from "./replay";
import { ResumableSessionHandle } from "./types";

/** Present recovery eligibility from the same snapshot used for replay. */
export function probeReplayState(
  state: ResumableReplayState,
): ResumeProbeResult {
  const result: ResumeProbeResult = {
    sessionId: state.sessionId,
    modelId: state.modelId,
    emittedTokens: state.emittedTokens,
    processedSeqLen: state.processedSeqLen,
    resumable: false,
    recoveryMode: state.emittedTokens > 0 ? "text_only" : "none",
  };
  const stopped = state.scanStoppedReason;
  if (state.recordCount === 0) {
    return {
      ...result,
      reason:
        stopped === undefined
          ? "missing journal records"
          : `journal scan stopped at ${stopped}`,
    };
  }
  if (stopped !== undefined) {
    return {
      ...result,
      reason: `journal scan stopped at ${stopped}; token replay unavailable`,
    };
  }
  if (state.finished) {
    return { ...result, recoveryMode: "none", reason: "generation finished" };
  }
  let reason: string | undefined;
  if (!state.hasPromptTokens) {
    reason = "missing prompt token record; token replay unavailable";
  } else if (!state.hasGenerationConfig) {
    reason = "missing generation config record; token replay unavailable";
  } else if (!state.hasResumableGenerationConfig) {
    reason = "missing resumable generation config; continuation unavailable";
  } else if (state.hasUnsupportedGrammarReplay) {
    reason = "unsupported grammar replay; token replay unavailable";
  } else if (
    state.emittedTokens > 0 &&
    state.generatedTokens.at(-1)?.rngState === undefined
  ) {
    reason = "missing RNG state; token replay unavailable";
  } else if (state.modelId === "") {
    reason = "missing model id; token replay unavailable";
  }
  if (reason !== undefined) return { ...result, reason };
  return {
    ...result,
    resumable: true,
    recoveryMode: "token_replay",
    reason: state.aborted
      ? "generation aborted"
      : state.engineErrorMessage !== undefined
        ? `engine error: ${state.engineErrorMessage}`
        : "generation incomplete",
  };
}

export async function probeResumableSession(
  files: OPFSFileStore,
  session: ResumableSessionHandle,
): Promise<ResumeProbeResult> {
  return probeReplayState(await readResumableReplayState(files, session));
}
