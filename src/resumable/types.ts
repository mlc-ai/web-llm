export const RESUMABLE_STORE_ROOT = "webllm-resume";

export interface ResumableSessionManifest {
  sessionId: string;
  createdAtMs: number;
  updatedAtMs: number;
  modelId?: string;
}

export type ResumableSessionManifestInit = Partial<
  Omit<ResumableSessionManifest, "sessionId">
>;

export interface ResumableSessionPaths {
  sessionDir: string;
  manifestPath: string;
  journalPath: string;
  lockPath: string;
  kvDir: string;
}

export interface ResumableCheckpointRef {
  checkpointId: string;
  path: string;
  completePath: string;
}

export interface ResumableSessionHandle {
  sessionId: string;
  paths: ResumableSessionPaths;
  manifest?: ResumableSessionManifest;
}

export interface SessionManifestRebuildInputs {
  sessionId: string;
  manifest?: ResumableSessionManifest;
  journalPath: string;
  hasJournal: boolean;
  checkpoints: ResumableCheckpointRef[];
}
