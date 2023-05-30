import { AppConfig, ChatConfig } from "./config"

/**
 * Custom options that can be used to
 * override known config values.
 */
export interface ChatOptions extends Partial<ChatConfig> {}

/**
 * Report during intialization.
 */
export interface InitProgressReport {
  progress: number;
  timeElapsed: number;
  text: string;
}

/**
 * Callbacks used to report initialization process.
 */
export type InitProgressCallback = (report: InitProgressReport) => void;

/**
 * Callbacks used to report initialization process.
 */
export type GenerateProgressCallback = (step: number, currentMessage: string) => void;

/**
 * Common interface of chat module that UI can interact with
 */
export interface ChatInterface {
  /**
   * Set an initialization progress callback function
   * which reports the progress of model loading.
   *
   * This function can be useful to implement an UI that
   * update as we loading the model.
   *
   * @param initProgressCallback The callback function
   */
  setInitProgressCallback: (initProgressCallback: InitProgressCallback) => void;

  /**
   * Reload the chat with a new model.
   *
   * @param localIdOrUrl local_id of the model or model artifact url.
   * @param chatOpts Extra options to overide chat behavior.
   * @param appConfig Override the app config in this load.
   * @returns A promise when reload finishes.
   * @note This is an async function.
   */
  reload: (localIdOrUrl: string, chatOpts?: ChatOptions, appConfig?: AppConfig) => Promise<void>;

  /**
   * Generate a response for a given input.
   *
   * @param input The input prompt.
   * @param progressCallback Callback that is being called to stream intermediate results.
   * @param streamInterval callback interval to call progresscallback
   * @returns The final result.
   */
  generate: (
    input: string,
    progressCallback?: GenerateProgressCallback,
    streamInterval?: number,
  ) => Promise<string>;

  /**
   * @returns A text summarizing the runtime stats.
   * @note This is an async function
   */
  runtimeStatsText: () => Promise<string>;

  /**
   * Interrupt the generate process if it is already running.
   */
  interruptGenerate: () => void;

  /**
   * Explicitly unload the current model and release the related resources.
   */
  unload: () => Promise<void>;

  /**
   * Reset the current chat session by clear all memories.
   */
  resetChat: () => Promise<void>;
}

