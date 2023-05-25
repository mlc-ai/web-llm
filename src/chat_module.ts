import * as tvmjs from "tvmjs";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig, AppConfig, prebuiltAppConfig } from "./config";
import { LLMChatPipeline } from "./llm_chat"

import {
  InitProgressCallback,
  ChatInterface,
  ChatOptions,
  GenerateProgressCallback
} from "./types";

/**
 * This is the main interface to the chat module.
 */
export class ChatModule implements ChatInterface {
  private logger: (msg: string) => void = console.log
  private pipeline?: LLMChatPipeline;
  private initProgressCallback?: InitProgressCallback;
  private interruptSignal = false;
  private artifactCache = new tvmjs.ArtifactCache();

  setInitProgressCallback(initProgressCallback: InitProgressCallback) {
    this.initProgressCallback = initProgressCallback;
  }

  async reload(localId: string, chatOpts?: ChatOptions, appConfig?: AppConfig): Promise<void> {
    this.unload();
    const tstart = performance.now();
    if (appConfig === undefined) {
      appConfig = prebuiltAppConfig;
    }

    const findModelUrl = () => {
      const matchedItem = appConfig?.model_list.find(
        item => item.local_id == localId
      );
      if (matchedItem !== undefined) return matchedItem.model_url;
      throw Error("Cannot find model_url for " + localId);
    }

    let modelUrl = findModelUrl();
    if (!modelUrl.startsWith("http")) {
      modelUrl = new URL(modelUrl, document.URL).href;
    }
    // load config
    const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
    const config = await (
      await this.artifactCache.fetchWithCache(configUrl)
    ).json() as ChatConfig;


    const findWasmUrl = () => {
      if (appConfig?.model_lib_map !== undefined) {
        const libUrl = appConfig?.model_lib_map[config.model_lib];
        if (libUrl !== undefined) return libUrl;
      } else {
        const libUrl = prebuiltAppConfig.model_lib_map[config.model_lib];
        if (libUrl !== undefined) return libUrl;
      }
      throw Error("Cannot find wasm for " + config.model_lib + " in supplied libMap");
    }

    // load tvm wasm
    const wasmUrl = findWasmUrl();
    const wasmSource = await (
      await this.artifactCache.fetchWithCache(wasmUrl)
    ).arrayBuffer();

    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      tvmjs.createPolyfillWASI(),
      this.logger
    );

    if (this.initProgressCallback !== undefined) {
      tvm.registerInitProgressCallback(this.initProgressCallback);
    }

    // detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw Error("Cannot find WebGPU in the environment");
    }
    let gpuLabel = "WebGPU";
    if (gpuDetectOutput.adapterInfo.description.length != 0) {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.description;
    } else {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.vendor;
    }
    tvm.initWebGPU(gpuDetectOutput.device);
    const tokenizer = await this.asyncLoadTokenizer(modelUrl, config);
    await tvm.fetchNDArrayCache(modelUrl, tvm.webgpu());

    this.pipeline = new LLMChatPipeline(tvm, tokenizer, config);
    await this.pipeline?.asyncLoadWebGPUPiplines();

    const tend = performance.now();

    if (this.initProgressCallback !== undefined) {
      const text = "Finish loading on " + gpuLabel;
      this.initProgressCallback({
        progress: 1,
        timeElapsed: (tend - tstart) / 1e3,
        text: text
      })
    }
  }

  async generate(
    input: string,
    progressCallback?: GenerateProgressCallback,
    streamInterval = 1,
  ) : Promise<string> {
    this.interruptSignal = false;
    await this.prefill(input);

    let counter = 1;
    while (!this.stopped()) {
      if (this.interruptSignal) {
        this.getPipeline().triggerStop();
        break;
      }
      counter += 1;
      await this.decode();
      if (counter % streamInterval == 0 && progressCallback !== undefined) {
        progressCallback(counter, this.getMessage());
      }
    }
    return this.getMessage();
  }

  async interruptGenerate() {
    this.interruptSignal = true;
  }

  async runtimeStatsText(): Promise<string> {
    return this.getPipeline().runtimeStatsText();
  }

  async resetChat() {
    this.pipeline?.resetChat();
  }

  async unload() {
    this.pipeline?.dispose();
    this.pipeline = undefined;
  }

  //--------------------------
  // Lower level API
  //--------------------------
  /**
   * @returns Whether the generation stopped.
   */
  stopped(): boolean {
    return this.getPipeline().stopped();
  }

  /**
   * Get the current generated response.
   *
   * @returns The current output message.
   */
  getMessage(): string {
    return this.getPipeline().getMessage();
  }

  /**
   * Run a prefill step with a given input.
   * @param input The input prompt.
   */
  async prefill(input: string) {
    return this.getPipeline().prefillStep(input);
  }

  /**
   * Run a decode step to decode the next token.
   */
  async decode() {
    return this.getPipeline().decodeStep();
  }

  private getPipeline(): LLMChatPipeline {
    if (this.pipeline === undefined) {
      throw Error("Chat module not yet initialized, did you call chat.reload?");
    }
    return this.pipeline;
  }

  private async asyncLoadTokenizer(
    baseUrl: string,
    config: ChatConfig
  ): Promise<Tokenizer> {
    if (config.tokenizer_files.includes("tokenizer.model")) {
      const url = new URL("tokenizer.model", baseUrl).href;
      const model = await (await this.artifactCache.fetchWithCache(url)).arrayBuffer();
      return Tokenizer.fromSentencePiece(model);
    } else if (config.tokenizer_files.includes("tokenizer.json")) {
      const url = new URL("tokenizer.json", baseUrl).href;
      const model = await (await this.artifactCache.fetchWithCache(url)).arrayBuffer();
      return Tokenizer.fromJSON(model);
    }
    throw Error("Cannot handle tokenizer files " + config.tokenizer_files)
  }
}
