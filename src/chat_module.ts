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

  setInitProgressCallback(initProgressCallback: InitProgressCallback) {
    this.initProgressCallback = initProgressCallback;
  }

  async reload(localId: string, chatOpts?: ChatOptions, appConfig?: AppConfig): Promise<void> {
    this.unload();
    const tstart = performance.now();
    if (appConfig === undefined) {
      appConfig = prebuiltAppConfig;
    }

    const findModelRecord = () => {
      const matchedItem = appConfig?.model_list.find(
        item => item.local_id == localId
      );
      if (matchedItem !== undefined) return matchedItem;
      throw Error("Cannot find model_url for " + localId);
    }

    const modelRecord = findModelRecord();
    let modelUrl = modelRecord.model_url;
    if (!modelUrl.startsWith("http")) {
      modelUrl = new URL(modelUrl, document.URL).href;
    }
    const configCache = new tvmjs.ArtifactCache("webllm/config");

    // load config
    const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
    const config = {
      ...(await (await configCache.fetchWithCache(configUrl)).json()),
      ...chatOpts
    } as ChatConfig;

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
    const wasmCache = new tvmjs.ArtifactCache("webllm/wasm");
    const wasmUrl = findWasmUrl();
    const fetchWasmSource = async () => {
      if (wasmUrl.includes("localhost")) {
        // do not cache wasm on local host as we might update code frequently
        return await fetch(wasmUrl);
      } else if (!wasmUrl.startsWith("http")) {
        // do not cache wasm on the same server as it can also refresh
        // rely on the normal caching strategy
        return await fetch(new URL(wasmUrl, document.URL).href);
      } else {
        // use cache
        return await wasmCache.fetchWithCache(wasmUrl);
      }
    };
    const wasmSource = await(await fetchWasmSource()).arrayBuffer();

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
    if (modelRecord.required_features !== undefined) {
      for (const feature of modelRecord.required_features) {
        if (!gpuDetectOutput.device.features.has(feature)) {
          if (feature == "shader-f16") {
            throw Error(
              "This model requires WebGPU extension shader-f16, " +
              "which is not enabled in this browser. " +
              "You can try Chrome Canary with flag --enable-dawn-features=allow_unsafe_apis"
            );
          }
          throw Error(
            "This model requires feature " + feature +
            ", which is not yet supported by this browser. "
          );
        }
      }
    }

    tvm.initWebGPU(gpuDetectOutput.device);
    const tokenizer = await this.asyncLoadTokenizer(modelUrl, config);
    await tvm.fetchNDArrayCache(modelUrl, tvm.webgpu(), "webllm/model");

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
    const modelCache = new tvmjs.ArtifactCache("webllm/model");
    if (config.tokenizer_files.includes("tokenizer.model")) {
      const url = new URL("tokenizer.model", baseUrl).href;
      const model = await (await modelCache.fetchWithCache(url)).arrayBuffer();
      return Tokenizer.fromSentencePiece(model);
    } else if (config.tokenizer_files.includes("tokenizer.json")) {
      const url = new URL("tokenizer.json", baseUrl).href;
      const model = await (await modelCache.fetchWithCache(url)).arrayBuffer();
      return Tokenizer.fromJSON(model);
    }
    throw Error("Cannot handle tokenizer files " + config.tokenizer_files)
  }
}

/**
 * This is the interface to the chat module that connects to the REST API.
 */
export class ChatRestModule implements ChatInterface {
  private logger: (msg: string) => void = console.log
  private initProgressCallback?: InitProgressCallback;

  setInitProgressCallback(initProgressCallback: InitProgressCallback) {
      this.initProgressCallback = initProgressCallback;
  }

  async reload(localId: string, chatOpts?: ChatOptions, appConfig?: AppConfig): Promise<void> {
      throw new Error("Method not implemented.");
  }

  async unload() {
      throw new Error("Method not supported.");
  }

  async interruptGenerate() {
    throw new Error("Method not supported.");
  }

  async generate(
      input: string,
      progressCallback?: GenerateProgressCallback,
      streamInterval = 1,
    ) : Promise<string> {
      if (streamInterval == 0) {
          const response = await fetch('http://localhost:8000/v1/chat/completions', {
                method: "POST",
                headers: { "Content-type": "application/json" },
                body: JSON.stringify({
                  model: "",
                  messages: [{"role": "user", "content": input}],
                  stream: false
                })
            })
            .then((response) => response.json())
            .then((json) => {
                let msg = json["choices"][0]["message"]["content"] as string;
                if (progressCallback !== undefined) {
                    progressCallback(0, msg);
                }
                return msg;
            });
            return response;
      } else {
          var msg = "";
          const response = await fetch('http://localhost:8000/v1/chat/completions', {
              method: "POST",
              headers: { "Content-type": "application/json" },
              body: JSON.stringify({
                  model: "",
                  messages: [{"role": "user", "content": input}],
                  stream: true
                })
            })
            .then((response) => {
              const reader = response.body!.getReader();
              reader.read().then(function pump({ done, value }): any {
                if (done) {
                  if (progressCallback !== undefined) {
                      progressCallback(0, msg);
                  }
                  return;
                }
                const jsonString = Buffer.from(value).toString('utf8').substring(6);
                const parsedData = JSON.parse(jsonString);
                const delta = parsedData["choices"][0]["delta"]["content"] as string;
                // Hack to ignore chunks once we get the EOS token
                if (delta.includes("<")) {
                    return;
                }
                msg += delta;
                if (progressCallback !== undefined) {
                    progressCallback(0, msg);
                }
                return reader.read().then(pump);
              });
            });
            return msg;
      }
    }

    async runtimeStatsText(): Promise<string> {
      const response = await fetch('http://localhost:8000/stats', {
          method: "GET"
      })
      .then((response) => response.json())
      .then((json) => {
          return json;
      });
      return response;
    }

    async resetChat() {
        await fetch('http://localhost:8000/chat/reset', {
          method: "POST"
        });
    }
}
