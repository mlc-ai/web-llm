import * as tvmjs from "@mlc-ai/web-runtime";
import log from "loglevel";
import { ChatOptions, MLCEngineConfig } from "./config";
import { ReloadParams, WorkerRequest } from "./message";
import {
  ChatWorker,
  WebWorkerMLCEngineHandler,
  WebWorkerMLCEngine,
} from "./web_worker";
import { areArraysEqual, areChatOptionsListEqual } from "./utils";
import { WebGPUNotFoundError } from "./error";

export interface ExtensionMLCEngineConfig extends MLCEngineConfig {
  extensionId?: string;
  onDisconnect?: () => void;
}

/**
 * Worker handler that can be used in a ServiceWorker.
 *
 * @example
 *
 * const engine = new MLCEngine();
 * let handler;
 * chrome.runtime.onConnect.addListener(function (port) {
 *   if (handler === undefined) {
 *     handler = new ServiceWorkerMLCEngineHandler(engine, port);
 *   } else {
 *     handler.setPort(port);
 *   }
 *   port.onMessage.addListener(handler.onmessage.bind(handler));
 * });
 */
export class ServiceWorkerMLCEngineHandler extends WebWorkerMLCEngineHandler {
  port: chrome.runtime.Port | null;

  constructor(port: chrome.runtime.Port) {
    super();
    this.port = port;
    port.onDisconnect.addListener(() => this.onPortDisconnect(port));
  }

  postMessage(msg: any) {
    this.port?.postMessage(msg);
  }

  setPort(port: chrome.runtime.Port) {
    this.port = port;
    port.onDisconnect.addListener(() => this.onPortDisconnect(port));
  }

  onPortDisconnect(port: chrome.runtime.Port) {
    if (port === this.port) {
      this.port = null;
    }
  }

  onmessage(event: any): void {
    if (event.type === "keepAlive") {
      return;
    }

    const msg = event as WorkerRequest;
    if (msg.kind === "reload") {
      this.handleTask(msg.uuid, async () => {
        const params = msg.content as ReloadParams;
        // If the modelId, chatOpts, and appConfig are the same, immediately return
        if (
          areArraysEqual(this.modelId, params.modelId) &&
          areChatOptionsListEqual(this.chatOpts, params.chatOpts)
        ) {
          log.info("Already loaded the model. Skip loading");
          const gpuDetectOutput = await tvmjs.detectGPUDevice();
          if (gpuDetectOutput == undefined) {
            throw new WebGPUNotFoundError();
          }
          let gpuLabel = "WebGPU";
          if (gpuDetectOutput.adapterInfo.description.length != 0) {
            gpuLabel += " - " + gpuDetectOutput.adapterInfo.description;
          } else {
            gpuLabel += " - " + gpuDetectOutput.adapterInfo.vendor;
          }
          this.engine.getInitProgressCallback()?.({
            progress: 1,
            timeElapsed: 0,
            text: "Finish loading on " + gpuLabel,
          });
          return null;
        }

        await this.engine.reload(params.modelId, params.chatOpts);
        this.modelId = params.modelId;
        this.chatOpts = params.chatOpts;
        return null;
      });
      return;
    }

    // All rest of message handling are the same as WebWorkerMLCEngineHandler
    super.onmessage(event);
  }
}

/**
 * Create a ServiceWorkerMLCEngine.
 *
 * @param modelId model_id of the model to load, either string or string[]. When multiple models
 *   are provided, we load all models sequentially. Each modelId needs to either be in
 *   `webllm.prebuiltAppConfig`, or in `engineCOnfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig` for more.
 * @param chatOpts Extra options to optionally override the `mlc-chat-config.json` of `modelId`.
 *   The size of which needs to match that of `modelId`; chatOpts[i] will be used for modelId[i].
 * @param keepAliveMs The interval to send keep alive messages to the service worker.
 * See [Service worker lifecycle](https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/lifecycle#idle-shutdown)
 * The default is 10s.
 * @returns An initialized `WebLLM.ServiceWorkerMLCEngine` with `modelId` loaded.
 */
export async function CreateServiceWorkerMLCEngine(
  modelId: string | string[],
  engineConfig?: ExtensionMLCEngineConfig,
  chatOpts?: ChatOptions | ChatOptions[],
  keepAliveMs = 10000,
): Promise<ServiceWorkerMLCEngine> {
  const serviceWorkerMLCEngine = new ServiceWorkerMLCEngine(
    engineConfig,
    keepAliveMs,
  );
  await serviceWorkerMLCEngine.reload(modelId, chatOpts);
  return serviceWorkerMLCEngine;
}

class PortAdapter implements ChatWorker {
  port: chrome.runtime.Port;
  private _onmessage!: (message: any) => void;

  constructor(port: chrome.runtime.Port) {
    this.port = port;
    this.port.onMessage.addListener(this.handleMessage.bind(this));
  }

  // Wrapper to handle incoming messages and delegate to onmessage if available
  private handleMessage(message: any) {
    if (this._onmessage) {
      this._onmessage(message);
    }
  }

  // Getter and setter for onmessage to manage adding/removing listeners
  get onmessage(): (message: any) => void {
    return this._onmessage;
  }

  set onmessage(listener: (message: any) => void) {
    this._onmessage = listener;
  }

  // Wrap port.postMessage to maintain 'this' context
  postMessage = (message: any): void => {
    this.port.postMessage(message);
  };
}

/**
 * A client of MLCEngine that exposes the same interface
 */
export class ServiceWorkerMLCEngine extends WebWorkerMLCEngine {
  port: chrome.runtime.Port;
  extensionId?: string;

  constructor(engineConfig?: ExtensionMLCEngineConfig, keepAliveMs = 10000) {
    const extensionId = engineConfig?.extensionId;
    const onDisconnect = engineConfig?.onDisconnect;
    const port = extensionId
      ? chrome.runtime.connect(extensionId, {
          name: "web_llm_service_worker",
        })
      : chrome.runtime.connect({ name: "web_llm_service_worker" });
    const chatWorker = new PortAdapter(port);
    super(chatWorker, engineConfig);
    this.port = port;
    this.extensionId = extensionId;

    // Keep alive through periodical heartbeat signals
    const keepAliveTimer = setInterval(() => {
      this.worker.postMessage({ kind: "keepAlive" });
    }, keepAliveMs);

    port.onDisconnect.addListener(() => {
      clearInterval(keepAliveTimer);
      if (onDisconnect) {
        onDisconnect();
      }
    });
  }
}
