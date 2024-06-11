import * as tvmjs from "tvmjs";
import log from "loglevel";
import { AppConfig, ChatOptions, MLCEngineConfig } from "./config";
import { ReloadParams, WorkerRequest } from "./message";
import { MLCEngineInterface } from "./types";
import {
  ChatWorker,
  MLCEngineWorkerHandler,
  WebWorkerMLCEngine,
  PostMessageHandler,
} from "./web_worker";
import { areAppConfigsEqual, areChatOptionsEqual } from "./utils";

/**
 * A post message handler that sends messages to a chrome.runtime.Port.
 */
export class PortPostMessageHandler implements PostMessageHandler {
  port: chrome.runtime.Port;
  enabled = true;

  constructor(port: chrome.runtime.Port) {
    this.port = port;
  }

  /**
   * Close the PortPostMessageHandler. This will prevent any further messages
   */
  close() {
    this.enabled = false;
  }

  postMessage(event: any) {
    if (this.enabled) {
      this.port.postMessage(event);
    }
  }
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
export class MLCEngineServiceWorkerHandler extends MLCEngineWorkerHandler {
  modelId?: string;
  chatOpts?: ChatOptions;
  appConfig?: AppConfig;

  private portPostMessageHandler: PortPostMessageHandler;

  constructor(engine: MLCEngineInterface, port: chrome.runtime.Port) {
    super(engine);
    const handler = new PortPostMessageHandler(port);
    this.portPostMessageHandler = handler;
    port.onDisconnect.addListener(() => {
      handler.close();
    });
  }

  postMessage(msg: any) {
    this.portPostMessageHandler.postMessage(msg);
  }

  setPort(port: chrome.runtime.Port) {
    const handler = new PortPostMessageHandler(port);
    this.portPostMessageHandler = handler;
    port.onDisconnect.addListener(() => {
      handler.close();
    });
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
          this.modelId === params.modelId &&
          areChatOptionsEqual(this.chatOpts, params.chatOpts)
        ) {
          log.info("Already loaded the model. Skip loading");
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
    } else if (msg.kind === "setAppConfig") {
      // TODO: ADD SPECIAL HANDLING HERE
      // something to replace areAppConfigsEqual(this.appConfig, params.appConfig) for init
    }
    super.onmessage(event);
  }
}

/**
 * Create a ServiceWorkerMLCEngine.
 *
 * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
 * `engineConfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig` for more.
 * @param keepAliveMs The interval to send keep alive messages to the service worker.
 * See [Service worker lifecycle](https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/lifecycle#idle-shutdown)
 * The default is 10s.
 * @returns An initialized `WebLLM.ServiceWorkerMLCEngine` with `modelId` loaded.
 */
export async function CreateServiceWorkerMLCEngine(
  modelId: string,
  engineConfig?: MLCEngineConfig,
  chatOpts?: ChatOptions,
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

  constructor(engineConfig?: MLCEngineConfig, keepAliveMs = 10000) {
    const port = chrome.runtime.connect({ name: "web_llm_service_worker" });
    const chatWorker = new PortAdapter(port);
    super(chatWorker, engineConfig);
    this.port = port;

    // Keep alive through periodical heartbeat signals
    setInterval(() => {
      this.worker.postMessage({ kind: "keepAlive" });
    }, keepAliveMs);
  }
}
