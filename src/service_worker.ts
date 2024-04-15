import * as tvmjs from "tvmjs";
import { AppConfig, ChatOptions, EngineConfig } from "./config";
import { ReloadParams, WorkerMessage } from "./message";
import { EngineInterface } from "./types";
import {
  ChatWorker,
  EngineWorkerHandler,
  WebWorkerEngine,
  PostMessageHandler,
} from "./web_worker";

/**
 * A post message handler that sends messages to a chrome.runtime.Port.
 */
export class PortPostMessageHandler implements PostMessageHandler {
  port: chrome.runtime.Port;
  enabled: boolean = true;

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
 * const engine = new Engine();
 * let handler;
 * chrome.runtime.onConnect.addListener(function (port) {
 *   if (handler === undefined) {
 *     handler = new ServiceWorkerEngineHandler(engine, port);
 *   } else {
 *     handler.setPort(port);
 *   }
 *   port.onMessage.addListener(handler.onmessage.bind(handler));
 * });
 */
export class ServiceWorkerEngineHandler extends EngineWorkerHandler {
  modelId?: string;
  chatOpts?: ChatOptions;
  appConfig?: AppConfig;

  constructor(engine: EngineInterface, port: chrome.runtime.Port) {
    let portHandler = new PortPostMessageHandler(port);
    super(engine, portHandler);

    port.onDisconnect.addListener(() => {
      portHandler.close();
    });
  }

  setPort(port: chrome.runtime.Port) {
    let portHandler = new PortPostMessageHandler(port);
    this.setPostMessageHandler(portHandler);
    port.onDisconnect.addListener(() => {
      portHandler.close();
    });
  }

  onmessage(event: any): void {
    if (event.type === "keepAlive") {
      return;
    }

    const msg = event as WorkerMessage;
    if (msg.kind === "init") {
      this.handleTask(msg.uuid, async () => {
        const params = msg.content as ReloadParams;
        // If the modelId, chatOpts, and appConfig are the same, immediately return
        if (
          this.modelId === params.modelId &&
          this.chatOpts === params.chatOpts &&
          this.appConfig === params.appConfig
        ) {
          console.log("Already loaded the model. Skip loading");
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

        await this.engine.reload(
          params.modelId,
          params.chatOpts,
          params.appConfig
        );
        this.modelId = params.modelId;
        this.chatOpts = params.chatOpts;
        this.appConfig = params.appConfig;
        return null;
      });
      return;
    }
    super.onmessage(event);
  }
}

/**
 * Create a ServiceWorkerEngine.
 * 
 * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
 * `engineConfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.EngineConfig` for more.
 * @param keepAliveMs The interval to send keep alive messages to the service worker.
 * See [Service worker lifecycle](https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/lifecycle#idle-shutdown)
 * The default is 10s.
 * @returns An initialized `WebLLM.ServiceWorkerEngine` with `modelId` loaded.
 */
export async function CreateServiceWorkerEngine(
  modelId: string,
  engineConfig?: EngineConfig,
  keepAliveMs: number = 10000
): Promise<ServiceWorkerEngine> {
  const serviceWorkerEngine = new ServiceWorkerEngine();
  serviceWorkerEngine.setInitProgressCallback(
    engineConfig?.initProgressCallback
  );
  await serviceWorkerEngine.init(
    modelId,
    engineConfig?.chatOpts,
    engineConfig?.appConfig
  );
  setInterval(() => {
    serviceWorkerEngine.keepAlive();
  }, keepAliveMs);
  return serviceWorkerEngine;
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
 * A client of Engine that exposes the same interface
 */
export class ServiceWorkerEngine extends WebWorkerEngine {
  port: chrome.runtime.Port;

  constructor() {
    let port = chrome.runtime.connect({ name: "web_llm_service_worker" });
    let chatWorker = new PortAdapter(port);
    super(chatWorker);
    this.port = port;
  }

  keepAlive() {
    this.port.postMessage({ type: "keepAlive" });
  }

  /**
   * Initialize the chat with a model.
   *
   * @param modelId model_id of the model to load.
   * @param chatOpts Extra options to overide chat behavior.
   * @param appConfig Override the app config in this load.
   * @returns A promise when reload finishes.
   * @note The difference between init and reload is that init
   * should be called only once when the engine is created, while reload
   * can be called multiple times to switch between models.
   */
  async init(
    modelId: string,
    chatOpts?: ChatOptions,
    appConfig?: AppConfig
  ): Promise<void> {
    const msg: WorkerMessage = {
      kind: "init",
      uuid: crypto.randomUUID(),
      content: {
        modelId: modelId,
        chatOpts: chatOpts,
        appConfig: appConfig,
      },
    };
    await this.getPromise<null>(msg);
  }
}
