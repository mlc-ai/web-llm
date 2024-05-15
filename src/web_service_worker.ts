import * as tvmjs from "tvmjs";
import { AppConfig, ChatOptions, EngineConfig } from "./config";
import { ReloadParams, WorkerMessage } from "./message";
import { EngineInterface } from "./types";
import {
  EngineWorkerHandler,
  WebWorkerEngine,
  PostMessageHandler,
  ChatWorker,
} from "./web_worker";
import { areAppConfigsEqual, areChatOptionsEqual } from "./utils";

const BROADCAST_CHANNEL_SERVICE_WORKER_ID = "@mlc-ai/web-llm-sw";
const BROADCAST_CHANNEL_CLIENT_ID = "@mlc-ai/web-llm-client";
export const serviceWorkerBroadcastChannel = new BroadcastChannel(
  BROADCAST_CHANNEL_SERVICE_WORKER_ID
);
export const clientBroadcastChannel = new BroadcastChannel(
  BROADCAST_CHANNEL_CLIENT_ID
);

/**
 * PostMessageHandler wrapper for sending message from service worker back to client
 */
class ClientPostMessageHandler implements PostMessageHandler {
  postMessage(message: any) {
    clientBroadcastChannel.postMessage(message);
  }
}

/**
 * PostMessageHandler wrapper for sending message from client to service worker
 */
class ServiceWorker implements ChatWorker {
  constructor() {
    serviceWorkerBroadcastChannel.onmessage = this.onmessage;
  }

  // ServiceWorkerEngine will later overwrite this
  onmessage() {}

  postMessage(message: any) {
    serviceWorkerBroadcastChannel.postMessage(message);
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

  constructor(engine: EngineInterface) {
    super(engine, new ClientPostMessageHandler());
    serviceWorkerBroadcastChannel.onmessage = this.onmessage.bind(this);
  }

  onmessage(event: any): void {
    const msgEvent = event as MessageEvent;
    const msg = msgEvent.data as WorkerMessage;

    if (msg.kind === "keepAlive") {
      const msg: WorkerMessage = {
        kind: "heartbeat",
        uuid: msgEvent.data.uuid,
        content: "",
      };
      this.postMessageInternal(msg);
      return;
    }

    if (msg.kind === "init") {
      this.handleTask(msg.uuid, async () => {
        const params = msg.content as ReloadParams;
        // If the modelId, chatOpts, and appConfig are the same, immediately return
        if (
          this.modelId === params.modelId &&
          areChatOptionsEqual(this.chatOpts, params.chatOpts) &&
          areAppConfigsEqual(this.appConfig, params.appConfig)
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
    super.onmessage(msg);
  }
}

/**
 * Create a ServiceWorkerEngine.
 *
 * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
 * `engineConfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.EngineConfig` for more.
 * @returns An initialized `WebLLM.ServiceWorkerEngine` with `modelId` loaded.
 */
export async function CreateServiceWorkerEngine(
  modelId: string,
  engineConfig?: EngineConfig
): Promise<ServiceWorkerEngine> {
  await navigator.serviceWorker.ready;
  const serviceWorkerEngine = new ServiceWorkerEngine(new ServiceWorker());
  serviceWorkerEngine.setInitProgressCallback(
    engineConfig?.initProgressCallback
  );
  await serviceWorkerEngine.init(
    modelId,
    engineConfig?.chatOpts,
    engineConfig?.appConfig
  );
  return serviceWorkerEngine;
}

/**
 * A client of Engine that exposes the same interface
 */
export class ServiceWorkerEngine extends WebWorkerEngine {
  missedHeatbeat = 0

  constructor(worker: ChatWorker, keepAliveMs=10000) {
    super(worker);
    clientBroadcastChannel.onmessage = (event) => {
      try {
        this.onevent.bind(this)(event)
      } catch (err: any) {
        // This is expected to throw if user has multiple windows open
        if (!err.message.startsWith("return from a unknown uuid")) {
          console.error("CreateWebServiceWorkerEngine.onmessage", err);
        }
      }
    };
    setInterval(() => {
      this.keepAlive();
    }, keepAliveMs);
  }

  keepAlive() {
    this.worker.postMessage({ kind: "keepAlive" });
  }

  onevent(event: MessageEvent): void {
    const msg = event.data as WorkerMessage;
    if (msg.kind === "heartbeat") {
      this.missedHeatbeat = 0
      return;
    }
    this.onmessage(msg);
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
