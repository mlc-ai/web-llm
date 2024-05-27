import * as tvmjs from "tvmjs";
import log from "loglevel";
import { AppConfig, ChatOptions, MLCEngineConfig } from "./config";
import { ReloadParams, WorkerRequest, WorkerResponse } from "./message";
import { MLCEngineInterface, InitProgressReport, LogLevel } from "./types";
import {
  MLCEngineWorkerHandler,
  WebWorkerMLCEngine,
  ChatWorker,
} from "./web_worker";
import { areAppConfigsEqual, areChatOptionsEqual } from "./utils";

/* Service Worker Script */

type IServiceWorker = globalThis.ServiceWorker;

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
export class ServiceWorkerMLCEngineHandler extends MLCEngineWorkerHandler {
  modelId?: string;
  chatOpts?: ChatOptions;
  appConfig?: AppConfig;

  private clientRegistry = new Map<
    string,
    IServiceWorker | Client | MessagePort
  >();
  private initReuqestUuid?: string;

  constructor(engine: MLCEngineInterface) {
    if (!self || !("addEventListener" in self)) {
      throw new Error(
        "ServiceWorkerGlobalScope is not defined. ServiceWorkerMLCEngineHandler must be created in service worker script.",
      );
    }
    const postMessageHandler = {
      postMessage: (message: WorkerResponse) => {
        if (this.clientRegistry.has(message.uuid)) {
          const client = this.clientRegistry.get(message.uuid);
          client?.postMessage(message);

          if (message.kind === "return" || message.kind === "throw") {
            this.clientRegistry.delete(message.uuid);
          } else {
            // TODO: Delete clientRegistry after complete to avoid memory leak?
          }
        }
      },
    };
    const initProgressCallback = (report: InitProgressReport) => {
      const msg: WorkerResponse = {
        kind: "initProgressCallback",
        uuid: this.initReuqestUuid || "",
        content: report,
      };
      this.postMessageInternal(msg);
    };
    super(engine, postMessageHandler, initProgressCallback);
    const onmessage = this.onmessage.bind(this);

    self.addEventListener("message", (event) => {
      const message = event as unknown as ExtendableMessageEvent;
      if (message.source) {
        this.clientRegistry.set(message.data.uuid, message.source);
      }
      message.waitUntil(
        new Promise((resolve, reject) => {
          onmessage(message, resolve, reject);
        }),
      );
    });
  }

  onmessage(
    event: ExtendableMessageEvent,
    onComplete?: (value: any) => void,
    onError?: () => void,
  ): void {
    const msg = event.data as WorkerRequest;
    log.trace(
      `ServiceWorker message: [${msg.kind}] ${JSON.stringify(msg.content)}`,
    );

    if (msg.kind === "keepAlive") {
      const reply: WorkerRequest = {
        kind: "heartbeat",
        uuid: msg.uuid,
        content: "",
      };
      this.postMessageInternal(reply);
      onComplete?.(reply);
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
          onComplete?.(null);
          return null;
        }

        this.initReuqestUuid = msg.uuid;
        await this.engine.reload(
          params.modelId,
          params.chatOpts,
          params.appConfig,
        );
        this.modelId = params.modelId;
        this.chatOpts = params.chatOpts;
        this.appConfig = params.appConfig;
        onComplete?.(null);
        return null;
      });
      return;
    }
    super.onmessage(msg, onComplete, onError);
  }
}

/* Webapp Client */
/**
 * PostMessageHandler wrapper for sending message from client to service worker
 */
export class ServiceWorker implements ChatWorker {
  serviceWorker: IServiceWorker;
  onmessage: () => void;

  constructor(serviceWorker: IServiceWorker) {
    this.serviceWorker = serviceWorker;
    this.onmessage = () => {};
  }

  postMessage(message: WorkerRequest) {
    if (!("serviceWorker" in navigator)) {
      throw new Error("Service worker API is not available");
    }
    const serviceWorker = (navigator.serviceWorker as ServiceWorkerContainer)
      .controller;
    if (!serviceWorker) {
      throw new Error("There is no active service worker");
    }
    serviceWorker.postMessage(message);
  }
}

/**
 * Create a ServiceWorkerMLCEngine.
 *
 * @param modelId The model to load, needs to either be in `webllm.prebuiltAppConfig`, or in
 * `engineConfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig` for more.
 * @returns An initialized `WebLLM.ServiceWorkerMLCEngine` with `modelId` loaded.
 */
export async function CreateServiceWorkerMLCEngine(
  modelId: string,
  engineConfig?: MLCEngineConfig,
): Promise<ServiceWorkerMLCEngine> {
  if (!("serviceWorker" in navigator)) {
    throw new Error(
      "Service worker API is not available in your browser. Please ensure that your browser supports service workers and that you are using a secure context (HTTPS). " +
        "Check the browser compatibility and ensure that service workers are not disabled in your browser settings.",
    );
  }
  const serviceWorkerAPI = navigator.serviceWorker as ServiceWorkerContainer;
  const registration = await serviceWorkerAPI.ready;
  const serviceWorker = registration.active || serviceWorkerAPI.controller;
  if (!serviceWorker) {
    throw new Error(
      "Service worker failed to initialize. This could be due to a failure in the service worker registration process or because the service worker is not active. " +
        "Please refresh the page to retry initializing the service worker.",
    );
  }
  const serviceWorkerMLCEngine = new ServiceWorkerMLCEngine(serviceWorker);
  if (engineConfig?.logLevel) {
    serviceWorkerMLCEngine.setLogLevel(engineConfig.logLevel);
  }
  serviceWorkerMLCEngine.setInitProgressCallback(
    engineConfig?.initProgressCallback,
  );
  await serviceWorkerMLCEngine.init(
    modelId,
    engineConfig?.chatOpts,
    engineConfig?.appConfig,
  );
  return serviceWorkerMLCEngine;
}

/**
 * A client of MLCEngine that exposes the same interface
 */
export class ServiceWorkerMLCEngine extends WebWorkerMLCEngine {
  missedHeatbeat = 0;

  constructor(worker: IServiceWorker, keepAliveMs = 10000) {
    if (!("serviceWorker" in navigator)) {
      throw new Error("Service worker API is not available");
    }
    super(new ServiceWorker(worker));
    const onmessage = this.onmessage.bind(this);

    (navigator.serviceWorker as ServiceWorkerContainer).addEventListener(
      "message",
      (event: MessageEvent) => {
        const msg = event.data;
        log.trace(
          `MLC client message: [${msg.kind}] ${JSON.stringify(msg.content)}`,
        );
        try {
          if (msg.kind === "heartbeat") {
            this.missedHeatbeat = 0;
            return;
          }
          onmessage(msg);
        } catch (err: any) {
          // This is expected to throw if user has multiple windows open
          if (!err.message.startsWith("return from a unknown uuid")) {
            log.error("CreateWebServiceWorkerMLCEngine.onmessage", err);
          }
        }
      },
    );

    setInterval(() => {
      this.worker.postMessage({ kind: "keepAlive", uuid: crypto.randomUUID() });
      this.missedHeatbeat += 1;
      log.trace("missedHeatbeat", this.missedHeatbeat);
    }, keepAliveMs);
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
    appConfig?: AppConfig,
  ): Promise<void> {
    const msg: WorkerRequest = {
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
