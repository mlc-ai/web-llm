import {
  ServiceWorkerMLCEngineHandler,
  MLCEngineInterface,
  MLCEngine,
} from "@mlc-ai/web-llm";

const engine: MLCEngineInterface = new MLCEngine();
let handler: ServiceWorkerMLCEngineHandler;

self.addEventListener("activate", function (event) {
  handler = new ServiceWorkerMLCEngineHandler(engine);
  console.log("Web-LLM Service Worker Activated")
});
