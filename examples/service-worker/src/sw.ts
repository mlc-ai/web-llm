import {
  ServiceWorkerEngineHandler,
  EngineInterface,
  Engine,
} from "@mlc-ai/web-llm";

const engine: EngineInterface = new Engine();
let handler: ServiceWorkerEngineHandler;

self.addEventListener("activate", function (event) {
  handler = new ServiceWorkerEngineHandler(engine);
  console.log("Web-LLM Service Worker Activated")
});
