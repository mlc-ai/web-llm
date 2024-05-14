import {
  WebServiceWorkerEngineHandler,
  EngineInterface,
  Engine,
} from "@mlc-ai/web-llm";

const engine: EngineInterface = new Engine();
let handler: WebServiceWorkerEngineHandler;

self.addEventListener("activate", function (event) {
  handler = new WebServiceWorkerEngineHandler(engine);
  console.log("Web-LLM Service Worker Activated")
});
