import {
  MLCEngineServiceWorkerHandler,
  MLCEngineInterface,
  MLCEngine,
} from "@mlc-ai/web-llm";

const engine: MLCEngineInterface = new MLCEngine();
let handler: MLCEngineServiceWorkerHandler;

self.addEventListener("activate", function (event) {
  handler = new MLCEngineServiceWorkerHandler(engine);
  console.log("Web-LLM Service Worker Activated");
});
