import { MLCEngineServiceWorkerHandler } from "@mlc-ai/web-llm";

let handler: MLCEngineServiceWorkerHandler;

self.addEventListener("activate", function (event) {
  handler = new MLCEngineServiceWorkerHandler();
  console.log("Web-LLM Service Worker Activated");
});
