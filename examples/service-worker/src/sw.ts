import { ServiceWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

// Service worker event listeners must be registered during initial script
// evaluation. An already-active worker can restart without another `activate`.
new ServiceWorkerMLCEngineHandler();
console.log("Web-LLM Service Worker Initialized");
