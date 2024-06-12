import { MLCEngineWorkerHandler } from "@mlc-ai/web-llm";

// Hookup an engine to a worker handler
const handler = new MLCEngineWorkerHandler();
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
