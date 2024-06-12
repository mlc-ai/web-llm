// Serve the engine workload through web worker
import { MLCEngineWorkerHandler } from "@mlc-ai/web-llm";

const handler = new MLCEngineWorkerHandler();
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
