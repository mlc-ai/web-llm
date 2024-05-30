// Serve the chat workload through web worker
import * as webllm from "@mlc-ai/web-llm";
import { MyLogitProcessor } from "./my_logit_processor";

console.log("Use web worker for logit processor");

const myLogitProcessor = new MyLogitProcessor();
const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
logitProcessorRegistry.set("phi-2-q4f32_1-MLC", myLogitProcessor);

const engine = new webllm.MLCEngine();
engine.setLogitProcessorRegistry(logitProcessorRegistry);
const handler = new webllm.MLCEngineWorkerHandler(engine);
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
