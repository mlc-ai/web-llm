import {
  MLCEngine,
  ServiceWorkerMLCEngineHandler,
  ExtensionServiceWorkerMLCEngineHandler,
  type ChatCompletionRequestNonStreaming,
} from "@mlc-ai/web-llm";

const request: ChatCompletionRequestNonStreaming = {
  messages: [{ role: "user", content: "Hello" }],
  stream: false,
};

const engine = new MLCEngine();
void engine.chat.completions.create(request);

const handler = new ServiceWorkerMLCEngineHandler();
handler.onmessage(new MessageEvent("message", { data: { kind: "keepAlive" } }));

declare const port: chrome.runtime.Port;
new ExtensionServiceWorkerMLCEngineHandler(port);
