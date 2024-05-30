import {
  ExtensionServiceWorkerMLCEngineHandler,
  MLCEngine,
} from "@mlc-ai/web-llm";

// Hookup an engine to a service worker handler
const engine = new MLCEngine();
let handler;

chrome.runtime.onConnect.addListener(function (port) {
  console.assert(port.name === "web_llm_service_worker");
  if (handler === undefined) {
    handler = new ExtensionServiceWorkerMLCEngineHandler(engine, port);
  } else {
    handler.setPort(port);
  }
  port.onMessage.addListener(handler.onmessage.bind(handler));
});
