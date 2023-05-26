import { ChatWorkerHandler, ChatModule } from "@mlc-ai/web-llm";

// Hookup a chat module to a worker handler
const chat = new ChatModule();
const handler = new ChatWorkerHandler(chat);
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
