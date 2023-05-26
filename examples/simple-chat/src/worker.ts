// Serve the chat workload through web worker
import { ChatWorkerHandler, ChatModule } from "@mlc-ai/web-llm";

const chat = new ChatModule();
const handler = new ChatWorkerHandler(chat);
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
