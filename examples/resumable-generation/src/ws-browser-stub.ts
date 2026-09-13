export default class NodeWebSocketStub {
  constructor() {
    throw new Error("Node ws is not available in the browser.");
  }
}

export class WebSocketServer {
  constructor() {
    throw new Error("Node ws server is not available in the browser.");
  }
}

export const Server = WebSocketServer;
