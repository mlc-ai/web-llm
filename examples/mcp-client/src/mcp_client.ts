import * as webllm from "@mlc-ai/web-llm";

interface MCPTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
}

interface MCPToolResult {
  content: Array<{ type: string; text?: string }>;
  isError?: boolean;
}

class MCPClient {
  private mode: "stateful" | "stateless" = "stateful";
  private sessionId: string | null = null;
  private nextId = 1;
  public tools: MCPTool[] = [];

  constructor(private url: string) {}

  private async rpc(
    method: string,
    params: Record<string, unknown>,
    isNotification = false,
  ): Promise<any> {
    const body: Record<string, unknown> = { jsonrpc: "2.0", method, params };
    if (!isNotification) body.id = this.nextId++;
    if (this.mode === "stateless") {
      body._meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
          name: "web-llm-mcp-example",
          version: "0.1.0",
        },
      };
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
    };
    if (this.mode === "stateful" && this.sessionId) {
      headers["Mcp-Session-Id"] = this.sessionId;
    }

    const res = await fetch(this.url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (this.mode === "stateful") {
      const sid = res.headers.get("Mcp-Session-Id");
      if (sid) this.sessionId = sid;
    }

    if (isNotification) return null;
    if (!res.ok) {
      throw new Error(
        `MCP HTTP ${res.status}: ${await res.text().catch(() => "")}`,
      );
    }

    const ctype = res.headers.get("Content-Type") || "";
    if (ctype.includes("text/event-stream")) return this.readSSE(res);

    const json = await res.json();
    if (json.error)
      throw new Error(`MCP error ${json.error.code}: ${json.error.message}`);
    return json.result;
  }

  private async readSSE(res: Response): Promise<any> {
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let result: any = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop()!;
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          const json = JSON.parse(payload);
          if (json.error)
            throw new Error(
              `MCP error ${json.error.code}: ${json.error.message}`,
            );
          if ("result" in json) result = json.result;
        } catch {
          /* ignore keep-alive / non-JSON lines */
        }
      }
    }
    return result;
  }

  async initialize(): Promise<void> {
    try {
      this.mode = "stateless";
      await this.rpc("server/discover", {});
      return;
    } catch {
      // Not a 2026-07-28+ server — fall back to the stateful handshake.
    }

    this.mode = "stateful";
    await this.rpc("initialize", {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "web-llm-mcp-example", version: "0.1.0" },
    });
    await this.rpc("notifications/initialized", {}, true);
  }

  async listTools(): Promise<MCPTool[]> {
    const result = await this.rpc("tools/list", {});
    this.tools = result?.tools ?? [];
    return this.tools;
  }

  async callTool(
    name: string,
    args: Record<string, unknown>,
  ): Promise<MCPToolResult> {
    return this.rpc("tools/call", { name, arguments: args ?? {} });
  }

  toOpenAITools(): webllm.ChatCompletionTool[] {
    return this.tools.map((t) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description || "",
        parameters: t.inputSchema || { type: "object", properties: {} },
      },
    }));
  }
}

function extractResultText(result: MCPToolResult): string {
  const parts = (result?.content ?? []).map((c) =>
    c.type === "text" ? (c.text ?? "") : JSON.stringify(c),
  );
  return parts.join("\n") || JSON.stringify(result);
}

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) throw Error("Cannot find label " + id);
  label.innerText = text;
}

function appendLog(text: string) {
  const log = document.getElementById("log");
  if (log != null) log.textContent += `${text}\n`;
  console.log(text);
}

let engine: webllm.MLCEngineInterface | undefined;

async function loadEngine() {
  if (engine) return engine;
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC";
  engine = await webllm.CreateMLCEngine(selectedModel, {
    initProgressCallback,
  });
  return engine;
}

async function run(mcpUrl: string) {
  try {
    appendLog(`Connecting to MCP server at ${mcpUrl} ...`);
    const mcp = new MCPClient(mcpUrl);
    await mcp.initialize();
    const tools = await mcp.listTools();
    appendLog(
      `Discovered ${tools.length} tool(s): ${tools.map((t) => t.name).join(", ")}`,
    );

    appendLog("Loading model...");
    const eng = await loadEngine();

    const messages: webllm.ChatCompletionMessageParam[] = [
      {
        role: "user",
        content:
          "Use the available tools to help answer this. Be concise once you have an answer.",
      },
    ];

    appendLog("Requesting tool call from the model...");
    const first = await eng.chat.completions.create({
      stream: false,
      messages,
      tools: mcp.toOpenAITools(),
      tool_choice: "auto",
    });

    const choice = first.choices[0].message;
    if (!choice.tool_calls?.length) {
      appendLog(`Model answered directly (no tool needed): ${choice.content}`);
      return;
    }

    messages.push({
      role: "assistant",
      content: choice.content ?? "",
      tool_calls: choice.tool_calls,
    } as webllm.ChatCompletionMessageParam);

    for (const call of choice.tool_calls) {
      const args = JSON.parse(call.function.arguments || "{}");
      appendLog(
        `Calling ${call.function.name}(${JSON.stringify(args)}) on the MCP server...`,
      );
      const result = await mcp.callTool(call.function.name, args);
      const resultText = extractResultText(result);
      appendLog(`Result: ${resultText}`);
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: resultText,
      } as webllm.ChatCompletionMessageParam);
    }

    appendLog("Requesting final assistant reply...");
    const final = await eng.chat.completions.create({
      stream: false,
      messages,
    });
    appendLog(`Final assistant message:\n${final.choices[0].message.content}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    appendLog(`Error: ${message}`);
    console.error(err);
  }
}

document.getElementById("connect-btn")!.addEventListener("click", () => {
  const input = document.getElementById("mcp-url") as HTMLInputElement;
  const url = input.value.trim();
  if (!url) {
    appendLog("Enter an MCP server URL first.");
    return;
  }
  void run(url);
});
