import * as webllm from "@mlc-ai/web-llm";

type ToolInvocation = {
  name: string;
  arguments: Record<string, unknown>;
};

type ToolDefinition = {
  name: string;
  description: string;
  schema: Record<string, unknown>;
};

const tools: ToolDefinition[] = [
  {
    name: "get_weather",
    description: "Fetch an approximate weather report for a city.",
    schema: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name, e.g. Tokyo" },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"],
          description: "Temperature unit",
        },
      },
      required: ["location"],
    },
  },
  {
    name: "get_time",
    description: "Return the current time in a given IANA timezone.",
    schema: {
      type: "object",
      properties: {
        timezone: {
          type: "string",
          description: "IANA timezone name, defaults to UTC",
        },
      },
      required: [],
    },
  },
];

const mcpStructuralTag = {
  type: "structural_tag",
  format: {
    type: "triggered_tags",
    triggers: ["<tool_call>"],
    tags: tools.map((tool) => ({
      begin: `<tool_call>\n{"name": "${tool.name}", "arguments": `,
      content: { type: "json_schema", json_schema: tool.schema },
      end: "}\n</tool_call>",
    })),
    at_least_one: true,
    stop_after_first: false,
  },
} as const;

const initProgressCallback = (report: webllm.InitProgressReport) => {
  setLabel("init-label", report.text);
};

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

function appendLog(text: string) {
  const log = document.getElementById("log");
  if (log != null) {
    log.textContent += `${text}\n`;
  }
  console.log(text);
}

function parseToolCallBlocks(
  content: string | null | undefined,
): ToolInvocation[] {
  if (!content) {
    throw new Error("Assistant reply did not contain a tool call.");
  }
  const regex = /<tool_call>\s*({[\s\S]*?})\s*<\/tool_call>/g;
  const calls: ToolInvocation[] = [];
  let match: RegExpExecArray | null;
  while ((match = regex.exec(content)) !== null) {
    const payload = JSON.parse(match[1]);
    if (typeof payload.name !== "string" || payload.arguments === undefined) {
      continue;
    }
    calls.push({ name: payload.name, arguments: payload.arguments });
  }
  if (calls.length === 0) {
    throw new Error("Failed to find any <tool_call> blocks.");
  }
  return calls;
}

async function runTool(call: ToolInvocation): Promise<Record<string, unknown>> {
  if (call.name === "get_weather") {
    const location = String(call.arguments.location ?? "").trim() || "unknown";
    const unit = (call.arguments.unit as string) ?? "celsius";
    return {
      location,
      unit,
      temperature: unit === "fahrenheit" ? 72.0 : 22.2,
      conditions: "Clear skies",
      source: "demo-weather-kit",
    };
  }
  if (call.name === "get_time") {
    const timezone = (call.arguments.timezone as string) ?? "UTC";
    return {
      timezone,
      iso_time: new Date().toISOString(),
      note: "Demo tool uses local clock only.",
    };
  }
  return { error: `Tool ${call.name} is not implemented in the demo.` };
}

async function main() {
  try {
    appendLog("Loading model...");
    const selectedModel = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
    const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
      selectedModel,
      { initProgressCallback: initProgressCallback, logLevel: "INFO" },
    );

    const systemPrompt =
      "You are a MCP assistant. " +
      'Use the provided tools and emit one or more <tool_call> blocks (one per tool you need) with a JSON body {"name": ..., "arguments": ...}. ' +
      "Do not add extra prose when calling a tool." +
      " Available tools: " +
      JSON.stringify(
        tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
        })),
        null,
        2,
      );

    const messages: webllm.ChatCompletionMessageParam[] = [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content:
          "Give me the weather in Paris in celsius and also tell me the current time in UTC.",
      },
    ];

    const responseFormat: webllm.ResponseFormat = {
      type: "structural_tag",
      structural_tag: mcpStructuralTag,
    };

    appendLog("Requesting constrained tool call...");
    const toolCallReply = await engine.chat.completions.create({
      stream: false,
      messages,
      max_tokens: 1024,
      response_format: responseFormat,
    });

    const toolCallContent = toolCallReply.choices[0].message.content ?? "";
    appendLog(`Assistant tool call:\n${toolCallContent}`);
    const parsedCalls = parseToolCallBlocks(toolCallContent);
    const toolCalls = parsedCalls.map((call, idx) => {
      const toolCallId = `${call.name}-call-${idx + 1}`;
      return { id: toolCallId, call };
    });
    messages.push({
      role: "assistant",
      content: toolCallContent,
      tool_calls: toolCalls.map(({ id, call }) => ({
        id,
        type: "function",
        function: {
          name: call.name,
          arguments: JSON.stringify(call.arguments),
        },
      })),
    } as webllm.ChatCompletionMessageParam);

    for (const { id, call } of toolCalls) {
      const toolResult = await runTool(call);
      messages.push({
        role: "tool",
        tool_call_id: id,
        content: JSON.stringify(toolResult),
      });
      appendLog(
        `Tool response for ${call.name}:\n${JSON.stringify(toolResult, null, 2)}`,
      );
    }

    messages.push({
      role: "user",
      content:
        "You have been given one or more tool responses above. Summarize ALL tool results in a single reply. Include both the weather details and the time information. Do not make up any values.",
    });

    appendLog("Requesting final assistant message...");
    const finalReply = await engine.chat.completions.create({
      stream: false,
      messages,
      max_tokens: 256,
    });
    const finalContent = finalReply.choices[0].message.content ?? "";
    appendLog(`Final assistant message:\n${finalContent}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    appendLog(`Error: ${message}`);
    console.error(err);
  }
}

void main();
