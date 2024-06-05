import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );

  const tools: Array<webllm.ChatCompletionTool> = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: ["location"],
        },
      },
    },
  ];

  const request: webllm.ChatCompletionRequest = {
    stream: true, // works with stream as well, where the last chunk returns tool_calls
    stream_options: { include_usage: true },
    messages: [
      {
        role: "user",
        content:
          "What is the current weather in celsius in Pittsburgh and Tokyo?",
      },
    ],
    tool_choice: "auto",
    tools: tools,
  };

  if (!request.stream) {
    const reply0 = await engine.chat.completions.create(request);
    console.log(reply0.choices[0]);
    console.log(reply0.usage);
  } else {
    // If streaming, the last chunk returns tool calls
    const asyncChunkGenerator = await engine.chat.completions.create(request);
    let message = "";
    let lastChunk: webllm.ChatCompletionChunk | undefined;
    let usageChunk: webllm.ChatCompletionChunk | undefined;
    for await (const chunk of asyncChunkGenerator) {
      console.log(chunk);
      message += chunk.choices[0]?.delta?.content || "";
      setLabel("generate-label", message);
      if (!chunk.usage) {
        lastChunk = chunk;
      }
      usageChunk = chunk;
    }
    console.log(lastChunk!.choices[0].delta);
    console.log(usageChunk!.usage);
  }
}

main();
