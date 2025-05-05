import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

// Helper method to stream responses from the engine
async function streamResponse(
  engine: webllm.MLCEngineInterface,
  request: webllm.ChatCompletionRequestStreaming,
): Promise<void> {
  console.log("Requesting chat completion with request:", request);
  const asyncChunkGenerator = await engine.chat.completions.create(request);
  let message = "";
  for await (const chunk of asyncChunkGenerator) {
    message += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label", message);
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }
  console.log("Final message:\n", await engine.getMessage()); // the concatenated message
}

/**
 * We demonstrate how Qwen3's best practices can be followed in WebLLM. For more, see
 * https://huggingface.co/Qwen/Qwen3-8B#best-practices.
 */
async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Qwen3-4B-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );

  /**
   * 1. Default behavior: enable thinking
   */
  let request: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      {
        role: "user",
        content: "How many r's are there in the word strawberry?",
      },
    ],
    // Specifying `enable_thinking` is optional, as it defaults to think.
    // extra_body: {
    //   enable_thinking: true,
    // }
  };
  await streamResponse(engine, request);

  /**
   * 2. Disable thinking with `enable_thinking: false`.
   */
  request = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      {
        role: "user",
        content: "How many r's are there in the word strawberry?",
      },
    ],
    extra_body: {
      enable_thinking: false,
    },
  };
  await streamResponse(engine, request);

  /**
   * 3. Disable thinking with soft switch /no_think
   * or enable thinking with soft switch /think.
   * Using soft switch: "When enable_thinking=True, regardless of whether the user
   * uses /think or /no_think, the model will always output a block wrapped in
   * <think>...</think>. However, the content inside this block may be empty if
   * thinking is disabled. When enable_thinking=False, the soft switches are not
   * valid. Regardless of any /think or /no_think tags input by the user, the
   * model will not generate think content and will not include a <think>...</think> block.
   */
  request = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      {
        role: "user",
        content: "How many r's are there in the word strawberry? /no_think",
        // content: "How many r's are there in the word strawberry? /think",
      },
    ],
  };
  await streamResponse(engine, request);

  /**
   * 4. For multi-turn messages, it is recommended to
   * parse out the thinking content in the history
   * messages as described in the Best Practices section.
   */
  const history: webllm.ChatCompletionMessageParam[] = [
    {
      role: "user",
      content: "How many r's are there in the word strawberry? /think",
    },
    {
      role: "assistant",
      content:
        "<think>Dummy thinking content here...</think>\n\nThe answer is 3.",
    },
  ];
  // Preprocess history to remove thinking content
  const preprocessedHistory = history.map((msg) => {
    if (msg.role === "assistant") {
      // Remove <think>...</think> block from assistant messages that is at the start
      // and may contain two \n\n line breaks.
      const thinkRegex = /<think>.*?<\/think>\n?\n?/s; // Match <think>...</think> with optional \n\n
      const contentWithoutThink = msg.content!.replace(thinkRegex, "").trim();
      return { ...msg, content: contentWithoutThink };
    }
    return msg; // User messages remain unchanged
  });
  console.log("Preprocessed history:", preprocessedHistory);

  // Now use the preprocessed history in the request
  const newMessage: webllm.ChatCompletionMessageParam = {
    role: "user",
    content: "What about blueberries?",
  };

  request = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [...preprocessedHistory, newMessage],
  };
  await streamResponse(engine, request);
}

main();
