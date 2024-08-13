import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

/**
 * Chat completion (OpenAI style) with streaming, with two models in the pipeline.
 */
async function mainStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel1 = "Phi-3-mini-4k-instruct-q4f32_1-MLC-1k";
  const selectedModel2 = "gemma-2-2b-it-q4f32_1-MLC-1k";

  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    [selectedModel1, selectedModel2],
    { initProgressCallback: initProgressCallback },
  );

  const request1: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      { role: "user", content: "Provide me three US states." },
      { role: "assistant", content: "California, New York, Pennsylvania." },
      { role: "user", content: "Two more please!" },
    ],
    model: selectedModel1, // without specifying it, error will throw due to ambiguity
  };

  const request2: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      { role: "user", content: "Provide me three cities in NY." },
      { role: "assistant", content: "New York, Binghamton, Buffalo." },
      { role: "user", content: "Two more please!" },
    ],
    model: selectedModel2, // without specifying it, error will throw due to ambiguity
  };

  const asyncChunkGenerator1 = await engine.chat.completions.create(request1);
  let message = "";
  for await (const chunk of asyncChunkGenerator1) {
    console.log(chunk);
    message += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label", message);
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }
  const asyncChunkGenerator2 = await engine.chat.completions.create(request2);
  message += "\n\n";
  for await (const chunk of asyncChunkGenerator2) {
    console.log(chunk);
    message += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label", message);
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }

  // without specifying from which model to get message, error will throw due to ambiguity
  console.log("Final message 1:\n", await engine.getMessage(selectedModel1));
  console.log("Final message 2:\n", await engine.getMessage(selectedModel2));
}

mainStreaming();
