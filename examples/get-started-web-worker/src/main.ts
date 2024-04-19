import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

// There are two demonstrations, pick one to run

/**
 * Chat completion (OpenAI style) without streaming, where we get the entire response at once.
 */
async function mainNonStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Llama-3-8B-Instruct-q4f32_1";

  const engine: webllm.EngineInterface = await webllm.CreateWebWorkerEngine(
    new Worker(
      new URL('./worker.ts', import.meta.url),
      { type: 'module' }
    ),
    selectedModel,
    { initProgressCallback: initProgressCallback }
  );

  const request: webllm.ChatCompletionRequest = {
    messages: [
      {
        "role": "system",
        "content": "You are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please. "
      },
      { "role": "user", "content": "Provide me three US states." },
      { "role": "assistant", "content": "California, New York, Pennsylvania." },
      { "role": "user", "content": "Two more please!" },
    ],
    n: 3,
    temperature: 1.5,
    max_gen_len: 256,
  };

  const reply0 = await engine.chat.completions.create(request);
  console.log(reply0);

  console.log(await engine.runtimeStatsText());
}

/**
 * Chat completion (OpenAI style) with streaming, where delta is sent while generating response.
 */
async function mainStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Llama-3-8B-Instruct-q4f32_1";

  const engine: webllm.EngineInterface = await webllm.CreateWebWorkerEngine(
    new Worker(
      new URL('./worker.ts', import.meta.url),
      { type: 'module' }
    ),
    selectedModel,
    { initProgressCallback: initProgressCallback }
  );

  const request: webllm.ChatCompletionRequest = {
    stream: true,
    messages: [
      {
        "role": "system",
        "content": "You are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please. "
      },
      { "role": "user", "content": "Provide me three US states." },
      { "role": "assistant", "content": "California, New York, Pennsylvania." },
      { "role": "user", "content": "Two more please!" },
    ],
    temperature: 1.5,
    max_gen_len: 256,
  };

  const asyncChunkGenerator = await engine.chat.completions.create(request);
  let message = "";
  for await (const chunk of asyncChunkGenerator) {
    console.log(chunk);
    if (chunk.choices[0].delta.content) {
      // Last chunk has undefined content
      message += chunk.choices[0].delta.content;
    }
    setLabel("generate-label", message);
    // engine.interruptGenerate();  // works with interrupt as well
  }
  console.log("Final message:\n", await engine.getMessage());  // the concatenated message
  console.log(await engine.runtimeStatsText());
}

// Run one of the function below
// mainNonStreaming();
mainStreaming();
