import * as webllm from "@mlc-ai/web-llm";

const SYSTEM_PROMPT_PREFIX =
  "You are a helpful assistant running in the user's browser, responsible for answering questions.";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function testPrefix() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO",
      // Prefilling KV cache for efficiency
      cachedPrefixes: [[{ role: "system", content: SYSTEM_PROMPT_PREFIX }]],
    },
    {
      context_window_size: 2048,
    },
  );

  const reply_using_prefix = await engine.chat.completions.create({
    messages: [
      { role: "system", content: SYSTEM_PROMPT_PREFIX },
      { role: "user", content: "List three US states." },
    ],
    // below configurations are all optional
    n: 1,
    temperature: 1.5,
    max_tokens: 64,
    logprobs: true,
    top_logprobs: 2,
  });
  console.log(reply_using_prefix);
  console.log(reply_using_prefix.usage);
}

async function testWithoutPrefix() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  // Engine Initialization without cachedPrefixes
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO",
    },
    {
      context_window_size: 2048,
    },
  );

  const reply_without_prefix = await engine.chat.completions.create({
    messages: [
      { role: "system", content: SYSTEM_PROMPT_PREFIX },
      { role: "user", content: "List three US states." },
    ],
    // below configurations are all optional
    n: 1,
    temperature: 1.5,
    max_tokens: 64,
    logprobs: true,
    top_logprobs: 2,
  });
  console.log(reply_without_prefix);
  console.log(reply_without_prefix.usage);
}

async function testMultiRound() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO",
      cachedPrefixes: [[{ role: "system", content: SYSTEM_PROMPT_PREFIX }]], // Prefilling KV cache for efficiency
    },
    {
      context_window_size: 2048,
    },
  );

  // First Completion with cachedPrefixes
  const reply0 = await engine.chat.completions.create({
    messages: [
      { role: "system", content: SYSTEM_PROMPT_PREFIX },
      { role: "user", content: "List three US states." },
    ],
    // below configurations are all optional
    n: 1,
    temperature: 1.5,
    max_tokens: 64,
    logprobs: true,
    top_logprobs: 2,
  });
  console.log(reply0);
  console.log(reply0.usage);

  // Second Completion with cachedPrefixes
  const reply1 = await engine.chat.completions.create({
    messages: [
      { role: "system", content: SYSTEM_PROMPT_PREFIX },
      { role: "user", content: "Where is the US capital?" },
    ],
    // below configurations are all optional
    n: 1,
    temperature: 1.5,
    max_tokens: 64,
    logprobs: true,
    top_logprobs: 2,
  });
  console.log(reply1);
  console.log(reply1.usage);
}

async function main() {
  await testPrefix();

  await testWithoutPrefix();

  await testMultiRound();
}

main();
