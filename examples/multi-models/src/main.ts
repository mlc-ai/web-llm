/**
 * This example demonstrates loading multiple models in the same engine concurrently.
 * sequentialGeneration() shows inference each model one at a time.
 * parallelGeneration() shows inference both models at the same time.
 * This example uses WebWorkerMLCEngine, but the same idea applies to MLCEngine and
 * ServiceWorkerMLCEngine as well.
 */

import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

const initProgressCallback = (report: webllm.InitProgressReport) => {
  setLabel("init-label", report.text);
};

// Prepare request for each model, same for both methods
const selectedModel1 = "Phi-3.5-mini-instruct-q4f32_1-MLC-1k";
const selectedModel2 = "gemma-2-2b-it-q4f32_1-MLC-1k";
const prompt1 = "Tell me about California in 3 short sentences.";
const prompt2 = "Tell me about New York City in 3 short sentences.";
setLabel("prompt-label-1", `(with model ${selectedModel1})\n` + prompt1);
setLabel("prompt-label-2", `(with model ${selectedModel2})\n` + prompt2);

const request1: webllm.ChatCompletionRequestStreaming = {
  stream: true,
  stream_options: { include_usage: true },
  messages: [{ role: "user", content: prompt1 }],
  model: selectedModel1, // without specifying it, error will throw due to ambiguity
  max_tokens: 128,
};

const request2: webllm.ChatCompletionRequestStreaming = {
  stream: true,
  stream_options: { include_usage: true },
  messages: [{ role: "user", content: prompt2 }],
  model: selectedModel2, // without specifying it, error will throw due to ambiguity
  max_tokens: 128,
};

/**
 * Chat completion (OpenAI style) with streaming, with two models in the pipeline.
 */
async function sequentialGeneration() {
  const engine = await webllm.CreateWebWorkerMLCEngine(
    new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
    [selectedModel1, selectedModel2],
    { initProgressCallback: initProgressCallback },
  );

  const asyncChunkGenerator1 = await engine.chat.completions.create(request1);
  let message1 = "";
  for await (const chunk of asyncChunkGenerator1) {
    // console.log(chunk);
    message1 += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label-1", message1);
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }
  const asyncChunkGenerator2 = await engine.chat.completions.create(request2);
  let message2 = "";
  for await (const chunk of asyncChunkGenerator2) {
    // console.log(chunk);
    message2 += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label-2", message2);
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }

  // without specifying from which model to get message, error will throw due to ambiguity
  console.log("Final message 1:\n", await engine.getMessage(selectedModel1));
  console.log("Final message 2:\n", await engine.getMessage(selectedModel2));
}

/**
 * Chat completion (OpenAI style) with streaming, with two models in the pipeline.
 */
async function parallelGeneration() {
  const engine = await webllm.CreateWebWorkerMLCEngine(
    new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
    [selectedModel1, selectedModel2],
    { initProgressCallback: initProgressCallback },
  );

  // We can serve the two requests concurrently
  async function getModel1Response() {
    let message1 = "";
    const asyncChunkGenerator1 = await engine.chat.completions.create(request1);
    for await (const chunk of asyncChunkGenerator1) {
      // console.log(chunk);
      message1 += chunk.choices[0]?.delta?.content || "";
      setLabel("generate-label-1", message1);
      if (chunk.usage) {
        console.log(chunk.usage); // only last chunk has usage
      }
      // engine.interruptGenerate();  // works with interrupt as well
    }
  }

  async function getModel2Response() {
    let message2 = "";
    const asyncChunkGenerator2 = await engine.chat.completions.create(request2);
    for await (const chunk of asyncChunkGenerator2) {
      // console.log(chunk);
      message2 += chunk.choices[0]?.delta?.content || "";
      setLabel("generate-label-2", message2);
      if (chunk.usage) {
        console.log(chunk.usage); // only last chunk has usage
      }
      // engine.interruptGenerate();  // works with interrupt as well
    }
  }

  await Promise.all([getModel1Response(), getModel2Response()]);
  // Note: concurrent requests to the same model are executed sequentially in FCFS,
  // unlike to different models like above
  // Fore more, see https://github.com/mlc-ai/web-llm/pull/549
  // await Promise.all([getModel1Response(), getModel1Response()]);

  // without specifying from which model to get message, error will throw due to ambiguity
  console.log("Final message 1:\n", await engine.getMessage(selectedModel1));
  console.log("Final message 2:\n", await engine.getMessage(selectedModel2));
}

// Pick one to run
sequentialGeneration();
// parallelGeneration();
