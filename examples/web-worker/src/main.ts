import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

// There are three demonstrations, simply pick one from below

/**
 * We domnstrate using WebLLM with `genereate()`.
 */
async function mainGenerate() {
  console.log("Using ChatModule.generate()");
  // Use a chat worker client instead of ChatModule here
  const chat: webllm.ChatInterface = new webllm.ChatWorkerClient(new Worker(
    new URL('./worker.ts', import.meta.url),
    { type: 'module' }
  ));

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const generateProgressCallback = (_step: number, message: string) => {
    setLabel("generate-label", message);
  };

  const prompt0 = "What is the capital of Canada?";
  setLabel("prompt-label", prompt0);
  const reply0 = await chat.generate(prompt0, generateProgressCallback);
  console.log(reply0);

  const prompt1 = "Can you write a poem about it?";
  setLabel("prompt-label", prompt1);
  const reply1 = await chat.generate(prompt1, generateProgressCallback);
  console.log(reply1);

  console.log(await chat.runtimeStatsText());
}

/**
 * Chat completion (OpenAI style) without streaming, where we get the entire response at once.
 */
async function mainOpenAIAPINonStreaming() {
  console.log("Using ChatModule.chatCompletion() without streaming.");
  // Use a chat worker client instead of ChatModule here
  const chat: webllm.ChatInterface = new webllm.ChatWorkerClient(new Worker(
    new URL('./worker.ts', import.meta.url),
    { type: 'module' }
  ));

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const request: webllm.ChatCompletionRequest = {
    // stateful: true,  // set this optionally to preserve chat history
    messages: [
      {
        "role": "system",
        "content": "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n "
      },
      { "role": "user", "content": "Provide me three US states." },
      { "role": "assistant", "content": "California, New York, Pennsylvania." },
      { "role": "user", "content": "Two more please!" },
    ],
    n: 3,
    temperature: 1.5,
    max_gen_len: 25,
  };

  const reply0 = await chat.chatCompletion(request);
  console.log(reply0);

  console.log(await chat.runtimeStatsText());
}

/**
 * Chat completion (OpenAI style) with streaming, where delta is sent while generating response.
 */
async function mainOpenAIAPIStreaming() {
  console.log("Using ChatModule.chatCompletion() with streaming.");
  // Use a chat worker client instead of ChatModule here
  const chat: webllm.ChatInterface = new webllm.ChatWorkerClient(new Worker(
    new URL('./worker.ts', import.meta.url),
    { type: 'module' }
  ));

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const request: webllm.ChatCompletionRequest = {
    // stateful: true,  // set this optionally to preserve chat history
    stream: true,
    messages: [
      {
        "role": "system",
        "content": "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n "
      },
      { "role": "user", "content": "Provide me three US states." },
      { "role": "assistant", "content": "California, New York, Pennsylvania." },
      { "role": "user", "content": "Two more please!" },
    ],
  };
  const asyncChunkGenerator = await chat.chatCompletion(request);
  let message = "";
  for await (const chunk of asyncChunkGenerator) {
    console.log(chunk);
    if (chunk.choices[0].delta.content) {
      // Last chunk has undefined content
      message += chunk.choices[0].delta.content;
    }
    setLabel("generate-label", message);
    // chat.interruptGenerate();  // works with interrupt as well
  }
  console.log("Final message:\n", await chat.getMessage());  // the concatenated message
  console.log(await chat.runtimeStatsText());
}

// Run one of the function below -- three different ChatModule APIs
// mainGenerate();
// mainOpenAIAPINonStreaming();
mainOpenAIAPIStreaming();
