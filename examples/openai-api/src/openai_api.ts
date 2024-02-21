import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function mainNonStreaming() {
  const chat: webllm.ChatInterface = new webllm.ChatModule();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const request: webllm.ChatCompletionRequest = {
    stream: false,
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

async function mainStreaming() {
  const chat: webllm.ChatInterface = new webllm.ChatModule();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const request: webllm.ChatCompletionRequest = {
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
    temperature: 1.5,
  };

  for await (const chunk of await chat.chatCompletion(request)) {
    console.log(chunk);
  }

  console.log(await chat.runtimeStatsText());
}

// Run one of the functions
// mainNonStreaming();
mainStreaming();
