import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

// There are four demonstrations, pick one from below to run

/**
 * We domnstrate chat completion without streaming, where we get the entire response at once.
 */
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
    max_gen_len: 50,
    // 13813 is "Florida", 10319 is "Texas", and 7660 is "Washington" in Llama-2-7b-chat
    // So we would have a higher chance of seeing the latter two, but never the first in the answer
    logit_bias: {
      "13813": -100,
      "10319": 5,
      "7660": 5,
    },
    logprobs: true,
    top_logprobs: 2,
  };

  const reply0 = await chat.chatCompletion(request);
  console.log(reply0);
  console.log(await chat.getMessage());  // the final response

  console.log(await chat.runtimeStatsText());
}


/**
 * We domnstrate chat completion with streaming, where delta is sent while generating response.
 */
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
    logprobs: true,
    top_logprobs: 2,
  };

  for await (const chunk of await chat.chatCompletion(request)) {
    console.log(chunk);
  }

  console.log(await chat.getMessage());  // the final response
  console.log(await chat.runtimeStatsText());
}

/**
 * We domnstrate stateful chat completion, where chat history is preserved across requests.
 */
async function mainStateful() {
  const chat: webllm.ChatInterface = new webllm.ChatModule();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const request0: webllm.ChatCompletionRequest = {
    stateful: true,
    // stream: true, // works with and without streaming
    messages: [
      {
        "role": "system",
        "content": "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n "
      },
      { "role": "user", "content": "Provide me three US states." },
    ],
  };

  const reply0 = await chat.chatCompletion(request0);
  console.log(reply0);
  console.log(await chat.getMessage());

  const request1: webllm.ChatCompletionRequest = {
    stateful: true,
    // stream: true, // works with and without streaming
    messages: [
      { "role": "user", "content": "Two more please!" },
    ],
  };

  const reply1 = await chat.chatCompletion(request1);
  console.log(reply1);
  console.log(await chat.getMessage());

  console.log(await chat.runtimeStatsText());
}

async function mainFunctionCalling() {
  const chat: webllm.ChatInterface = new webllm.ChatModule();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  const myAppConfig: webllm.AppConfig = {
    model_list: [
      {
        "model_url": "https://huggingface.co/mlc-ai/gorilla-openfunctions-v2-q4f16_1-MLC/resolve/main/",
        "local_id": "gorilla-openfunctions-v2-q4f16_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/gorilla-openfunctions-v2/gorilla-openfunctions-v2-q4f16_1.wasm",
      },
    ]
  }
  const selectedModel = "gorilla-openfunctions-v2-q4f16_1"
  await chat.reload(selectedModel, undefined, myAppConfig);

  const tools: Array<webllm.ChatCompletionTool> = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
          },
          "required": ["location"],
        },
      },
    }
  ]

  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      { "role": "user", "content": "What is the current weather in celsius in Pittsburgh and Tokyo?" },
    ],
    tool_choice: 'auto',
    tools: tools,
  };

  const reply0 = await chat.chatCompletion(request);
  console.log(reply0.choices[0].message.content);

  console.log(await chat.runtimeStatsText());
}

// Run one of the functions
// mainNonStreaming();
// mainStreaming();
mainFunctionCalling();
