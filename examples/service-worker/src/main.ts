import * as webllm from "@mlc-ai/web-llm";

const registerServiceWorker = async () => {
  if ("serviceWorker" in navigator) {
    try {
      const registration = await navigator.serviceWorker.register(
        new URL("sw.ts", import.meta.url),
        { type: "module" },
      );
      if (registration.installing) {
        console.log("Service worker installing");
      } else if (registration.waiting) {
        console.log("Service worker installed");
      } else if (registration.active) {
        console.log("Service worker active");
      }
    } catch (error) {
      console.error(`Registration failed with ${error}`);
    }
  }
};

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
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";

  const engine: webllm.MLCEngineInterface =
    await webllm.CreateServiceWorkerMLCEngine(selectedModel, {
      initProgressCallback: initProgressCallback,
    });

  const request: webllm.ChatCompletionRequest = {
    messages: [
      {
        role: "system",
        content:
          "You are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please. ",
      },
      { role: "user", content: "Provide me three US states." },
      { role: "assistant", content: "California, New York, Pennsylvania." },
      { role: "user", content: "Two more please!" },
    ],
    n: 3,
    temperature: 1.5,
    max_tokens: 256,
  };

  const reply0 = await engine.chat.completions.create(request);
  console.log(reply0);
  setLabel("generate-label", reply0.choices[0].message.content || "");

  console.log(reply0.usage);
}

/**
 * Chat completion (OpenAI style) with streaming, where delta is sent while generating response.
 */
async function mainStreaming() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";

  const engine: webllm.ServiceWorkerMLCEngine =
    await webllm.CreateServiceWorkerMLCEngine(selectedModel, {
      initProgressCallback: initProgressCallback,
    });

  const request: webllm.ChatCompletionRequest = {
    stream: true,
    stream_options: { include_usage: true },
    messages: [
      {
        role: "system",
        content:
          "You are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please. ",
      },
      { role: "user", content: "Provide me three US states." },
      { role: "assistant", content: "California, New York, Pennsylvania." },
      { role: "user", content: "Two more please!" },
    ],
    temperature: 1.5,
    max_tokens: 256,
  };

  const asyncChunkGenerator = await engine.chat.completions.create(request);
  let message = "";
  for await (const chunk of asyncChunkGenerator) {
    console.log(chunk);
    message += chunk.choices[0]?.delta?.content || "";
    setLabel("generate-label", message);
    if (chunk.usage) {
      console.log(chunk.usage); // only last chunk has usage
    }
    // engine.interruptGenerate();  // works with interrupt as well
  }
  console.log("Final message:\n", await engine.getMessage()); // the concatenated message
}

registerServiceWorker();
// Run one of the function below
// mainNonStreaming();
mainStreaming();
