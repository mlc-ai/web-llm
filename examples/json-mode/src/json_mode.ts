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
  // Pick any one of these models to start trying -- most models in WebLLM support grammar
  const selectedModel = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );
  // Note that you'd need to prompt the model to answer in JSON either in
  // user's message or the system prompt
  const request: webllm.ChatCompletionRequest = {
    stream: false, // works with streaming, logprobs, top_logprobs as well
    messages: [
      {
        role: "user",
        content: "Write a short JSON file introducing yourself.",
      },
    ],
    n: 2,
    max_tokens: 128,
    response_format: { type: "json_object" } as webllm.ResponseFormat,
  };

  const reply0 = await engine.chatCompletion(request);
  console.log(reply0);
  console.log("First reply's last choice:\n" + (await engine.getMessage()));
  console.log(reply0.usage);
}

main();
