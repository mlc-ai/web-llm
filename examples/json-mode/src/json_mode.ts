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
  const selectedModel = "Llama-3-8B-Instruct-q4f32_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );

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
