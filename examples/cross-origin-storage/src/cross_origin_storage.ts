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

  const selectedModel = "Qwen2-0.5B-Instruct-q4f16_1-MLC";

  const appConfig: webllm.AppConfig = {
    ...webllm.prebuiltAppConfig,
    // 👇 Enable cross-origin storage cache.
    useCrossOriginStorageCache: true,
  };

  console.log("Initializing engine with Cross-Origin Storage Cache enabled...");
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback, appConfig: appConfig },
  );

  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      {
        role: "user",
        content: "Write a short poem about storage.",
      },
    ],
  };

  const reply0 = await engine.chatCompletion(request);
  console.log(reply0);
  console.log("Reply:\n" + (await engine.getMessage()));
  console.log(reply0.usage);
}

main();
