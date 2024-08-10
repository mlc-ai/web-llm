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

  // Unlike "Llama-3.1-8B-Instruct-q4f32_1-MLC", this is a base model
  const selectedModel = "Llama-3.1-8B-q4f32_1-MLC";

  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-q4f32_1-MLC", // a base model
        model_id: selectedModel,
        model_lib:
          webllm.modelLibURLPrefix +
          webllm.modelVersion +
          "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
        overrides: {
          context_window_size: 2048,
        },
      },
    ],
  };
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      appConfig: appConfig,
      initProgressCallback: initProgressCallback,
      logLevel: "INFO",
    },
  );

  const reply0 = await engine.completions.create({
    prompt: "List 3 US states: ",
    // below configurations are all optional
    echo: true,
    n: 2,
    max_tokens: 64,
    logprobs: true,
    top_logprobs: 2,
  });
  console.log(reply0);
  console.log(reply0.usage);

  // To change model, either create a new engine via `CreateMLCEngine()`, or call `engine.reload(modelId)`
}

main();
