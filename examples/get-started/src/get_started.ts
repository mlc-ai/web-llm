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
  // Option 1: If we do not specify appConfig, we use `prebuiltAppConfig` defined in `config.ts`
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO", // specify the log level
    },
    // customize kv cache, use either context_window_size or sliding_window_size (with attention sink)
    {
      context_window_size: 2048,
      // sliding_window_size: 1024,
      // attention_sink_size: 4,
    },
  );

  // Option 2: Specify your own model other than the prebuilt ones
  // const appConfig: webllm.AppConfig = {
  //   model_list: [
  //     {
  //       model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f32_1-MLC",
  //       model_id: "Llama-3.1-8B-Instruct-q4f32_1-MLC",
  //       model_lib:
  //         webllm.modelLibURLPrefix +
  //         webllm.modelVersion +
  //         "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
  //       overrides: {
  //         context_window_size: 2048,
  //       },
  //     },
  //   ],
  // };
  // const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
  //   selectedModel,
  //   { appConfig: appConfig, initProgressCallback: initProgressCallback },
  // );

  // Option 3: Instantiate MLCEngine() and call reload() separately
  // const engine: webllm.MLCEngineInterface = new webllm.MLCEngine({
  //   appConfig: appConfig, // if do not specify, we use webllm.prebuiltAppConfig
  //   initProgressCallback: initProgressCallback,
  // });
  // await engine.reload(selectedModel);

  const reply0 = await engine.chat.completions.create({
    messages: [{ role: "user", content: "List three US states." }],
    // below configurations are all optional
    n: 3,
    temperature: 1.5,
    max_tokens: 256,
    // 46510 and 7188 are "California", and 8421 and 51325 are "Texas" in Llama-3.1-8B-Instruct
    // So we would have a higher chance of seeing the latter two, but never the first in the answer
    logit_bias: {
      "46510": -100,
      "7188": -100,
      "8421": 5,
      "51325": 5,
    },
    logprobs: true,
    top_logprobs: 2,
  });
  console.log(reply0);
  console.log(reply0.usage);

  // To change model, either create a new engine via `CreateMLCEngine()`, or call `engine.reload(modelId)`
}

main();
