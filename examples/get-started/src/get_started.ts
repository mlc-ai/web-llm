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
  const selectedModel = "Llama-3-8B-Instruct-q4f32_1";
  const engine: webllm.EngineInterface = await webllm.CreateEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback }
  );

  // Option 2: Specify your own model other than the prebuilt ones
  // const appConfig: webllm.AppConfig = {
  //   model_list: [
  //     {
  //       "model_url": "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f32_1-MLC/resolve/main/",
  //       "model_id": "Llama-3-8B-Instruct-q4f32_1",
  //       "model_lib_url": webllm.modelLibURLPrefix + webllm.modelVersion + "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
  //     },
  //   ]
  // };
  // const engine: webllm.EngineInterface = await webllm.CreateEngine(
  //   selectedModel,
  //   { appConfig: appConfig, initProgressCallback: initProgressCallback }
  // );

  const reply0 = await engine.chat.completions.create({
    messages: [
      { "role": "user", "content": "List three US states." },
    ],
    // below configurations are all optional
    n: 3,
    temperature: 1.5,
    max_gen_len: 256,
    // 46510 and 7188 are "California", and 8421 and 51325 are "Texas" in Llama-3-8B-Instruct
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
  console.log(await engine.runtimeStatsText());

  // To change model, either create a new engine via `CreateEngine()`, or call `engine.reload(modelId)`
}

main();
