import * as webllm from "@mlc-ai/web-llm";
import { MyLogitProcessor } from "./my_logit_processor";

const USE_WEB_WORKER = true;  // Toggle this to use Logit Processor without a web worker
const AUTOREGRESS_LIMIT = 32;  // How many tokens to generate for this test

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  // Define modelRecord
  const myAppConfig: webllm.AppConfig = {
    model_list: [
      {
        "model_url": "https://huggingface.co/mlc-ai/phi-2-q4f32_1-MLC/resolve/main/",
        "local_id": "Phi2-q4f32_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/phi-2/phi-2-q4f32_1-ctx2k-webgpu.wasm",
      },
    ]
  }

  // Instantiate myLogitProcessor, registering in the logitProcessorRegistry
  const myLogitProcessor = new MyLogitProcessor();
  const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
  logitProcessorRegistry.set("Phi2-q4f32_1", myLogitProcessor);

  let chat: webllm.ChatInterface;
  
  // Depending on whether we use a web worker, the code is slightly different
  if (USE_WEB_WORKER) {
    chat = new webllm.ChatWorkerClient(new Worker(
      // see worker.ts on how LogitProcessor plays a role there
      new URL('./worker.ts', import.meta.url),
      { type: 'module' }
    ));
  } else {
    chat = new webllm.ChatModule(logitProcessorRegistry);
  }

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  // Reload chat module with a logit processor
  await chat.reload("Phi2-q4f32_1", undefined, myAppConfig);

  // Below we demonstrate the usage of a low-level API `forwardTokensAndSample()`
  const prompt: Array<number> = [42];
  let nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, /*isPrefill=*/true);
  console.log(nextToken);

  let counter = prompt.length;
  while (counter < AUTOREGRESS_LIMIT) {
    counter += 1;
    nextToken = await chat.forwardTokensAndSample([nextToken], counter, /*isPrefill=*/false);
    console.log(nextToken);
  }

  // By calling `chat.resetChat()`, we triggers MyLogitProcessor.resetState()
  chat.resetChat();
  counter = prompt.length;
  nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, /*isPrefill=*/true);
  console.log(nextToken);
  while (counter < AUTOREGRESS_LIMIT) {
    counter += 1;
    nextToken = await chat.forwardTokensAndSample([nextToken], counter, /*isPrefill=*/false);
    console.log(nextToken);
  }

  // `forwardTokensAndSample()` is made compatible with registering runtime stats.
  console.log(await chat.runtimeStatsText());
}

main();
