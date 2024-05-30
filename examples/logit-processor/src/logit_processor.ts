import * as webllm from "@mlc-ai/web-llm";
import { MyLogitProcessor } from "./my_logit_processor";

const USE_WEB_WORKER = true; // Toggle this to use Logit Processor without a web worker
const AUTOREGRESS_LIMIT = 32; // How many tokens to generate for this test

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
  // Instantiate myLogitProcessor, registering in the logitProcessorRegistry
  const myLogitProcessor = new MyLogitProcessor();
  const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
  logitProcessorRegistry.set("phi-2-q4f32_1-MLC", myLogitProcessor);

  let engine: webllm.MLCEngineInterface;

  // Depending on whether we use a web worker, the code is slightly different
  if (USE_WEB_WORKER) {
    // see worker.ts on how LogitProcessor plays a role there
    engine = await webllm.CreateWebWorkerMLCEngine(
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
      "phi-2-q4f32_1-MLC",
      { initProgressCallback: initProgressCallback },
    );
  } else {
    engine = await webllm.CreateMLCEngine("phi-2-q4f32_1-MLC", {
      initProgressCallback: initProgressCallback,
      logitProcessorRegistry: logitProcessorRegistry,
    });
  }

  // Below we demonstrate the usage of a low-level API `forwardTokensAndSample()`
  const prompt: Array<number> = [42];
  let nextToken = await engine.forwardTokensAndSample(
    prompt,
    /*isPrefill=*/ true,
  );
  console.log(nextToken);

  let counter = prompt.length;
  while (counter < AUTOREGRESS_LIMIT) {
    counter += 1;
    nextToken = await engine.forwardTokensAndSample(
      [nextToken],
      /*isPrefill=*/ false,
    );
    console.log(nextToken);
  }

  // By calling `engine.resetChat()`, we triggers MyLogitProcessor.resetState()
  engine.resetChat();
  counter = prompt.length;
  nextToken = await engine.forwardTokensAndSample(prompt, /*isPrefill=*/ true);
  console.log(nextToken);
  while (counter < AUTOREGRESS_LIMIT) {
    counter += 1;
    nextToken = await engine.forwardTokensAndSample(
      [nextToken],
      /*isPrefill=*/ false,
    );
    console.log(nextToken);
  }

  // `forwardTokensAndSample()` is made compatible with registering runtime stats.
  console.log(await engine.runtimeStatsText());
}

main();
