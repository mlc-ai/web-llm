import * as webllm from "@mlc-ai/web-llm";
import { error } from "loglevel";

let engine;

function setLabel(id, text) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  const initProgressCallback = (report) => {
    console.log(report.text);
    setLabel("init-label", report.text);
  };
  // Option 1: If we do not specify appConfig, we use `prebuiltAppConfig` defined in `config.ts`
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  engine = new webllm.MLCEngine({
    initProgressCallback,
  });
  engine.reload(selectedModel);
}
main();
setTimeout(() => {
  console.log("calling unload");
  engine.unload().catch((err) => {
    console.log(err);
  });
}, 5000);
