import * as webllm from "@mlc-ai/web-llm";


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

  // Define LogitProcessor
  class MyLogitProcessor implements webllm.LogitProcessor {
    private tokenSequence: Array<number> = [];

    processLogits(logits: Float32Array): Float32Array {
      logits[0] = 100.0;  // should be enough so that we always sample token 0 below
      return logits;
    }

    processSampledToken(token: number): void {
      this.tokenSequence.push(token);
      console.log("processSampledToken: " + this.tokenSequence.length);
    }

    resetState(): void {
      this.tokenSequence = [];
      console.log("resetState");
    }
  }
  const myLogitProcessor = new MyLogitProcessor();
  const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
  logitProcessorRegistry.set("Phi2-q4f32_1", myLogitProcessor);
  const chat = new webllm.ChatModule(logitProcessorRegistry);

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  // Reload chat module with a logit processor
  await chat.reload("Phi2-q4f32_1", undefined, myAppConfig);

  // Get next token
  const prompt: Array<number> = [42];
  setLabel("prompt-label", prompt.toString());
  let nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, /*isPrefill=*/true);
  console.log(nextToken);

  let counter = prompt.length;
  while (counter < 64) {
    counter += 1;
    nextToken = await chat.forwardTokensAndSample([nextToken], counter, /*isPrefill=*/false);
    console.log(nextToken);
  }

  chat.resetChat();  // triggers MyLogitProcessor.resetState()
  counter = prompt.length;
  nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, /*isPrefill=*/true);
  console.log(nextToken);
  while (counter < 64) {
    counter += 1;
    nextToken = await chat.forwardTokensAndSample([nextToken], counter, /*isPrefill=*/false);
    console.log(nextToken);
  }

  console.log(await chat.runtimeStatsText());
}

main();
