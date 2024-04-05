import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  const chat = new webllm.ChatModule();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  // Option 1: Specify appConfig to decide what models to include
  const selectedModel = "Llama-2-7b-chat-hf-q4f32_1"
  // const selectedModel = "Mistral-7B-Instruct-v0.2-q4f16_1"
  await chat.reload(selectedModel);

  // Option 2: If we do not specify appConfig, we use `prebuiltAppConfig` defined in `config.ts`
  // await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const generateProgressCallback = (_step: number, message: string) => {
    setLabel("generate-label", message);
  };

  // Per-generation configuration
  let genConfig: webllm.GenerationConfig = {
    presence_penalty: 0.1,
    frequency_penalty: 0.1,
    // stop: ["is", "Canada"]  // for demonstration purpose
  }

  const prompt0 = "What is the capital of Canada?";
  setLabel("prompt-label", prompt0);
  const reply0 = await chat.generate(prompt0, generateProgressCallback, 1, genConfig);
  console.log(reply0);

  genConfig = {
    presence_penalty: 0.2,
    frequency_penalty: 0.2,
  }
  const prompt1 = "Can you write a poem about it?";
  setLabel("prompt-label", prompt1);
  const reply1 = await chat.generate(prompt1, generateProgressCallback, 1, genConfig);
  console.log(reply1);

  console.log(await chat.runtimeStatsText());
}

main();
