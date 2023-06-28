import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
      throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
async function main() {
  const chat = new webllm.ChatRestModule();
  chat.resetChat();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
      setLabel("init-label", report.text);
  });

  const generateProgressCallback = (_step: number, message: string) => {
      setLabel("generate-label", message);
  };

  const prompt0 = "Write a song about Pittsburgh.";
  setLabel("prompt-label", prompt0);
  const reply0 = await chat.generate(prompt0, generateProgressCallback);
  console.log(reply0);

  console.log(await chat.runtimeStatsText());
}

main();
