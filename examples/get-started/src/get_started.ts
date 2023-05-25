import { ChatModule, InitProgressReport } from "@mlc-ai/web-llm";

// We use label to intentionally keep it simple
function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  // create a ChatModule,
  const chat = new ChatModule();

  // This callback allows us to report initialization progress
  chat.setInitProgressCallback((report: InitProgressReport) => {
    setLabel("init-label", report.text);
  });
  // pick a model, here we use red-pajama
  // at any time point, you can call reload
  // to switch the underlying model
  const localId = "RedPajama-INCITE-Chat-3B-v1-q4f32_0";
  await chat.reload(localId);

  // this callback allows us to stream result back
  const generateProgressCallback = (_step: number, message: string) => {
    setLabel("generate-label", message);
  };
  const prompt0 = "What is the capital of Canada?";
  setLabel("prompt-label", prompt0);

  // generate  response
  const reply0 = await chat.generate(prompt0, generateProgressCallback);
  console.log(reply0);

  const prompt1 = "How about France?";
  setLabel("prompt-label", prompt1);
  const reply1 = await chat.generate(prompt1, generateProgressCallback)
  console.log(reply1);

  // We can print out the statis
  console.log(await chat.runtimeStatsText());
}

main()
