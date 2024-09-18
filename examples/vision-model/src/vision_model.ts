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
  const selectedModel = "Phi-3.5-vision-instruct-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO", // specify the log level
    },
  );

  // 1. Single image input (with choices)
  const messages: webllm.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "You are a helpful and honest assistant that answers question concisely.",
    },
    {
      role: "user",
      content: [
        { type: "text", text: "List the items in the image concisely." },
        {
          type: "image_url",
          image_url: {
            url: "https://www.ilankelman.org/stopsigns/australia.jpg",
          },
        },
      ],
    },
  ];
  const request0: webllm.ChatCompletionRequest = {
    stream: false, // can be streaming, same behavior
    messages: messages,
  };
  const reply0 = await engine.chat.completions.create(request0);
  const replyMessage0 = await engine.getMessage();
  console.log(reply0);
  console.log(replyMessage0);
  console.log(reply0.usage);

  // 2. A follow up text-only question
  messages.push({ role: "assistant", content: replyMessage0 });
  messages.push({ role: "user", content: "What is special about this image?" });
  const request1: webllm.ChatCompletionRequest = {
    stream: false, // can be streaming, same behavior
    messages: messages,
  };
  const reply1 = await engine.chat.completions.create(request1);
  const replyMessage1 = await engine.getMessage();
  console.log(reply1);
  console.log(replyMessage1);
  console.log(reply1.usage);

  // 3. A follow up multi-image question
  messages.push({ role: "assistant", content: replyMessage1 });
  messages.push({
    role: "user",
    content: [
      { type: "text", text: "What about these two images? Answer concisely." },
      {
        type: "image_url",
        image_url: { url: "https://www.ilankelman.org/eiffeltower.jpg" },
      },
      {
        type: "image_url",
        image_url: { url: "https://www.ilankelman.org/sunset.jpg" },
      },
    ],
  });
  const request2: webllm.ChatCompletionRequest = {
    stream: false, // can be streaming, same behavior
    messages: messages,
  };
  const reply2 = await engine.chat.completions.create(request2);
  const replyMessage2 = await engine.getMessage();
  console.log(reply2);
  console.log(replyMessage2);
  console.log(reply2.usage);
}

main();
