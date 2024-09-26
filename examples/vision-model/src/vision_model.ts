import * as webllm from "@mlc-ai/web-llm";
import { imageURLToBase64 } from "./utils";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

const USE_WEB_WORKER = true;

const proxyUrl = "https://cors-anywhere.herokuapp.com/";
const url_https_street = "https://www.ilankelman.org/stopsigns/australia.jpg";
const url_https_tree = "https://www.ilankelman.org/sunset.jpg";
const url_https_sea =
  "https://www.islandvulnerability.org/index/silhouette.jpg";

async function main() {
  // can feed request with either base64 or http url
  const url_base64_street = await imageURLToBase64(proxyUrl + url_https_street);

  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Phi-3.5-vision-instruct-q4f16_1-MLC";

  const engineConfig: webllm.MLCEngineConfig = {
    initProgressCallback: initProgressCallback,
    logLevel: "INFO", // specify the log level
  };
  const chatOpts = {
    context_window_size: 6144,
  };

  const engine: webllm.MLCEngineInterface = USE_WEB_WORKER
    ? await webllm.CreateWebWorkerMLCEngine(
        new Worker(new URL("./worker.ts", import.meta.url), {
          type: "module",
        }),
        selectedModel,
        engineConfig,
        chatOpts,
      )
    : await webllm.CreateMLCEngine(selectedModel, engineConfig, chatOpts);

  // 1. Prefill two images
  const messages: webllm.ChatCompletionMessageParam[] = [
    {
      role: "user",
      content: [
        { type: "text", text: "List the items in each image concisely." },
        {
          type: "image_url",
          image_url: {
            url: url_base64_street,
          },
        },
        {
          type: "image_url",
          image_url: {
            url: proxyUrl + url_https_sea,
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
  messages.push({ role: "user", content: "What is special about each image?" });
  const request1: webllm.ChatCompletionRequest = {
    stream: false, // can be streaming, same behavior
    messages: messages,
  };
  const reply1 = await engine.chat.completions.create(request1);
  const replyMessage1 = await engine.getMessage();
  console.log(reply1);
  console.log(replyMessage1);
  console.log(reply1.usage);

  // 3. A follow up single-image question
  messages.push({ role: "assistant", content: replyMessage1 });
  messages.push({
    role: "user",
    content: [
      { type: "text", text: "What about this image? Answer concisely." },
      {
        type: "image_url",
        image_url: { url: proxyUrl + url_https_tree },
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
