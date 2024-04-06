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
  const appConfig = webllm.prebuiltAppConfig;
  // CHANGE THIS TO SEE EFFECTS OF BOTH, CODE BELOW DO NOT NEED TO CHANGE
  appConfig.useIndexedDBCache = true;

  if (appConfig.useIndexedDBCache) {
    console.log("Using IndexedDB Cache");
  } else {
    console.log("Using Cache API");
  }

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });
  const selectedModel = "Phi2-q4f16_1";

  // 1. This triggers downloading and caching the model with either Cache or IndexedDB Cache
  await chat.reload(selectedModel, undefined, appConfig);

  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      { "role": "user", "content": "Write an analogy between mathematics and a lighthouse." },
    ],
    n: 1,
  };
  let reply = await chat.chatCompletion(request);
  console.log(reply);

  // 2. Check whether model weights are cached
  let modelCached = await webllm.hasModelInCache(selectedModel, appConfig);
  console.log("hasModelInCache: ", modelCached);

  // 3. We reload, and we should see this time it is much faster because the weights are cached.
  console.log("Reload model start");
  await chat.reload(selectedModel, undefined, appConfig);
  console.log("Reload model end");
  reply = await chat.chatCompletion(request);
  console.log(reply);

  // 4. Delete every thing about this model from cache
  // You can also delete only the model library wasm, only the model weights, or only the config file
  await webllm.deleteModelAllInfoInCache(selectedModel, appConfig);
  modelCached = await webllm.hasModelInCache(selectedModel, appConfig);
  console.log("After deletion, hasModelInCache: ", modelCached);

  // 5. If we reload, we should expect the model to start downloading again
  console.log("Reload model start");
  await chat.reload(selectedModel, undefined, appConfig);
  console.log("Reload model end");
  reply = await chat.chatCompletion(request);
  console.log(reply);
}

main();
