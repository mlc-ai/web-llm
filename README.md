<div align="center">

# WebLLM
[![NPM Package](https://img.shields.io/badge/NPM_Package-Published-cc3534)](https://www.npmjs.com/package/@mlc-ai/web-llm)
[!["WebLLM Chat Deployed"](https://img.shields.io/badge/WebLLM_Chat-Deployed-%2332a852)](https://chat.webllm.ai/)
[![Join Discoard](https://img.shields.io/badge/Join-Discord-7289DA?logo=discord&logoColor=white)]("https://discord.gg/9Xpy2HGBuD")
[![Related Repository: WebLLM Chat](https://img.shields.io/badge/Related_Repo-WebLLM_Chat-fafbfc?logo=github)](https://github.com/mlc-ai/web-llm-chat/)
[![Related Repository: MLC LLM](https://img.shields.io/badge/Related_Repo-MLC_LLM-fafbfc?logo=github)](https://github.com/mlc-ai/mlc-llm/)

**High-Performance In-Browser LLM Inference Engine.**


[Get Started](#get-started) | [Examples](examples) | [Documentation](https://mlc.ai/mlc-llm/docs/deploy/javascript.html)

</div>

## Overview
WebLLM is a high-performance in-browser LLM inference engine that directly
brings language model inference directly onto web browsers with hardware acceleration.
Everything runs inside the browser with no server support and is accelerated with WebGPU.

WebLLM is **fully compatible with [OpenAI API](https://platform.openai.com/docs/api-reference/chat).**
That is, you can use the same OpenAI API on **any open source models** locally, with functionalities
including json-mode, function-calling, streaming, etc.

We can bring a lot of fun opportunities to build AI assistants for everyone and enable privacy while enjoying GPU acceleration.

You can use WebLLM as a base [npm package](https://www.npmjs.com/package/@mlc-ai/web-llm) and build your own web application on top of it by following the [documentation](https://mlc.ai/mlc-llm/docs/deploy/javascript.html) and checking out [Get Started](#get-started).
This project is a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm), which enables universal deployment of LLM across hardware environments.

<div align="center">

**[Check out WebLLM Chat to try it out!](https://chat.webllm.ai/)**

[WebLLM Chat Demo Video](https://github.com/mlc-ai/web-llm-chat/assets/23090573/f700e27e-bb88-4068-bc8b-8a33ea5a4300)

</div>

## Get Started

WebLLM offers a minimalist and modular interface to access the chatbot in the browser.
The WebLLM package itself does not come with UI, and is designed in a
modular way to hook to any of the UI components. The following code snippet
demonstrate a simple example that generates a streaming response on a webpage.
You can check out [examples/get-started](examples/get-started/) to see the complete example.

```typescript
import * as webllm from "@mlc-ai/web-llm";

async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    const label = document.getElementById("init-label");
    label.innerText = report.text;
  };
  const selectedModel = "Llama-3-8B-Instruct-q4f32_1";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    /*engineConfig=*/ { initProgressCallback: initProgressCallback },
  );

  const reply0 = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Tell me about Pittsburgh." }],
  });
  console.log(reply0);
  console.log(await engine.runtimeStatsText());
}

main();
```

Note that if you need to separate the instantiation of `webllm.MLCEngine` from loading a model, you could substitute

```typescript
const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
  selectedModel,
  /*engineConfig=*/ { initProgressCallback: initProgressCallback },
);
```

with the equivalent

```typescript
const engine: webllm.MLCEngineInterface = new webllm.MLCEngine();
engine.setInitProgressCallback(initProgressCallback);
await engine.reload(selectedModel, chatConfig, appConfig);
```

### CDN Delivery

Thanks to [jsdelivr.com](https://www.jsdelivr.com/package/npm/@mlc-ai/web-llm), the following Javascript code should work out-of-the-box on sites like [jsfiddle.net](https://jsfiddle.net/):

```javascript
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

async function main() {
  const initProgressCallback = (report) => {
    console.log(report.text);
  };
  const selectedModel = "TinyLlama-1.1B-Chat-v0.4-q4f16_1-1k";
  const engine = await webllm.CreateMLCEngine(selectedModel, {
    initProgressCallback: initProgressCallback,
  });

  const reply = await engine.chat.completions.create({
    messages: [
      {
        role: "user",
        content: "Tell me about Pittsburgh.",
      },
    ],
  });
  console.log(reply);
  console.log(await engine.runtimeStatsText());
}

main();
```

### Using Web Worker

WebLLM comes with API support for WebWorker so you can hook
the generation process into a separate worker thread so that
the compute in the webworker won't disrupt the UI.

We first create a worker script that created a MLCEngine and
hook it up to a handler that handles requests.

```typescript
// worker.ts
import { MLCEngineWorkerHandler, MLCEngine } from "@mlc-ai/web-llm";

// Hookup an MLCEngine to a worker handler
const engine = new MLCEngine();
const handler = new MLCEngineWorkerHandler(engine);
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
```

Then in the main logic, we create a `WebWorkerMLCEngine` that
implements the same `MLCEngineInterface`. The rest of the logic remains the same.

```typescript
// main.ts
import * as webllm from "@mlc-ai/web-llm";

async function main() {
  // Use a WebWorkerMLCEngine instead of MLCEngine here
  const engine: webllm.MLCEngineInterface =
    await webllm.CreateWebWorkerMLCEngine(
      /*worker=*/ new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
      }),
      /*modelId=*/ selectedModel,
      /*engineConfig=*/ { initProgressCallback: initProgressCallback },
    );
  // everything else remains the same
}
```

### Use Service Worker

WebLLM comes with API support for ServiceWorker so you can hook the generation process
into a service worker to avoid reloading the model in every page visit and optimize
your application's offline experience.

We first create a service worker script that created a MLCEngine and hook it up to a handler
that handles requests when the service worker is ready.

```typescript
// sw.ts
import {
  ServiceWorkerMLCEngineHandler,
  MLCEngineInterface,
  MLCEngine,
} from "@mlc-ai/web-llm";

const engine: MLCEngineInterface = new MLCEngine();
let handler: ServiceWorkerMLCEngineHandler;

self.addEventListener("activate", function (event) {
  handler = new ServiceWorkerMLCEngineHandler(engine);
  console.log("Service Worker is ready");
});
```

Then in the main logic, we register the service worker and then create the engine using
`CreateServiceWorkerMLCEngine` function. The rest of the logic remains the same.

```typescript
// main.ts
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register(
    /*workerScriptURL=*/ new URL("sw.ts", import.meta.url),
    { type: "module" },
  );
}

const engine: webllm.MLCEngineInterface =
  await webllm.CreateServiceWorkerMLCEngine(
    /*modelId=*/ selectedModel,
    /*engineConfig=*/ { initProgressCallback: initProgressCallback },
  );
```

You can find a complete example on how to run WebLLM in service worker in [examples/service-worker](examples/service-worker/).

### Build a ChatApp

You can find a complete chat app example in [examples/simple-chat](examples/simple-chat/).

### Chrome Extension

You can also find examples on building chrome extension with WebLLM in [examples/chrome-extension](examples/chrome-extension/) and [examples/chrome-extension-webgpu-service-worker](examples/chrome-extension-webgpu-service-worker/). The latter one leverages service worker, so the extension is persistent in the background.

## Full OpenAI Compatibility

WebLLM is designed to be fully compatible with [OpenAI API](https://platform.openai.com/docs/api-reference/chat). Thus, besides building simple chat bot, you can also have the following functionalities with WebLLM:

- [streaming](examples/streaming): return output as chunks in real-time in the form of an AsyncGenerator
- [json-mode](examples/json-mode): efficiently ensure output is in json format, see [OpenAI Reference](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) for more.
- [function-calling](examples/function-calling): function calling with fields `tools` and `tool_choice`.
- [seed-to-reproduce](examples/seed-to-reproduce): use seeding to ensure reproducible output with fields `seed`.

## Model Support

We export all supported models in `webllm.prebuiltAppConfig`, where you can see a list of models
that you can simply call `const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(anyModel)` with.
Prebuilt models include:

- Llama-2
- Llama-3
- Gemma
- Phi-1.5 and Phi-2
- Mistral-7B-Instruct
- OpenHermes-2.5-Mistral-7B
- NeuralHermes-2.5-Mistral-7B
- TinyLlama
- RedPajama

Alternatively, you can compile your own model and weights as described below.

WebLLM works as a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm).
It reuses the model artifact and builds flow of MLC LLM, please check out
[MLC LLM document](https://llm.mlc.ai/docs/deploy/javascript.html)
on how to add new model weights and libraries to WebLLM.

Here, we go over the high-level idea. There are two elements of the WebLLM package that enables new models and weight variants.

- `model_url`: Contains a URL to model artifacts, such as weights and meta-data.
- `model_lib_url`: A URL to the web assembly library (i.e. wasm file) that contains the executables to accelerate the model computations.

Both are customizable in the WebLLM.

```typescript
async main() {
  const appConfig = {
    "model_list": [
      {
        "model_url": "/url/to/my/llama",
        "model_id": "MyLlama-3b-v1-q4f32_0"
        "model_lib_url": "/url/to/myllama3b.wasm",
      }
    ],
  };
  // override default
  const chatOpts = {
    "repetition_penalty": 1.01
  };

  const chat = new ChatModule();
  // load a prebuilt model
  // with a chat option override and app config
  // under the hood, it will load the model from myLlamaUrl
  // and cache it in the browser cache
  // The chat will also load the model library from "/url/to/myllama3b.wasm",
  // assuming that it is compatible to the model in myLlamaUrl.
  const engine = await webllm.CreateMLCEngine(
    "MyLlama-3b-v1-q4f32_0",
    /*engineConfig=*/{ chatOpts: chatOpts, appConfig: appConfig }
  );
}
```

In many cases, we only want to supply the model weight variant, but
not necessarily a new model (e.g. `NeuralHermes-Mistral` can reuse `Mistral`'s
model library). For examples on how a model library can be shared by different model variants,
see `prebuiltAppConfig`.

## Build WebLLM Package From Source

NOTE: you don't need to build by yourself unless you would
like to change the WebLLM package. To simply use the npm, follow [Get Started](#get-started) or any of the [examples](examples) instead.

WebLLM package is a web runtime designed for [MLC LLM](https://github.com/mlc-ai/mlc-llm).

1. Install all the prerequisites for compilation:

   1. [emscripten](https://emscripten.org). It is an LLVM-based compiler that compiles C/C++ source code to WebAssembly.
      - Follow the [installation instruction](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended) to install the latest emsdk.
      - Source `emsdk_env.sh` by `source path/to/emsdk_env.sh`, so that `emcc` is reachable from PATH and the command `emcc` works.
   2. Install jekyll by following the [official guides](https://jekyllrb.com/docs/installation/). It is the package we use for website. This is not needed if you're using nextjs (see next-simple-chat in the examples).
   3. Install jekyll-remote-theme by command. Try [gem mirror](https://gems.ruby-china.com/) if install blocked.
      `shell
gem install jekyll-remote-theme
`
      We can verify the successful installation by trying out `emcc` and `jekyll` in terminal, respectively.

2. Setup necessary environment

   Prepare all the necessary dependencies for web build:

   ```shell
   ./scripts/prep_deps.sh
   ```

3. Buld WebLLM Package

   ```shell
   npm run build
   ```

4. Validate some of the sub-packages

   You can then go to the subfolders in [examples](examples) to validate some of the sub-packages.
   We use Parcelv2 for bundling. Although Parcel is not very good at tracking parent directory
   changes sometimes. When you make a change in the WebLLM package, try to edit the `package.json`
   of the subfolder and save it, which will trigger Parcel to rebuild.

## Links

- [Demo page](https://webllm.mlc.ai/)
- If you want to run LLM on native runtime, check out [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- You might also be interested in [Web Stable Diffusion](https://github.com/mlc-ai/web-stable-diffusion/).

## Acknowledgement

This project is initiated by members from CMU catalyst, UW SAMPL, SJTU, OctoML and the MLC community. We would love to continue developing and supporting the open-source ML community.

This project is only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort. The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities make these models accessible. We would like to thank the teams behind vicuna, SentencePiece, LLaMA, Alpaca. We also would like to thank the WebAssembly, Emscripten, and WebGPU communities. Finally, thanks to Dawn and WebGPU developers.
