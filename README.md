<div align="center">

# WebLLM
[![NPM Package](https://img.shields.io/badge/NPM_Package-Published-cc3534)](https://www.npmjs.com/package/@mlc-ai/web-llm)
[!["WebLLM Chat Deployed"](https://img.shields.io/badge/WebLLM_Chat-Deployed-%2332a852)](https://chat.webllm.ai/)
[![Join Discoard](https://img.shields.io/badge/Join-Discord-7289DA?logo=discord&logoColor=white)]("https://discord.gg/9Xpy2HGBuD")
[![Related Repository: WebLLM Chat](https://img.shields.io/badge/Related_Repo-WebLLM_Chat-fafbfc?logo=github)](https://github.com/mlc-ai/web-llm-chat/)
[![Related Repository: MLC LLM](https://img.shields.io/badge/Related_Repo-MLC_LLM-fafbfc?logo=github)](https://github.com/mlc-ai/mlc-llm/)

**High-Performance In-Browser LLM Inference Engine.**


[Get Started](#get-started) | [Examples](examples) | [Documentation](https://mlc.ai/mlc-llm/docs/deploy/webllm.html)

</div>

## Overview
WebLLM is a high-performance in-browser LLM inference engine that brings language model inference directly onto web browsers with hardware acceleration.
Everything runs inside the browser with no server support and is accelerated with WebGPU.

WebLLM is **fully compatible with [OpenAI API](https://platform.openai.com/docs/api-reference/chat).**
That is, you can use the same OpenAI API on **any open source models** locally, with functionalities
including streaming, JSON-mode, function-calling (WIP), etc.

We can bring a lot of fun opportunities to build AI assistants for everyone and enable privacy while enjoying GPU acceleration.

You can use WebLLM as a base [npm package](https://www.npmjs.com/package/@mlc-ai/web-llm) and build your own web application on top of it by following the examples below. This project is a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm), which enables universal deployment of LLM across hardware environments.

<div align="center">

**[Check out WebLLM Chat to try it out!](https://chat.webllm.ai/)**

</div>

## Key Features
- **In-Browser Inference**: WebLLM is a high-performance, in-browser language model inference engine that leverages WebGPU for hardware acceleration, enabling powerful LLM operations directly within web browsers without server-side processing.

- [**Full OpenAI API Compatibility**](#full-openai-compatibility): Seamlessly integrate your app with WebLLM using OpenAI API with functionalities such as streaming, JSON-mode, logit-level control, seeding, and more.

- **Structured JSON Generation**: WebLLM supports state-of-the-art JSON mode structured generation, implemented in the WebAssembly portion of the model library for optimal performance. Check [WebLLM JSON Playground](https://huggingface.co/spaces/mlc-ai/WebLLM-JSON-Playground) on HuggingFace to try generating JSON output with custom JSON schema.

- [**Extensive Model Support**](#built-in-models): WebLLM natively supports a range of models including Llama 3, Phi 3, Gemma, Mistral, Qwen(通义千问), and many others, making it versatile for various AI tasks. For the complete supported model list, check [MLC Models](https://mlc.ai/models).

- [**Custom Model Integration**](#custom-models): Easily integrate and deploy custom models in MLC format, allowing you to adapt WebLLM to specific needs and scenarios, enhancing flexibility in model deployment.

- **Plug-and-Play Integration**: Easily integrate WebLLM into your projects using package managers like NPM and Yarn, or directly via CDN, complete with comprehensive [examples](./examples/) and a modular design for connecting with UI components.

- **Streaming & Real-Time Interactions**: Supports streaming chat completions, allowing real-time output generation which enhances interactive applications like chatbots and virtual assistants.

- **Web Worker & Service Worker Support**: Optimize UI performance and manage the lifecycle of models efficiently by offloading computations to separate worker threads or service workers.

- **Chrome Extension Support**: Extend the functionality of web browsers through custom Chrome extensions using WebLLM, with examples available for building both basic and advanced extensions.

## Built-in Models

Check the complete list of available models on [MLC Models](https://mlc.ai/models). WebLLM supports a subset of these available models and the list can be accessed at [`prebuiltAppConfig.model_list`](https://github.com/mlc-ai/web-llm/blob/main/src/config.ts#L293).

Here are the primary families of models currently supported:

- **Llama**: Llama 3, Llama 2, Hermes-2-Pro-Llama-3
- **Phi**: Phi 3, Phi 2, Phi 1.5
- **Gemma**: Gemma-2B
- **Mistral**: Mistral-7B-v0.3, Hermes-2-Pro-Mistral-7B, NeuralHermes-2.5-Mistral-7B, OpenHermes-2.5-Mistral-7B
- **Qwen (通义千问)**: Qwen2 0.5B, 1.5B, 7B

If you need more models, [request a new model via opening an issue](https://github.com/mlc-ai/web-llm/issues/new/choose) or check [Custom Models](#custom-models) for how to compile and use your own models with WebLLM.

## Jumpstart with Examples

Learn how to use WebLLM to integrate large language models into your application and generate chat completions through this simple Chatbot example: 

[![Example Chatbot on JSFiddle](https://img.shields.io/badge/Example-JSFiddle-blue?logo=jsfiddle&logoColor=white)](https://jsfiddle.net/neetnestor/4nmgvsa2/)
[![Example Chatbot on Codepen](https://img.shields.io/badge/Example-Codepen-gainsboro?logo=codepen)](https://codepen.io/neetnestor/pen/vYwgZaG)

For an advanced example of a larger, more complicated project, check [WebLLM Chat](https://github.com/mlc-ai/web-llm-chat/blob/main/app/client/webllm.ts).

More examples for different use cases are available in the [examples](./examples/) folder.

## Get Started

WebLLM offers a minimalist and modular interface to access the chatbot in the browser.
The package is designed in a modular way to hook to any of the UI components.

### Installation

#### Package Manager

```sh
# npm
npm install @mlc-ai/web-llm
# yarn
yarn add @mlc-ai/web-llm
# or pnpm
pnpm install @mlc-ai/web-llm
```

Then import the module in your code.

```typescript
// Import everything
import * as webllm from "@mlc-ai/web-llm";
// Or only import what you need
import { CreateMLCEngine } from "@mlc-ai/web-llm";
```

#### CDN Delivery

Thanks to [jsdelivr.com](https://www.jsdelivr.com/package/npm/@mlc-ai/web-llm), WebLLM can be imported directly through URL and work out-of-the-box on cloud development platforms like [jsfiddle.net](https://jsfiddle.net/) and [Codepen.io](https://codepen.io/):

```javascript
import * as webllm from "https://esm.run/@mlc-ai/web-llm";
```

### Create MLCEngine

Most operations in WebLLM are invoked through the `MLCEngine` interface. You can create an `MLCEngine` instance and loading the model by calling the `CreateMLCEngine()` factory function.

(Note that loading models requires downloading and it can take a significant amount of time for the very first run without caching previously. You should properly handle this asynchronous call.)

```typescript
import { CreateMLCEngine } from "@mlc-ai/web-llm";

// Callback function to update model loading progress
const initProgressCallback = (initProgress) => {
  console.log(initProgress);
}
const selectedModel = "Llama-3-8B-Instruct-q4f32_1-MLC";

const engine = await CreateMLCEngine(
  selectedModel,
  { initProgressCallback: initProgressCallback }, // engineConfig
);
```

Under the hood, this factory function does the following steps for first creating an engine instance (synchrounous) and then loading the model (asynchrounous). You can also do them separately in your application.

```typescript
import { MLCEngine } from "@mlc-ai/web-llm";

// This is a synchrounous call that returns immediately
const engine = new MLCEngine({
  initProgressCallback: initProgressCallback
});

// This is an asynchrounous call and can take a long time to finish
await engine.reload(selectedModel);
```

### Chat Completion
After successfully initializing the engine, you can now invoke chat completions using OpenAI style chat APIs through the `engine.chat.completions` interface. For the full list of parameters and their descriptions, check [section below](#full-openai-compatibility) and [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create).

(Note: The `model` parameter is not supported and will be ignored here. Instead, call `CreateMLCEngine(model)` or `engine.reload(model)` instead as shown in the [Create MLCEngine](#create-mlcengine) above.)


```typescript
const messages = [
  { role: "system", content: "You are a helpful AI assistant." },
  { role: "user", content: "Hello!" },
]

const reply = await engine.chat.completions.create({
  messages,
});
console.log(reply.choices[0].message);
console.log(reply.usage);
```

### Streaming

WebLLM also supports streaming chat completion generating. To use it, simply pass `stream: true` to the `engine.chat.completions.create` call.

```typescript
const messages = [
  { role: "system", content: "You are a helpful AI assistant." },
  { role: "user", content: "Hello!" },
]

// Chunks is an AsyncGenerator object
const chunks = await engine.chat.completions.create({
  messages,
  temperature: 1,
  stream: true, // <-- Enable streaming
  stream_options: { include_usage: true },
});

let reply = "";
for await (const chunk of chunks) {
  reply += chunk.choices[0]?.delta.content || "";
  console.log(reply);
  if (chunk.usage) {
    console.log(chunk.usage); // only last chunk has usage
  }
}

const fullReply = await engine.getMessage();
console.log(fullReply);
```

## Advanced Usage

### Using Workers

You can put the heavy computation in a worker script to optimize your application performance. To do so, you need to:

1. Create a handler in the worker thread that communicates with the frontend while handling the requests.
2. Create a Worker Engine in your main application, which under the hood sends messages to the handler in the worker thread.

For detailed implementation for different kinds of Workers, check the following sections.

#### Dedicated Web Worker

WebLLM comes with API support for WebWorker so you can hook
the generation process into a separate worker thread so that
the computing in the worker thread won't disrupt the UI.

We create a handler in the worker thread that communicates with the frontend while handling the requests.

```typescript
// worker.ts
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

// A handler that resides in the worker thread
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
```

In the main logic, we create a `WebWorkerMLCEngine` that
implements the same `MLCEngineInterface`. The rest of the logic remains the same.

```typescript
// main.ts
import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";

async function main() {
  // Use a WebWorkerMLCEngine instead of MLCEngine here
  const engine = await CreateWebWorkerMLCEngine(
    new Worker(
      new URL("./worker.ts", import.meta.url), 
      {
        type: "module",
      }
    ),
    selectedModel,
    { initProgressCallback }, // engineConfig
  );

  // everything else remains the same
}
```

### Use Service Worker

WebLLM comes with API support for ServiceWorker so you can hook the generation process
into a service worker to avoid reloading the model in every page visit and optimize
your application's offline experience.

(Note, Service Worker's life cycle is managed by the browser and can be killed any time without notifying the webapp. `ServiceWorkerMLCEngine` will try to keep the service worker thread alive by periodically sending heartbeat events, but your application should also include proper error handling. Check `keepAliveMs` and `missedHeatbeat` in [`ServiceWorkerMLCEngine`](https://github.com/mlc-ai/web-llm/blob/main/src/service_worker.ts#L234) for more details.)

We create a handler in the worker thread that communicates with the frontend while handling the requests.


```typescript
// sw.ts
import { ServiceWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

let handler: ServiceWorkerMLCEngineHandler;

self.addEventListener("activate", function (event) {
  handler = new ServiceWorkerMLCEngineHandler();
  console.log("Service Worker is ready");
});
```

Then in the main logic, we register the service worker and create the engine using
`CreateServiceWorkerMLCEngine` function. The rest of the logic remains the same.

```typescript
// main.ts
import { MLCEngineInterface, CreateServiceWorkerMLCEngine } from "@mlc-ai/web-llm";

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register(
    new URL("sw.ts", import.meta.url),  // worker script
    { type: "module" },
  );
}

const engine: MLCEngineInterface =
  await CreateServiceWorkerMLCEngine(
    selectedModel,
    { initProgressCallback }, // engineConfig
  );
```

You can find a complete example on how to run WebLLM in service worker in [examples/service-worker](examples/service-worker/).

### Chrome Extension
You can also find examples of building Chrome extension with WebLLM in [examples/chrome-extension](examples/chrome-extension/) and [examples/chrome-extension-webgpu-service-worker](examples/chrome-extension-webgpu-service-worker/). The latter one leverages service worker, so the extension is persistent in the background.

## Full OpenAI Compatibility
WebLLM is designed to be fully compatible with [OpenAI API](https://platform.openai.com/docs/api-reference/chat). Thus, besides building a simple chatbot, you can also have the following functionalities with WebLLM:

- [streaming](examples/streaming): return output as chunks in real-time in the form of an AsyncGenerator
- [json-mode](examples/json-mode): efficiently ensure output is in JSON format, see [OpenAI Reference](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) for more.
- [seed-to-reproduce](examples/seed-to-reproduce): use seeding to ensure a reproducible output with fields `seed`.
- [function-calling](examples/function-calling) (WIP): function calling with fields `tools` and `tool_choice` (with preliminary support).

## Custom Models

WebLLM works as a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm) and it supports custom models in MLC format. 
It reuses the model artifact and builds the flow of MLC LLM. To compile and use your own models with WebLLM, please check out
[MLC LLM document](https://llm.mlc.ai/docs/deploy/webllm.html)
on how to compile and deploy new model weights and libraries to WebLLM. 

Here, we go over the high-level idea. There are two elements of the WebLLM package that enable new models and weight variants.

- `model`: Contains a URL to model artifacts, such as weights and meta-data.
- `model_lib`: A URL to the web assembly library (i.e. wasm file) that contains the executables to accelerate the model computations.

Both are customizable in the WebLLM.

```typescript
import { CreateMLCEngine } from "@mlc-ai/web-llm";

async main() {
  const appConfig = {
    "model_list": [
      {
        "model": "/url/to/my/llama",
        "model_id": "MyLlama-3b-v1-q4f32_0"
        "model_lib": "/url/to/myllama3b.wasm",
      }
    ],
  };
  // override default
  const chatOpts = {
    "repetition_penalty": 1.01
  };

  // load a prebuilt model
  // with a chat option override and app config
  // under the hood, it will load the model from myLlamaUrl
  // and cache it in the browser cache
  // The chat will also load the model library from "/url/to/myllama3b.wasm",
  // assuming that it is compatible to the model in myLlamaUrl.
  const engine = await CreateMLCEngine(
    "MyLlama-3b-v1-q4f32_0",
    { appConfig }, // engineConfig
    chatOpts,
  );
}
```

In many cases, we only want to supply the model weight variant, but
not necessarily a new model (e.g. `NeuralHermes-Mistral` can reuse `Mistral`'s
model library). For examples of how a model library can be shared by different model variants,
see `webllm.prebuiltAppConfig`.

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

- [Demo App: WebLLM Chat](https://chat.webllm.ai/)
- If you want to run LLM on native runtime, check out [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- You might also be interested in [Web Stable Diffusion](https://github.com/mlc-ai/web-stable-diffusion/).

## Acknowledgement

This project is initiated by members from CMU Catalyst, UW SAMPL, SJTU, OctoML, and the MLC community. We would love to continue developing and supporting the open-source ML community.

This project is only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort. The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities make these models accessible. We would like to thank the teams behind Vicuna, SentencePiece, LLaMA, and Alpaca. We also would like to thank the WebAssembly, Emscripten, and WebGPU communities. Finally, thanks to Dawn and WebGPU developers.
