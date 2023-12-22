[discord-url]: https://discord.gg/9Xpy2HGBuD

# Web LLM
| [NPM Package](https://www.npmjs.com/package/@mlc-ai/web-llm) | [Get Started](#get-started) | [Examples](examples) | [Documentation](https://mlc.ai/mlc-llm/docs/deploy/javascript.html) | [MLC LLM](https://github.com/mlc-ai/mlc-llm) | [Discord][discord-url] |

WebLLM is a modular, customizable javascript package that directly
brings language model chats directly onto web browsers with hardware acceleration.
**Everything runs inside the browser with no server support and is accelerated with WebGPU.**
We can bring a lot of fun opportunities to build AI assistants for everyone and enable privacy while enjoying GPU acceleration.

**[Check out our demo webpage to try out!](https://webllm.mlc.ai/)**
You can use WebLLM as a base [npm package](https://www.npmjs.com/package/@mlc-ai/web-llm) and build your own web application on top of it by following the [documentation](https://mlc.ai/mlc-llm/docs/deploy/javascript.html) and checking out [Get Started](#get-started).
This project is a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm),
which runs LLMs natively on iPhone and other native local environments.


<img src="site/img/fig/demo.gif">

## Get Started

WebLLM offers a minimalist and modular interface to access the chatbot in the browser.
The WebLLM package itself does not come with UI, and is designed in a
modular way to hook to any of the UI components. The following code snippet
demonstrate a simple example that generates a streaming response on a webpage.
You can check out [examples/get-started](examples/get-started/) to see the complete example.

```typescript
import * as webllm from "@mlc-ai/web-llm";

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
  const chat = new webllm.ChatModule();
  // This callback allows us to report initialization progress
  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });
  // You can also try out "RedPajama-INCITE-Chat-3B-v1-q4f32_1"
  await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

  const generateProgressCallback = (_step: number, message: string) => {
    setLabel("generate-label", message);
  };

  const prompt0 = "What is the capital of Canada?";
  setLabel("prompt-label", prompt0);
  const reply0 = await chat.generate(prompt0, generateProgressCallback);
  console.log(reply0);

  const prompt1 = "Can you write a poem about it?";
  setLabel("prompt-label", prompt1);
  const reply1 = await chat.generate(prompt1, generateProgressCallback);
  console.log(reply1);

  console.log(await chat.runtimeStatsText());
}

main();
```

### Using Web Worker

WebLLM comes with API support for WebWorker so you can hook
the generation process into a separate worker thread so that
the compute in the webworker won't disrupt the UI.

We first create a worker script that created a ChatModule and
hook it up to a handler that handles requests.

```typescript
// worker.ts
import { ChatWorkerHandler, ChatModule } from "@mlc-ai/web-llm";

// Hookup a chat module to a worker handler
const chat = new ChatModule();
const handler = new ChatWorkerHandler(chat);
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
```

Then in the main logic, we create a `ChatWorkerClient` that
implements the same `ChatInterface`. The rest of the logic remains the same.

```typescript
// main.ts
import * as webllm from "@mlc-ai/web-llm";

async function main() {
  // Use a chat worker client instead of ChatModule here
  const chat = new webllm.ChatWorkerClient(new Worker(
    new URL('./worker.ts', import.meta.url),
    {type: 'module'}
  ));
  // everything else remains the same
}
```


### Build a ChatApp

You can find a complete
a complete chat app example in [examples/simple-chat](examples/simple-chat/).


## Customized Model Weights

WebLLM works as a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm).
It reuses the model artifact and builds flow of MLC LLM, please check out
[MLC LLM document](https://llm.mlc.ai/docs/deploy/javascript.html)
on how to add new model weights and libraries to WebLLM.

Here, we go over the high-level idea. There are two elements of the WebLLM package that enables new models and weight variants.

- model_url: Contains a URL to model artifacts, such as weights and meta-data.
- model_lib_url: A URL to the web assembly library (i.e. wasm file) that contains the executables to accelerate the model computations.

Both are customizable in the WebLLM.

```typescript
async main() {
  const myLlamaUrl = "/url/to/my/llama";
  const appConfig = {
    "model_list": [
      {
        "model_url": myLlamaUrl,
        "local_id": "MyLlama-3b-v1-q4f32_0"
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
  await chat.reload("MyLlama-3b-v1-q4f32_0", chatOpts, appConfig);
}
```

In many cases, we only want to supply the model weight variant, but
not necessarily a new model (e.g. `NeuralHermes-Mistral` can reuse `Mistral`'s
model library; `WizardMath` can reuse `Llama-2`'s model library). For
an example of how a model library is shared by different model variants,
see `examples/simple-chat/src/gh-config.js`. We also provide
a plethora of prebuilt model libraries, including:

- `Llama-2-7b-chat-hf-q4f32_1`: Llama-7b models.
- `RedPajama-INCITE-Chat-3B-v1-q4f32_1`: RedPajama-3B variants.
- `Mistral-7B-Instruct-v0.1-q4f16_1`: Mistral-7B variants.
- and many more at [binary-mlc-llm-libs](https://github.com/mlc-ai/binary-mlc-llm-libs).

## Use WebLLM Package

You can directly use WebLLM in your package via npm. Checkout instructions
in the following project

- [get-started](examples/get-started): minimum get started example.
- [web-worker](examples/web-worker): get started with web worker backed chat.
- [simple-chat](examples/simple-chat): a mininum and complete chat app.

## Build WebLLM Package From Source

NOTE: you don't need to build by yourself unless you would
like to change the WebLLM package, follow [use WebLLM](#use-web-llm-package) instead.

WebLLM package is a web runtime designed for [MLC LLM](https://github.com/mlc-ai/mlc-llm).

1. Install all the prerequisites for compilation:
    1. [emscripten](https://emscripten.org). It is an LLVM-based compiler that compiles C/C++ source code to WebAssembly.
        - Follow the [installation instruction](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended) to install the latest emsdk.
        - Source `emsdk_env.sh` by `source path/to/emsdk_env.sh`, so that `emcc` is reachable from PATH and the command `emcc` works.
    4. Install jekyll by following the [official guides](https://jekyllrb.com/docs/installation/). It is the package we use for website. This is not needed if you're using nextjs (see next-simple-chat in the examples).
    5. Install jekyll-remote-theme by command. Try [gem mirror](https://gems.ruby-china.com/) if install blocked.
        ```shell
        gem install jekyll-remote-theme
        ```
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
