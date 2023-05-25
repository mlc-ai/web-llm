[discord-url]: https://discord.gg/9Xpy2HGBuD

# Web LLM
| [NPM Package](https://www.npmjs.com/package/@mlc-ai/web-llm) | [Get Started](#get-started) | [MLC LLM](https://github.com/mlc-ai/mlc-llm) | [Discord][discord-url]

WebLLM is a modular, customizable javascript package that directly
brings language model chats directly onto web browsers with hardware acceleration.
**Everything runs inside the browser with no server support and accelerated with WebGPU.**
We can bring a lot of fun opportunities to build AI assistants for everyone and enable privacy while enjoying GPU acceleration.

**[Check out our demo webpage to try out!](https://mlc.ai/web-llm/)**
This project is a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm),
our companion project that runs LLMs natively on iPhone and other native local environments.


<img src="site/img/fig/demo.gif">

## Get Started

WebLLM offers a minimalist and modular interface to access the chatbot in the browser.
The following code demonstrates the basic usage.

```typescript
import { ChatModule } from "@mlc-ai/web-llm";

async function main() {
  const chat = new ChatModule();
  // load a prebuilt model
  await chat.reload("RedPajama-INCITE-Chat-3B-v1-q4f32_0");
  // generate a reply base on input
  const prompt = "What is the capital of Canada?";
  const reply = await chat.generate(prompt);
  console.log(reply);
}
```

The WebLLM package itself does not come with UI, and is designed in a
modular way to hook to any of the UI components. The following code snippet
contains part of the program that generates a streaming response on a webpage.
You can check out [examples/get-started](examples/get-started/) to see the complete example.

```typescript
async function main() {
  // create a ChatModule,
  const chat = new ChatModule();
  // This callback allows us to report initialization progress
  chat.setInitProgressCallback((report: InitProgressReport) => {
    setLabel("init-label", report.text);
  });
  // pick a model. Here we use red-pajama
  const localId = "RedPajama-INCITE-Chat-3B-v1-q4f32_0";
  await chat.reload(localId);

  // callback to refresh the streaming response
  const generateProgressCallback = (_step: number, message: string) => {
    setLabel("generate-label", message);
  };
  const prompt0 = "What is the capital of Canada?";
  // generate  response
  const reply0 = await chat.generate(prompt0, generateProgressCallback);
  console.log(reply0);

  const prompt1 = "How about France?";
  const reply1 = await chat.generate(prompt1, generateProgressCallback)
  console.log(reply1);

  // We can print out the status
  console.log(await chat.runtimeStatsText());
}
```

Finally, you can find a complete
You can also find a complete chat app in [examples/simple-chat](examples/simple-chat/).

## Customized Model Weights

WebLLM works as a companion project of [MLC LLM](https://github.com/mlc-ai/mlc-llm).
It reuses the model artifact and builds flow of MLC LLM, please check out MLC LLM document
on how to build new model weights and libraries (MLC LLM document will come in the incoming weeks).
To generate the wasm needed by WebLLM, you can run with `--target webgpu` in the mlc llm build.
There are two elements of the WebLLM package that enables new models and weight variants.

- model_url: Contains a URL to model artifacts, such as weights and meta-data.
- model_lib: The web assembly libary that contains the executables to accelerate the model computations.

Both are customizable in the WebLLM.

```typescript
async main() {
  const myLlamaUrl = "/url/to/my/llama";
  const appConfig = {
  "model_list": [
    {
      "model_url": myLlamaUrl,
      "local_id": "MyLlama-3b-v1-q4f32_0"
    }
  ],
  "model_lib_map": {
    "llama-v1-3b-q4f32_0": "/url/to/myllama3b.wasm",
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
  //
  // Let us assume that myLlamaUrl/mlc-config.json contains a model_lib
  // field that points to "llama-v1-3b-q4f32_0"
  // then chat module will initialize with these information
  await chat.reload("MyLlama-3b-v1-q4f32_0", chatOpts, appConfig);
}
```

In many cases, we only want to supply the model weight variant, but
not necessarily a new model. In such cases, we can reuse the model lib.
In such cases, we can just pass in the `model_list` field and skip the model lib,
and make sure the `mlc-chat-config.json` in the model url has a model lib
that points to a prebuilt version, right now the prebuilt lib includes

- `vicuna-v1-7b-q4f32_0`: llama-7b models.
- `RedPajama-INCITE-Chat-3B-v1-q4f32_0`: RedPajama-3B variant.


## Build WebLLM Package From Source

WebLLM package is a web runtime designed for [MLC LLM](https://github.com/mlc-ai/mlc-llm).

1. Install all the prerequisites for compilation:
    1. [emscripten](https://emscripten.org). It is an LLVM-based compiler that compiles C/C++ source code to WebAssembly.
        - Follow the [installation instruction](https://emscripten.org/docs/getting_started/downloads.html#installation-instructions-using-the-emsdk-recommended) to install the latest emsdk.
        - Source `emsdk_env.sh` by `source path/to/emsdk_env.sh`, so that `emcc` is reachable from PATH and the command `emcc` works.
    4. Install jekyll by following the [official guides](https://jekyllrb.com/docs/installation/). It is the package we use for website.
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

    You can then go to the subfolders in [examples] to validate some of the sub-packages.
    We use Parcelv2 for bundling. Although Parcel is not very good at tracking parent directory
    changes sometimes. When you make a change in the WebLLM package, try to edit the `package.json`
    of the subfolder and save it, which will trigger Parcel to rebuild.


## Links

- [Demo page](https://mlc.ai/web-llm/)
- If you want to run LLM on native runtime, check out [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- You might also be interested in [Web Stable Diffusion](https://github.com/mlc-ai/web-stable-diffusion/).

## Acknowledgement

This project is initiated by members from CMU catalyst, UW SAMPL, SJTU, OctoML and the MLC community. We would love to continue developing and supporting the open-source ML community.

This project is only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort. The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities make these models accessible. We would like to thank the teams behind vicuna, SentencePiece, LLaMA, Alpaca. We also would like to thank the WebAssembly, Emscripten, and WebGPU communities. Finally, thanks to Dawn and WebGPU developers.
