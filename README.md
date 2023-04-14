# Web LLM

This project brings language model chats directly onto web browsers. Everything runs inside the browser with no server support.  

<screen shot>

We have been seeing amazing progress in generative AI and LLM recently. Thanks to the open-source efforts like Llama, alpaca, vicuna and dolly, we start to see an exciting future of building our own open source language models and personal AI assistant.

These models are usually big and compute-heavy. To build a chat service, we will need a large cluster to run an inference server, while clients send requests to servers and retrieve the inference output. We also usually have to run on a specific type of GPUs where popular deep-learning frameworks are readily available.

This project is our step to bring more diversity to the ecosystem. Specifically, can we simply bake LLMs directly into the client side and directly run them inside a browser? If that can be realized, we could offer support for client personal AI models with the benefit of cost reduction, enhancement for personalization and privacy protection. The client side is getting pretty powerful. For example, the latest MacBook Pro can have more than 60G+ unified GPU RAM that can be used to store the model weights and a reasonably powerful GPU to run many of the workloads.

Won’t it be even more amazing if we can simply open up a browser and directly bring AI natively to your browser tab? There is some level of readiness in the ecosystem. WebGPU just shipped and enables native GPU executions on the browser.

Still, there are big hurdles to cross, to name a few:

- We need to bring the models somewhere without the relevant GPU accelerated python frameworks.
- Most of the AI frameworks have a heavy reliance on optimized computed libraries that are maintained by hardware vendors. We need to start from zero.
- Careful planning of memory usage, and aggressive compression of weights so we can fit the models into memory. 

We also do not want to only do it for just one model. Instead, we would like to present a repeatable and hackable workflow that enables anyone to easily develop and optimize these models in a productive python first approach, and universally deploy them everywhere, including the web. 

Besides supporting WebGPU, this project also provides the harness for other kinds of GPU backends that TVM supports (such as CUDA, OpenCL, vulkan) and really enables accessible deployment of LLM models.

## Inference on native GPU runtime with command line interface (Coming soon)

- Install MLC package. 

  ```
  pip3 install mlc-ai-nightly -f https://mlc.ai/wheels
  ```

- Get Model

  You need to retrieve model weights first before building model.

  - Vicuna: See [here](https://github.com/lm-sys/FastChat#vicuna-weights)  for instructions on getting vicuna weights.
  - LLaMA: Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama)

  Create a soft link of your model path under dist/models

  ```
  ln -s your_model_path dist/models/your_model
  ```

  Supported models for now: vicuna-7b, llama-7b

- Import, optimize and build model

  ```
  python3 build.py —model vicuna-7b
  ```

- To run chat bot on a native GPU runtime 

  ```
  python3 chat.py --model vicuna-7b [--max-gen-len 512 (default to 128)]
  ```



## How 

The key technology here is machine learning compilation (MLC). Our solution builts on the shoulders of the open source ecosystem, including huggingface, model variants from llama and vicuna, wasm and webgpu. The main flow builds on on Apache TVM Unity, an exciting on going development in the [Apache TVM Community](https://tvm.apache.org/) 

- We bake a language model's IRModule in TVM with native dynamic shape support, avoiding the need of padding to max length and reducing both computation amount and memory usage.
- Each function in TVM’s IRModule can be further transformed and generate runnable code that can be deployed universally on any environment that is supported by minimum tvm runtime (javascript being one of them).
- [TensorIR](https://arxiv.org/abs/2207.04296) is the key technique used to generate optimized programs. We provide productive solutions by quickly transforming TensorIR programs based on the combination of expert knowledge and automated scheduler.
- Heuristics are used when optimizing light-weight operators in order to reduce the engineering pressure..
- We utilize int4 quantization technique to compress the model weights so that they can fit into memory.
- We build static memory planning optimizations to reuse memory across multiple layers.
- [Emscripten](https://emscripten.org/) and typescript to build a TVM web runtime that can deploy generated modules.
- We also leveraged a wasm port of sentencepiece tokenizer.

<img src="site/img/fig/web-llm.svg" alt="web-llm" />


All parts of this workflow are done in Python, with the exception of course, of the last part that builds a 600 loc javascript app that connects things together. This is also a fun process of interactive development, bringing new models.

All these are made possible by the open-source ecosystem that we leverage. Specifically, we make heavy use of [TVM unity](https://discuss.tvm.apache.org/t/establish-tvm-unity-connection-a-technical-strategy/13344), an exciting latest development in the TVM project that enables such Python-first interactive MLC development experiences that allows us to easily compose new optimizations, all in Python, and incrementally bring our app to the web. 

TVM unity also provides an easy way to compose new solutions in the ecosystem. We will continue to bring further optimizations such as fused quantization kernels, and bring them to more platforms.

One key characteristic of LLM models is the dynamic nature of the model. As the decoding and encoding process depends on computations that grow with the size of tokens. We leveraged the first-class dynamic shape support in TVM unity that represents sequence dimensions through symbolic integers. This allows us to plan ahead to statically allocate all the memory needed for the sequence window of interest without padding.

We also leveraged the integration of tensor expressions to quickly express partial-tensor computations such as rotary embedding directly without materializing them into full-tensor matrix computations.


## Comparison to Native GPU Runtime, Limitations and Opportunities

Besides the webgpu runtime, we also provide options for native deployment with local GPU runtime. So they can be used both as a tool to deploy on native environment as well as a reference point to compare native GPU driver performance and WebGPU.

WebGPU works by translating WGSL shaders to native shaders. We observed that there are opportunities to reach zero gap  between  the web gpu runtime and native environment.

Some of the current gaps are caused by chrome's WebGPU implementation inserts bound clips for all array index access, such that `a[i]` becomes `a[min(i, a.size)]`. This can be optimized out as the webgpu support continues to mature

You can get around this by using a special flag to launch chrome(Thanks to Dawn developers for providing the pointers), by exiting chrome completely, then in command line, type

```
/path/to/chrome --enable-dawn-features=disable_robustness
```

Then you will find that the execution speed is as fast as native gpu environment. We anticipate this problem will get resolved as WebGPU matures. WebGPU just shipped and we are excited to see opportunities it can unblock. There are also a lot of exciting upcoming features we can leverage to further improve things such as fp16 extensions.


## Acknowledgement

This project is made possible thanks to collaboration with 

<a href="https://www.scs.cmu.edu">
<img src="site/img/logo/cmuscs.png" alt="CMU School of Computer Science" height="50"/>
</a>
<a href="https://catalyst.cs.cmu.edu">
<img src="site/img/logo/catalyst.svg" alt="Catalyst" height="50"/>
</a>
<a href="https://mlc.ai">
<img src="site/img/logo/mlc-logo-with-text-landscape.svg" alt="MLC" height="50"/>
</a>
<a href="https://octoml.ai">
<img src="site/img/logo/octoml.png" alt="OctoML" height="50"/>
</a>
<a href="https://www.cs.washington.edu/">
<img src="site/img/logo/uw.jpg" alt="UW" height="50"/>
</a>
<a href="https://en.sjtu.edu.cn/">
<img src="site/img/logo/sjtu.png" alt="SJTU" height="50"/>
</a>

This project is only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort. The open-source ML community members made these models publicly available. PyTorch and hugging face communities that make these models accessible. We would like to thank the teams behind vicuna, sentencepiece, Llama, alpaca. We also would like to thank the WebAssembly, Emscripten, and WebGPU communities. Finally, thanks to Dawn and WebGPU developers.