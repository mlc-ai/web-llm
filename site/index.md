---
layout: default
title: Home
notitle: true
---
<script>
  $(function(){
    $("#llm_chat").load("llm_chat.html");
  });
</script>

# Web LLM

This project brings large-language model and LLM-based chatbot to web browsers. **Everything runs inside the browser with no server support and accelerated with WebGPU.** This opens up a lot of fun opportunities to build AI assistants for everyone and enable privacy while enjoying GPU acceleration. Please check out our [GitHub repo](https://github.com/mlc-ai/web-llm) to see how we did it. There is also a [demo](#chat-demo) which you can try out.


<img src="img/fig/pitts.png" width="100%">

We have been seeing amazing progress in generative AI and LLM recently. Thanks to the open-source efforts like LLaMA, Alpaca, Vicuna and Dolly, we start to see an exciting future of building our own open source language models and personal AI assistant.

These models are usually big and compute-heavy. To build a chat service, we will need a large cluster to run an inference server, while clients send requests to servers and retrieve the inference output. We also usually have to run on a specific type of GPUs where popular deep-learning frameworks are readily available.

This project is our step to bring more diversity to the ecosystem. Specifically, can we simply bake LLMs directly into the client side and directly run them inside a browser? If that can be realized, we could offer support for client personal AI models with the benefit of cost reduction, enhancement for personalization and privacy protection. The client side is getting pretty powerful.

Won’t it be even more amazing if we can simply open up a browser and directly bring AI natively to your browser tab? There is some level of readiness in the ecosystem. This project provides an affirmative answer to the question.


## Instructions

WebGPU just shipped to Chrome. You can try out the latest Chrome 113. Chrome version ≤ 112 is not supported, and if you are using it,
the demo will raise an error like `Find an error initializing the WebGPU device OperationError: Required limit (1073741824) is greater than the supported limit (268435456). - While validating maxBufferSize - While validating required limits.`
We have tested it on Windows and Mac, you will need a GPU with about 6GB memory to run Vicuna-7B and about 3GB memory to run RedPajama-3B.
Some of the models requires fp16 support. To enable fp16 shaders, you will need to use the following instruction(`allow_unsafe_apis`) to turn it on in Chrome Canary.

If you have a Mac computer with Apple silicon, here are the instructions for you to run the chatbot demo on your browser locally:

- Upgrade Chrome to version ≥ 113.
- Launch Chrome. You are recommended to launch from terminal with the following command:
  ```
  /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --enable-dawn-features=allow_unsafe_apis,disable_robustness
  ```
  This command turns off the robustness check from Chrome that slows down chatbot reply to times. It is not necessary, but we strongly recommend you to start Chrome with this command.
- Select the model you want to try out. Enter your inputs, click “Send” – we are ready to go! The chat bot will first fetch model parameters into local cache. The download may take a few minutes, only for the first run. The subsequent refreshes and runs will be faster.

## Chat Demo

The chat demo is based on [vicuna-7b-v1.1](https://huggingface.co/lmsys/vicuna-7b-delta-v1.1) model and [RedPajama-INCITE-Chat-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) model. More model supports are on the way.

<div id="llm_chat"></div>



## Links

- [Web LLM Github](https://github.com/mlc-ai/web-llm)
- You might also be interested in [Web Stable Diffusion](https://websd.mlc.ai/).

## Disclaimer

This demo site is for research purposes only, subject to the model License of LLaMA, Vicuna and RedPajama. Please contact us if you find any potential violation.
