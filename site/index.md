---
layout: default
title: Home
notitle: true
---


# Web LLM

This project brings chat with open large language models (LLMs)  directly onto web browsers. Everything runs inside the browser with no server support.  

<screen shot>

We have been seeing amazing progress in generative AI and LLM recently. Thanks to the open-source efforts like Llama, alpaca, vicuna and dolly, we start to see an exciting future of building our own open source language models and personal AI assistant.

These models are usually big and compute-heavy. To build a chat service, we will need a large cluster to run an inference server, while clients send requests to servers and retrieve the inference output. We also usually have to run on a specific type of GPUs where popular deep-learning frameworks are readily available.

This project is our step to bring more diversity to the ecosystem. Specifically, can we simply bake LLMs directly into the client side and directly run them inside a browser? If that can be realized, we could offer support for client personal AI models with the benefit of cost reduction, enhancement for personalization and privacy protection. The client side is getting pretty powerful. For example, the latest MacBook Pro can have more than 60G+ unified GPU RAM that can be used to store the model weights and a reasonably powerful GPU to run many of the workloads.

Won’t it be even more amazing if we can simply open up a browser and directly bring AI natively to your browser tab? There is some level of readiness in the ecosystem. This project provides an affirmative answer to the question

#### Additional Information

WebGPU just shipped to Chrome and is in beta. We do our experiments in [Chrome Canary](https://www.google.com/chrome/canary/).  You can also try out the latest chrome 113.

To get the maximum performance, 

##### Instructions

If you have a Mac computer with Apple silicon, here are the instructions for you to run stable diffusion on your browser locally:

- Install [Chrome Canary](https://www.google.com/chrome/canary/), a developer version of Chrome that enables the use of WebGPU.

- Launch Chrome Canary. You are recommended to launch from terminal with the following command (or replace canary with chrome):

  ```
  /Applications/Google\ Chrome\ Canary.app/Contents/MacOS/Google\ Chrome\ Canary --enable-dawn-features=disable_robustness
  ```

  This command turns off the robustness check from Chrome Canary that slows down image generation to times. It is not necessary, but we strongly recommend you to start Chrome with this command.

- Enter your inputs, click “Send” – we are ready to go! The chat bot will first fetch model parameters into local cache. The download may take a few minutes, only for the first run. The subsequent refreshes and runs will be faster.

#### Chat Demo
{% include llm_chat.html %}

#### Disclaimer
This demo site is for research purposes only, subject to the model License of LLaMA and Vicuna. Please contact us if you find any potential violation.


