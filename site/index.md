---
layout: default
title: Home
notitle: true
---

{% include hero.html %}

## Overview

We have been seeing amazing progress in generative AI and LLM recently. Thanks to the open-source efforts like LLaMA, Alpaca, Vicuna and Dolly, we start to see an exciting future of building our own open source language models and personal AI assistant.

These models are usually big and compute-heavy. To build a chat service, we will need a large cluster to run an inference server, while clients send requests to servers and retrieve the inference output. We also usually have to run on a specific type of GPUs where popular deep-learning frameworks are readily available.

This project is our step to bring more diversity to the ecosystem. Specifically, can we simply bake LLMs directly into the client side and directly run them inside a browser? If that can be realized, we could offer support for client personal AI models with the benefit of cost reduction, enhancement for personalization and privacy protection. The client side is getting pretty powerful.

Won’t it be even more amazing if we can simply open up a browser and directly bring AI natively to your browser tab? There is some level of readiness in the ecosystem. This project provides an affirmative answer to the question.

## Key Features
- **In-Browser Inference**: WebLLM is a high-performance, in-browser language model inference engine that leverages WebGPU for hardware acceleration, enabling powerful LLM operations directly within web browsers without server-side processing.

- [**Full OpenAI API Compatibility**](https://github.com/mlc-ai/web-llm?tab=readme-ov-file#full-openai-compatibility): Seamlessly integrate your app with WebLLM using OpenAI API with functionalities such as JSON-mode, function-calling, streaming, and more.

- [**Extensive Model Support**](https://github.com/mlc-ai/web-llm?tab=readme-ov-file#built-in-models): WebLLM natively supports a range of models including Llama, Phi, Gemma, RedPajama, Mistral, Qwen(通义千问), and many others, making it versatile for various AI tasks.

- [**Custom Model Integration**](https://github.com/mlc-ai/web-llm?tab=readme-ov-file#custom-models): Easily integrate and deploy custom models in MLC format, allowing you to adapt WebLLM to specific needs and scenarios, enhancing flexibility in model deployment.

- **Plug-and-Play Integration**: Easily integrate WebLLM into your projects using package managers like NPM and Yarn, or directly via CDN, complete with comprehensive [examples](https://github.com/mlc-ai/web-llm/tree/main/examples) and a modular design for connecting with UI components.

- **Streaming & Real-Time Interactions**: Supports streaming chat completions, allowing real-time output generation which enhances interactive applications like chatbots and virtual assistants.

- **Web Worker & Service Worker Support**: Optimize UI performance and manage the lifecycle of models efficiently by offloading computations to separate worker threads or service workers.

- **Chrome Extension Support**: Extend the functionality of web browsers through custom Chrome extensions using WebLLM, with examples available for building both basic and advanced extensions.

## Disclaimer

The [demo site](https://chat.webllm.ai) is for research purposes only, subject to the model License of LLaMA, Vicuna and RedPajama. Please contact us if you find any potential violation.
