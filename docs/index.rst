👋 Welcome to WebLLM
====================

`GitHub <https://github.com/mlc-ai/web-llm>`_ | `WebLLM Chat <https://chat.webllm.ai/>`_ | `NPM <https://www.npmjs.com/package/@mlc-ai/web-llm>`_ | `Discord <https://discord.gg/9Xpy2HGBuD>`_

WebLLM is a high-performance in-browser language model inference engine that brings large language models (LLMs) to web browsers with hardware acceleration. With WebGPU support, it allows developers to build AI-powered applications directly within the browser environment, removing the need for server-side processing and ensuring privacy.

It provides a specialized runtime for the web backend of MLCEngine, leverages
`WebGPU <https://www.w3.org/TR/webgpu/>`_ for local acceleration, offers OpenAI-compatible API,
and provides built-in support for web workers to separate heavy computation from the UI flow.

Key Features
------------
- 🌐 In-Browser Inference: Run LLMs directly in the browser
- 🚀 WebGPU Acceleration: Leverage hardware acceleration for optimal performance
- 🔄 OpenAI API Compatibility: Seamless integration with standard AI workflows
- 📦 Multiple Model Support: Works with Llama, Phi, Gemma, Mistral, and more

Start exploring WebLLM by `chatting with WebLLM Chat <https://chat.webllm.ai/>`_, and start building webapps with high-performance local LLM inference with the following guides and tutorials.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/get_started.rst
   user/basic_usage.rst
   user/advanced_usage.rst
   user/api_reference.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/building_from_source.rst
   developer/add_models.rst
