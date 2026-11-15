Getting Started with WebLLM
===========================

This guide will help you set up WebLLM in your project, install necessary dependencies, and verify your setup.


WebLLM Chat
-----------

If you want to experience AI Chat supported by local LLM inference and understand how WebLLM works, try out `WebLLM Chat <https://chat.webllm.ai/>`__, which provides a great example
of integrating WebLLM into a full web application.

A WebGPU-compatible browser is needed to run WebLLM-powered web applications.
You can download the latest Google Chrome and use `WebGPU Report <https://webgpureport.org/>`__
to verify the functionality of WebGPU on your browser.

Installation
------------

WebLLM offers a minimalist and modular interface to access the chatbot in the browser. The package is designed in a modular way to hook to any of the UI components.

WebLLM is available as an `npm package <https://www.npmjs.com/package/@mlc-ai/web-llm>`_ and is also CDN-delivered. Therefore, you can install WebLLM using Node.js package managers like npm, yarn, or pnpm, or directly import the pacakge via CDN.

Using Package Managers
^^^^^^^^^^^^^^^^^^^^^^
Install WebLLM via your preferred package manager:

.. code-block:: bash

   # npm
   npm install @mlc-ai/web-llm
   # yarn
   yarn add @mlc-ai/web-llm
   # pnpm
   pnpm install @mlc-ai/web-llm

Import WebLLM into your project:

.. code-block:: javascript

   // Import everything
   import * as webllm from "@mlc-ai/web-llm";

   // Or only import what you need
   import { CreateMLCEngine } from "@mlc-ai/web-llm";

Using CDN
^^^^^^^^^
Thanks to `jsdelivr.com <https://www.jsdelivr.com/package/npm/@mlc-ai/web-llm>`_, WebLLM can be imported directly through URL and work out-of-the-box on cloud development platforms like `jsfiddle.net <https://jsfiddle.net/>`_, `Codepen.io <https://codepen.io/>`_, and `Scribbler <https://scribbler.live/>`_:

.. code-block:: javascript

   import * as webllm from "https://esm.run/@mlc-ai/web-llm";

This method is especially useful for online environments like CodePen, JSFiddle, or local experiments.

Verifying Installation
^^^^^^^^^^^^^^^^^^^^^^
Run the following script to verify the installation:

.. code-block:: javascript

   import { CreateMLCEngine } from "@mlc-ai/web-llm";
   console.log("WebLLM loaded successfully!");


Online IDE Sandbox
------------------

Instead of setting WebLLM locally, you can also try it on online Javascript IDE sandboxes like:

- `Example in JSFiddle <https://jsfiddle.net/neetnestor/4nmgvsa2/>`_
- `Example in CodePen <https://codepen.io/neetnestor/pen/vYwgZaG>`_


