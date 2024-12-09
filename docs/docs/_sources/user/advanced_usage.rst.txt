Advanced Use Cases
==================

Using Workers
-------------

You can put the heavy computation in a worker script to optimize your application performance. To do so, you need to:

Create a handler in the worker thread that communicates with the frontend while handling the requests.
Create a Worker Engine in your main application, which under the hood sends messages to the handler in the worker thread.
For detailed implementations of different kinds of Workers, check the following sections.

Using Web Workers
^^^^^^^^^^^^^^^^^
WebLLM comes with API support for `Web Workers <https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers>`_ so you can offload the computation-heavy generation work into a separate worker thread. WebLLM has implemented the cross-thread communication through messages under the hood so you don't need to manually implement it any more.

In the worker script, import and instantiate ``WebWorkerMLCEngineHandler``, which handles the communications with other scripts and processes incoming requests.

.. code-block:: typescript

   // worker.ts
   import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

   const handler = new WebWorkerMLCEngineHandler();
   self.onmessage = (msg: MessageEvent) => {
       handler.onmessage(msg);
   };

In the main script, import and instantiate a ``WebWorkerMLCEngine`` that implements the same ``MLCEngineInterface`` and exposes the same APIs, then simply use it as how you would use a normal ``MLCEngine`` in your application.

.. code-block:: typescript

   import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";

   async function runWorker() {
       const engine = await CreateWebWorkerMLCEngine(
           new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
           "Llama-3.1-8B-Instruct"
       );

       const messages = [{ role: "user", content: "How does WebLLM use workers?" }];
       const reply = await engine.chat.completions.create({ messages });
       console.log(reply.choices[0].message.content);
   }

   runWorker();


Under the hood, ``WebWorkerMLCEngine`` does **not** actual doing any computation, but instead serves as a proxy to translate all calls into messages and send to the ``WebWorkerMLCEngineHandler`` to process. The worker thread will receive these messages and process the actual computation using a hidden engine, and return the result back to the main thread using messages.

Service Workers
^^^^^^^^^^^^^^^
WebLLM also support offloading the computation in `Service Workers <https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API>`_ to avoid reloading the model between page refreshes and optimize your application's offline experience.

(Note, Service Worker's life cycle is managed by the browser and can be killed any time without notifying the webapp. WebLLM's ``ServiceWorkerMLCEngine`` will try to keep the service worker thread alive by periodically sending heartbeat events, but the script could still be killed any time by Chrome and your application should include proper error handling. Check `keepAliveMs` and `missedHeatbeat` in `ServiceWorkerMLCEngine <https://github.com/mlc-ai/web-llm/blob/main/src/service_worker.ts#L234>`_ for more details.)

In the worker script, import and instantiate ``ServiceWorkerMLCEngineHandler``, which handles the communications with page scripts and processes incoming requests.

.. code-block:: typescript

   // sw.ts
   import { ServiceWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

   self.addEventListener("activate", () => {
       const handler = new ServiceWorkerMLCEngineHandler();
       console.log("Service Worker activated!");
   });


Then in the main page script, register the service worker and instantiate the engine using ``CreateServiceWorkerMLCEngine`` factory function. The Engine implements the same ``MLCEngineInterface`` and exposes the same APIs, then simply use it as how you would use a normal ``MLCEngine`` in your application.

.. code-block:: typescript

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

Similar to the ``WebWorkerMLCEngine`` above, the ``ServiceWorkerMLCEngine`` is also a proxy and does not do any actual computation. Instead it sends all calls to the service worker thread to handle and receives the result back through messages.

Chrome Extension
----------------

WebLLM can be used in Chrome extensions to empower local LLM inference. You can find examples of building Chrome extension using WebLLM in `examples/chrome-extension <https://github.com/mlc-ai/web-llm/blob/main/examples/chrome-extension>`_ and `examples/chrome-extension-webgpu-service-worker <https://github.com/mlc-ai/web-llm/blob/main/examples/chrome-extension-webgpu-service-worker>`_. The latter one leverages service worker, so the extension is persistent in the background.

Additionally, we have a full Chrome extension project, `WebLLM Assistant <https://github.com/mlc-ai/web-llm-assistant>`_, which leverages WebLLM to provide personal web browsing copilot assistance experience. Free to to check it out and contribute if you are interested.


Other Customization
-------------------

Using IndexedDB Cache
^^^^^^^^^^^^^^^^^^^^^

Set `appConfig` in `MLCEngineConfig` to enable caching for faster subsequent model loads.

.. code-block:: typescript

   const engine = await CreateMLCEngine("Llama-3.1-8B-Instruct", {
       appConfig: {
           useIndexedDB: true,
           models: [
               { model_id: "Llama-3.1-8B", model_path: "/models/llama3" },
           ],
       },
   });

Customizing Token Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^

Modify `logit_bias` in `GenerationConfig` to influence token likelihood:

.. code-block:: typescript

   const messages = [
       { role: "user", content: "Describe WebLLM in detail." },
   ];

   const response = await engine.chatCompletion({
       messages,
       logit_bias: { "50256": -100 }, // Example: Prevent specific token generation
   });
