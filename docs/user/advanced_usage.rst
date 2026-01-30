Advanced Use Cases
==================

Using Workers
-------------

You can put the heavy computation in a worker script to optimize your application performance. To do so, you need to:

Create a handler in the worker thread that communicates with the frontend while handling the requests.
Create a worker engine in your main application that sends messages to the handler in the worker thread under the hood.
For detailed implementations of different kinds of workers, look at the following sections.

Using Web Workers
^^^^^^^^^^^^^^^^^
WebLLM comes with API support for `Web Workers <https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers>`_ so you can offload the computation-heavy generation work into a separate worker thread. WebLLM has implemented cross-thread communication through messages under the hood, so manual implementation is not required.

In the worker script, import and instantiate a ``WebWorkerMLCEngineHandler``, which handles communication with other scripts and processes incoming requests.

.. code-block:: typescript

   // worker.ts
   import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

   const handler = new WebWorkerMLCEngineHandler();
   self.onmessage = (msg: MessageEvent) => {
       handler.onmessage(msg);
   };

In the main script, import and instantiate a ``WebWorkerMLCEngine`` that implements the same ``MLCEngineInterface`` and exposes the same APIs. Then, simply use it as you would a normal ``MLCEngine``.

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


Under the hood, ``WebWorkerMLCEngine`` does **not** perform any computation. It translates all calls into messages and sends them to the ``WebWorkerMLCEngineHandler`` for processing. The worker thread receives these messages and processes the actual computation using a hidden engine, and returns the result to the main thread using messages.

Service Workers
^^^^^^^^^^^^^^^
WebLLM also supports offloading computation using `Service Workers <https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API>`_. This allows you to avoid reloading the model between page refreshes and optimize your application's offline experience.

(Note, the lifecycle of a Service Worker is managed by the browser and can be killed any time without notifying the web application. WebLLM's ``ServiceWorkerMLCEngine`` attempts to keep the service worker thread alive by periodically sending heartbeat events. However, the script could still be killed at any time by Chrome, and your application should include proper error handling. Check `keepAliveMs` and `missedHeartbeat` in `ServiceWorkerMLCEngine <https://github.com/mlc-ai/web-llm/blob/main/src/service_worker.ts#L218>`_ for more details.)

In the worker script, import and instantiate ``ServiceWorkerMLCEngineHandler``, which handles communication with page scripts and processes incoming requests.

.. code-block:: typescript

   // sw.ts
   import { ServiceWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

   self.addEventListener("activate", () => {
       const handler = new ServiceWorkerMLCEngineHandler();
       console.log("Service Worker activated!");
   });


Then, in the main page script, register the service worker and instantiate the engine using the ``CreateServiceWorkerMLCEngine`` factory function that implements the same ``MLCEngineInterface`` and exposes the same APIs. Then, simply use it as you would a normal ``MLCEngine``.

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

Similar to the ``WebWorkerMLCEngine`` above, the ``ServiceWorkerMLCEngine`` is also a proxy and does not perform any actual computation. Instead, it forwards all calls to the service worker thread and receives the result through messages.

Chrome Extension
----------------

WebLLM can be used in Chrome extensions to empower local LLM inference. You can find examples of building Chrome extension using WebLLM in `examples/chrome-extension <https://github.com/mlc-ai/web-llm/blob/main/examples/chrome-extension>`_ and `examples/chrome-extension-webgpu-service-worker <https://github.com/mlc-ai/web-llm/blob/main/examples/chrome-extension-webgpu-service-worker>`_. The latter leverages Service Worker, so the extension is persistent in the background.

Additionally, we have a full Chrome extension project, `WebLLM Assistant <https://github.com/mlc-ai/web-llm-assistant>`_, which leverages WebLLM to provide a personal web browsing copilot assistant experience. Feel free to check it out and contribute if you are interested.


Additional Customization
------------------------

Using IndexedDB Cache
^^^^^^^^^^^^^^^^^^^^^

By default, WebLLM caches model artifacts using the `Cache API <https://developer.mozilla.org/en-US/docs/Web/API/Cache>`_ for faster subsequent model loads. You can alternatively use `IndexedDB caching <https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API>`_ by setting the `useIndexedDBCache` field in `appConfig` of `MLCEngineConfig` to `true`.

.. code-block:: typescript

   const engine = await CreateMLCEngine("Llama-3.1-8B-Instruct", {
       appConfig: {
           useIndexedDBCache: true,
           models: [
               { model_id: "Llama-3.1-8B", model_path: "/models/llama3" },
           ],
       },
   });

Customizing Token Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can modify `logit_bias` in `GenerationConfig` to control token likelihood. Setting a token's bias to a positive value increases its likelihood of being generated, while a negative value decreases it. A large negative value (e.g., -100) can effectively prevent the token from being generated.

.. code-block:: typescript

   const messages = [
       { role: "user", content: "Describe WebLLM in detail." },
   ];

   const response = await engine.chatCompletion({
       messages,
       logit_bias: { "50256": -100 }, // Example: Prevent specific token generation
   });
