Basic Usage
================

Model Records in WebLLM
-----------------------

Each of the model available WebLLM is registered as an instance of
``ModelRecord`` and can be accessed at
`webllm.prebuiltAppConfig.model_list <https://github.com/mlc-ai/web-llm/blob/main/src/config.ts#L293>`__.

Creating an MLCEngine
---------------------

WebLLM APIs are exposed through the ``MLCEngine`` interface. You can create an ``MLCEngine`` instance and loading the model by calling the CreateMLCEngine() factory function.

(Note that loading models requires downloading and it can take a significant amount of time for the very first run without caching previously. You should properly handle this asynchronous call.)

``MLCEngine`` can be instantiated in two ways:
1. Using the factory function ``CreateMLCEngine``.
2. Instantiating the ``MLCEngine`` class directly and using ``reload()`` to load models.

.. code-block:: typescript

   import { CreateMLCEngine, MLCEngine } from "@mlc-ai/web-llm";

    // Initialize with a progress callback
    const initProgressCallback = (progress) => {
        console.log("Model loading progress:", progress);
    };

   // Using CreateMLCEngine
   const engine = await CreateMLCEngine("Llama-3.1-8B-Instruct", { initProgressCallback });

   // Direct instantiation
   const engineInstance = new MLCEngine({ initProgressCallback });
   await engineInstance.reload("Llama-3.1-8B-Instruct");

Under the hood, this factory function ``CreateMLCEngine`` does the following steps for first creating an engine instance (synchronous) and then loading the model (asynchronous). You can also do them separately in your application.

.. code-block:: typescript

    import { MLCEngine } from "@mlc-ai/web-llm";

    // This is a synchronous call that returns immediately
    const engine = new MLCEngine({
        initProgressCallback: initProgressCallback
    });

    // This is an asynchronous call and can take a long time to finish
    await engine.reload(selectedModel);


Chat Completion
---------------

Chat completions can be invoked using OpenAI style chat APIs through the ``engine.chat.completions`` interface of an initialized ``MLCEgnine``. For the full list of parameters and their descriptions, check :ref:`api-reference` for full list of parameters.

(Note: As model is determined at the ``MLCEngine`` initialization time, ``model`` parameter is not supported and will be **ignored**. Instead, call ``CreateMLCEngine(model)`` or ``engine.reload(model)`` to reinitialize the engine to use a specific model.)

.. code-block:: typescript

    const messages = [
        { role: "system", content: "You are a helpful AI assistant." },
        { role: "user", content: "Hello!" }
    ];

    const reply = await engine.chat.completions.create({
        messages,
    });

    console.log(reply.choices[0].message);
    console.log(reply.usage);


Streaming Chat Completion
-------------------------

Streaming chat completion could be enabled by passsing ``stream: true`` parameter to the `engine.chat.completions.create` call configuration. Check :ref:`api-reference` for full list of parameters.

.. code-block:: typescript

    const messages = [
        { role: "system", content: "You are a helpful AI assistant." },
        { role: "user", content: "Hello!" },
    ]

    // Chunks is an AsyncGenerator object
    const chunks = await engine.chat.completions.create({
        messages,
        temperature: 1,
        stream: true, // <-- Enable streaming
        stream_options: { include_usage: true },
    });

    let reply = "";
    for await (const chunk of chunks) {
        reply += chunk.choices[0]?.delta.content || "";
        console.log(reply);
        if (chunk.usage) {
            console.log(chunk.usage); // only last chunk has usage
        }
    }

    const fullReply = await engine.getMessage();
    console.log(fullReply);


Chatbot Examples
----------------

Learn how to use WebLLM to integrate large language models into your applications and generate chat completions through this simple Chatbot example:

- `Example in JSFiddle <https://jsfiddle.net/neetnestor/4nmgvsa2/>`_
- `Example in CodePen <https://codepen.io/neetnestor/pen/vYwgZaG>`_

For an advanced example of a larger, more complicated project, check `WebLLM Chat <https://github.com/mlc-ai/web-llm-chat/blob/main/app/client/webllm.ts>`_.

More examples for different use cases are available in the examples folder.


