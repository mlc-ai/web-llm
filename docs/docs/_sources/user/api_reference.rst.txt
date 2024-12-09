.. _api-reference:

WebLLM API Reference
====================

The ``MLCEngine`` class is the core interface of WebLLM. It enables model loading, chat completions, embeddings, and other operations. Below, we document its methods, along with the associated configuration interfaces.

Interfaces
----------

The following interfaces are used as parameters or configurations within ``MLCEngine`` methods. They are linked to their respective methods for reference.

MLCEngineConfig
^^^^^^^^^^^^^^^

Optional configurations for ``CreateMLCEngine()`` and ``CreateWebWorkerMLCEngine()``.


- **Fields**:
    - ``appConfig``: Configure the app, including the list of models and whether to use IndexedDB cache.
    - ``initProgressCallback``: A callback for showing the progress of loading the model.
    - ``logitProcessorRegistry``: A register for stateful logit processors, see ``webllm.LogitProcessor``.


- **Usage**:
    - ``appConfig``: Contains application-specific settings, including:
        - Model configurations.
        - IndexedDB caching preferences.
    - ``initProgressCallback``: Allows developers to visualize model loading progress by implementing a callback.
    - ``logitProcessorRegistry``: A ``Map`` object for registering custom logit processors. Only applies to ``MLCEngine``.


.. note:: All fields are optional, and ``logitProcessorRegistry`` is only used for ``MLCEngine``.


Example:

.. code-block:: typescript

   const engine = await CreateMLCEngine("Llama-3.1-8B-Instruct", {
       appConfig: { /* app-specific config */ },
       initProgressCallback: (progress) => console.log(progress),
   });


GenerationConfig
^^^^^^^^^^^^^^^^

Configurations for a single generation task, primarily used in chat completions.

- **Fields**:
    - ``repetition_penalty``, ``ignore_eos``: Specific to MLC models.
    - ``top_p``, ``temperature``, ``max_tokens``, ``stop``: Common with OpenAI APIs.
    - ``logit_bias``, ``n``: Additional parameters for sampling control.

- **Usage**:
    - Fields like ``repetition_penalty`` and ``ignore_eos`` allow fine control over the output generation behavior.
    - Common parameters shared with OpenAI APIs (e.g., ``temperature``, ``top_p``) ensure compatibility.


Example:

.. code-block:: typescript

   const messages = [
       { role: "system", content: "You are a helpful assistant." },
       { role: "user", content: "Explain WebLLM." },
   ];

   const response = await engine.chatCompletion({
       messages,
       top_p: 0.9,
       temperature: 0.8,
       max_tokens: 150,
   });

ChatCompletionRequest
^^^^^^^^^^^^^^^^^^^^^

Defines the structure for chat completion requests.

- **Base Interface**: ``ChatCompletionRequestBase``
    - Contains parameters like ``messages``, ``stream``, ``frequency_penalty``, and ``presence_penalty``.
- **Variants**:
    - ``ChatCompletionRequestNonStreaming``: For non-streaming completions.
    - ``ChatCompletionRequestStreaming``: For streaming completions.

- **Usage**:
    - Combines settings from ``GenerationConfig`` and ``ChatCompletionRequestBase`` to provide complete control over chat behavior.
    - The ``stream`` parameter enables dynamic streaming responses, improving interactivity in conversational agents.
    - The ``logit_bias`` feature allows fine-tuning of token generation probabilities, providing a mechanism to restrict or encourage specific outputs.


Example:

.. code-block:: typescript

   const response = await engine.chatCompletion({
       messages: [
           { role: "user", content: "Tell me about WebLLM." },
       ],
       stream: true,
   });

Model Loading
-------------

``MLCEngine.reload(modelId: string | string[], chatOpts?: ChatOptions | ChatOptions[]): Promise<void>``

Loads the specified model(s) into the engine. Uses ``MLCEngineConfig`` during initialization.

- Parameters:
    - ``modelId``: Identifier(s) for the model(s) to load.
    - ``chatOpts``: Configuration for generation (see ``GenerationConfig``).

Example:

.. code-block:: typescript

   await engine.reload(["Llama-3.1-8B", "Gemma-2B"], [
       { temperature: 0.7 },
       { top_p: 0.9 },
   ]);

``MLCEngine.unload(): Promise<void>``

Unloads all loaded models and clears their associated configurations.

Example:

.. code-block:: typescript

   await engine.unload();

---

Chat Completions
----------------

``MLCEngine.chat.completions.create(request: ChatCompletionRequest): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>``

Generates chat-based completions using a specified request configuration.

- Parameters:
  - ``request``: A ``ChatCompletionRequest`` instance.

Example:

.. code-block:: typescript

   const response = await engine.chat.completions.create({
       messages: [
           { role: "system", content: "You are a helpful AI assistant." },
           { role: "user", content: "What is WebLLM?" },
       ],
       temperature: 0.8,
       stream: false,
   });

---

Utility Methods
^^^^^^^^^^^^^^^

``MLCEngine.getMessage(modelId?: string): Promise<string>``

Retrieves the current output message from the specified model.

``MLCEngine.resetChat(keepStats?: boolean, modelId?: string): Promise<void>``

Resets the chat history and optionally retains usage statistics.

GPU Information
----------------

The following methods provide detailed information about the GPU used for WebLLM computations.

``MLCEngine.getGPUVendor(): Promise<string>``

Retrieves the vendor name of the GPU used for computations. Useful for understanding the hardware capabilities during inference.

- **Returns**: A string indicating the GPU vendor (e.g., "Intel", "NVIDIA").

Example:

.. code-block:: typescript

   const gpuVendor = await engine.getGPUVendor();
   console.log(``GPU Vendor: ${gpuVendor}``);

``MLCEngine.getMaxStorageBufferBindingSize(): Promise<number>``

Returns the maximum storage buffer size supported by the GPU. This is important when working with larger models that require significant memory for processing.

- **Returns**: A number representing the maximum size in bytes.

Example:

.. code-block:: typescript

   const maxBufferSize = await engine.getMaxStorageBufferBindingSize();
   console.log(``Max Storage Buffer Binding Size: ${maxBufferSize}``);
