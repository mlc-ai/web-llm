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
    - ``initProgressCallback``: A callback for showing model loading progress.
    - ``logitProcessorRegistry``: A registry for stateful logit processors (see ``webllm.LogitProcessor``).


- **Usage**:
    - ``appConfig``: Contains application-specific settings, including:
        - Model configurations.
        - IndexedDB caching preferences.
    - ``initProgressCallback``: Allows developers to visualize model loading progress by implementing a callback.
    - ``logitProcessorRegistry``: A ``Map`` object for registering custom logit processors. Only applies to ``MLCEngine``.


.. note:: All fields are optional, and ``logitProcessorRegistry`` is only used in ``MLCEngine``.


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
    - ``repetition_penalty``, ``ignore_eos``: Parameters specific to MLC models.
    - ``top_p``, ``temperature``, ``max_tokens``, ``stop``: Common parameters shared with OpenAI APIs.
    - ``frequency_penalty``, ``presence_penalty``: Tune repetition behavior following OpenAI semantics.
    - ``logit_bias``, ``n``, ``logprobs``, ``top_logprobs``: Advanced sampling controls.
    - ``response_format``, ``enable_thinking``, ``enable_latency_breakdown``: Additional OpenAI-style request features.

- **Usage**:
    - Fields like ``repetition_penalty`` and ``ignore_eos`` give explicit control over repetition handling and whether the model stops at the EOS token, respectively.
    - Common parameters shared with OpenAI APIs (e.g., ``temperature``, ``top_p``) ensure compatibility while still falling back to the values configured during ``MLCEngine.reload()`` when omitted.
    - ``frequency_penalty`` and ``presence_penalty`` mirror OpenAI's bounds ``[-2, 2]``; providing only one will default the other to ``0``.
    - ``response_format`` (for JSON or other schema outputs), ``enable_thinking``, and ``enable_latency_breakdown`` pass through directly to the engine and surface enhanced telemetry or structured responses when the underlying model supports them.


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

ChatConfig
^^^^^^^^^^

Model's baseline configuration loaded from ``mlc-chat-config.json`` when ``MLCEngine.reload()`` runs. ``ChatOptions`` (and therefore the ``chatOpts`` argument to ``reload``) can override any subset of these fields.

- **Fields** (subset):
    - ``tokenizer_files``, ``tokenizer_info``: Files and parameters required to initialize the tokenizer.
    - ``conv_template``, ``conv_config``: Conversation templates that define prompts, separators, and role formatting.
    - ``context_window_size``, ``sliding_window_size``, ``attention_sink_size``: KV-cache and memory settings.
    - Default generation knobs such as ``repetition_penalty``, ``frequency_penalty``, ``presence_penalty``, ``top_p``, and ``temperature``.

- **Usage**:
    - Loaded automatically for each model; provides defaults that ``GenerationConfig`` falls back to when fields are omitted.
    - Override selected values per model load by supplying ``chatOpts`` (``Partial<ChatConfig>``) to ``MLCEngine.reload()``.


Example:

.. code-block:: typescript

   await engine.reload("Llama-3.1-8B-Instruct", {
       temperature: 0.7,
       repetition_penalty: 1.1,
       context_window_size: 4096,
   });

ChatCompletionRequest
^^^^^^^^^^^^^^^^^^^^^

Defines the structure for chat completion requests.

- **Base Interface**: ``ChatCompletionRequestBase``
    - Contains parameters such as ``messages``, ``stream``, ``frequency_penalty``, and ``presence_penalty``.
- **Sub-interfaces**:
    - ``ChatCompletionRequestNonStreaming``: For non-streaming completions.
    - ``ChatCompletionRequestStreaming``: For streaming completions.

- **Usage**:
    - Combines settings from ``GenerationConfig`` and ``ChatCompletionRequestBase`` to provide complete control over chat behavior.
    - The ``stream`` parameter enables streaming responses, improving interactivity in conversational agents.
    - The ``logit_bias`` feature allows controlling token generation probabilities, providing a mechanism to restrict or encourage specific outputs.


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
    - ``chatOpts``: Configuration for generation (see ``ChatConfig``).

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

- Parameters:
    - ``modelId``: (Optional) Identifier of model to query. Omitting modelId only works when the engine currently has a single model loaded.

``MLCEngine.resetChat(keepStats?: boolean, modelId?: string): Promise<void>``

Resets the chat history and optionally retains usage statistics.

- Parameters:
    - ``keepStats``: (Optional) If true, retains usage statistics.
    - ``modelId``: (Optional) Identifier of the model to reset. Omitting modelId only works when the engine currently has a single model loaded.

GPU Information
----------------

The following methods provide detailed information about the GPU used for WebLLM computations.

``MLCEngine.getGPUVendor(): Promise<string>``

Retrieves the vendor name of the GPU used for computations. This is useful for understanding hardware capabilities during inference.

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
