Experimental component graph execution
======================================

``createArtifactGraphExecution`` runs a validated schema-version-2 component
pipeline with caller-supplied backends.  It is a host scheduler, not a Qwen Omni
model port or a WebGPU binding.  Normal ``MLCEngine`` loading continues to reject
v2 manifests before downloading model configurations, libraries or weights.
Version 1 model loading is unchanged.

Execution accepts a matching package/compiled contract pair, a task name,
named input streams, a component factory and an optional set of output names.
It verifies the canonical interface identity and component specifications before
creating backend instances.  Only components needed by the selected outputs
are created.  Omitting an optional speech output therefore avoids creating its
speech and waveform components when no selected output depends on them.

Each input is an ``AsyncIterable<GraphTensor>``.  A tensor contains a typed-array
payload and its shape.  ``int32``, ``float32`` and ``uint8`` use their corresponding
JavaScript typed arrays; ``float16`` uses ``Uint16Array`` containing binary16 bits.
The scheduler validates dtype, rank, fixed dimensions, repeated dimension names
within a tensor, element count and the configured per-tensor byte limit.  A
``value`` port carries exactly one tensor; a ``stream`` port can carry zero or
more.  Cross-port temporal alignment and model-specific symbolic dimensions
remain the backend's semantic responsibility.

The factory creates an isolated backend for each selected component and request.
It is responsible for resolving module exports and weights and checking device
requirements; the scheduler validates their declared contracts, not the binary
contents or the capabilities of a GPU.
Its ``run`` method consumes a merged stream of tagged ``data`` and ``end`` events
and yields named output tensors.  Input order is preserved per port; arrival
order across distinct ports is not prescribed.  Every input port receives an
explicit end event.  A backend must consume its inputs through completion, then
yield any final buffered or decoder-flush outputs before returning.  It must
honor the abort signal and release request-local state in ``dispose``.  A failed
factory creation must release its own partially allocated resources.

Queues are bounded by ``queueCapacity`` and propagate backpressure.  Fan-out
copies isolate consumers from each other's tensor mutations.  The per-tensor
``maxTensorBytes`` limit and queue capacities bound individual queued payloads;
they do not estimate total model residency or memory allocated inside a backend.
Consumers must drain all selected output streams concurrently.  Awaiting one
stream to completion before reading another can block the producer by design.

.. code-block:: typescript

   const execution = await createArtifactGraphExecution({
     model: packageManifest,
     compiled: compiledManifest,
     task: "chat.completions",
     inputs: inputStreams,
     outputs: ["text", "audio"],
     factory: componentFactory,
     signal: abortController.signal,
   });

   await Promise.all([
     ...Object.entries(execution.outputs).map(async ([name, stream]) => {
       for await (const tensor of stream) {
         await consumeOutput(name, tensor);
       }
     }),
     execution.done,
   ]);

``done`` settles after output drain and backend disposal.  Input failures,
backend failures, early output-consumer return and explicit cancellation stop
the whole request and unblock its queues.  Pending foreign iterators are raced
against cancellation, and their cleanup is requested best-effort.  Backends
remain responsible for stopping their own asynchronous work and settling their
disposal operations; the scheduler cannot forcibly terminate arbitrary caller
code.  The original request failure is preserved if cleanup also fails.

The deterministic test backends demonstrate mixed text, audio, image and video
inputs feeding both text and audio outputs.  They do not verify any real model's
processors, logits, codec output, waveform quality or browser memory fit.
