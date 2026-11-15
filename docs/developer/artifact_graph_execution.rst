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

Qwen2.5 Omni audio component binding
----------------------------------

``createAudioEncoderTVMFactory`` binds the
``qwen2_5_omni.audio_encoder.v1`` contract to actual TVM VM exports. The caller
provides a dedicated initialized runtime, device, validated package/compiled
manifests, device capabilities and an already populated parameter cache. The
adapter verifies embedded ``_metadata.artifact``, hashes the parameter schema,
and checks every cached parameter's shape and dtype before binding packed
parameters in compiler order. These checks establish schema agreement, not
weight-file integrity: acquisition must independently verify the weight bytes.

The initial ABI uses float32 mel features ``[mel_bins, frames]`` and returns
float32 embeddings ``[tokens, output_dim]``. ``_metadata.audio_encoder`` supplies
``hidden_size``, ``n_window`` and ``max_pooled_frames``. The caller's clip limit must fit the compiled projection bound. ``encode_chunk`` receives a zero-padded
``[1, mel_bins, 2 * n_window]`` tensor, an int32 ``[1]`` valid-frame tensor, and the
packed parameter tuple. Its output is ``[n_window, hidden_size]``; only the first
``ceil(valid_frames / 2)`` rows are retained. Padding is masked inside the model.
The adapter concatenates valid chunk outputs for one clip before calling
``project(pairs, packed_params)`` once, with pairs shaped
``[floor(valid_hidden_rows / 2), 2, hidden_size]`` and any odd final row dropped.
``project`` performs pair pooling,
normalization and projection. Clips with fewer than two hidden rows yield an
empty result without a zero-extent GPU call. The required manifest ``run`` export
aliases ``encode_chunk`` as the primary entry point; this contract-specific
adapter invokes both exports and does not treat ``run`` as a whole-clip function.

Each backend owns its VM, parameter tuple and temporary tensors. The caller
retains ownership of the runtime, device and cached weights. Only one request may
use the factory at a time, and its dedicated runtime must not be shared with
other work. Consume through ``createArtifactGraphExecution`` for cancellation
and disposal. GPU synchronization is cooperative, not forcibly interruptible.
``maxFrames`` bounds accepted clips; it is not a global GPU memory admission
policy. Chunk outputs are explicitly copied through host memory, so this binding
does not claim zero-copy execution or throughput improvements.

This is an audio encoder binding, not a complete Omni model: PCM preprocessing,
Thinker/Talker/vocoder integration and model-quality validation are separate.
The ordinary MLCEngine v2 loading gate remains unchanged.

``createWhisperAudioTVMFactory`` binds the separate
``qwen2_5_omni.audio_features.v1`` transform contract. Its value input ``pcm``
contains finite float32 mono samples at 16 kHz; empty clips, oversized clips and
unsupported processor metadata are rejected. It pads to the compiled processor
capacity, calls ``forward(padded_pcm, packed_params)``, then crops each mel row
to ``ceil(valid_samples / 160)`` frames. It does not silently truncate, normalize
or resample input. Metadata ``audio_processor`` specifies ``num_samples``,
``sampling_rate``, ``hop_length`` and ``num_mel_bins``. The pinned Qwen2.5 Omni
processor capacity is 4,800,000 samples (300 seconds); this differs from the
ordinary Whisper default of 30 seconds.

A caller may compose this transform with the audio encoder through the
``mel_features`` edge. Each factory validates its own module's embedded standalone
manifest, while the host executor independently validates the composed graph.
Route each component ID to its corresponding factory; resource URLs and weight
integrity remain the caller's responsibility. Separate TVM runtimes may share one
GPU device, but their lifetimes must be managed together: disposing a runtime may
destroy that shared device. Dispose runtimes only after all graph work has ended.
The initial graph copies mel features through host memory; it is not a zero-copy
multimodal pipeline or complete Omni chat integration.
