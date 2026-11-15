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

Text/audio Thinker binding
~~~~~~~~~~~~~~~~~~~~~~~~~

``createThinkerTVMFactory`` binds the restricted
``qwen2_5_omni.thinker_audio.v1`` contract to a preloaded TVM runtime. It accepts
explicit token IDs and audio embeddings, verifies that audio placeholder counts
match, and rejects image/video tokens. Callers remain responsible for the pinned
chat template and audio preprocessing. The adapter replaces each audio placeholder
with its corresponding embedding before chunked prefill, then streams greedy token
IDs until an EOS token or the configured generation/context limit. EOS tokens are
excluded from the output.

The compiled contract must map ``embed``, ``prefill``, ``decode``,
``create_state``, and ``reset_state`` roles to module exports. The initial
``create_state`` role maps to ``create_tir_paged_kv_cache``. Cache creation uses the standard paged-KV
shape arguments; each request gets a fresh cache, resets it, registers sequence
zero, and brackets every forward operation with the TVM cache globals. Metadata
binds model dtype, hidden/vocabulary sizes, special tokens, and context/prefill
limits. Float32 audio is converted to float16 with nearest-even rounding when
required; nonfinite or overflowing values are rejected before embedding.

This initial implementation stages prompt embeddings and logits through CPU memory
for correctness and does not promise optimized throughput. Runtime and GPU-device
ownership follow the audio bindings above. Synthetic-weight Chrome tests cover
real VM execution, audio insertion, greedy decoding, and repeated requests; those
tests do not establish pretrained Omni quality or complete multimodal support.

Bounded waveform decoder
~~~~~~~~~~~~~~~~~~~~~~~~

``createWaveformDecoderTVMFactory`` binds the separate
``qwen2_5_omni.waveform_decoder.v1`` contract to a preloaded TVM runtime.
Its ``log_mel`` value input contains finite float32 synthesis features shaped
``[80, frames]``; these are distinct from the 128-bin Whisper input features.
The factory calls ``forward(log_mel, packed_params)`` and returns one ``pcm``
value containing ``240 * frames`` float32 samples at 24 kHz. Metadata
``waveform_decoder`` binds ``mel_dim``, ``sampling_rate``, ``samples_per_frame``
and ``max_mel_frames``. The caller's positive ``maxFrames`` option must not exceed
that compiled limit. Empty, oversized, nonfinite and malformed clips are rejected.

The factory validates the named ports, embedded manifest, parameter schema and
cached parameter shapes before execution. Runtime/device ownership and cancellation
follow the audio bindings above. A request owns one complete bounded clip; splitting
clips into independent calls changes convolution boundary behavior and is not
streaming synthesis. Talker codec generation and DiT codec-to-mel sampling are
separate stages; this binding alone does not turn text or codec tokens into speech.

Reference conditioning
~~~~~~~~~~~~~~~~~~~~~~

``createReferenceConditionerTVMFactory`` binds the separate
``qwen2_5_omni.reference_conditioner.v1`` encoder. It accepts finite float32
reference mels shaped ``[frames, 80]`` and returns a 128-dimensional embedding.
The frame count must be at least five for reflection padding and no greater than
the compiled limit. Null conditioning evaluates the same network on zero mels;
it does not replace the output with zeros. The factory checks the artifact,
parameter schema, input/output shapes and cancellation like the audio bindings.

Explicit flow prediction and integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``createFlowPredictorTVMFactory`` binds
``qwen2_5_omni.flow_predictor.v1`` to a preloaded runtime. Its five value inputs
are ``noise[frames, mel_dim]``, ``time[1]``, int32 ``code[frames / repeats]``,
``reference[reference_dim]`` and ``speaker[speaker_dim]``. It validates finite
float32 values, time within ``[0, 1]``, codec IDs within the checkpoint vocabulary,
exact repeat counts and the compiled frame limit before invoking the VM. The
``velocity`` output has the same shape as the noise input.

``integrateFlowRK4`` orchestrates a supplied asynchronous predictor using the
pinned RK4 3/8 method. The caller supplies initial noise, a strictly increasing
float32 time grid, a time-point limit, model dimensions/bounds, guidance scale,
and distinct conditional/null reference embeddings. A predictor can execute the
standalone component graph for each call. Every conditional and null result is
checked independently for dtype, shape and finite values before guidance.
Cancellation is checked before and after each prediction. No random noise or
time warping is implicit. This implementation stages values through host memory
and does not promise optimized sampling throughput.

The null reference embedding must come from evaluating the reference conditioner
on zero reference mels, rather than substituting a zero vector. The output mel
array is shaped ``[frames, mel_dim]`` and must be transposed before passing it to
the separate waveform decoder. The codec-generation binding below is a separate
component; callers still supply the cross-component conditioning and sequencing.

Cached codec generation
~~~~~~~~~~~~~~~~~~~~~~~

``createTalkerTVMFactory`` binds ``qwen2_5_omni.talker.v1`` to a preloaded FP32
Talker runtime. Inputs are value tensors ``prefix_embeddings[prefix, 2048]`` and
``reply_embeddings[reply, 2048]``; the value output is int32 ``code[codes]``. The
caller prepares Thinker hidden-state/token-embedding alignment, speaker BOS and
text EOS/PAD conditioning using the pinned HF generation contract. The factory
adds codec PAD/BOS embeddings to the final two prefix rows and advances through
reply conditioning as it decodes, repeating the last row when needed. This
component does not infer alignment from text or implement vision/video input.

Options require ``contextWindowSize`` and ``maxCodes`` (at most 256 for the
current 512-frame flow predictor). Neutral greedy decoding is the default.
Optional ``sampling`` takes ``temperature``, ``topK``, ``topP``,
``repetitionPenalty`` and a caller-provided ``random()`` returning a finite value
in ``[0, 1)``. Sampling applies full-history repetition penalty, codec BOS
suppression, temperature, top-k and top-p, then inverse-CDF selection. Controlled
uniforms reproduce the categorical policy, not PyTorch's seeded RNG algorithm.
Equal-score ties use token-ID order; floating-point filtering near boundaries can
differ from HF.
Codec PAD/EOS stop generation and are stripped; unsupported codec IDs reject.
A cap-limited result is bounded output and does not establish natural EOS.

The verified completed-answer fixture uses captured FP32 Chrome Thinker states
and matches all 125 HF codec IDs and natural EOS. FP16 Thinker hidden-state
parity is not established by token agreement. The full 60,000-sample waveform
experiment has RMSE 8.01e-6 and maximum absolute error 5.57e-4; nine samples exceed
the predeclared pointwise tolerance. That composed waveform gate remains failed,
separate from the passing codec comparison. One independent ASR smoke check
recovers the expected sentence; it is not a speech-quality benchmark.
