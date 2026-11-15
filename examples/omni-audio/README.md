# Audio input to text output

A local developer example for the restricted Qwen2.5-Omni audio graph:
PCM → Whisper features → audio encoder → Thinker text generation. It uses the
public component factories and validates the exact standalone and composed
manifests. It does not enable ordinary `MLCEngine` v2 loading, image/video input,
or speech output.

## Run

Build the three component artifacts with the matching compiler/runtime. Place
(or symlink for local development) these directories in `public/artifacts/`:

- `omni-features/`: feature WASM, model/compiled manifests and tensor cache.
- `omni-audio/`: pretrained audio encoder WASM, manifests and tensor cache.
- `omni-thinker-f32accum-q0f16/`: pretrained Thinker WASM, manifests and tensor cache.
- `omni-reference/tokenizer.json`: tokenizer from the same pinned base model.

Each component directory must retain its generated filenames and weight shards.
The encoder output and Thinker audio input ports must match exactly, including
symbolic dimension names. Edit `src/config.ts` for other local artifact URLs;
no hosted or published model package is assumed.

```sh
npm install
npm run dev
```

Open the displayed localhost URL in Google Chrome with WebGPU and `shader-f16`
support. Select an audio file, enter a plain-text prompt, and click Generate.
Web Audio decodes, downmixes to mono, and resamples to 16 kHz. This example limits
clips to **30 seconds**, despite the feature component's 300-second capacity,
and reserves room for up to 128 generated tokens in the 4096-token context.
The single-turn prompt follows Qwen2.5-Omni's pinned audio-then-text template;
reserved control/media tokens in user text are rejected.

The unquantized components require several GB of downloads and GPU memory.
Each request releases its runtimes afterward. Cancel interrupts model downloads
and graph generation; cleanup waits for in-flight work to settle before releasing
its runtime. Closing the page also releases resources. No audio is uploaded: artifacts
are fetched from your configured server and inference runs locally.

The initial verification uses fixed synthetic PCM probes and independent HF
references; it is not a general audio-quality benchmark.

## Source of truth

The prompt is the audio-then-text subset of the pinned
[Qwen2.5-Omni-3B chat template](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/chat_template.json).
Audio preprocessing follows that revision's
[preprocessor configuration](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/preprocessor_config.json).
The example's 30-second input limit is deliberately smaller than its 300-second
processor capacity. Each template audio marker expands to
`floor(ceil(ceil(samples / 160) / 2) / 2)` encoder embeddings.

Follow the companion MLC-LLM
[`examples/python/omni_audio/README.md`](https://github.com/JCNTH/mlc-llm/tree/codex/omni-audio-review/examples/python/omni_audio)
for pinned source download, weight conversion, and the three component-builder
commands. Use the companion integration branch containing that example until it
is merged; these standalone builders are not installed `mlc_llm` CLI commands.
Use its matching artifact builder and runtime revision; ordinary chat libraries
do not provide these component contracts. See
[the graph execution guide](../../docs/developer/artifact_graph_execution.rst)
for the required export roles, metadata, parameter-schema checks, and ownership.
