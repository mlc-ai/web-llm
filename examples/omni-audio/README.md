# Text and audio to speech

This local Qwen2.5-Omni-3B example generates text and speech from a text prompt,
with an optional audio clip. It uses the public component factories and
`createOmniSpeechGraph`: PCM features and audio encoding feed the Thinker;
its actual hidden states condition the Talker, followed by reference conditioning,
DiT flow integration and waveform decoding. No prerecorded hidden states are used.

## Run

Build the component artifacts with the matching compiler/runtime from the
[companion MLC integration](https://github.com/mlc-ai/mlc-llm/pull/3600).
Place these directories in `public/artifacts/` (local symlinks are also suitable):

- `omni-features/` and `omni-audio/`: PCM features and audio encoder artifacts.
- `omni-public-thinker-speech/`: `build_thinker.py --talker --quantization q0f32` artifacts.
- `omni-talker-graph-webgpu/`: cached Talker artifacts.
- `omni-ecapa-webgpu/`: reference conditioner artifacts.
- `omni-dit-webgpu-512/`: flow predictor compiled for 512 mel frames.
- `omni-vocoder-webgpu-512/`: waveform decoder compiled for 512 mel frames.
- `omni-public-speakers/`: `speakers.json` from the pinned `export_speakers.py`.
- `omni-reference/tokenizer.json`: tokenizer from the same pinned model revision.

Retain each component's generated filenames, manifests and tensor-cache shards.
Edit `src/config.ts` to use other local artifact URLs. No hosted model package is
assumed. Ordinary chat libraries do not expose these component contracts.

```sh
npm install
npm run dev
```

Open the localhost URL in Google Chrome with WebGPU, enter a prompt, select Ethan
or Chelsie, and click Generate. Audio input is optional; Web Audio decodes,
downmixes and resamples uploaded clips to mono 16 kHz. Clips are limited to
30 seconds. The example uses a 4096-token context, at most 64 Thinker tokens and
256 codec tokens. Reserved control/media tokens in user text are rejected.

Text appears while the Thinker runs. Speech becomes available as a 24 kHz WAV
once synthesis finishes. An immediate text EOS produces a clear empty-response
message instead of fabricated speech conditioning. A generation cap is reported
separately from natural completion.

The FP32 Thinker requires a storage buffer larger than 1 GiB; the example checks
adapter limits and requests sufficient limits before loading weights. These
unquantized components require substantial downloads and GPU memory. Each request
releases its runtimes. Cancel interrupts downloads and generation, and cleanup
waits for queued GPU work before disposing resources. Input audio stays local;
only configured model artifacts are downloaded.

## Numerical scope

The fixed text and synthetic-audio probes produced 125 and 166 reference codec
tokens respectively, both with natural EOS. Their composed waveforms exceeded
the strict pointwise gate at 9 of 60,000 and 41 of 79,680 samples respectively
(maximum absolute errors 0.000557 and 0.001775; RMSE 0.00000801 and 0.00001728).
This is a component integration example, not an audio-quality benchmark.
Sampling uses temperature 0.9, top-k 40, top-p 0.8 and repetition penalty 1.05.
Equal-score ties use token-ID order; floating-point filtering near boundaries can
differ from HF. The example uses fresh randomness rather than fixture uniforms.

## Source of truth

The single-turn prompt follows the pinned
[Qwen2.5-Omni-3B chat template](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/chat_template.json).
Audio preprocessing follows the same revision's
[preprocessor configuration](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/preprocessor_config.json).
The audio marker count is `floor(ceil(ceil(samples / 160) / 2) / 2)`.
See the compiler integration's `examples/python/omni_audio` builders and
[graph execution guide](../../docs/developer/artifact_graph_execution.rst) for
export roles, manifest validation and resource ownership. Image/video input and
ordinary `MLCEngine` loading are outside this example's contract.
