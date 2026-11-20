import * as tvmjs from "@mlc-ai/web-runtime";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import {
  createAudioEncoderTVMFactory,
  createWhisperAudioTVMFactory,
  createThinkerSpeechTVMFactory,
  createTalkerTVMFactory,
  createReferenceConditionerTVMFactory,
  createFlowPredictorTVMFactory,
  createWaveformDecoderTVMFactory,
  createOmniSpeechGraph,
  renderOmniPrompt,
  EmptyThinkerResponseError,
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  validateGraphArtifactPair,
  type OmniSpeechPart,
  type OmniSpeechParts,
  type GraphTensor,
} from "@mlc-ai/web-llm";
import { config } from "./config";

const audio = document.querySelector<HTMLInputElement>("#audio")!;
const prompt = document.querySelector<HTMLTextAreaElement>("#prompt")!;
const generate = document.querySelector<HTMLButtonElement>("#generate")!;
const cancel = document.querySelector<HTMLButtonElement>("#cancel")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
const playback = document.querySelector<HTMLAudioElement>("#playback")!;
const voice = document.querySelector<HTMLSelectElement>("#voice")!;
let playbackURL: string | undefined;
const runtimes: tvmjs.Instance[] = [];
let controller: AbortController | undefined;
let tokenizer: Tokenizer | undefined;
let device: GPUDevice | undefined;
function dispose() {
  controller?.abort();
  // Each runtime owns the same device: dispose only after all graph work ends.
  for (const runtime of runtimes.splice(0)) {
    try {
      runtime.dispose();
    } catch (error) {
      console.error(error);
    }
  }
  device?.destroy();
  device = undefined;
  tokenizer?.dispose();
  tokenizer = undefined;
}
window.addEventListener("pagehide", () => {
  if (playbackURL) URL.revokeObjectURL(playbackURL);
  // Let the active request settle before freeing its VM resources.
  if (controller) controller.abort();
  else dispose();
});
cancel.onclick = () => controller?.abort();

async function fetchBytes(url: string, signal: AbortSignal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}
async function readPCM(file: File, signal: AbortSignal) {
  if (file.size > 50 * 1024 * 1024)
    throw new Error("Choose a file below 50 MB.");
  const context = new AudioContext({ sampleRate: 16000 });
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    signal.throwIfAborted();
    if (!buffer.duration || buffer.duration > config.maxSeconds) {
      throw new Error(
        `Choose a clip between 0 and ${config.maxSeconds} seconds.`,
      );
    }
    const offline = new OfflineAudioContext(
      1,
      Math.ceil(buffer.duration * 16000),
      16000,
    );
    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start();
    const result = await offline.startRendering();
    signal.throwIfAborted();
    const pcm = result.getChannelData(0).slice();
    if (!pcm.every(Number.isFinite))
      throw new Error("Audio contains nonfinite samples.");
    return pcm;
  } finally {
    await context.close();
  }
}

function normalNoise(count: number): Float32Array {
  const output = new Float32Array(count),
    words = new Uint32Array(16384);
  let offset = words.length;
  const uniform = () => {
    if (offset === words.length) {
      crypto.getRandomValues(words);
      offset = 0;
    }
    return (words[offset++] + 0.5) / 4294967296;
  };
  for (let i = 0; i < count; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(uniform())),
      angle = 2 * Math.PI * uniform();
    output[i] = radius * Math.cos(angle);
    if (i + 1 < count) output[i + 1] = radius * Math.sin(angle);
  }
  return output;
}
function waveformURL(samples: Float32Array): string {
  const bytes = new ArrayBuffer(44 + samples.length * 4),
    view = new DataView(bytes);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++)
      view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 4, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 24000, true);
  view.setUint32(28, 96000, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 32, true);
  text(36, "data");
  view.setUint32(40, samples.length * 4, true);
  for (let i = 0; i < samples.length; i++)
    view.setFloat32(44 + i * 4, samples[i], true);
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}
generate.onclick = async () => {
  if (controller) return;
  controller = new AbortController();
  const requestController = controller,
    { signal } = requestController;
  let gpuFailure: Error | undefined;
  generate.disabled = true;
  cancel.disabled = false;
  output.textContent = "";
  playback.pause();
  playback.removeAttribute("src");
  if (playbackURL) {
    URL.revokeObjectURL(playbackURL);
    playbackURL = undefined;
  }
  try {
    const pcm = audio.files?.[0]
      ? await readPCM(audio.files[0], signal)
      : undefined;
    const rendered = renderOmniPrompt(prompt.value, pcm?.length);
    tokenizer = await Tokenizer.fromJSON(
      await fetchBytes(config.tokenizer, signal),
    );
    signal.throwIfAborted();
    const ids = tokenizer.encode(rendered);
    if (
      pcm &&
      ids.filter((id) => id === 151646).length !==
        Math.floor(Math.ceil(Math.ceil(pcm.length / 160) / 2) / 2)
    )
      throw new Error("Tokenizer does not match the audio template");
    if (
      ids.length + config.maxNewTokens + 2 + config.maxCodes >
      config.contextWindowSize
    )
      throw new Error(
        "Prompt and speech exceed the configured context. Shorten the clip or prompt.",
      );
    const speakers = JSON.parse(
        new TextDecoder().decode(await fetchBytes(config.speakers, signal)),
      ),
      profile = speakers[voice.value];
    if (
      !profile ||
      ![151870, 151872].includes(profile.bos_token) ||
      !Array.isArray(profile.cond) ||
      profile.cond.length !== 1 ||
      profile.cond[0].length !== 192 ||
      !Array.isArray(profile.ref_mel) ||
      profile.ref_mel.length !== 1 ||
      profile.ref_mel[0].length !== 400 ||
      !profile.ref_mel[0].every(
        (row: unknown) => Array.isArray(row) && row.length === 80,
      )
    )
      throw new Error("Invalid exported speaker profile");
    const speaker = new Float32Array(profile.cond[0]),
      reference = new Float32Array(profile.ref_mel[0].flat());
    if (!speaker.every(Number.isFinite) || !reference.every(Number.isFinite))
      throw new Error("Invalid speaker conditioning");
    status.textContent = "Loading speech components…";
    const parts: Record<string, OmniSpeechPart> = {};
    for (const [name, base] of Object.entries(config.components)) {
      if (!pcm && (name === "features" || name === "audio")) continue;
      const json = async (file: string) =>
        JSON.parse(
          new TextDecoder().decode(await fetchBytes(base + file, signal)),
        );
      const model = parseModelPackageManifestV2(
          await json("model-manifest.json"),
        ),
        compiled = parseCompiledProgramArtifactV2(
          await json("compiled-manifest.json"),
        );
      await validateGraphArtifactPair(model, compiled);
      const names = Object.keys(compiled.components);
      if (names.length !== 1)
        throw new Error("Expected a standalone component package");
      parts[name] = {
        model,
        compiled,
        componentId: names[0],
        factory: undefined!,
      };
    }
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("WebGPU is unavailable");
    const required = Math.max(
      ...Object.values(parts).map(
        (p) =>
          p.compiled.components[p.componentId].resources
            .max_storage_buffer_binding_size,
      ),
    );
    const bufferLimit = Math.max(2 ** 30, required);
    if (
      bufferLimit > adapter.limits.maxBufferSize ||
      bufferLimit > adapter.limits.maxStorageBufferBindingSize
    )
      throw new Error(
        "This FP32 speech model needs larger GPU buffers than this device supports",
      );
    const features = new Set<GPUFeatureName>();
    for (const part of Object.values(parts))
      for (const feature of part.compiled.components[part.componentId].resources
        .required_features) {
        if (!adapter.features.has(feature as GPUFeatureName))
          throw new Error(`Missing GPU feature ${feature}`);
        features.add(feature as GPUFeatureName);
      }
    for (const feature of ["shader-f16", "subgroups"] as GPUFeatureName[])
      if (adapter.features.has(feature)) features.add(feature);
    device = await adapter.requestDevice({
      requiredFeatures: [...features],
      requiredLimits: {
        maxBufferSize: bufferLimit,
        maxStorageBufferBindingSize: bufferLimit,
        maxComputeWorkgroupStorageSize: 32768,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeInvocationsPerWorkgroup: Math.min(
          1024,
          adapter.limits.maxComputeInvocationsPerWorkgroup,
        ),
      },
    });
    device.addEventListener("uncapturederror", (event) => {
      gpuFailure = new Error(event.error.message);
      requestController.abort();
    });
    void device.lost.then((info) => {
      if (info.reason !== "destroyed") {
        gpuFailure = new Error(`GPU device lost: ${info.message}`);
        requestController.abort();
      }
    });
    const capabilities = {
      features: device.features,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    };
    for (const [name, part] of Object.entries(parts)) {
      signal.throwIfAborted();
      const base = config.components[name as keyof typeof config.components];
      const bytes = await fetchBytes(
        new URL(
          part.compiled.components[part.componentId].module,
          new URL(base, location.href),
        ).href,
        signal,
      );
      const runtime = await tvmjs.instantiate(
        bytes,
        tvmjs.createPolyfillWASI(),
        console.log,
      );
      runtimes.push(runtime);
      runtime.initWebGPU(device);
      await runtime.fetchTensorCache(
        new URL(base, location.href).href,
        runtime.webgpu(),
        { signal },
      );
      signal.throwIfAborted();
      const common = {
        runtime,
        model: part.model,
        compiled: part.compiled,
        componentId: part.componentId,
        device: runtime.webgpu(),
        capabilities,
      };
      if (name === "features")
        part.factory = await createWhisperAudioTVMFactory({
          ...common,
          maxFrames: Math.ceil(config.maxSeconds * 100),
        });
      else if (name === "audio")
        part.factory = await createAudioEncoderTVMFactory({
          ...common,
          maxFrames: Math.ceil(config.maxSeconds * 100),
        });
      else if (name === "thinker")
        part.factory = await createThinkerSpeechTVMFactory({
          ...common,
          contextWindowSize: config.contextWindowSize,
          maxNewTokens: config.maxNewTokens,
          speakerBosTokenId: profile.bos_token,
        });
      else if (name === "talker")
        part.factory = await createTalkerTVMFactory({
          ...common,
          contextWindowSize: config.contextWindowSize,
          maxCodes: config.maxCodes,
          sampling: {
            temperature: 0.9,
            topK: 40,
            topP: 0.8,
            repetitionPenalty: 1.05,
            random: () =>
              crypto.getRandomValues(new Uint32Array(1))[0] / 4294967296,
          },
        });
      else if (name === "reference")
        part.factory = await createReferenceConditionerTVMFactory({
          ...common,
          maxFrames: 512,
        });
      else if (name === "flow")
        part.factory = await createFlowPredictorTVMFactory({
          ...common,
          maxFrames: config.maxFrames,
        });
      else
        part.factory = await createWaveformDecoderTVMFactory({
          ...common,
          maxFrames: config.maxFrames,
        });
    }
    const timePoints = Float32Array.from({ length: 10 }, (_, i) => {
      const t = Math.fround(i / 9);
      const angle = Math.fround(Math.fround(Math.PI / 2) * t);
      const adjustment = Math.fround(
        Math.fround(Math.fround(Math.cos(angle)) - 1) + t,
      );
      return Math.fround(t - adjustment);
    });
    const graph = await createOmniSpeechGraph({
      parts: parts as unknown as OmniSpeechParts,
      maxFrames: config.maxFrames,
      timePoints,
      guidanceScale: 0.5,
    });
    const stream = (tensor: GraphTensor) =>
      (async function* () {
        yield tensor;
      })();
    status.textContent = "Generating text and speech…";
    const execution = await graph.execute({
      mode: pcm ? "audio" : "text",
      signal,
      inputs: {
        token_ids: stream({ shape: [ids.length], data: ids }),
        ...(pcm ? { pcm: stream({ shape: [pcm.length], data: pcm }) } : {}),
        reference_mel: stream({ shape: [400, 80], data: reference }),
        speaker: stream({ shape: [192], data: speaker }),
        noise: stream({
          shape: [config.maxFrames, 80],
          data: normalNoise(config.maxFrames * 80),
        }),
      },
    });
    const tokens: number[] = [];
    let waveform: Float32Array | undefined,
      finishReason = 1;
    try {
      await Promise.all([
        execution.done,
        (async () => {
          for await (const value of execution.outputs.token_ids) {
            tokens.push(...value.data);
            output.textContent = tokenizer!.decode(new Int32Array(tokens));
          }
        })(),
        (async () => {
          for await (const value of execution.outputs.finish_reason)
            finishReason = value.data[0];
        })(),
        (async () => {
          for await (const value of execution.outputs.pcm) {
            if (waveform || !(value.data instanceof Float32Array))
              throw new Error("Invalid speech output");
            waveform = value.data;
          }
        })(),
      ]);
    } catch (error) {
      execution.abort(
        error instanceof Error ? error : new Error(String(error)),
      );
      await execution.done.catch(() => {});
      throw error;
    }
    if (!waveform) throw new Error("No speech output");
    playbackURL = waveformURL(waveform);
    playback.src = playbackURL;
    status.textContent =
      finishReason === 0
        ? "Finished. Play the response below."
        : "Text reached its limit. Play the bounded response below.";
  } catch (error) {
    status.textContent = gpuFailure
      ? gpuFailure.message
      : signal.aborted
        ? "Cancelled."
        : error instanceof EmptyThinkerResponseError
          ? "The model ended without a spoken response. Try a different prompt."
          : String(error);
  } finally {
    dispose();
    controller = undefined;
    generate.disabled = false;
    cancel.disabled = true;
  }
};
