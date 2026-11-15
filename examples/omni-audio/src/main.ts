import * as tvmjs from "@mlc-ai/web-runtime";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import {
  createAudioEncoderTVMFactory,
  createWhisperAudioTVMFactory,
  createThinkerTVMFactory,
  createArtifactGraphExecution,
  computeGraphInterfaceId,
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  validateGraphArtifactPair,
  type ModelPackageManifestV2,
  type CompiledProgramArtifactV2,
  type GraphComponentFactory,
} from "@mlc-ai/web-llm";
import { config } from "./config";
import { renderAudioPrompt } from "./prompt";

const audio = document.querySelector<HTMLInputElement>("#audio")!;
const prompt = document.querySelector<HTMLTextAreaElement>("#prompt")!;
const generate = document.querySelector<HTMLButtonElement>("#generate")!;
const cancel = document.querySelector<HTMLButtonElement>("#cancel")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const output = document.querySelector<HTMLPreElement>("#output")!;
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

generate.onclick = async () => {
  if (controller) return;
  controller = new AbortController();
  const requestController = controller;
  const { signal } = requestController;
  let gpuFailure: Error | undefined;
  generate.disabled = true;
  cancel.disabled = false;
  output.textContent = "";
  try {
    if (!audio.files?.[0]) throw new Error("Choose an audio file first.");
    status.textContent = "Decoding audio…";
    const pcm = await readPCM(audio.files[0], signal);
    const rendered = renderAudioPrompt(prompt.value, pcm.length);
    tokenizer = await Tokenizer.fromJSON(
      await fetchBytes(config.tokenizer, signal),
    );
    signal.throwIfAborted();
    const ids = tokenizer.encode(rendered);
    const count = Math.floor(Math.ceil(Math.ceil(pcm.length / 160) / 2) / 2);
    if (ids.filter((id) => id === 151646).length !== count) {
      throw new Error(
        "Tokenizer does not match the Qwen2.5-Omni audio template.",
      );
    }
    if (ids.length + config.maxNewTokens > config.contextWindowSize) {
      throw new Error(
        "Prompt and audio exceed the configured context. Shorten the clip or prompt.",
      );
    }
    // Validate prompts before downloading model weights or doing GPU work.
    status.textContent = "Loading local components…";
    const parts: Record<
      string,
      {
        model: ModelPackageManifestV2;
        compiled: CompiledProgramArtifactV2;
        factory?: GraphComponentFactory;
      }
    > = {};
    for (const [id, base] of Object.entries(config.components)) {
      const json = async (name: string) =>
        JSON.parse(
          new TextDecoder().decode(await fetchBytes(base + name, signal)),
        );
      const model = parseModelPackageManifestV2(
        await json("model-manifest.json"),
      );
      const compiled = parseCompiledProgramArtifactV2(
        await json("compiled-manifest.json"),
      );
      await validateGraphArtifactPair(model, compiled);
      parts[id] = { model, compiled };
    }
    const spec = (id: string) => parts[id].model.components[id];
    const model: ModelPackageManifestV2 = {
      schema: "mlc.model-package",
      schema_version: 2,
      interface_id: "sha256:" + "0".repeat(64),
      components: Object.fromEntries(
        Object.keys(parts).map((id) => [id, spec(id)]),
      ),
      tasks: {
        "audio.generate": {
          inputs: {
            pcm: {
              spec: spec("features").inputs.pcm,
              targets: [{ component: "features", port: "pcm" }],
            },
            token_ids: {
              spec: spec("thinker").inputs.token_ids,
              targets: [{ component: "thinker", port: "token_ids" }],
            },
          },
          outputs: {
            token_ids: {
              spec: spec("thinker").outputs.token_ids,
              source: { component: "thinker", port: "token_ids" },
              optional: false,
            },
          },
          edges: [
            {
              source: { component: "features", port: "mel_features" },
              target: { component: "audio", port: "mel_features" },
            },
            {
              source: { component: "audio", port: "embeddings" },
              target: { component: "thinker", port: "audio_embeddings" },
            },
          ],
        },
      },
    };
    model.interface_id = await computeGraphInterfaceId(model);
    const compiled: CompiledProgramArtifactV2 = {
      schema: "mlc.compiled-program",
      schema_version: 2,
      interface_id: model.interface_id,
      components: Object.fromEntries(
        Object.entries(parts).map(([id, p]) => [id, p.compiled.components[id]]),
      ),
    };
    await validateGraphArtifactPair(model, compiled);
    signal.throwIfAborted();
    const gpu = await tvmjs.detectGPUDevice();
    if (!gpu) throw new Error("WebGPU is unavailable.");
    device = gpu.device;
    signal.throwIfAborted();
    gpu.device.addEventListener("uncapturederror", (event) => {
      gpuFailure = new Error(event.error.message);
      requestController.abort();
    });
    void gpu.device.lost.then((info) => {
      if (info.reason !== "destroyed") {
        gpuFailure = new Error(`GPU device lost: ${info.message}`);
        requestController.abort();
      }
    });
    const capabilities = {
      features: gpu.device.features,
      maxStorageBufferBindingSize:
        gpu.device.limits.maxStorageBufferBindingSize,
    };
    for (const [id, base] of Object.entries(config.components)) {
      signal.throwIfAborted();
      const part = parts[id];
      const bytes = await fetchBytes(
        new URL(
          part.compiled.components[id].module,
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
      runtime.initWebGPU(gpu.device);
      await runtime.fetchTensorCache(
        new URL(base, location.href).href,
        runtime.webgpu(),
      );
      signal.throwIfAborted();
      const options = {
        runtime,
        ...part,
        componentId: id,
        device: runtime.webgpu(),
        capabilities,
        maxFrames: Math.ceil(config.maxSeconds * 100),
        contextWindowSize: config.contextWindowSize,
        maxNewTokens: config.maxNewTokens,
      };
      part.factory = await (
        id === "features"
          ? createWhisperAudioTVMFactory
          : id === "audio"
            ? createAudioEncoderTVMFactory
            : createThinkerTVMFactory
      )(options);
    }
    const factory: GraphComponentFactory = {
      create: (id, component, abort) =>
        parts[id].factory!.create(id, component, abort),
    };
    async function* input(data: Float32Array | Int32Array) {
      yield { shape: [data.length], data };
    }
    status.textContent = "Generating…";
    const execution = await createArtifactGraphExecution({
      model,
      compiled,
      task: "audio.generate",
      inputs: { pcm: input(pcm), token_ids: input(ids) },
      factory,
      signal,
    });
    const tokens: number[] = [];
    try {
      for await (const value of execution.outputs.token_ids) {
        tokens.push(...value.data);
        output.textContent = tokenizer.decode(new Int32Array(tokens));
      }
      await execution.done;
    } catch (error) {
      execution.abort(
        error instanceof Error ? error : new Error(String(error)),
      );
      await execution.done.catch(() => {});
      throw error;
    }
    status.textContent = "Finished.";
  } catch (error) {
    status.textContent = gpuFailure
      ? gpuFailure.message
      : signal.aborted
        ? "Cancelled."
        : String(error);
  } finally {
    dispose();
    controller = undefined;
    generate.disabled = false;
    cancel.disabled = true;
  }
};
