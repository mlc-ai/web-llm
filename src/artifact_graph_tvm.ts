/** TVM binding for the stateless Qwen2.5 Omni audio encoder component. */
import type * as tvmjs from "@mlc-ai/web-runtime";
import {
  createTVMRequestDrain,
  readTVMFloat32,
} from "./artifact_graph_tvm_lifecycle";
import {
  GraphComponentBackend,
  GraphComponentFactory,
  GraphInputEvent,
  GraphTensor,
} from "./artifact_graph_executor";
import {
  CompiledProgramArtifactV2,
  ModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  parseModelPackageManifestV2,
  validateGraphArtifactPair,
} from "./artifact_manifest_v2";
type Component = CompiledProgramArtifactV2["components"][string];
interface Parameter {
  name: string;
  shape: number[];
  dtype: string;
}
export interface AudioEncoderTVMOptions {
  /** Dedicated initialized runtime with the exact component weights already cached. */
  runtime: tvmjs.Instance;
  model: ModelPackageManifestV2;
  compiled: CompiledProgramArtifactV2;
  componentId: string;
  device: tvmjs.DLDevice;
  capabilities: {
    features: ReadonlySet<string>;
    maxStorageBufferBindingSize: number;
  };
  /** Bound clip length and host staging allocation. */
  maxFrames: number;
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonical(object[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}
const same = (a: unknown, b: unknown) => canonical(a) === canonical(b);
async function schemaId(params: Parameter[]): Promise<string> {
  const bytes = new TextEncoder().encode(
    canonical(
      [...params].sort((a, b) =>
        a.name < b.name ? -1 : a.name > b.name ? 1 : 0,
      ),
    ),
  );
  const digest = new Uint8Array(
    await globalThis.crypto.subtle.digest(
      "SHA-256",
      bytes.buffer as ArrayBuffer,
    ),
  );
  return `sha256:${Array.from(digest, (x) => x.toString(16).padStart(2, "0")).join("")}`;
}
function aborted(signal: AbortSignal): void {
  if (signal.aborted) throw new Error("Audio encoder execution aborted");
}
interface ComponentMetadata {
  talker: {
    model_dtype: string;
    hidden_size: number;
    embedding_size: number;
    vocab_size: number;
    codec_bos_token_id: number;
    codec_pad_token_id: number;
    codec_eos_token_id: number;
    context_window_size: number;
    prefill_chunk_size: number;
  };
  flow_predictor: {
    mel_dim: number;
    reference_dim: number;
    speaker_dim: number;
    repeats: number;
    num_embeds: number;
    max_mel_frames: number;
  };
  reference_conditioner: {
    mel_dim: number;
    embedding_dim: number;
    min_mel_frames: number;
    max_mel_frames: number;
  };
  waveform_decoder: {
    mel_dim: number;
    sampling_rate: number;
    samples_per_frame: number;
    max_mel_frames: number;
  };
  thinker: {
    speech_profile?: {
      speaker_bos_token_ids: number[];
      text_eos_token_id: number;
      text_pad_token_id: number;
    };
    model_dtype: string;
    hidden_size: number;
    vocab_size: number;
    active_vocab_size: number | null;
    audio_token_id: number;
    image_token_id: number;
    video_token_id: number;
    eos_token_ids: number[];
    context_window_size: number;
    prefill_chunk_size: number;
  };
  artifact: unknown;
  params: Parameter[];
  audio_encoder: {
    hidden_size: number;
    n_window: number;
    max_pooled_frames: number;
  };
  audio_processor: {
    num_samples: number;
    sampling_rate: number;
    hop_length: number;
    num_mel_bins: number;
  };
}
async function loadComponent(
  tvm: tvmjs.Instance,
  device: tvmjs.DLDevice,
  compiled: CompiledProgramArtifactV2,
  component: Component,
  own: <T extends { dispose(): void }>(value: T) => T,
): Promise<{
  vm: tvmjs.VirtualMachine;
  metadata: ComponentMetadata;
  params: Parameter[];
}> {
  let vm: tvmjs.VirtualMachine;
  let metadata: ComponentMetadata;
  tvm.beginScope();
  try {
    vm = own(tvm.createVirtualMachine(device));
    metadata = JSON.parse(vm.getFunction("_metadata")().toString());
  } finally {
    tvm.endScope();
  }
  if (!same(parseCompiledProgramArtifactV2(metadata.artifact), compiled))
    throw new Error("VM metadata does not match compiled graph");
  if (!Array.isArray(metadata.params) || !metadata.params.length)
    throw new Error("Missing parameter metadata");
  const params: Parameter[] = metadata.params.map((p) => {
    if (
      typeof p.name !== "string" ||
      !p.name ||
      typeof p.dtype !== "string" ||
      !Array.isArray(p.shape) ||
      p.shape.some((n) => !Number.isSafeInteger(n) || n < 0)
    )
      throw new Error("Invalid parameter metadata");
    return { name: p.name, shape: p.shape, dtype: p.dtype };
  });
  if (
    new Set(params.map((p) => p.name)).size !== params.length ||
    (await schemaId(params)) !== component.spec.weights!.parameter_schema_id
  )
    throw new Error("Parameter schema mismatch");
  return { vm, metadata, params };
}

/**
 * Bind real VM exports and packed parameters. Caller owns runtime/device/cache;
 * backend owns VM, parameter tuple and temporaries. Use a dedicated runtime and
 * only one request per factory. Initial ABI is float32 with explicit host staging.
 */
export async function createAudioEncoderTVMFactory(
  options: AudioEncoderTVMOptions,
): Promise<GraphComponentFactory> {
  const { runtime: tvm, device, componentId, maxFrames } = options;
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId];
  const input = component?.spec.inputs.mel_features,
    output = component?.spec.outputs.embeddings;
  if (
    !component ||
    component.spec.contract !== "qwen2_5_omni.audio_encoder.v1" ||
    component.spec.kind !== "encoder" ||
    component.spec.state.kind !== "none" ||
    !component.spec.weights ||
    Object.keys(component.spec.inputs).length !== 1 ||
    Object.keys(component.spec.outputs).length !== 1 ||
    input?.dtype !== "float32" ||
    output?.dtype !== "float32" ||
    input.delivery !== "value" ||
    output.delivery !== "value" ||
    input.shape.length !== 2 ||
    output.shape.length !== 2 ||
    typeof input.shape[0] !== "number" ||
    typeof output.shape[1] !== "number" ||
    !component.exports.encode_chunk ||
    !component.exports.project ||
    component.exports.run !== component.exports.encode_chunk ||
    !Number.isSafeInteger(maxFrames) ||
    maxFrames < 1
  )
    throw new Error("Unsupported audio encoder component ABI");
  const melBins = input.shape[0],
    outputWidth = output.shape[1];
  if (melBins < 1 || outputWidth < 1)
    throw new Error("Invalid audio encoder port dimensions");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error("Audio encoder exceeds device buffer binding limit");
  let active = false;
  return {
    async create(
      id: string,
      requested: Component,
      signal: AbortSignal,
    ): Promise<GraphComponentBackend> {
      aborted(signal);
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active) throw new Error("Audio encoder runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(value: T): T => {
        tvm.detachFromCurrentScope(value);
        owned.push(value);
        return value;
      };
      let disposed = false;
      const drain = createTVMRequestDrain(device);
      let disposal: Promise<void> | undefined;
      const dispose = (): Promise<void> => {
        if (disposal) return disposal;
        disposed = true;
        disposal = (async () => {
          await drain.wait();
          let failure: unknown;
          for (const value of owned.reverse()) {
            try {
              value.dispose();
            } catch (error) {
              failure ??= error;
            }
          }
          active = false;
          if (failure) throw failure;
        })();
        return disposal;
      };
      try {
        const { vm, metadata, params } = await loadComponent(
          tvm,
          device,
          compiled,
          component,
          own,
        );
        const width = metadata.audio_encoder?.hidden_size,
          window = metadata.audio_encoder?.n_window;
        if (
          !Number.isSafeInteger(width) ||
          width <= 0 ||
          !Number.isSafeInteger(window) ||
          window <= 0
        )
          throw new Error("Missing audio encoder dimensions");
        const maxPooled = metadata.audio_encoder.max_pooled_frames;
        if (
          !Number.isSafeInteger(maxPooled) ||
          maxPooled < 1 ||
          Math.floor(Math.ceil(maxFrames / 2) / 2) > maxPooled
        )
          throw new Error("Audio clip limit exceeds compiled projection bound");
        aborted(signal);
        let packed: tvmjs.TVMObject,
          encode: tvmjs.PackedFunc,
          project: tvmjs.PackedFunc;
        tvm.beginScope();
        try {
          for (const p of params) {
            const tensor = tvm.tensorCacheGet(p.name);
            if (
              !tensor ||
              tensor.dtype !== p.dtype ||
              !same(tensor.shape, p.shape)
            )
              throw new Error(`Cached parameter mismatch: ${p.name}`);
          }
          packed = own(tvm.getParamsFromCacheByName(params.map((p) => p.name)));
          encode = own(vm.getFunction(component.exports.encode_chunk));
          project = own(vm.getFunction(component.exports.project));
        } finally {
          tvm.endScope();
        }
        let executionSignal: AbortSignal | undefined;
        const checkAbort = () => {
          if (disposed) throw new Error("TVM request disposed");
          aborted(signal);
          if (executionSignal) aborted(executionSignal);
        };
        const call = async (
          fn: tvmjs.PackedFunc,
          data: Float32Array,
          shape: number[],
          resultShape: number[],
          validFrames?: number,
        ) => {
          checkAbort();
          return readTVMFloat32(
            tvm,
            drain,
            resultShape,
            () => {
              const arg = tvm.empty(shape, "float32", device).copyFrom(data);
              const valid =
                validFrames === undefined
                  ? undefined
                  : tvm
                      .empty([1], "int32", device)
                      .copyFrom(new Int32Array([validFrames]));
              const result = (
                valid ? fn(arg, valid, packed) : fn(arg, packed)
              ) as tvmjs.Tensor;
              if (
                result.dtype !== "float32" ||
                !same(result.shape, resultShape)
              )
                throw new Error(
                  "Audio encoder export returned unexpected shape or dtype",
                );
              return result;
            },
            checkAbort,
          );
        };
        let used = false;
        return {
          dispose,
          async *run(
            events: AsyncIterable<GraphInputEvent>,
            runSignal: AbortSignal,
          ) {
            if (used || disposed)
              throw new Error("Audio encoder backend can run once");
            used = true;
            executionSignal = runSignal;
            let clip: GraphTensor | undefined,
              ended = false;
            for await (const event of events) {
              checkAbort();
              if (event.port !== "mel_features" || ended)
                throw new Error("Invalid audio encoder input event");
              if (event.kind === "end") {
                ended = true;
                continue;
              }
              if (clip) throw new Error("Audio encoder expects one clip");
              clip = event.tensor;
            }
            if (
              !clip ||
              !ended ||
              !(clip.data instanceof Float32Array) ||
              clip.shape.length !== 2 ||
              clip.shape[0] !== melBins ||
              !Number.isSafeInteger(clip.shape[1]) ||
              clip.shape[1] < 1 ||
              clip.shape[1] > maxFrames ||
              clip.data.length !== melBins * clip.shape[1]
            )
              throw new Error("Invalid mel features");
            const frames = clip.shape[1],
              total = Math.ceil(frames / 2);
            if (total < 2) {
              yield {
                port: "embeddings",
                tensor: {
                  shape: [0, outputWidth],
                  data: new Float32Array(0),
                },
              };
              return;
            }
            const hidden = new Float32Array(total * width);
            let offset = 0;
            for (let start = 0; start < frames; start += 2 * window) {
              const count = Math.min(2 * window, frames - start),
                chunk = new Float32Array(melBins * 2 * window);
              for (let row = 0; row < melBins; row++)
                chunk.set(
                  clip.data.subarray(
                    row * frames + start,
                    row * frames + start + count,
                  ),
                  row * 2 * window,
                );
              const result = await call(
                encode,
                chunk,
                [1, melBins, 2 * window],
                [window, width],
                count,
              );
              const valid = result.subarray(0, Math.ceil(count / 2) * width);
              hidden.set(valid, offset);
              offset += valid.length;
            }
            const rows = Math.floor(total / 2),
              embeddings = await call(
                project,
                hidden.subarray(0, rows * 2 * width),
                [rows, 2, width],
                [rows, outputWidth],
              );
            yield {
              port: "embeddings",
              tensor: {
                shape: [rows, outputWidth],
                data: embeddings,
              },
            };
          },
        };
      } catch (error) {
        try {
          await dispose();
        } catch {
          /* Preserve the setup failure. */
        }
        throw error;
      }
    },
  };
}

export type WhisperAudioTVMOptions = Omit<AudioEncoderTVMOptions, "maxFrames">;

/** Bind the pinned mono 16 kHz PCM frontend; no implicit resampling or truncation. */
export async function createWhisperAudioTVMFactory(
  options: WhisperAudioTVMOptions,
): Promise<GraphComponentFactory> {
  const { runtime: tvm, device, componentId } = options;
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId],
    input = component?.spec.inputs.pcm,
    output = component?.spec.outputs.mel_features;
  if (
    !component ||
    component.spec.contract !== "qwen2_5_omni.audio_features.v1" ||
    component.spec.kind !== "transform" ||
    component.spec.state.kind !== "none" ||
    !component.spec.weights ||
    Object.keys(component.spec.inputs).length !== 1 ||
    Object.keys(component.spec.outputs).length !== 1 ||
    input?.dtype !== "float32" ||
    input.delivery !== "value" ||
    input.shape.length !== 1 ||
    output?.dtype !== "float32" ||
    output.delivery !== "value" ||
    output.shape.length !== 2 ||
    output.shape[0] !== 128 ||
    !component.exports.forward ||
    component.exports.run !== component.exports.forward
  )
    throw new Error("Unsupported Whisper audio component ABI");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error("Whisper frontend exceeds device buffer binding limit");
  let active = false;
  return {
    async create(id, requested, signal) {
      aborted(signal);
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active) throw new Error("Whisper runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(value: T): T => {
        tvm.detachFromCurrentScope(value);
        owned.push(value);
        return value;
      };
      let disposed = false;
      const drain = createTVMRequestDrain(device);
      let disposal: Promise<void> | undefined;
      const dispose = (): Promise<void> => {
        if (disposal) return disposal;
        disposed = true;
        disposal = (async () => {
          await drain.wait();
          let error: unknown;
          for (const value of owned.reverse()) {
            try {
              value.dispose();
            } catch (e) {
              error ??= e;
            }
          }
          active = false;
          if (error) throw error;
        })();
        return disposal;
      };
      try {
        const { vm, metadata, params } = await loadComponent(
          tvm,
          device,
          compiled,
          component,
          own,
        );
        const processor = metadata.audio_processor;
        if (
          !processor ||
          processor.sampling_rate !== 16000 ||
          processor.hop_length !== 160 ||
          processor.num_mel_bins !== 128 ||
          !Number.isSafeInteger(processor.num_samples) ||
          processor.num_samples < 400 ||
          processor.num_samples % 160 !== 0
        )
          throw new Error("Invalid Whisper processor metadata");
        const capacity = processor.num_samples,
          frames = capacity / 160;
        aborted(signal);
        let packed: tvmjs.TVMObject, forward: tvmjs.PackedFunc;
        tvm.beginScope();
        try {
          for (const p of params) {
            const tensor = tvm.tensorCacheGet(p.name);
            if (
              !tensor ||
              tensor.dtype !== p.dtype ||
              !same(tensor.shape, p.shape)
            )
              throw new Error(`Cached parameter mismatch: ${p.name}`);
          }
          packed = own(tvm.getParamsFromCacheByName(params.map((p) => p.name)));
          forward = own(vm.getFunction(component.exports.forward));
        } finally {
          tvm.endScope();
        }
        let used = false;
        return {
          dispose,
          async *run(events, runSignal) {
            if (used || disposed)
              throw new Error("Whisper backend can run once");
            used = true;
            const checkAbort = () => {
              if (disposed) throw new Error("TVM request disposed");
              aborted(signal);
              aborted(runSignal);
            };
            let samples: Float32Array | undefined,
              ended = false;
            for await (const event of events) {
              checkAbort();
              if (event.port !== "pcm" || ended)
                throw new Error("Invalid PCM input event");
              if (event.kind === "end") {
                ended = true;
                continue;
              }
              const tensor = event.tensor;
              if (
                samples ||
                !(tensor.data instanceof Float32Array) ||
                tensor.shape.length !== 1 ||
                tensor.shape[0] !== tensor.data.length ||
                tensor.data.length === 0 ||
                tensor.data.length > capacity ||
                !tensor.data.every(Number.isFinite)
              )
                throw new Error(
                  "Invalid PCM samples; empty and oversized clips are unsupported",
                );
              samples = new Float32Array(tensor.data);
            }
            checkAbort();
            if (!samples || !ended)
              throw new Error("Missing PCM samples or end event");
            const padded = new Float32Array(capacity);
            padded.set(samples);
            const raw = await readTVMFloat32(
              tvm,
              drain,
              [128, frames],
              () => {
                const arg = tvm
                  .empty([capacity], "float32", device)
                  .copyFrom(padded);
                const result = forward(arg, packed) as tvmjs.Tensor;
                if (
                  result.dtype !== "float32" ||
                  !same(result.shape, [128, frames])
                )
                  throw new Error(
                    "Whisper export returned unexpected shape or dtype",
                  );
                return result;
              },
              checkAbort,
            );
            const valid = Math.ceil(samples.length / 160),
              mel = new Float32Array(128 * valid);
            for (let bin = 0; bin < 128; bin++)
              mel.set(
                raw.subarray(bin * frames, bin * frames + valid),
                bin * valid,
              );
            yield {
              port: "mel_features",
              tensor: { shape: [128, valid], data: mel },
            };
          },
        };
      } catch (error) {
        try {
          await dispose();
        } catch {
          /* Preserve original setup error. */
        }
        throw error;
      }
    },
  };
}

// Shared internally by component-specific adapters; not a generic artifact loader.
export { loadComponent, same, aborted };
