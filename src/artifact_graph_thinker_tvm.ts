/** Restricted text/audio Thinker generation through actual TVM exports. */
import type * as tvmjs from "@mlc-ai/web-runtime";
import {
  AudioEncoderTVMOptions,
  loadComponent,
  same,
  aborted,
} from "./artifact_graph_tvm";
import { GraphComponentFactory, GraphTensor } from "./artifact_graph_executor";
import {
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  validateGraphArtifactPair,
} from "./artifact_manifest_v2";
export interface ThinkerTVMOptions
  extends Omit<AudioEncoderTVMOptions, "maxFrames"> {
  contextWindowSize: number;
  maxNewTokens: number;
}
/** IEEE binary32 to binary16, round to nearest with ties to even. */
const halfFloat = new Float32Array(1);
const halfBits = new Uint32Array(halfFloat.buffer);
export function float32ToHalf(value: number): number {
  halfFloat[0] = value;
  const bits = halfBits[0];
  const sign = (bits >>> 16) & 0x8000,
    exponent = (bits >>> 23) & 255,
    mantissa = bits & 0x7fffff;
  if (exponent === 255) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  if (exponent > 142) return sign | 0x7c00;
  if (exponent < 102) return sign;
  const shift = exponent < 113 ? 126 - exponent : 13;
  const significand = exponent < 113 ? mantissa | 0x800000 : mantissa;
  let rounded = significand >>> shift;
  const remainder = significand & ((1 << shift) - 1),
    half = 1 << (shift - 1);
  if (remainder > half || (remainder === half && rounded & 1)) rounded++;
  return sign + (exponent < 113 ? rounded : ((exponent - 112) << 10) + rounded);
}
export async function createThinkerTVMFactory(
  options: ThinkerTVMOptions,
): Promise<GraphComponentFactory> {
  const {
    runtime: tvm,
    device,
    componentId,
    contextWindowSize,
    maxNewTokens,
  } = options;
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId],
    ids = component?.spec.inputs.token_ids,
    audio = component?.spec.inputs.audio_embeddings,
    output = component?.spec.outputs.token_ids;
  if (
    !component ||
    component.spec.contract !== "qwen2_5_omni.thinker_audio.v1" ||
    component.spec.kind !== "autoregressive" ||
    component.spec.state.kind !== "kv_cache" ||
    !component.spec.weights ||
    Object.keys(component.spec.inputs).length !== 2 ||
    Object.keys(component.spec.outputs).length !== 1 ||
    ids?.dtype !== "int32" ||
    ids.delivery !== "value" ||
    ids.shape.length !== 1 ||
    audio?.dtype !== "float32" ||
    audio.delivery !== "value" ||
    audio.shape.length !== 2 ||
    typeof audio.shape[1] !== "number" ||
    output?.dtype !== "int32" ||
    output.delivery !== "stream" ||
    !same(output.shape, [1]) ||
    !["embed", "prefill", "decode", "create_state", "reset_state"].every(
      (name) => component.exports[name],
    ) ||
    component.exports.run !== component.exports.prefill ||
    !Number.isSafeInteger(contextWindowSize) ||
    contextWindowSize < 1 ||
    !Number.isSafeInteger(maxNewTokens) ||
    maxNewTokens < 1
  )
    throw new Error("Unsupported Thinker component ABI or generation limits");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error("Thinker exceeds buffer binding limit");
  let active = false;
  return {
    async create(id, requested, signal) {
      aborted(signal);
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active) throw new Error("Thinker runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(v: T): T => {
        tvm.detachFromCurrentScope(v);
        owned.push(v);
        return v;
      };
      let disposed = false;
      const dispose = () => {
        if (disposed) return;
        disposed = true;
        let failure: unknown;
        for (const v of owned.reverse()) {
          try {
            v.dispose();
          } catch (e) {
            failure ??= e;
          }
        }
        active = false;
        if (failure) throw failure;
      };
      try {
        const { vm, metadata, params } = await loadComponent(
          tvm,
          device,
          compiled,
          component,
          own,
        );
        const config = metadata.thinker;
        if (
          !config ||
          !["float16", "float32"].includes(config.model_dtype) ||
          config.hidden_size !== audio.shape[1] ||
          !Number.isSafeInteger(config.vocab_size) ||
          config.vocab_size < 1 ||
          !Number.isSafeInteger(config.prefill_chunk_size) ||
          config.prefill_chunk_size < 1 ||
          !Number.isSafeInteger(config.context_window_size) ||
          contextWindowSize > config.context_window_size ||
          config.audio_token_id !== 151646 ||
          config.image_token_id !== 151655 ||
          config.video_token_id !== 151656 ||
          !same(config.eos_token_ids, [151643, 151645])
        )
          throw new Error("Invalid Thinker metadata");
        const vocab = config.active_vocab_size ?? config.vocab_size;
        if (
          !Number.isSafeInteger(vocab) ||
          vocab < 1 ||
          vocab > config.vocab_size
        )
          throw new Error("Invalid active vocabulary");
        aborted(signal);
        let packed: tvmjs.TVMObject, cache: tvmjs.TVMObject;
        const functions: Record<string, tvmjs.PackedFunc> = {};
        tvm.beginScope();
        try {
          for (const p of params) {
            const value = tvm.tensorCacheGet(p.name);
            if (
              !value ||
              value.dtype !== p.dtype ||
              !same(value.shape, p.shape)
            )
              throw new Error(`Cached parameter mismatch: ${p.name}`);
          }
          packed = own(tvm.getParamsFromCacheByName(params.map((p) => p.name)));
          for (const name of [
            "embed",
            "prefill",
            "decode",
            "create_state",
            "reset_state",
          ])
            functions[name] = own(vm.getFunction(component.exports[name]));
          for (const [name, global] of [
            ["add", "vm.builtin.kv_state_add_sequence"],
            ["begin", "vm.builtin.kv_state_begin_forward"],
            ["end", "vm.builtin.kv_state_end_forward"],
          ])
            functions[name] = own(tvm.getGlobalFunc(global));
          cache = own(
            functions.create_state(
              ...[1, contextWindowSize, config.prefill_chunk_size, 16, 0].map(
                (x) => tvm.makeShapeTuple([x]),
              ),
            ),
          );
          functions.reset_state(cache);
          functions.add(cache, tvm.scalar(0, "int64"));
        } finally {
          tvm.endScope();
        }
        let used = false;
        return {
          dispose,
          async *run(events, runSignal) {
            if (used || disposed)
              throw new Error("Thinker backend can run once");
            used = true;
            const check = () => {
              aborted(signal);
              aborted(runSignal);
            };
            const values: Record<string, GraphTensor> = {},
              ended = new Set<string>();
            for await (const event of events) {
              check();
              if (
                !["token_ids", "audio_embeddings"].includes(event.port) ||
                ended.has(event.port)
              )
                throw new Error("Invalid Thinker input event");
              if (event.kind === "end") {
                ended.add(event.port);
                continue;
              }
              if (values[event.port])
                throw new Error("Duplicate Thinker input");
              values[event.port] = {
                shape: [...event.tensor.shape],
                data: event.tensor.data.slice(),
              };
            }
            check();
            const tokens = values.token_ids,
              embeddings = values.audio_embeddings,
              width = config.hidden_size;
            if (
              ended.size !== 2 ||
              !tokens ||
              !(tokens.data instanceof Int32Array) ||
              tokens.shape.length !== 1 ||
              tokens.shape[0] !== tokens.data.length ||
              tokens.data.length < 1 ||
              tokens.data.length >= contextWindowSize ||
              !embeddings ||
              !(embeddings.data instanceof Float32Array) ||
              embeddings.shape.length !== 2 ||
              embeddings.shape[1] !== width ||
              !Number.isSafeInteger(embeddings.shape[0]) ||
              embeddings.shape[0] < 0 ||
              embeddings.data.length !== embeddings.shape[0] * width ||
              !embeddings.data.every(Number.isFinite)
            )
              throw new Error("Invalid Thinker prompt tensors");
            let audioCount = 0;
            for (const token of tokens.data) {
              if (token === config.audio_token_id) audioCount++;
              else if (
                token < 0 ||
                token >= vocab ||
                [
                  151652,
                  151653,
                  151654,
                  config.image_token_id,
                  config.video_token_id,
                ].includes(token)
              )
                throw new Error("Unsupported Thinker token or modality");
            }
            if (audioCount !== embeddings.shape[0])
              throw new Error(
                "Audio placeholder count does not match embeddings",
              );
            const dtype = config.model_dtype;
            if (
              dtype === "float16" &&
              !embeddings.data.every(
                (v) => (float32ToHalf(v) & 0x7c00) !== 0x7c00,
              )
            )
              throw new Error("Audio embeddings overflow float16");
            // Audio placeholders are replaced before embedding, so their semantic
            // ID need not occur in a reduced vocabulary used by test models.
            const copyLogits = (result: tvmjs.TVMObject) => {
              const logits = (
                result as unknown as { get(i: number): tvmjs.Tensor }
              ).get(0);
              if (
                logits.dtype !== "float32" ||
                !same(logits.shape, [1, 1, config.vocab_size])
              )
                throw new Error("Unexpected Thinker logits");
              return tvm.detachFromCurrentScope(
                tvm.empty(logits.shape, "float32", tvm.cpu()).copyFrom(logits),
              );
            };
            const readLogits = async (cpu: tvmjs.Tensor) => {
              try {
                await device.sync();
                check();
                const data = cpu.toArray() as Float32Array;
                let best = 0;
                for (let i = 0; i < vocab; i++) {
                  if (!Number.isFinite(data[i]))
                    throw new Error("Nonfinite Thinker logits");
                  if (data[i] > data[best]) best = i;
                }
                return best;
              } finally {
                cpu.dispose();
              }
            };
            let next = 0,
              audioOffset = 0;
            for (
              let start = 0;
              start < tokens.data.length;
              start += config.prefill_chunk_size
            ) {
              check();
              const count = Math.min(
                  config.prefill_chunk_size,
                  tokens.data.length - start,
                ),
                chunk = tokens.data.slice(start, start + count),
                safe = chunk.map((x) => (x === config.audio_token_id ? 0 : x));
              let host: tvmjs.Tensor;
              tvm.beginScope();
              try {
                const embedded = functions.embed(
                  tvm.empty([count], "int32", device).copyFrom(safe),
                  packed,
                ) as tvmjs.Tensor;
                if (
                  embedded.dtype !== dtype ||
                  !same(embedded.shape, [count, width])
                )
                  throw new Error("Unexpected Thinker embeddings");
                host = tvm.detachFromCurrentScope(
                  tvm
                    .empty([count, width], dtype, tvm.cpu())
                    .copyFrom(embedded),
                );
              } finally {
                tvm.endScope();
              }
              let data: Float32Array | Uint16Array;
              try {
                await device.sync();
                check();
                data =
                  dtype === "float16"
                    ? new Uint16Array(host.toRawBytes().slice().buffer)
                    : (host.toArray() as Float32Array);
              } finally {
                host.dispose();
              }
              for (let i = 0; i < count; i++)
                if (chunk[i] === config.audio_token_id) {
                  for (let j = 0; j < width; j++) {
                    const v = embeddings.data[audioOffset * width + j];
                    data[i * width + j] =
                      dtype === "float16" ? float32ToHalf(v) : v;
                  }
                  audioOffset++;
                }
              let cpu: tvmjs.Tensor;
              tvm.beginScope();
              try {
                const input = tvm.empty([1, count, width], dtype, device);
                if (dtype === "float16")
                  input.copyFromRawBytes(
                    new Uint8Array(
                      data.buffer,
                      data.byteOffset,
                      data.byteLength,
                    ),
                  );
                else input.copyFrom(data as Float32Array);
                functions.begin(
                  cache,
                  tvm.makeShapeTuple([0]),
                  tvm.makeShapeTuple([count]),
                );
                try {
                  cpu = copyLogits(functions.prefill(input, cache, packed));
                } finally {
                  functions.end(cache);
                }
              } finally {
                tvm.endScope();
              }
              next = await readLogits(cpu);
            }
            const budget = Math.min(
              maxNewTokens,
              contextWindowSize - tokens.data.length,
            );
            for (let generated = 0; generated < budget; generated++) {
              check();
              if (config.eos_token_ids.includes(next)) break;
              yield {
                port: "token_ids",
                tensor: { shape: [1], data: new Int32Array([next]) },
              };
              check();
              if (generated + 1 === budget) break;
              let cpu: tvmjs.Tensor;
              tvm.beginScope();
              try {
                const embedded = functions.embed(
                  tvm
                    .empty([1], "int32", device)
                    .copyFrom(new Int32Array([next])),
                  packed,
                ) as tvmjs.Tensor;
                if (
                  embedded.dtype !== dtype ||
                  !same(embedded.shape, [1, width])
                )
                  throw new Error("Unexpected decode embeddings");
                functions.begin(
                  cache,
                  tvm.makeShapeTuple([0]),
                  tvm.makeShapeTuple([1]),
                );
                try {
                  cpu = copyLogits(
                    functions.decode(
                      embedded.view([1, 1, width]),
                      cache,
                      packed,
                    ),
                  );
                } finally {
                  functions.end(cache);
                }
              } finally {
                tvm.endScope();
              }
              next = await readLogits(cpu);
            }
          },
        };
      } catch (error) {
        try {
          dispose();
        } catch {
          /* Preserve setup failure. */
        }
        throw error;
      }
    },
  };
}
