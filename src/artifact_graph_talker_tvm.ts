/** Cached codec generation from caller-aligned text/audio Thinker states. */
import {
  sampleTalkerToken,
  TalkerSamplingOptions,
} from "./artifact_graph_talker_sampling";
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

export interface TalkerTVMOptions
  extends Omit<AudioEncoderTVMOptions, "maxFrames"> {
  contextWindowSize: number;
  maxCodes: number;
  sampling?: TalkerSamplingOptions;
}
function cleanup(body: () => void, release: () => void): void {
  let failed = false,
    error: unknown;
  try {
    body();
  } catch (e) {
    failed = true;
    error = e;
  }
  try {
    release();
  } catch (e) {
    if (!failed) {
      failed = true;
      error = e;
    }
  }
  if (failed) throw error;
}
/** Caller-aligned codec generation; only codec BOS is suppressed. */
export async function createTalkerTVMFactory(
  options: TalkerTVMOptions,
): Promise<GraphComponentFactory> {
  const {
    runtime: tvm,
    device,
    componentId,
    contextWindowSize,
    maxCodes,
  } = options;
  const sampling = options.sampling ? { ...options.sampling } : undefined;
  if (
    sampling &&
    (!Number.isFinite(sampling.temperature) ||
      sampling.temperature <= 0 ||
      !Number.isSafeInteger(sampling.topK) ||
      sampling.topK < 0 ||
      sampling.topK > 8448 ||
      !Number.isFinite(sampling.topP) ||
      sampling.topP <= 0 ||
      sampling.topP > 1 ||
      !Number.isFinite(sampling.repetitionPenalty) ||
      sampling.repetitionPenalty <= 0 ||
      typeof sampling.random !== "function")
  )
    throw new Error("Invalid Talker sampling options");
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId],
    inputs = component?.spec.inputs,
    output = component?.spec.outputs.code;
  const embeddingSize = inputs?.prefix_embeddings?.shape[1];
  if (
    !component ||
    component.spec.contract !== "qwen2_5_omni.talker.v1" ||
    component.spec.kind !== "autoregressive" ||
    component.spec.state.kind !== "kv_cache" ||
    !component.spec.weights ||
    (embeddingSize !== 2048 && embeddingSize !== 3584) ||
    Object.keys(inputs).length !== 2 ||
    Object.keys(component.spec.outputs).length !== 1 ||
    !["prefix_embeddings", "reply_embeddings"].every(
      (name) =>
        inputs[name]?.dtype === "float32" &&
        inputs[name].delivery === "value" &&
        inputs[name].shape.length === 2 &&
        inputs[name].shape[1] === embeddingSize,
    ) ||
    inputs.prefix_embeddings.semantic !== "audio.talker_prefix.v1" ||
    inputs.reply_embeddings.semantic !== "audio.talker_reply.v1" ||
    output?.semantic !== "audio.codec_ids.v1" ||
    output.dtype !== "int32" ||
    output.delivery !== "value" ||
    output.shape.length !== 1 ||
    ![
      "embed",
      "project",
      "prefill",
      "decode",
      "create_state",
      "reset_state",
    ].every((name) => component.exports[name]) ||
    component.exports.run !== component.exports.prefill ||
    !Number.isSafeInteger(contextWindowSize) ||
    contextWindowSize < 2 ||
    !Number.isSafeInteger(maxCodes) ||
    maxCodes < 1 ||
    maxCodes > 256
  )
    throw new Error("Unsupported Talker ABI or generation limits");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error("Talker exceeds buffer binding limit");
  let active = false;
  return {
    async create(id, requested, signal) {
      aborted(signal);
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active) throw new Error("Talker runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(value: T): T => {
        tvm.detachFromCurrentScope(value);
        owned.push(value);
        return value;
      };
      let disposed = false;
      let inFlight: Promise<void> | undefined;
      let disposal: Promise<void> | undefined;
      const dispose = (): Promise<void> => {
        if (disposal) return disposal;
        disposed = true;
        disposal = (async () => {
          // Cancellation can detach the pending generator.next() before GPU work drains.
          await inFlight;
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
        const config = metadata.talker;
        if (
          !config ||
          config.model_dtype !== "float32" ||
          config.embedding_size !== embeddingSize ||
          config.hidden_size !== 896 ||
          config.vocab_size !== 8448 ||
          config.codec_bos_token_id !== 8293 ||
          config.codec_pad_token_id !== 8292 ||
          config.codec_eos_token_id !== 8294 ||
          !Number.isSafeInteger(config.context_window_size) ||
          contextWindowSize > config.context_window_size ||
          !Number.isSafeInteger(config.prefill_chunk_size) ||
          config.prefill_chunk_size < 1
        )
          throw new Error("Invalid Talker metadata");
        aborted(signal);
        const functions: Record<string, tvmjs.PackedFunc> = {};
        let packed: tvmjs.TVMObject, cache: tvmjs.TVMObject;
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
            "project",
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
              throw new Error("Talker backend can run once");
            used = true;
            const check = () => {
              if (disposed) throw new Error("Talker backend disposed");
              aborted(signal);
              aborted(runSignal);
            };
            const values: Record<string, GraphTensor> = {},
              ended = new Set<string>();
            for await (const event of events) {
              check();
              if (
                !["prefix_embeddings", "reply_embeddings"].includes(
                  event.port,
                ) ||
                ended.has(event.port)
              )
                throw new Error("Invalid Talker input event");
              if (event.kind === "end") {
                ended.add(event.port);
                continue;
              }
              if (
                values[event.port] ||
                !(event.tensor.data instanceof Float32Array)
              )
                throw new Error("Invalid or duplicate Talker input");
              values[event.port] = {
                shape: [...event.tensor.shape],
                data: new Float32Array(event.tensor.data),
              };
            }
            check();
            const prefix = values.prefix_embeddings,
              reply = values.reply_embeddings;
            for (const tensor of [prefix, reply])
              if (
                !tensor ||
                tensor.shape.length !== 2 ||
                tensor.shape[1] !== embeddingSize ||
                !Number.isSafeInteger(tensor.shape[0]) ||
                tensor.shape[0] < 1 ||
                tensor.data.length !== tensor.shape[0] * embeddingSize ||
                !tensor.data.every(Number.isFinite)
              )
                throw new Error("Invalid Talker conditioning tensors");
            if (
              ended.size !== 2 ||
              prefix.shape[0] < 2 ||
              prefix.shape[0] + maxCodes - 1 > contextWindowSize
            )
              throw new Error(
                "Talker conditioning exceeds context or lacks EOF",
              );
            const readHost = async (
              operation: () => tvmjs.Tensor,
              shape: number[],
            ): Promise<Float32Array> => {
              let cpu: tvmjs.Tensor | undefined;
              tvm.beginScope();
              try {
                cleanup(
                  () => {
                    const value = operation();
                    if (value.dtype !== "float32" || !same(value.shape, shape))
                      throw new Error("Unexpected Talker tensor");
                    cpu = tvm.detachFromCurrentScope(
                      tvm.empty(shape, "float32", tvm.cpu()),
                    );
                    cpu.copyFrom(value);
                  },
                  () => tvm.endScope(),
                );
              } catch (error) {
                try {
                  await device.sync();
                } catch {
                  /* Preserve the original operation error. */
                }
                try {
                  cpu?.dispose();
                } catch {
                  /* Preserve operation failure. */
                }
                throw error;
              }
              if (!cpu) throw new Error("Missing Talker output");
              let data: Float32Array | undefined,
                failed = false,
                failure: unknown;
              try {
                await device.sync();
                check();
                data = new Float32Array(cpu.toArray() as Float32Array);
                if (!data.every(Number.isFinite))
                  throw new Error("Nonfinite Talker tensor");
              } catch (error) {
                failed = true;
                failure = error;
              }
              cleanup(
                () => {
                  if (failed) throw failure;
                },
                () => cpu!.dispose(),
              );
              return data!;
            };
            const host = async (
              operation: () => tvmjs.Tensor,
              shape: number[],
            ): Promise<Float32Array> => {
              check();
              const pending = readHost(operation, shape);
              inFlight = pending.then(
                () => {},
                () => {},
              );
              try {
                return await pending;
              } finally {
                inFlight = undefined;
              }
            };
            const embed = (ids: number[]) =>
              host(
                () =>
                  functions.embed(
                    tvm
                      .empty([ids.length], "int32", device)
                      .copyFrom(new Int32Array(ids)),
                    packed,
                  ) as tvmjs.Tensor,
                [ids.length, embeddingSize],
              );
            const specials = await embed([8292, 8293]);
            const initial = prefix.data as Float32Array;
            for (let i = 0; i < specials.length; i++)
              initial[initial.length - specials.length + i] += specials[i];
            const forward = (data: Float32Array, decode: boolean) =>
              host(() => {
                if (!data.every(Number.isFinite))
                  throw new Error("Talker conditioning overflow");
                const count = data.length / embeddingSize;
                const projected = functions.project(
                  tvm
                    .empty([count, embeddingSize], "float32", device)
                    .copyFrom(data),
                  packed,
                ) as tvmjs.Tensor;
                if (
                  projected.dtype !== "float32" ||
                  !same(projected.shape, [count, 896])
                )
                  throw new Error("Unexpected Talker projection");
                functions.begin(
                  cache,
                  tvm.makeShapeTuple([0]),
                  tvm.makeShapeTuple([count]),
                );
                let logits: tvmjs.Tensor | undefined;
                cleanup(
                  () => {
                    logits = (
                      functions[decode ? "decode" : "prefill"](
                        projected.view([1, count, 896]),
                        cache,
                        packed,
                      ) as unknown as { get(i: number): tvmjs.Tensor }
                    ).get(0);
                  },
                  () => functions.end(cache),
                );
                return logits!;
              }, [1, 1, 8448]);
            let logits: Float32Array | undefined;
            for (
              let start = 0;
              start < prefix.shape[0];
              start += config.prefill_chunk_size
            ) {
              check();
              logits = await forward(
                initial.subarray(
                  start * embeddingSize,
                  Math.min(prefix.shape[0], start + config.prefill_chunk_size) *
                    embeddingSize,
                ),
                false,
              );
            }
            const codes: number[] = [];
            const history: number[] = [
              ...new Array<number>(prefix.shape[0] - 2).fill(8296),
              8292,
              8293,
            ];
            for (let step = 0; step < maxCodes; step++) {
              check();
              let token = 0;
              if (sampling)
                token = sampleTalkerToken(logits!, history, sampling);
              else
                for (let i = 1; i < logits!.length; i++)
                  if (i !== 8293 && logits![i] > logits![token]) token = i;
              if (token === 8292 || token === 8294) break;
              if (token > 8193)
                throw new Error("Talker selected an unsupported codec token");
              codes.push(token);
              history.push(token);
              if (step + 1 === maxCodes) break;
              const aligned = await embed([token]);
              const offset = Math.min(step, reply.shape[0] - 1) * embeddingSize;
              for (let i = 0; i < aligned.length; i++)
                aligned[i] += reply.data[offset + i];
              logits = await forward(aligned, true);
            }
            check();
            if (!codes.length)
              throw new Error("Talker produced no codec tokens");
            yield {
              port: "code",
              tensor: { shape: [codes.length], data: new Int32Array(codes) },
            };
          },
        };
      } catch (error) {
        try {
          await dispose();
        } catch {
          /* Preserve creation failure. */
        }
        throw error;
      }
    },
  };
}
