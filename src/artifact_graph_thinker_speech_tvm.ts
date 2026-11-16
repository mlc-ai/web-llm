/** Opt-in FP32 Thinker binding that emits text and real Talker conditioning. */
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createTVMRequestDrain } from "./artifact_graph_tvm_lifecycle";
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
import { alignThinkerSpeechTrace } from "./omni_thinker_alignment";

/** No forwarded reply state exists from which to construct speech conditioning. */
export class EmptyThinkerResponseError extends Error {
  constructor() {
    super(
      "Thinker ended before producing text; no speech conditioning is available",
    );
    this.name = "EmptyThinkerResponseError";
  }
}

export interface ThinkerSpeechTVMOptions
  extends Omit<AudioEncoderTVMOptions, "maxFrames"> {
  contextWindowSize: number;
  maxNewTokens: number;
  speakerBosTokenId: number;
}

export async function createThinkerSpeechTVMFactory(
  options: ThinkerSpeechTVMOptions,
): Promise<GraphComponentFactory> {
  const {
    runtime: tvm,
    device,
    componentId,
    contextWindowSize,
    maxNewTokens,
    speakerBosTokenId,
  } = options;
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId],
    spec = component?.spec;
  const ids = spec?.inputs.token_ids,
    audio = spec?.inputs.audio_embeddings,
    outputs = spec?.outputs;
  if (
    !component ||
    spec.contract !== "qwen2_5_omni.thinker_audio_talker.v1" ||
    spec.kind !== "autoregressive" ||
    spec.state.kind !== "kv_cache" ||
    !spec.weights ||
    Object.keys(spec.inputs).length !== 2 ||
    Object.keys(outputs).length !== 4 ||
    ids?.semantic !== "token.ids.v1" ||
    ids.dtype !== "int32" ||
    ids.delivery !== "value" ||
    ids.shape.length !== 1 ||
    audio?.semantic !== "audio.embeddings.v1" ||
    audio.dtype !== "float32" ||
    audio.delivery !== "value" ||
    audio.shape.length !== 2 ||
    typeof audio.shape[1] !== "number" ||
    outputs.token_ids?.semantic !== "token.ids.v1" ||
    outputs.token_ids.dtype !== "int32" ||
    outputs.token_ids.delivery !== "stream" ||
    !same(outputs.token_ids.shape, [1]) ||
    outputs.finish_reason?.dtype !== "int32" ||
    outputs.finish_reason.delivery !== "value" ||
    outputs.finish_reason.semantic !== "generation.finish_reason.v1" ||
    !same(outputs.finish_reason.shape, [1]) ||
    !["prefix_embeddings", "reply_embeddings"].every(
      (name, index) =>
        outputs[name]?.dtype === "float32" &&
        outputs[name].delivery === "value" &&
        same(outputs[name].shape, [
          index === 0 ? "prefix" : "reply",
          audio.shape[1],
        ]) &&
        outputs[name].semantic ===
          (index === 0 ? "audio.talker_prefix.v1" : "audio.talker_reply.v1"),
    ) ||
    !["embed", "prefill", "decode", "create_state", "reset_state"].every(
      (name) => component.exports[name],
    ) ||
    component.exports.run !== component.exports.prefill ||
    !Number.isSafeInteger(contextWindowSize) ||
    contextWindowSize < 3 ||
    !Number.isSafeInteger(maxNewTokens) ||
    maxNewTokens < 2 ||
    !Number.isSafeInteger(speakerBosTokenId)
  )
    throw new Error("Unsupported speech Thinker ABI or limits");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error("Speech Thinker exceeds buffer binding limit");
  let active = false;
  return {
    async create(id, requested, signal) {
      aborted(signal);
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active) throw new Error("Speech Thinker runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(value: T): T => {
        tvm.detachFromCurrentScope(value);
        owned.push(value);
        return value;
      };
      const drain = createTVMRequestDrain(device);
      let disposed = false,
        disposal: Promise<void> | undefined;
      const dispose = (): Promise<void> => {
        if (disposal) return disposal;
        disposed = true;
        disposal = (async () => {
          await drain.wait();
          let first: unknown;
          for (const value of owned.reverse())
            try {
              value.dispose();
            } catch (error) {
              first ??= error;
            }
          active = false;
          if (first) throw first;
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
        const config = metadata.thinker,
          profile = config?.speech_profile;
        if (
          !config ||
          config.model_dtype !== "float32" ||
          config.hidden_size !== audio.shape[1] ||
          !Number.isSafeInteger(config.hidden_size) ||
          config.hidden_size < 1 ||
          !Number.isSafeInteger(config.vocab_size) ||
          config.vocab_size < 1 ||
          !Number.isSafeInteger(config.prefill_chunk_size) ||
          config.prefill_chunk_size < 1 ||
          !Number.isSafeInteger(config.context_window_size) ||
          contextWindowSize > config.context_window_size ||
          config.audio_token_id !== 151646 ||
          config.image_token_id !== 151655 ||
          config.video_token_id !== 151656 ||
          !same(config.eos_token_ids, [151643, 151645]) ||
          !profile ||
          !same(profile.speaker_bos_token_ids, [151870, 151872]) ||
          profile.text_eos_token_id !== 151861 ||
          profile.text_pad_token_id !== 151859 ||
          !profile.speaker_bos_token_ids.includes(speakerBosTokenId) ||
          [
            speakerBosTokenId,
            profile.text_eos_token_id,
            profile.text_pad_token_id,
          ].some((x) => x < 0 || x >= config.vocab_size)
        )
          throw new Error("Invalid FP32 speech Thinker metadata");
        const vocab = config.active_vocab_size ?? config.vocab_size,
          width = config.hidden_size;
        if (
          !Number.isSafeInteger(vocab) ||
          vocab < 1 ||
          vocab > config.vocab_size
        )
          throw new Error("Invalid active vocabulary");
        let packed: tvmjs.TVMObject, cache: tvmjs.TVMObject;
        const functions: Record<string, tvmjs.PackedFunc> = {};
        let setupFailed = false,
          setupError: unknown;
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
        } catch (error) {
          setupFailed = true;
          setupError = error;
        }
        try {
          tvm.endScope();
        } catch (error) {
          if (!setupFailed) {
            setupFailed = true;
            setupError = error;
          }
        }
        if (setupFailed) throw setupError;
        let used = false;
        return {
          dispose,
          async *run(events, runSignal) {
            if (used || disposed)
              throw new Error("Speech Thinker backend can run once");
            used = true;
            const check = () => {
              if (disposed) throw new Error("TVM request disposed");
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
                throw new Error("Invalid speech Thinker input event");
              if (event.kind === "end") {
                ended.add(event.port);
                continue;
              }
              if (values[event.port])
                throw new Error("Duplicate speech Thinker input");
              values[event.port] = {
                shape: [...event.tensor.shape],
                data: event.tensor.data.slice(),
              };
            }
            check();
            const tokens = values.token_ids,
              embeddings = values.audio_embeddings;
            if (
              ended.size !== 2 ||
              !tokens ||
              !(tokens.data instanceof Int32Array) ||
              !same(tokens.shape, [tokens.data.length]) ||
              tokens.data.length < 1 ||
              tokens.data.length + 2 > contextWindowSize ||
              !embeddings ||
              !(embeddings.data instanceof Float32Array) ||
              embeddings.shape.length !== 2 ||
              embeddings.shape[1] !== width ||
              !Number.isSafeInteger(embeddings.shape[0]) ||
              embeddings.shape[0] < 0 ||
              embeddings.data.length !== embeddings.shape[0] * width ||
              !embeddings.data.every(Number.isFinite)
            )
              throw new Error("Invalid speech Thinker prompt tensors");
            const audioProvided = embeddings.shape[0] > 0;
            let audioCount = 0;
            for (const token of tokens.data) {
              if (token === config.audio_token_id && audioProvided)
                audioCount++;
              else if (
                token < 0 ||
                token >= config.vocab_size ||
                [151652, 151653, 151654, 151655, 151656].includes(token)
              )
                throw new Error("Unsupported speech Thinker token or modality");
            }
            if (audioCount !== embeddings.shape[0])
              throw new Error(
                "Audio placeholder count does not match embeddings",
              );
            const read = (shapes: number[][], body: () => tvmjs.Tensor[]) =>
              drain.run(async () => {
                check();
                const cpus: tvmjs.Tensor[] = [];
                let failure: unknown,
                  failed = false,
                  syncAttempted = false,
                  result: Float32Array[] | undefined;
                try {
                  tvm.beginScope();
                  try {
                    const tensors = body();
                    if (tensors.length !== shapes.length)
                      throw new Error("Unexpected Thinker result cardinality");
                    for (let i = 0; i < tensors.length; i++) {
                      const tensor = tensors[i];
                      if (
                        tensor.dtype !== "float32" ||
                        !same(tensor.shape, shapes[i])
                      )
                        throw new Error(
                          "Unexpected speech Thinker result shape or dtype",
                        );
                      const cpu = tvm.detachFromCurrentScope(
                        tvm.empty(shapes[i], "float32", tvm.cpu()),
                      );
                      cpus.push(cpu);
                      cpu.copyFrom(tensor);
                    }
                  } catch (error) {
                    failed = true;
                    failure = error;
                  }
                  try {
                    tvm.endScope();
                  } catch (error) {
                    if (!failed) {
                      failed = true;
                      failure = error;
                    }
                  }
                  if (failed) throw failure;
                  syncAttempted = true;
                  await drain.sync();
                  check();
                  result = cpus.map((cpu) => {
                    const data = cpu.toArray();
                    if (
                      !(data instanceof Float32Array) ||
                      !data.every(Number.isFinite)
                    )
                      throw new Error("Invalid speech Thinker values");
                    return data;
                  });
                } catch (error) {
                  failed = true;
                  failure = error;
                }
                if (failed && !syncAttempted)
                  try {
                    await drain.sync();
                  } catch {
                    /* Preserve the first operation error. */
                  }
                for (const cpu of cpus)
                  try {
                    cpu.dispose();
                  } catch (error) {
                    if (!failed) {
                      failed = true;
                      failure = error;
                    }
                  }
                if (failed) throw failure;
                return result!;
              });
            const embed = async (ids: Int32Array) =>
              (
                await read([[ids.length, width]], () => [
                  functions.embed(
                    tvm.empty([ids.length], "int32", device).copyFrom(ids),
                    packed,
                  ) as tvmjs.Tensor,
                ])
              )[0];
            const forward = async (input: Float32Array, decode: boolean) => {
              const count = input.length / width;
              return read(
                [
                  [1, 1, config.vocab_size],
                  [1, count, width],
                ],
                () => {
                  const tensor = tvm
                    .empty([1, count, width], "float32", device)
                    .copyFrom(input);
                  functions.begin(
                    cache,
                    tvm.makeShapeTuple([0]),
                    tvm.makeShapeTuple([count]),
                  );
                  let outputs: tvmjs.Tensor[] | undefined,
                    failed = false,
                    failure: unknown;
                  try {
                    const value = functions[decode ? "decode" : "prefill"](
                      tensor,
                      cache,
                      packed,
                    ) as unknown as { get(i: number): tvmjs.Tensor };
                    outputs = [value.get(0), value.get(1)];
                  } catch (error) {
                    failed = true;
                    failure = error;
                  }
                  try {
                    functions.end(cache);
                  } catch (error) {
                    if (!failed) {
                      failed = true;
                      failure = error;
                    }
                  }
                  if (failed) throw failure;
                  return outputs!;
                },
              );
            };
            const promptEmbedding = new Float32Array(
                tokens.data.length * width,
              ),
              promptHidden = new Float32Array(promptEmbedding.length);
            let logits: Float32Array | undefined,
              audioOffset = 0;
            for (
              let start = 0;
              start < tokens.data.length;
              start += config.prefill_chunk_size
            ) {
              check();
              const chunk = tokens.data.slice(
                  start,
                  start + config.prefill_chunk_size,
                ),
                safe = chunk.map((id) =>
                  audioProvided && id === config.audio_token_id ? 0 : id,
                );
              const input = await embed(safe);
              for (let i = 0; i < chunk.length; i++)
                if (audioProvided && chunk[i] === config.audio_token_id) {
                  input.set(
                    embeddings.data.subarray(
                      audioOffset * width,
                      (audioOffset + 1) * width,
                    ),
                    i * width,
                  );
                  audioOffset++;
                }
              promptEmbedding.set(input, start * width);
              const result = await forward(input, false);
              logits = result[0];
              promptHidden.set(result[1], start * width);
            }
            const inputEmbeddings: Float32Array[] = [promptEmbedding],
              hiddenStates: Float32Array[] = [promptHidden],
              generatedIds: number[] = [];
            const budget = Math.min(
              maxNewTokens,
              contextWindowSize - tokens.data.length,
            );
            let naturalEOS = false;
            for (let step = 0; step < budget; step++) {
              check();
              let next = 0;
              for (let i = 1; i < vocab; i++)
                if (logits![i] > logits![next]) next = i;
              generatedIds.push(next);
              naturalEOS = config.eos_token_ids.includes(next);
              if (naturalEOS) break;
              yield {
                port: "token_ids",
                tensor: { shape: [1], data: new Int32Array([next]) },
              };
              check();
              if (step + 1 === budget) break;
              const input = await embed(new Int32Array([next]));
              const result = await forward(input, true);
              logits = result[0];
              inputEmbeddings.push(input);
              hiddenStates.push(result[1]);
            }
            check();
            if (generatedIds.length < 2) throw new EmptyThinkerResponseError();
            const specialEmbeddings = await embed(
              new Int32Array([
                speakerBosTokenId,
                profile.text_eos_token_id,
                profile.text_pad_token_id,
              ]),
            );
            const aligned = alignThinkerSpeechTrace({
              inputIds: tokens.data,
              generatedIds: new Int32Array(generatedIds),
              inputEmbeddings,
              hiddenStates,
              specialEmbeddings,
              width,
              vocabSize: config.vocab_size,
              audioFeaturesProvided: audioProvided,
              eosIds: config.eos_token_ids,
            });
            check();
            yield { port: "prefix_embeddings", tensor: aligned.prefix };
            check();
            yield { port: "reply_embeddings", tensor: aligned.reply };
            check();
            yield {
              port: "finish_reason",
              tensor: {
                shape: [1],
                data: new Int32Array([naturalEOS ? 0 : 1]),
              },
            };
          },
        };
      } catch (error) {
        try {
          await dispose();
        } catch {
          /* Preserve setup failure. */
        }
        throw error;
      }
    },
  };
}
