/** Bounded reference-mel conditioner; null mel is evaluated by the same network. */
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createTVMRequestDrain } from "./artifact_graph_tvm_lifecycle";
import {
  AudioEncoderTVMOptions,
  loadComponent,
  same,
} from "./artifact_graph_tvm";
import { GraphComponentFactory } from "./artifact_graph_executor";
import {
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  validateGraphArtifactPair,
} from "./artifact_manifest_v2";
export type ReferenceConditionerTVMOptions = AudioEncoderTVMOptions;

/** Caller owns the dedicated runtime, device and cache; the backend owns its VM. */
export async function createReferenceConditionerTVMFactory(
  options: ReferenceConditionerTVMOptions,
): Promise<GraphComponentFactory> {
  const { runtime: tvm, device, componentId, maxFrames } = options;
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId],
    input = component?.spec.inputs.mel,
    output = component?.spec.outputs.embedding;
  if (
    !component ||
    component.spec.contract !== "qwen2_5_omni.reference_conditioner.v1" ||
    component.spec.kind !== "encoder" ||
    component.spec.state.kind !== "none" ||
    !component.spec.weights ||
    Object.keys(component.spec.inputs).length !== 1 ||
    Object.keys(component.spec.outputs).length !== 1 ||
    input?.semantic !== "audio.reference_mel.v1" ||
    input.dtype !== "float32" ||
    input.delivery !== "value" ||
    input.shape.length !== 2 ||
    !Number.isSafeInteger(input.shape[1]) ||
    Number(input.shape[1]) < 1 ||
    output?.semantic !== "audio.reference_embedding.v1" ||
    output.dtype !== "float32" ||
    output.delivery !== "value" ||
    output.shape.length !== 1 ||
    !Number.isSafeInteger(output.shape[0]) ||
    Number(output.shape[0]) < 1 ||
    !component.exports.forward ||
    component.exports.run !== component.exports.forward ||
    !Number.isSafeInteger(maxFrames) ||
    maxFrames < 5
  )
    throw new Error("Unsupported reference conditioner ABI or frame bound");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error(
      "Reference conditioner exceeds device buffer binding limit",
    );
  let active = false;
  return {
    async create(id, requested, signal) {
      let disposed = false;
      const check = (runSignal?: AbortSignal) => {
        if (disposed) throw new Error("TVM request disposed");
        if (signal.aborted || runSignal?.aborted)
          throw new Error("Reference conditioner execution aborted");
      };
      check();
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active)
        throw new Error("Reference conditioner runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(v: T): T => {
        tvm.detachFromCurrentScope(v);
        owned.push(v);
        return v;
      };
      const drain = createTVMRequestDrain(device);
      let disposal: Promise<void> | undefined;
      const dispose = (): Promise<void> => {
        if (disposal) return disposal;
        disposed = true;
        disposal = (async () => {
          await drain.wait();
          let error: unknown;
          for (const v of owned.reverse()) {
            try {
              v.dispose();
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
        const config = metadata.reference_conditioner;
        if (
          !config ||
          config.mel_dim !== input.shape[1] ||
          config.embedding_dim !== output.shape[0] ||
          !Number.isSafeInteger(config.min_mel_frames) ||
          config.min_mel_frames < 5 ||
          maxFrames < config.min_mel_frames ||
          !Number.isSafeInteger(config.max_mel_frames) ||
          maxFrames > config.max_mel_frames ||
          !Number.isSafeInteger(maxFrames * config.mel_dim)
        )
          throw new Error(
            "Invalid reference conditioner metadata or frame bound",
          );
        check();
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
              throw new Error("Reference conditioner backend can run once");
            used = true;
            let mel: Float32Array | undefined,
              frames = 0,
              ended = false;
            for await (const event of events) {
              check(runSignal);
              if (event.port !== "mel" || ended)
                throw new Error("Invalid reference-mel input event");
              if (event.kind === "end") {
                ended = true;
                continue;
              }
              const { shape, data } = event.tensor;
              if (
                mel ||
                !(data instanceof Float32Array) ||
                shape.length !== 2 ||
                shape[1] !== config.mel_dim ||
                !Number.isSafeInteger(shape[0]) ||
                shape[0] < config.min_mel_frames ||
                shape[0] > maxFrames ||
                data.length !== shape[0] * shape[1] ||
                !data.every(Number.isFinite)
              )
                throw new Error("Invalid reference-mel clip");
              frames = shape[0];
              mel = new Float32Array(data);
            }
            check(runSignal);
            if (!mel || !ended)
              throw new Error("Missing reference-mel data or end event");
            const dimensions = config.embedding_dim;
            const embedding = await drain.run(async () => {
              let syncAttempted = false;
              let cpu: tvmjs.Tensor | undefined;
              let embedding: Float32Array | undefined;
              let failure: unknown,
                failed = false;
              try {
                tvm.beginScope();
                try {
                  const arg = tvm
                    .empty([frames, config.mel_dim], "float32", device)
                    .copyFrom(mel);
                  const result = forward(arg, packed) as tvmjs.Tensor;
                  if (
                    result.dtype !== "float32" ||
                    !same(result.shape, [dimensions])
                  )
                    throw new Error(
                      "Reference conditioner export returned unexpected shape or dtype",
                    );
                  cpu = tvm.detachFromCurrentScope(
                    tvm.empty([dimensions], "float32", tvm.cpu()),
                  );
                  cpu.copyFrom(result);
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
                check(runSignal);
                const raw = cpu!.toArray();
                if (!(raw instanceof Float32Array))
                  throw new Error("Invalid embedding dtype");
                embedding = raw;
                if (
                  !(embedding instanceof Float32Array) ||
                  embedding.length !== dimensions ||
                  !embedding.every((v) => Number.isFinite(v))
                )
                  throw new Error("Invalid embedding output");
              } catch (error) {
                failed = true;
                failure = error;
              }
              if (failed && !syncAttempted) {
                try {
                  await drain.sync();
                } catch {
                  /* Preserve the original operation error. */
                }
              }
              try {
                cpu?.dispose();
              } catch (error) {
                if (!failed) {
                  failed = true;
                  failure = error;
                }
              }
              if (failed) throw failure;
              return embedding!;
            });
            yield {
              port: "embedding",
              tensor: { shape: [dimensions], data: embedding! },
            };
          },
        };
      } catch (error) {
        try {
          await dispose();
        } catch {
          /* Preserve setup error. */
        }
        throw error;
      }
    },
  };
}
