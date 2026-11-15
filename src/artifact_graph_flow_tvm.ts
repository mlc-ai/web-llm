/** Explicit-time flow prediction; reference conditioning and sampling are separate. */
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createTVMRequestDrain } from "./artifact_graph_tvm_lifecycle";
import {
  AudioEncoderTVMOptions,
  loadComponent,
  same,
} from "./artifact_graph_tvm";
import { GraphComponentFactory, GraphTensor } from "./artifact_graph_executor";
import {
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  validateGraphArtifactPair,
} from "./artifact_manifest_v2";
export type FlowPredictorTVMOptions = AudioEncoderTVMOptions;
export interface FlowInputs {
  noise: GraphTensor;
  time: GraphTensor;
  code: GraphTensor;
  reference: GraphTensor;
  speaker: GraphTensor;
}
export interface FlowLimits {
  maxFrames: number;
  melDim: number;
  referenceDim: number;
  speakerDim: number;
  repeats: number;
  numEmbeds: number;
}
function abort(signal: AbortSignal) {
  if (signal.aborted) throw new Error("Flow execution aborted");
}
function finite(
  t: GraphTensor,
  shape: number[],
): t is GraphTensor & { data: Float32Array } {
  return (
    t.data instanceof Float32Array &&
    same(t.shape, shape) &&
    t.data.length === shape.reduce((a, b) => a * b, 1) &&
    t.data.every(Number.isFinite)
  );
}
function limitsValid(c: FlowLimits) {
  return (
    [
      c.maxFrames,
      c.melDim,
      c.referenceDim,
      c.speakerDim,
      c.repeats,
      c.numEmbeds,
    ].every((n) => Number.isSafeInteger(n) && n > 0) &&
    Number.isSafeInteger(c.maxFrames * c.melDim)
  );
}
function validate(inputs: FlowInputs, c: FlowLimits): number {
  const { noise, time, code, reference, speaker } = inputs;
  const frames = noise.shape[0];
  if (
    !limitsValid(c) ||
    !Number.isSafeInteger(frames) ||
    frames < 1 ||
    frames > c.maxFrames ||
    frames % c.repeats ||
    !finite(noise, [frames, c.melDim]) ||
    !finite(time, [1]) ||
    time.data[0] < 0 ||
    time.data[0] > 1 ||
    !(code.data instanceof Int32Array) ||
    !same(code.shape, [frames / c.repeats]) ||
    code.data.length !== frames / c.repeats ||
    !code.data.every((n) => n >= 0 && n <= c.numEmbeds) ||
    !finite(reference, [c.referenceDim]) ||
    !finite(speaker, [c.speakerDim])
  )
    throw new Error("Invalid bounded flow inputs");
  return frames;
}
const ports = ["noise", "time", "code", "reference", "speaker"] as const;

export async function createFlowPredictorTVMFactory(
  options: FlowPredictorTVMOptions,
): Promise<GraphComponentFactory> {
  const { runtime: tvm, device, componentId, maxFrames } = options;
  const model = parseModelPackageManifestV2(options.model),
    compiled = parseCompiledProgramArtifactV2(options.compiled);
  await validateGraphArtifactPair(model, compiled);
  const component = compiled.components[componentId];
  const expected = {
    noise: ["audio.flow_state.v1", "float32", 2],
    time: ["flow.time.v1", "float32", 1],
    code: ["audio.codec_ids.v1", "int32", 1],
    reference: ["audio.reference_embedding.v1", "float32", 1],
    speaker: ["audio.speaker_embedding.v1", "float32", 1],
  } as const;
  if (
    !component ||
    component.spec.contract !== "qwen2_5_omni.flow_predictor.v1" ||
    component.spec.kind !== "transform" ||
    component.spec.state.kind !== "none" ||
    !component.spec.weights ||
    Object.keys(component.spec.inputs).length !== 5 ||
    Object.keys(component.spec.outputs).length !== 1 ||
    !component.exports.forward ||
    component.exports.run !== component.exports.forward ||
    !Number.isSafeInteger(maxFrames) ||
    maxFrames < 1
  )
    throw new Error("Unsupported flow predictor ABI");
  for (const name of ports) {
    const port = component.spec.inputs[name],
      e = expected[name];
    if (
      !port ||
      port.semantic !== e[0] ||
      port.dtype !== e[1] ||
      port.shape.length !== e[2] ||
      port.delivery !== "value"
    )
      throw new Error("Unsupported flow input port");
  }
  const output = component.spec.outputs.velocity;
  if (
    !output ||
    output.semantic !== "audio.flow_velocity.v1" ||
    output.dtype !== "float32" ||
    output.delivery !== "value" ||
    output.shape.length !== 2 ||
    !same(output.shape, component.spec.inputs.noise.shape) ||
    !same(component.spec.inputs.time.shape, [1])
  )
    throw new Error("Unsupported flow output port");
  for (const feature of component.resources.required_features)
    if (!options.capabilities.features.has(feature))
      throw new Error(`Missing device feature ${feature}`);
  if (
    component.resources.max_storage_buffer_binding_size >
    options.capabilities.maxStorageBufferBindingSize
  )
    throw new Error("Flow predictor exceeds device buffer binding limit");
  let active = false;
  return {
    async create(id, requested, signal) {
      abort(signal);
      if (id !== componentId || !same(requested, component))
        throw new Error("Unexpected compiled component");
      if (active) throw new Error("Flow runtime is already in use");
      active = true;
      const owned: { dispose(): void }[] = [];
      const own = <T extends { dispose(): void }>(v: T): T => {
        tvm.detachFromCurrentScope(v);
        owned.push(v);
        return v;
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
          ),
          m = metadata.flow_predictor;
        const limits: FlowLimits = {
          maxFrames,
          melDim: m?.mel_dim,
          referenceDim: m?.reference_dim,
          speakerDim: m?.speaker_dim,
          repeats: m?.repeats,
          numEmbeds: m?.num_embeds,
        };
        if (
          !m ||
          !limitsValid(limits) ||
          !Number.isSafeInteger(m.max_mel_frames) ||
          maxFrames > m.max_mel_frames ||
          component.spec.inputs.noise.shape[1] !== m.mel_dim ||
          component.spec.inputs.reference.shape[0] !== m.reference_dim ||
          component.spec.inputs.speaker.shape[0] !== m.speaker_dim
        )
          throw new Error("Invalid flow predictor metadata or frame bound");
        abort(signal);
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
            if (used || disposed) throw new Error("Flow backend can run once");
            used = true;
            const check = () => {
              if (disposed) throw new Error("TVM request disposed");
              abort(signal);
              abort(runSignal);
            };
            const values: Partial<FlowInputs> = {},
              ended = new Set<string>();
            for await (const event of events) {
              check();
              const name = event.port as keyof FlowInputs;
              if (!ports.includes(name) || ended.has(name))
                throw new Error("Invalid flow input event");
              if (event.kind === "end") {
                ended.add(name);
                continue;
              }
              if (values[name]) throw new Error("Duplicate flow input");
              const t = event.tensor;
              values[name] = { shape: [...t.shape], data: t.data.slice() };
            }
            check();
            if (ports.some((name) => !values[name] || !ended.has(name)))
              throw new Error("Missing flow data or end event");
            const inputs = values as FlowInputs,
              frames = validate(inputs, limits);
            let cpu: tvmjs.Tensor | undefined, result: Float32Array | undefined;
            let failed = false,
              failure: unknown;
            try {
              tvm.beginScope();
              try {
                const args = ports.map((name) => {
                  const t = inputs[name];
                  return tvm
                    .empty(
                      t.shape,
                      name === "code" ? "int32" : "float32",
                      device,
                    )
                    .copyFrom(t.data as Float32Array | Int32Array);
                });
                const out = forward(...args, packed) as tvmjs.Tensor;
                if (
                  out.dtype !== "float32" ||
                  !same(out.shape, [frames, m.mel_dim])
                )
                  throw new Error("Unexpected flow velocity shape or dtype");
                cpu = tvm.detachFromCurrentScope(
                  tvm.empty(out.shape, "float32", tvm.cpu()).copyFrom(out),
                );
              } catch (e) {
                failed = true;
                failure = e;
              }
              try {
                tvm.endScope();
              } catch (e) {
                if (!failed) {
                  failed = true;
                  failure = e;
                }
              }
              if (failed) throw failure;
              await drain.sync();
              check();
              const data = cpu!.toArray();
              if (
                !(data instanceof Float32Array) ||
                data.length !== frames * m.mel_dim ||
                !data.every(Number.isFinite)
              )
                throw new Error("Invalid flow velocity");
              result = data;
            } catch (e) {
              failed = true;
              failure = e;
            }
            try {
              cpu?.dispose();
            } catch (e) {
              if (!failed) {
                failed = true;
                failure = e;
              }
            }
            if (failed) throw failure;
            yield {
              port: "velocity",
              tensor: { shape: [frames, m.mel_dim], data: result! },
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

export interface FlowSamplingOptions extends FlowLimits {
  noise: GraphTensor;
  code: GraphTensor;
  reference: GraphTensor;
  nullReference: GraphTensor;
  speaker: GraphTensor;
  timePoints: Float32Array;
  guidanceScale: number;
  /** Explicit limit on time points, including the initial point. */
  maxTimePoints: number;
}
export type FlowPrediction = (
  inputs: FlowInputs,
  signal: AbortSignal,
) => Promise<GraphTensor>;

/** Pinned RK4 3/8 integration; no implicit noise, time warping or reference encoding. */
export async function integrateFlowRK4(
  predict: FlowPrediction,
  options: FlowSamplingOptions,
  signal: AbortSignal,
): Promise<GraphTensor> {
  abort(signal);
  options = { ...options };
  const { timePoints: suppliedTimes, guidanceScale: guidance } = options;
  if (
    !(suppliedTimes instanceof Float32Array) ||
    !Number.isSafeInteger(options.maxTimePoints) ||
    options.maxTimePoints < 2 ||
    suppliedTimes.length < 2 ||
    suppliedTimes.length > options.maxTimePoints ||
    !suppliedTimes.every(
      (t, i) =>
        Number.isFinite(t) &&
        t >= 0 &&
        t <= 1 &&
        (i === 0 || t > suppliedTimes[i - 1]),
    ) ||
    !Number.isFinite(guidance) ||
    guidance < 0
  )
    throw new Error("Invalid flow time grid or guidance");
  const frames = validate(
    {
      ...options,
      time: { shape: [1], data: new Float32Array([suppliedTimes[0]]) },
    },
    options,
  );
  if (!finite(options.nullReference, [options.referenceDim]))
    throw new Error("Invalid null reference embedding");
  const copy = (tensor: GraphTensor): GraphTensor => ({
    shape: [...tensor.shape],
    data: tensor.data.slice(),
  });
  const times = new Float32Array(suppliedTimes),
    code = copy(options.code),
    reference = copy(options.reference),
    nullReference = copy(options.nullReference),
    speaker = copy(options.speaker);
  let state = new Float32Array(options.noise.data as Float32Array);
  const shape = [frames, options.melDim],
    zeroCode = {
      shape: [...code.shape],
      data: new Int32Array(code.data.length),
    },
    zeroSpeaker = {
      shape: [...speaker.shape],
      data: new Float32Array(speaker.data.length),
    };
  const f = Math.fround;
  const velocity = async (time: number, value: Float32Array) => {
    abort(signal);
    const inputs = {
      noise: { shape, data: value },
      time: { shape: [1], data: new Float32Array([time]) },
      code,
      reference,
      speaker,
    };
    const conditional = await predict(inputs, signal);
    abort(signal);
    if (!finite(conditional, shape))
      throw new Error("Invalid conditional flow velocity");
    const out = new Float32Array(conditional.data);
    if (guidance >= 1e-5) {
      const unconditioned = await predict(
        {
          ...inputs,
          code: zeroCode,
          reference: nullReference,
          speaker: zeroSpeaker,
        },
        signal,
      );
      abort(signal);
      if (!finite(unconditioned, shape))
        throw new Error("Invalid null flow velocity");
      for (let i = 0; i < out.length; i++)
        out[i] = f(out[i] + f(f(out[i] - unconditioned.data[i]) * f(guidance)));
    }
    if (!out.every(Number.isFinite))
      throw new Error("Nonfinite guided velocity");
    return out;
  };
  for (let step = 0; step < times.length - 1; step++) {
    const start = times[step],
      end = times[step + 1],
      dt = f(end - start),
      third = f(1 / 3),
      twoThirds = f(2 / 3);
    const k1 = await velocity(start, state);
    const x2 = Float32Array.from(state, (v, i) =>
      f(v + f(f(dt * k1[i]) * third)),
    );
    const k2 = await velocity(f(start + f(dt * third)), x2);
    const x3 = Float32Array.from(state, (v, i) =>
      f(v + f(dt * f(k2[i] - f(k1[i] * third)))),
    );
    const k3 = await velocity(f(start + f(dt * twoThirds)), x3);
    const x4 = Float32Array.from(state, (v, i) =>
      f(v + f(dt * f(f(k1[i] - k2[i]) + k3[i]))),
    );
    const k4 = await velocity(end, x4);
    state = Float32Array.from(state, (v, i) =>
      f(v + f(f(f(f(k1[i] + f(3 * f(k2[i] + k3[i]))) + k4[i]) * dt) / 8)),
    );
    if (!state.every(Number.isFinite)) throw new Error("Nonfinite flow state");
  }
  return { shape, data: state };
}
