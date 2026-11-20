/** Explicit text/audio → Thinker → Talker → flow → waveform graph orchestration. */
import {
  ArtifactGraphExecution,
  GraphComponentFactory,
  GraphInputEvent,
  GraphTensor,
  createArtifactGraphExecution,
} from "./artifact_graph_executor";
import {
  CompiledProgramArtifactV2,
  GraphComponent,
  GraphTask,
  ModelPackageManifestV2,
  computeGraphInterfaceId,
  parseCompiledProgramArtifactV2,
  parseModelPackageManifestV2,
  validateGraphArtifactPair,
} from "./artifact_manifest_v2";
import { integrateFlowRK4 } from "./artifact_graph_flow_tvm";
import { aborted, same } from "./artifact_graph_tvm";

export interface OmniSpeechPart {
  model: ModelPackageManifestV2;
  compiled: CompiledProgramArtifactV2;
  componentId: string;
  factory: GraphComponentFactory;
  /** Required when the standalone package has more than one task. */
  task?: string;
}
export interface OmniSpeechParts {
  thinker: OmniSpeechPart;
  talker: OmniSpeechPart;
  reference: OmniSpeechPart;
  flow: OmniSpeechPart;
  vocoder: OmniSpeechPart;
  features?: OmniSpeechPart;
  audio?: OmniSpeechPart;
}
export interface OmniSpeechOptions {
  parts: OmniSpeechParts;
  maxFrames: number;
  timePoints: Float32Array;
  guidanceScale: number;
}
export interface OmniSpeechGraph {
  model: ModelPackageManifestV2;
  compiled: CompiledProgramArtifactV2;
  factory: GraphComponentFactory;
  /** Task inputs are token IDs, optional PCM, reference mels, speaker embedding and explicit noise. */
  execute(options: {
    mode: "text" | "audio";
    inputs: Record<string, AsyncIterable<GraphTensor>>;
    signal?: AbortSignal;
    queueCapacity?: number;
  }): Promise<ArtifactGraphExecution>;
}

/** Run one validated standalone component task and release its request state. */
async function single(
  part: OmniSpeechPart,
  inputs: Record<string, GraphTensor>,
  output: string,
  signal: AbortSignal,
): Promise<GraphTensor> {
  const names = Object.keys(part.model.tasks),
    task = part.task ?? (names.length === 1 ? names[0] : undefined);
  if (!task) throw new Error("Ambiguous standalone Omni task");
  const execution = await createArtifactGraphExecution({
    model: part.model,
    compiled: part.compiled,
    task,
    factory: part.factory,
    outputs: [output],
    signal,
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([name, tensor]) => [
        name,
        (async function* () {
          yield tensor;
        })(),
      ]),
    ),
  });
  // Observe failure immediately even if the output iterator reports it first.
  const done = execution.done.then(
    () => ({ error: undefined as unknown }),
    (error) => ({ error }),
  );
  let value: GraphTensor | undefined;
  try {
    for await (const tensor of execution.outputs[output]) {
      if (value) throw new Error("Expected one Omni component output");
      value = tensor;
    }
    const result = await done;
    if (result.error) throw result.error;
    if (!value) throw new Error("Missing Omni component output");
    return value;
  } catch (error) {
    execution.abort();
    await done;
    throw error;
  }
}

export async function createOmniSpeechGraph(
  options: OmniSpeechOptions,
): Promise<OmniSpeechGraph> {
  const { maxFrames, guidanceScale } = options,
    timePoints = new Float32Array(options.timePoints);
  if (
    !Number.isSafeInteger(maxFrames) ||
    maxFrames < 2 ||
    maxFrames > 512 ||
    maxFrames % 2 ||
    !Number.isFinite(guidanceScale) ||
    guidanceScale < 0 ||
    timePoints.length < 2 ||
    timePoints.length > 64 ||
    !timePoints.every(
      (t, i) =>
        Number.isFinite(t) &&
        t >= 0 &&
        t <= 1 &&
        (i === 0 || t > timePoints[i - 1]),
    )
  )
    throw new Error("Invalid Omni synthesis policy");
  const parts = {} as Record<string, OmniSpeechPart>;
  for (const [name, part] of Object.entries(options.parts)) {
    if (!part) continue;
    const model = parseModelPackageManifestV2(part.model),
      compiled = parseCompiledProgramArtifactV2(part.compiled);
    await validateGraphArtifactPair(model, compiled);
    if (
      !model.components[part.componentId] ||
      !compiled.components[part.componentId]
    )
      throw new Error(`Missing Omni component ${name}`);
    parts[name] = { ...part, model, compiled };
  }
  if (
    ["thinker", "talker", "reference", "flow", "vocoder"].some(
      (name) => !parts[name],
    ) ||
    Boolean(parts.features) !== Boolean(parts.audio)
  )
    throw new Error("Incomplete Omni component set");
  const spec = (name: string) =>
    parts[name].model.components[parts[name].componentId];
  const contracts: Record<string, string> = {
    thinker: "qwen2_5_omni.thinker_audio_talker.v1",
    talker: "qwen2_5_omni.talker.v1",
    reference: "qwen2_5_omni.reference_conditioner.v1",
    flow: "qwen2_5_omni.flow_predictor.v1",
    vocoder: "qwen2_5_omni.waveform_decoder.v1",
  };
  if (parts.features) {
    contracts.features = "qwen2_5_omni.audio_features.v1";
    contracts.audio = "qwen2_5_omni.audio_encoder.v1";
  }
  for (const [name, contract] of Object.entries(contracts))
    if (spec(name).contract !== contract)
      throw new Error(`Unsupported ${name} contract`);
  if (
    !same(spec("flow").inputs.speaker.shape, [192]) ||
    !same(spec("reference").outputs.embedding.shape, [128]) ||
    !same(spec("vocoder").inputs.log_mel.shape, [80, "frames"])
  )
    throw new Error("Unsupported Omni synthesis dimensions");
  const synthesis: GraphComponent = {
    kind: "transform",
    contract: "qwen2_5_omni.flow_sampler.v1",
    weights: null,
    state: { kind: "none", scope: "request" },
    inputs: {
      code: spec("talker").outputs.code,
      reference_mel: spec("reference").inputs.mel,
      speaker: spec("flow").inputs.speaker,
      noise: {
        semantic: "audio.flow_initial_noise.v1",
        dtype: "float32",
        shape: [maxFrames, 80],
        delivery: "value",
      },
    },
    outputs: { log_mel: spec("vocoder").inputs.log_mel },
  };
  const components = Object.fromEntries(
    Object.keys(parts).map((name) => [name, spec(name)]),
  );
  components.synthesis = synthesis;
  const task = (audio: boolean): GraphTask => ({
    inputs: {
      token_ids: {
        spec: spec("thinker").inputs.token_ids,
        targets: [{ component: "thinker", port: "token_ids" }],
      },
      ...(audio
        ? {
            pcm: {
              spec: spec("features").inputs.pcm,
              targets: [{ component: "features", port: "pcm" }],
            },
          }
        : {
            audio_embeddings: {
              spec: spec("thinker").inputs.audio_embeddings,
              targets: [{ component: "thinker", port: "audio_embeddings" }],
            },
          }),
      reference_mel: {
        spec: synthesis.inputs.reference_mel,
        targets: [{ component: "synthesis", port: "reference_mel" }],
      },
      speaker: {
        spec: synthesis.inputs.speaker,
        targets: [{ component: "synthesis", port: "speaker" }],
      },
      noise: {
        spec: synthesis.inputs.noise,
        targets: [{ component: "synthesis", port: "noise" }],
      },
    },
    outputs: {
      token_ids: {
        spec: spec("thinker").outputs.token_ids,
        source: { component: "thinker", port: "token_ids" },
        optional: false,
      },
      finish_reason: {
        spec: spec("thinker").outputs.finish_reason,
        source: { component: "thinker", port: "finish_reason" },
        optional: false,
      },
      pcm: {
        spec: spec("vocoder").outputs.pcm,
        source: { component: "vocoder", port: "pcm" },
        optional: false,
      },
    },
    edges: [
      ...(audio
        ? [
            {
              source: { component: "features", port: "mel_features" },
              target: { component: "audio", port: "mel_features" },
            },
            {
              source: { component: "audio", port: "embeddings" },
              target: { component: "thinker", port: "audio_embeddings" },
            },
          ]
        : []),
      {
        source: { component: "thinker", port: "prefix_embeddings" },
        target: { component: "talker", port: "prefix_embeddings" },
      },
      {
        source: { component: "thinker", port: "reply_embeddings" },
        target: { component: "talker", port: "reply_embeddings" },
      },
      {
        source: { component: "talker", port: "code" },
        target: { component: "synthesis", port: "code" },
      },
      {
        source: { component: "synthesis", port: "log_mel" },
        target: { component: "vocoder", port: "log_mel" },
      },
    ],
  });
  let model: ModelPackageManifestV2 = {
    schema: "mlc.model-package",
    schema_version: 2,
    interface_id: "sha256:" + "0".repeat(64),
    components,
    tasks: {
      "omni.text_to_speech": task(false),
      ...(parts.features ? { "omni.audio_to_speech": task(true) } : {}),
    },
  };
  model.interface_id = await computeGraphInterfaceId(model);
  model = parseModelPackageManifestV2(model);
  const compiled = parseCompiledProgramArtifactV2({
    schema: "mlc.compiled-program",
    schema_version: 2,
    interface_id: model.interface_id,
    components: {
      ...Object.fromEntries(
        Object.entries(parts).map(([name, part]) => [
          name,
          part.compiled.components[part.componentId],
        ]),
      ),
      synthesis: {
        spec: synthesis,
        module: "web-llm.js",
        exports: { run: "integrateFlowRK4" },
        resources: {
          required_features: [],
          max_storage_buffer_binding_size: 0,
          estimated_device_memory_bytes: 0,
        },
      },
    },
  });
  await validateGraphArtifactPair(model, compiled);
  const synthesisFactory: GraphComponentFactory = {
    create(id, component, signal) {
      aborted(signal);
      if (id !== "synthesis" || !same(component, compiled.components.synthesis))
        throw new Error("Unexpected synthesis component");
      let disposed = false,
        used = false,
        work: Promise<GraphTensor> | undefined,
        disposal: Promise<void> | undefined;
      const controller = new AbortController();
      const abort = () => controller.abort();
      signal.addEventListener("abort", abort, { once: true });
      const check = (runSignal: AbortSignal) => {
        if (disposed) throw new Error("Omni synthesis disposed");
        aborted(signal);
        aborted(runSignal);
        aborted(controller.signal);
      };
      return {
        dispose() {
          if (disposal) return disposal;
          disposed = true;
          controller.abort();
          disposal = (async () => {
            try {
              await work;
            } catch {
              /* Keep the operation's original error. */
            } finally {
              signal.removeEventListener("abort", abort);
            }
          })();
          return disposal;
        },
        async *run(
          events: AsyncIterable<GraphInputEvent>,
          runSignal: AbortSignal,
        ) {
          if (used || disposed)
            throw new Error("Synthesis backend can run once");
          used = true;
          const onAbort = () => controller.abort();
          runSignal.addEventListener("abort", onAbort, { once: true });
          try {
            const values: Record<string, GraphTensor> = {},
              ended = new Set<string>();
            for await (const event of events) {
              check(runSignal);
              if (
                !Object.hasOwn(synthesis.inputs, event.port) ||
                ended.has(event.port)
              )
                throw new Error("Invalid synthesis input event");
              if (event.kind === "end") {
                ended.add(event.port);
                continue;
              }
              if (values[event.port])
                throw new Error("Duplicate synthesis input");
              values[event.port] = {
                shape: [...event.tensor.shape],
                data: event.tensor.data.slice(),
              };
            }
            check(runSignal);
            if (ended.size !== 4 || Object.keys(values).length !== 4)
              throw new Error("Missing synthesis data or EOF");
            const code = values.code,
              mel = values.reference_mel,
              speaker = values.speaker,
              noise = values.noise;
            const frames = code.data.length * 2;
            if (
              !(code.data instanceof Int32Array) ||
              !same(code.shape, [code.data.length]) ||
              frames < 2 ||
              frames > maxFrames ||
              !code.data.every((x) => x >= 0 && x <= 8193) ||
              !(mel.data instanceof Float32Array) ||
              mel.shape.length !== 2 ||
              mel.shape[1] !== 80 ||
              !Number.isSafeInteger(mel.shape[0]) ||
              mel.shape[0] < 5 ||
              mel.shape[0] > 512 ||
              mel.data.length !== mel.shape[0] * 80 ||
              !mel.data.every(Number.isFinite) ||
              !(speaker.data instanceof Float32Array) ||
              !same(speaker.shape, [192]) ||
              speaker.data.length !== 192 ||
              !speaker.data.every(Number.isFinite) ||
              !(noise.data instanceof Float32Array) ||
              !same(noise.shape, [maxFrames, 80]) ||
              noise.data.length !== maxFrames * 80 ||
              !noise.data.every(Number.isFinite)
            )
              throw new Error("Invalid synthesis tensors");
            work = (async () => {
              const reference = await single(
                parts.reference,
                { mel },
                "embedding",
                controller.signal,
              );
              check(runSignal);
              const nullReference = await single(
                parts.reference,
                {
                  mel: {
                    shape: [...mel.shape],
                    data: new Float32Array(mel.data.length),
                  },
                },
                "embedding",
                controller.signal,
              );
              check(runSignal);
              const solved = await integrateFlowRK4(
                (input, abortSignal) =>
                  single(parts.flow, { ...input }, "velocity", abortSignal),
                {
                  noise: {
                    shape: [frames, 80],
                    data: noise.data.slice(0, frames * 80),
                  },
                  code,
                  reference,
                  nullReference,
                  speaker,
                  timePoints,
                  guidanceScale,
                  maxTimePoints: timePoints.length,
                  maxFrames,
                  melDim: 80,
                  referenceDim: 128,
                  speakerDim: 192,
                  repeats: 2,
                  numEmbeds: 8193,
                },
                controller.signal,
              );
              check(runSignal);
              const channelMajor = new Float32Array(frames * 80);
              for (let t = 0; t < frames; t++)
                for (let c = 0; c < 80; c++)
                  channelMajor[c * frames + t] = solved.data[t * 80 + c];
              return { shape: [80, frames], data: channelMajor };
            })();
            const result = await work;
            check(runSignal);
            yield { port: "log_mel", tensor: result };
          } finally {
            runSignal.removeEventListener("abort", onAbort);
          }
        },
      };
    },
  };
  const factory: GraphComponentFactory = {
    create(id, component, signal) {
      if (
        !Object.hasOwn(compiled.components, id) ||
        !same(component, compiled.components[id])
      )
        throw new Error("Unexpected Omni graph component");
      if (id === "synthesis")
        return synthesisFactory.create(id, component, signal);
      const part = parts[id];
      return part.factory.create(
        part.componentId,
        part.compiled.components[part.componentId],
        signal,
      );
    },
  };
  return {
    model,
    compiled,
    factory,
    execute({ mode, inputs, signal, queueCapacity }) {
      if (mode !== "text" && mode !== "audio")
        throw new Error("Unsupported Omni input mode");
      const supplied = { ...inputs };
      if (mode === "text") {
        if (Object.hasOwn(supplied, "audio_embeddings"))
          throw new Error("Text mode supplies its own empty audio embeddings");
        supplied.audio_embeddings = (async function* () {
          yield {
            shape: [
              0,
              spec("thinker").inputs.audio_embeddings.shape[1] as number,
            ],
            data: new Float32Array(0),
          };
        })();
      }
      return createArtifactGraphExecution({
        model,
        compiled,
        factory,
        task: mode === "audio" ? "omni.audio_to_speech" : "omni.text_to_speech",
        inputs: supplied,
        signal,
        queueCapacity,
      });
    },
  };
}
