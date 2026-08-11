import { ArtifactManifestError } from "./error";

export const MODEL_PACKAGE_MANIFEST_FILENAME = "mlc-model-manifest.json";
export const ARTIFACT_SCHEMA_VERSION = 1;

export interface PromptInsertion {
  prefix_token_ids: number[];
  placeholder_token_id: number;
  suffix_token_ids: number[];
}

export interface AudioDecodeProcessor {
  kind: "audio_decode";
  format: "pcm_f32";
  sample_rate_hz: number;
  channels: 1;
  min_samples: number;
  max_samples: number;
}

export interface TaskInput {
  processor: string | AudioDecodeProcessor;
  adapter?: string;
  prompt?: PromptInsertion;
}

export interface TaskSpec {
  executor: string;
  inputs: Record<string, TaskInput>;
  output: string;
}

export interface WeightContract {
  manifest: "ndarray-cache.json";
  parameter_schema_id: string;
}

export interface ModelPackageManifest {
  schema: "mlc.model-package";
  schema_version: 1;
  chat_config: "mlc-chat-config.json";
  interface_id: string;
  weights: WeightContract;
  tasks: Record<string, TaskSpec>;
}

export interface ProgramSpec {
  kind: string;
  exports: Record<string, string>;
  adapters: Record<string, string>;
}

export interface ResourceRequirements {
  required_features: string[];
  max_storage_buffer_binding_size: number;
  estimated_device_memory_bytes: number;
}

export interface CompiledProgramArtifact {
  schema: "mlc.compiled-program";
  schema_version: 1;
  interface_id: string;
  parameter_schema_id: string;
  programs: Record<string, ProgramSpec>;
  resources: ResourceRequirements;
}

export interface ResolvedChatCompletionArtifact {
  task: TaskSpec;
  program: ProgramSpec;
  textInput: TaskInput;
  audioInput?: TaskInput & {
    processor: AudioDecodeProcessor;
    adapter: string;
    prompt: PromptInsertion;
  };
  compiled: CompiledProgramArtifact;
}

type JsonRecord = Record<string, unknown>;

function fail(path: string, message: string): never {
  throw new ArtifactManifestError(`${path}: ${message}`);
}

function record(value: unknown, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(path, "expected an object");
  }
  return value as JsonRecord;
}

function exactKeys(
  value: JsonRecord,
  allowed: readonly string[],
  required: readonly string[],
  path: string,
): void {
  const allowedSet = new Set(allowed);
  for (const key of Object.keys(value)) {
    if (!allowedSet.has(key)) {
      fail(path, `unknown field ${JSON.stringify(key)}`);
    }
  }
  for (const key of required) {
    if (!(key in value)) {
      fail(path, `missing field ${JSON.stringify(key)}`);
    }
  }
}

function stringValue(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    fail(path, "expected a non-empty string");
  }
  return value;
}

function literal<T extends string>(
  value: unknown,
  expected: T,
  path: string,
): T {
  if (value !== expected) {
    fail(path, `expected ${JSON.stringify(expected)}`);
  }
  return expected;
}

function integer(
  value: unknown,
  path: string,
  minimum = Number.MIN_SAFE_INTEGER,
): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    fail(path, `expected an integer greater than or equal to ${minimum}`);
  }
  return value as number;
}

function sha256(value: unknown, path: string): string {
  const result = stringValue(value, path);
  if (!/^sha256:[0-9a-f]{64}$/.test(result)) {
    fail(path, "expected a lowercase sha256:<64 hex digits> identifier");
  }
  return result;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const object = value as JsonRecord;
    const entries = Object.keys(object)
      .filter((key) => object[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`);
    return `{${entries.join(",")}}`;
  }
  const result = JSON.stringify(value);
  if (result === undefined) {
    throw new ArtifactManifestError(
      "Artifact identity contains a non-JSON value",
    );
  }
  return result;
}

/** Compute the identity MLC uses to bind canonical task semantics. */
export async function computeInterfaceId(
  tasks: Record<string, TaskSpec>,
): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalJson({ tasks }));
  // TextEncoder always owns an ArrayBuffer, while TypeScript models the view
  // more broadly as ArrayBufferLike.
  const buffer = bytes.buffer as ArrayBuffer;
  const digest = new Uint8Array(
    await globalThis.crypto.subtle.digest("SHA-256", buffer),
  );
  return `sha256:${Array.from(digest, (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("")}`;
}

async function verifyInterfaceId(
  manifest: ModelPackageManifest,
): Promise<void> {
  const actual = await computeInterfaceId(manifest.tasks);
  if (actual !== manifest.interface_id) {
    throw new ArtifactManifestError(
      `${MODEL_PACKAGE_MANIFEST_FILENAME}: interface_id does not match its tasks`,
    );
  }
}

function integerArray(value: unknown, path: string): number[] {
  if (!Array.isArray(value)) {
    fail(path, "expected an array");
  }
  return value.map((item, index) => integer(item, `${path}[${index}]`, 0));
}

function stringArray(value: unknown, path: string): string[] {
  if (!Array.isArray(value)) {
    fail(path, "expected an array");
  }
  return value.map((item, index) => stringValue(item, `${path}[${index}]`));
}

function parsePromptInsertion(value: unknown, path: string): PromptInsertion {
  const obj = record(value, path);
  exactKeys(
    obj,
    ["prefix_token_ids", "placeholder_token_id", "suffix_token_ids"],
    ["placeholder_token_id"],
    path,
  );
  return {
    prefix_token_ids:
      obj.prefix_token_ids === undefined
        ? []
        : integerArray(obj.prefix_token_ids, `${path}.prefix_token_ids`),
    placeholder_token_id: integer(
      obj.placeholder_token_id,
      `${path}.placeholder_token_id`,
      0,
    ),
    suffix_token_ids:
      obj.suffix_token_ids === undefined
        ? []
        : integerArray(obj.suffix_token_ids, `${path}.suffix_token_ids`),
  };
}

function parseAudioDecodeProcessor(
  value: unknown,
  path: string,
): AudioDecodeProcessor {
  const obj = record(value, path);
  exactKeys(
    obj,
    [
      "kind",
      "format",
      "sample_rate_hz",
      "channels",
      "min_samples",
      "max_samples",
    ],
    ["kind", "format", "sample_rate_hz", "channels", "max_samples"],
    path,
  );
  const minSamples =
    obj.min_samples === undefined
      ? 1
      : integer(obj.min_samples, `${path}.min_samples`, 1);
  const maxSamples = integer(obj.max_samples, `${path}.max_samples`, 1);
  if (minSamples > maxSamples) {
    fail(path, "min_samples must not exceed max_samples");
  }
  return {
    kind: literal(obj.kind, "audio_decode", `${path}.kind`),
    format: literal(obj.format, "pcm_f32", `${path}.format`),
    sample_rate_hz: integer(obj.sample_rate_hz, `${path}.sample_rate_hz`, 1),
    channels:
      integer(obj.channels, `${path}.channels`, 1) === 1
        ? 1
        : fail(path, "channels must be 1"),
    min_samples: minSamples,
    max_samples: maxSamples,
  };
}

function parseTaskInput(value: unknown, path: string): TaskInput {
  const obj = record(value, path);
  exactKeys(obj, ["processor", "adapter", "prompt"], ["processor"], path);
  let processor: string | AudioDecodeProcessor;
  if (typeof obj.processor === "string") {
    processor = stringValue(obj.processor, `${path}.processor`);
  } else {
    processor = parseAudioDecodeProcessor(obj.processor, `${path}.processor`);
  }
  const adapter =
    obj.adapter === undefined
      ? undefined
      : stringValue(obj.adapter, `${path}.adapter`);
  const prompt =
    obj.prompt === undefined
      ? undefined
      : parsePromptInsertion(obj.prompt, `${path}.prompt`);
  if ((adapter === undefined) !== (prompt === undefined)) {
    fail(path, "adapter and prompt must be declared together");
  }
  return { processor, adapter, prompt };
}

function parseTaskSpec(value: unknown, path: string): TaskSpec {
  const obj = record(value, path);
  exactKeys(
    obj,
    ["executor", "inputs", "output"],
    ["executor", "inputs", "output"],
    path,
  );
  const inputsObj = record(obj.inputs, `${path}.inputs`);
  if (Object.keys(inputsObj).length === 0) {
    fail(`${path}.inputs`, "expected at least one input");
  }
  const inputs: Record<string, TaskInput> = {};
  for (const name of Object.keys(inputsObj).sort()) {
    stringValue(name, `${path}.inputs key`);
    inputs[name] = parseTaskInput(inputsObj[name], `${path}.inputs.${name}`);
  }
  return {
    executor: stringValue(obj.executor, `${path}.executor`),
    inputs,
    output: stringValue(obj.output, `${path}.output`),
  };
}

function parseTasks(value: unknown, path: string): Record<string, TaskSpec> {
  const obj = record(value, path);
  if (Object.keys(obj).length === 0) {
    fail(path, "expected at least one task");
  }
  const tasks: Record<string, TaskSpec> = {};
  for (const name of Object.keys(obj).sort()) {
    stringValue(name, `${path} key`);
    tasks[name] = parseTaskSpec(obj[name], `${path}.${name}`);
  }
  return tasks;
}

function parseStringMap(
  value: unknown,
  path: string,
  allowEmpty: boolean,
): Record<string, string> {
  const obj = record(value, path);
  if (!allowEmpty && Object.keys(obj).length === 0) {
    fail(path, "expected at least one entrypoint");
  }
  const result: Record<string, string> = {};
  for (const name of Object.keys(obj).sort()) {
    stringValue(name, `${path} key`);
    result[name] = stringValue(obj[name], `${path}.${name}`);
  }
  return result;
}

function parseProgramSpec(value: unknown, path: string): ProgramSpec {
  const obj = record(value, path);
  exactKeys(obj, ["kind", "exports", "adapters"], ["kind", "exports"], path);
  return {
    kind: stringValue(obj.kind, `${path}.kind`),
    exports: parseStringMap(obj.exports, `${path}.exports`, false),
    adapters:
      obj.adapters === undefined
        ? {}
        : parseStringMap(obj.adapters, `${path}.adapters`, true),
  };
}

function parsePrograms(
  value: unknown,
  path: string,
): Record<string, ProgramSpec> {
  const obj = record(value, path);
  if (Object.keys(obj).length === 0) {
    fail(path, "expected at least one program");
  }
  const result: Record<string, ProgramSpec> = {};
  for (const name of Object.keys(obj).sort()) {
    stringValue(name, `${path} key`);
    result[name] = parseProgramSpec(obj[name], `${path}.${name}`);
  }
  return result;
}

export function parseModelPackageManifest(
  value: unknown,
): ModelPackageManifest {
  const path = MODEL_PACKAGE_MANIFEST_FILENAME;
  const obj = record(value, path);
  exactKeys(
    obj,
    [
      "schema",
      "schema_version",
      "chat_config",
      "interface_id",
      "weights",
      "tasks",
    ],
    ["schema", "schema_version", "interface_id", "weights", "tasks"],
    path,
  );
  const weights = record(obj.weights, `${path}.weights`);
  exactKeys(
    weights,
    ["manifest", "parameter_schema_id"],
    ["manifest", "parameter_schema_id"],
    `${path}.weights`,
  );
  const version = integer(obj.schema_version, `${path}.schema_version`, 1);
  if (version !== ARTIFACT_SCHEMA_VERSION) {
    fail(`${path}.schema_version`, `unsupported version ${version}`);
  }
  return {
    schema: literal(obj.schema, "mlc.model-package", `${path}.schema`),
    schema_version: 1,
    chat_config:
      obj.chat_config === undefined
        ? "mlc-chat-config.json"
        : literal(
            obj.chat_config,
            "mlc-chat-config.json",
            `${path}.chat_config`,
          ),
    interface_id: sha256(obj.interface_id, `${path}.interface_id`),
    weights: {
      manifest: literal(
        weights.manifest,
        "ndarray-cache.json",
        `${path}.weights.manifest`,
      ),
      parameter_schema_id: sha256(
        weights.parameter_schema_id,
        `${path}.weights.parameter_schema_id`,
      ),
    },
    tasks: parseTasks(obj.tasks, `${path}.tasks`),
  };
}

export function parseCompiledProgramArtifact(
  value: unknown,
): CompiledProgramArtifact {
  const path = "_metadata.artifact";
  const obj = record(value, path);
  exactKeys(
    obj,
    [
      "schema",
      "schema_version",
      "interface_id",
      "parameter_schema_id",
      "programs",
      "resources",
    ],
    [
      "schema",
      "schema_version",
      "interface_id",
      "parameter_schema_id",
      "programs",
      "resources",
    ],
    path,
  );
  const version = integer(obj.schema_version, `${path}.schema_version`, 1);
  if (version !== ARTIFACT_SCHEMA_VERSION) {
    fail(`${path}.schema_version`, `unsupported version ${version}`);
  }
  const resources = record(obj.resources, `${path}.resources`);
  exactKeys(
    resources,
    [
      "required_features",
      "max_storage_buffer_binding_size",
      "estimated_device_memory_bytes",
    ],
    [
      "required_features",
      "max_storage_buffer_binding_size",
      "estimated_device_memory_bytes",
    ],
    `${path}.resources`,
  );
  return {
    schema: literal(obj.schema, "mlc.compiled-program", `${path}.schema`),
    schema_version: 1,
    interface_id: sha256(obj.interface_id, `${path}.interface_id`),
    parameter_schema_id: sha256(
      obj.parameter_schema_id,
      `${path}.parameter_schema_id`,
    ),
    programs: parsePrograms(obj.programs, `${path}.programs`),
    resources: {
      required_features: stringArray(
        resources.required_features,
        `${path}.resources.required_features`,
      ),
      max_storage_buffer_binding_size: integer(
        resources.max_storage_buffer_binding_size,
        `${path}.resources.max_storage_buffer_binding_size`,
        0,
      ),
      estimated_device_memory_bytes: integer(
        resources.estimated_device_memory_bytes,
        `${path}.resources.estimated_device_memory_bytes`,
        0,
      ),
    },
  };
}

export function resolveChatCompletionArtifact(
  modelPackage: ModelPackageManifest,
  compiled: CompiledProgramArtifact,
): ResolvedChatCompletionArtifact {
  if (modelPackage.interface_id !== compiled.interface_id) {
    fail(
      "artifact contract",
      "model-package and compiled-program interface IDs differ",
    );
  }
  if (
    modelPackage.weights.parameter_schema_id !== compiled.parameter_schema_id
  ) {
    fail(
      "artifact contract",
      "weight and compiled-program parameter schema IDs differ",
    );
  }
  const task = modelPackage.tasks["chat.completions"];
  if (task === undefined) {
    fail("artifact contract", "task chat.completions is not declared");
  }
  if (task.output !== "text") {
    fail("artifact contract", "WebLLM chat.completions requires text output");
  }
  const program = compiled.programs[task.executor];
  if (program === undefined) {
    fail(
      "artifact contract",
      `task executor ${JSON.stringify(task.executor)} is missing`,
    );
  }
  if (program.kind !== "token_generation") {
    fail(
      "artifact contract",
      `unsupported executor kind ${JSON.stringify(program.kind)}`,
    );
  }
  for (const role of [
    "embed_tokens",
    "prefill_prompt",
    "decode_tokens",
    "create_kv_cache",
  ]) {
    if (program.exports[role] === undefined) {
      fail(
        "artifact contract",
        `executor is missing export role ${JSON.stringify(role)}`,
      );
    }
  }
  const textInput = task.inputs.text;
  if (textInput === undefined || textInput.processor !== "tokenizer") {
    fail(
      "artifact contract",
      "chat.completions requires a text tokenizer input",
    );
  }
  if (textInput.adapter !== undefined || textInput.prompt !== undefined) {
    fail(
      "artifact contract",
      "the tokenizer input cannot declare an adapter or prompt insertion",
    );
  }

  let audioInput: ResolvedChatCompletionArtifact["audioInput"];
  for (const [name, input] of Object.entries(task.inputs)) {
    if (
      typeof input.processor !== "string" &&
      input.processor.kind === "audio_decode"
    ) {
      if (audioInput !== undefined) {
        fail(
          "artifact contract",
          "WebLLM supports one canonical audio input per task",
        );
      }
      if (input.adapter === undefined || input.prompt === undefined) {
        fail(
          "artifact contract",
          `audio input ${JSON.stringify(name)} has no adapter prompt`,
        );
      }
      if (program.adapters[input.adapter] === undefined) {
        fail(
          "artifact contract",
          `audio input ${JSON.stringify(name)} references a missing adapter`,
        );
      }
      audioInput = {
        ...input,
        processor: input.processor,
        adapter: input.adapter,
        prompt: input.prompt,
      };
    }
  }
  return { task, program, textInput, audioInput, compiled };
}

export async function fetchOptionalModelPackageManifest(
  url: string,
  signal?: AbortSignal,
): Promise<ModelPackageManifest | undefined> {
  const response = await fetch(url, { signal });
  if (response.status === 404) {
    return undefined;
  }
  if (!response.ok) {
    throw new ArtifactManifestError(
      `Failed to fetch ${MODEL_PACKAGE_MANIFEST_FILENAME}: HTTP ${response.status}`,
    );
  }
  let value: unknown;
  try {
    value = await response.json();
  } catch (error) {
    throw new ArtifactManifestError(
      `${MODEL_PACKAGE_MANIFEST_FILENAME}: invalid JSON`,
      { cause: error },
    );
  }
  const manifest = parseModelPackageManifest(value);
  await verifyInterfaceId(manifest);
  return manifest;
}
