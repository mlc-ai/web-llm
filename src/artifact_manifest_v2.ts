/** Validated multi-component contracts. MLCEngine graph execution is not implemented. */
import { ArtifactManifestError } from "./error";
import {
  loadModelPackageManifest,
  ModelPackageManifest,
} from "./artifact_manifest";

export interface GraphPort {
  semantic: string;
  dtype: "int32" | "float16" | "float32" | "uint8";
  shape: (number | string)[];
  delivery: "value" | "stream";
}
export interface GraphEndpoint {
  component: string;
  port: string;
}
export interface GraphComponent {
  kind:
    | "encoder"
    | "autoregressive"
    | "code_predictor"
    | "waveform_decoder"
    | "transform";
  contract: string;
  inputs: Record<string, GraphPort>;
  outputs: Record<string, GraphPort>;
  weights: { manifest: string; parameter_schema_id: string } | null;
  state: {
    kind: "none" | "kv_cache" | "recurrent" | "hybrid";
    scope: "request";
  };
}
export interface GraphTask {
  inputs: Record<string, { spec: GraphPort; targets: GraphEndpoint[] }>;
  outputs: Record<
    string,
    { spec: GraphPort; source: GraphEndpoint; optional: boolean }
  >;
  edges: { source: GraphEndpoint; target: GraphEndpoint }[];
}
export interface ModelPackageManifestV2 {
  schema: "mlc.model-package";
  schema_version: 2;
  interface_id: string;
  components: Record<string, GraphComponent>;
  tasks: Record<string, GraphTask>;
}
export interface CompiledProgramArtifactV2 {
  schema: "mlc.compiled-program";
  schema_version: 2;
  interface_id: string;
  components: Record<
    string,
    {
      spec: GraphComponent;
      module: string;
      exports: Record<string, string>;
      resources: {
        required_features: string[];
        max_storage_buffer_binding_size: number;
        estimated_device_memory_bytes: number;
      };
    }
  >;
}
const ID = /^[A-Za-z][A-Za-z0-9_]*$/;
const TASK = /^[A-Za-z][A-Za-z0-9_.]*$/;
const FEATURE = /^[a-z][a-z0-9-]*$/;
const EXPORT = /^[A-Za-z_][A-Za-z0-9_]*$/;
const SEMANTIC = /^[a-z][a-z0-9_.]*\.v[1-9][0-9]*$/;
const SHA = /^sha256:[0-9a-f]{64}$/;
const forbidden = new Set(["__proto__", "prototype", "constructor"]);
function fail(message: string): never {
  throw new ArtifactManifestError(`Artifact v2: ${message}`);
}
function object(value: unknown): Record<string, unknown> {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    ![Object.prototype, null].includes(Object.getPrototypeOf(value))
  )
    fail("expected plain object");
  for (const key of Object.keys(value))
    if (forbidden.has(key)) fail("unsafe object key");
  return value as Record<string, unknown>;
}
function fields(value: unknown, names: string[]): Record<string, unknown> {
  const obj = object(value);
  if (
    Object.keys(obj).length !== names.length ||
    names.some((k) => !Object.hasOwn(obj, k))
  )
    fail(`expected fields ${names.join(", ")}`);
  return obj;
}
function string(value: unknown, pattern = ID): string {
  if (
    typeof value !== "string" ||
    pattern.exec(value)?.[0] !== value ||
    forbidden.has(value)
  )
    fail("invalid identifier");
  return value;
}
function choice<T extends string>(value: unknown, choices: readonly T[]): T {
  if (typeof value !== "string" || !choices.includes(value as T))
    fail("invalid enum value");
  return value as T;
}
function integer(value: unknown, min = 0): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min)
    fail("expected safe integer");
  return value;
}
function array<T>(
  value: unknown,
  parse: (x: unknown) => T,
  nonempty = false,
): T[] {
  if (!Array.isArray(value) || (nonempty && !value.length))
    fail("invalid array");
  return value.map(parse);
}
function record<T>(
  value: unknown,
  parse: (x: unknown) => T,
  nonempty = true,
  keyPattern = ID,
): Record<string, T> {
  const obj = object(value);
  if (nonempty && !Object.keys(obj).length) fail("empty map");
  return Object.fromEntries(
    Object.keys(obj)
      .sort()
      .map((k) => [string(k, keyPattern), parse(obj[k])]),
  );
}
function path(value: unknown): string {
  if (
    typeof value !== "string" ||
    !value
      .split("/")
      .every((s) => /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.exec(s)?.[0] === s)
  )
    fail("unsafe relative path");
  return value;
}
function port(value: unknown): GraphPort {
  const o = fields(value, ["semantic", "dtype", "shape", "delivery"]);
  return {
    semantic: string(o.semantic, SEMANTIC),
    dtype: choice(o.dtype, ["int32", "float16", "float32", "uint8"]),
    shape: array(o.shape, (x) =>
      typeof x === "string" ? string(x) : integer(x, 1),
    ),
    delivery: choice(o.delivery, ["value", "stream"]),
  };
}
function endpoint(value: unknown): GraphEndpoint {
  const o = fields(value, ["component", "port"]);
  return { component: string(o.component), port: string(o.port) };
}
function component(value: unknown): GraphComponent {
  const o = fields(value, [
    "kind",
    "contract",
    "inputs",
    "outputs",
    "weights",
    "state",
  ]);
  let weights = null;
  if (o.weights !== null) {
    const w = fields(o.weights, ["manifest", "parameter_schema_id"]);
    weights = {
      manifest: path(w.manifest),
      parameter_schema_id: string(w.parameter_schema_id, SHA),
    };
  }
  const s = fields(o.state, ["kind", "scope"]);
  return {
    kind: choice(o.kind, [
      "encoder",
      "autoregressive",
      "code_predictor",
      "waveform_decoder",
      "transform",
    ]),
    contract: string(o.contract, SEMANTIC),
    inputs: record(o.inputs, port, false),
    outputs: record(o.outputs, port),
    weights,
    state: {
      kind: choice(s.kind, ["none", "kv_cache", "recurrent", "hybrid"]),
      scope: choice(s.scope, ["request"]),
    },
  };
}
function task(value: unknown): GraphTask {
  const o = fields(value, ["inputs", "outputs", "edges"]);
  return {
    inputs: record(o.inputs, (x) => {
      const i = fields(x, ["spec", "targets"]);
      return { spec: port(i.spec), targets: array(i.targets, endpoint, true) };
    }),
    outputs: record(o.outputs, (x) => {
      const a = fields(x, ["spec", "source", "optional"]);
      if (typeof a.optional !== "boolean") fail("optional must be boolean");
      return {
        spec: port(a.spec),
        source: endpoint(a.source),
        optional: a.optional,
      };
    }),
    edges: array(o.edges, (x) => {
      const e = fields(x, ["source", "target"]);
      return { source: endpoint(e.source), target: endpoint(e.target) };
    }),
  };
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const o = value as Record<string, unknown>;
    return `{${Object.keys(o)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}
function equal(a: unknown, b: unknown): boolean {
  return canonical(a) === canonical(b);
}
function graphOrder(
  components: Record<string, GraphComponent>,
  task: GraphTask,
): string[] {
  const active = new Set<string>(),
    bound = new Set<string>();
  const deps = new Map<string, Set<string>>();
  function resolve(e: GraphEndpoint, side: "inputs" | "outputs"): GraphPort {
    if (!Object.hasOwn(components, e.component)) fail("unresolved endpoint");
    const c = components[e.component];
    if (!c || !Object.hasOwn(c[side], e.port)) fail("unresolved endpoint");
    active.add(e.component);
    return c[side][e.port];
  }
  function bind(e: GraphEndpoint, spec: GraphPort) {
    if (!equal(resolve(e, "inputs"), spec)) fail("incompatible ports");
    const key = `${e.component}.${e.port}`;
    if (bound.has(key)) fail("duplicate input binding");
    bound.add(key);
  }
  for (const i of Object.values(task.inputs))
    for (const e of i.targets) bind(e, i.spec);
  for (const e of task.edges) {
    bind(e.target, resolve(e.source, "outputs"));
    if (!deps.has(e.target.component)) deps.set(e.target.component, new Set());
    deps.get(e.target.component)!.add(e.source.component);
  }
  for (const o of Object.values(task.outputs))
    if (!equal(resolve(o.source, "outputs"), o.spec))
      fail("incompatible output");
  if (!Object.values(task.outputs).some((o) => !o.optional))
    fail("required output missing");
  for (const id of active)
    for (const p of Object.keys(components[id].inputs))
      if (!bound.has(`${id}.${p}`)) fail("unbound component input");
  const order: string[] = [],
    visiting = new Set<string>(),
    done = new Set<string>();
  function visit(id: string) {
    if (visiting.has(id)) fail("graph cycle");
    if (done.has(id)) return;
    visiting.add(id);
    for (const d of [...(deps.get(id) ?? [])].sort()) visit(d);
    visiting.delete(id);
    done.add(id);
    order.push(id);
  }
  for (const o of Object.values(task.outputs)) visit(o.source.component);
  if (done.size !== active.size) fail("dead graph branch");
  return order;
}
export function parseModelPackageManifestV2(
  value: unknown,
): ModelPackageManifestV2 {
  const o = fields(value, [
    "schema",
    "schema_version",
    "interface_id",
    "components",
    "tasks",
  ]);
  if (o.schema_version !== 2) fail("unsupported schema version");
  const result: ModelPackageManifestV2 = {
    schema: choice(o.schema, ["mlc.model-package"]),
    schema_version: 2,
    interface_id: string(o.interface_id, SHA),
    components: record(o.components, component),
    tasks: record(o.tasks, task, true, TASK),
  };
  for (const t of Object.values(result.tasks)) graphOrder(result.components, t);
  return result;
}
export async function computeGraphInterfaceId(value: unknown): Promise<string> {
  const { interface_id: ignored, ...payload } = parseModelPackageManifestV2({
    ...object(value),
    interface_id: `sha256:${"0".repeat(64)}`,
  });
  void ignored;
  const bytes = new TextEncoder().encode(canonical(payload));
  const digest = new Uint8Array(
    await globalThis.crypto.subtle.digest(
      "SHA-256",
      bytes.buffer as ArrayBuffer,
    ),
  );
  return `sha256:${Array.from(digest, (x) => x.toString(16).padStart(2, "0")).join("")}`;
}
export function parseCompiledProgramArtifactV2(
  value: unknown,
): CompiledProgramArtifactV2 {
  const o = fields(value, [
    "schema",
    "schema_version",
    "interface_id",
    "components",
  ]);
  if (o.schema_version !== 2) fail("unsupported schema version");
  return {
    schema: choice(o.schema, ["mlc.compiled-program"]),
    schema_version: 2,
    interface_id: string(o.interface_id, SHA),
    components: record(o.components, (v) => {
      const c = fields(v, ["spec", "module", "exports", "resources"]),
        spec = component(c.spec),
        exports = record(c.exports, (x) => string(x, EXPORT));
      for (const role of spec.state.kind === "none"
        ? ["run"]
        : ["run", "create_state", "reset_state"])
        if (!Object.hasOwn(exports, role)) fail(`missing export ${role}`);
      const r = fields(c.resources, [
        "required_features",
        "max_storage_buffer_binding_size",
        "estimated_device_memory_bytes",
      ]);
      const features = array(r.required_features, (x) => string(x, FEATURE));
      if (new Set(features).size !== features.length) fail("duplicate feature");
      return {
        spec,
        module: path(c.module),
        exports,
        resources: {
          required_features: features,
          max_storage_buffer_binding_size: integer(
            r.max_storage_buffer_binding_size,
          ),
          estimated_device_memory_bytes: integer(
            r.estimated_device_memory_bytes,
          ),
        },
      };
    }),
  };
}
export async function validateGraphArtifactPair(
  model: ModelPackageManifestV2,
  compiled: CompiledProgramArtifactV2,
): Promise<void> {
  model = parseModelPackageManifestV2(model);
  compiled = parseCompiledProgramArtifactV2(compiled);
  if ((await computeGraphInterfaceId(model)) !== model.interface_id)
    fail("interface hash mismatch");
  if (
    model.interface_id !== compiled.interface_id ||
    !equal(Object.keys(model.components), Object.keys(compiled.components))
  )
    fail("artifact interface mismatch");
  for (const id of Object.keys(model.components))
    if (!equal(model.components[id], compiled.components[id].spec))
      fail(`component contract mismatch: ${id}`);
}
export async function planTaskOutputs(
  model: ModelPackageManifestV2,
  taskName: string,
  requestedOutputs?: string[],
): Promise<{ components: string[]; inputs: string[] }> {
  model = parseModelPackageManifestV2(model);
  if ((await computeGraphInterfaceId(model)) !== model.interface_id)
    fail("interface hash mismatch");
  string(taskName, TASK);
  if (!Object.hasOwn(model.tasks, taskName)) fail("unknown task");
  const t = model.tasks[taskName];
  if (!t) fail("unknown task");
  const required = Object.keys(t.outputs).filter((k) => !t.outputs[k].optional);
  const selected =
    requestedOutputs === undefined
      ? required
      : array(requestedOutputs, (x) => string(x), true);
  if (
    !selected.length ||
    new Set(selected).size !== selected.length ||
    required.some((k) => !selected.includes(k)) ||
    selected.some((k) => !Object.hasOwn(t.outputs, k))
  )
    fail("invalid selected outputs");
  const needed = new Set(selected.map((k) => t.outputs[k].source.component));
  let changed = true;
  while (changed) {
    changed = false;
    for (const e of t.edges)
      if (needed.has(e.target.component) && !needed.has(e.source.component)) {
        needed.add(e.source.component);
        changed = true;
      }
  }
  return {
    components: graphOrder(model.components, t).filter((id) => needed.has(id)),
    inputs: Object.keys(t.inputs)
      .filter((id) => t.inputs[id].targets.some((e) => needed.has(e.component)))
      .sort(),
  };
}
/** Recognize valid v2 packages but refuse execution before any dependent artifact is fetched. */
export async function loadExecutableModelPackageManifest(
  data: ArrayBuffer,
): Promise<ModelPackageManifest> {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder().decode(data));
  } catch {
    fail("invalid JSON");
  }
  if (object(value).schema_version === 2) {
    const model = parseModelPackageManifestV2(value);
    if ((await computeGraphInterfaceId(model)) !== model.interface_id)
      fail("interface hash mismatch");
    fail(
      "graph contracts are recognized, but graph execution is not supported",
    );
  }
  return loadModelPackageManifest(data);
}
