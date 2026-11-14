import { readFileSync } from "fs";
import { join } from "path";
import { webcrypto } from "crypto";
import {
  computeGraphInterfaceId,
  loadExecutableModelPackageManifest,
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
  validateGraphArtifactPair,
  planTaskOutputs,
} from "../src/artifact_manifest_v2";
const fixture = (name: string) =>
  JSON.parse(
    readFileSync(
      join(__dirname, "fixtures/artifact_manifest_v2", name + ".json"),
      "utf8",
    ),
  );
beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  }),
);
test("shared graph identity, compiled agreement, and optional audio closure", async () => {
  const m = parseModelPackageManifestV2(fixture("model"));
  const expected = fixture("expected");
  expect(await computeGraphInterfaceId(m)).toBe(expected.interface_id);
  await validateGraphArtifactPair(
    m,
    parseCompiledProgramArtifactV2(fixture("compiled")),
  );
  expect(await planTaskOutputs(m, "chat.completions")).toEqual({
    components: expected.text_components,
    inputs: expected.inputs,
  });
  expect(
    (await planTaskOutputs(m, "chat.completions", ["text", "audio"]))
      .components,
  ).toEqual(expected.all_components);
  await expect(
    planTaskOutputs(m, "chat.completions", ["audio"]),
  ).rejects.toThrow();
});
test("recognized graph cannot execute", async () => {
  const bytes = new TextEncoder().encode(JSON.stringify(fixture("model")));
  await expect(
    loadExecutableModelPackageManifest(bytes.buffer as ArrayBuffer),
  ).rejects.toThrow(/graph execution is not supported/);
});
test.each([
  (m: any) => {
    m.schema_version = true;
  },
  (m: any) => {
    m.components.thinker.inputs.text.shape = [true];
  },
  (m: any) => {
    m.components.thinker.inputs.text.shape = [Number.MAX_SAFE_INTEGER + 1];
  },
  (m: any) => {
    m.components.thinker.weights.manifest = "../weights.json";
  },
  (m: any) => {
    m.components.thinker.weights.manifest = "https://host/weights.json";
  },
  (m: any) => {
    m.tasks["chat.completions"].edges.push(
      m.tasks["chat.completions"].edges[0],
    );
  },
  (m: any) => {
    m.tasks["chat.completions"].inputs.text.spec.shape = ["different_symbol"];
  },
  (m: any) => {
    m.tasks["chat.completions"].outputs.text.optional = 1;
  },
  (m: any) => {
    m.components.constructor = m.components.thinker;
  },
  (m: any) => {
    m.components.thinker.extra = 1;
  },
  (m: any) => {
    m.tasks["chat.completions"].edges[0].source.component = "missing";
  },
])("rejects malformed graph %i", (mutate) => {
  const m = fixture("model");
  mutate(m);
  expect(() => parseModelPackageManifestV2(m)).toThrow();
});
test("rejects cycles and dead branches", () => {
  const m = fixture("model"),
    t = m.tasks["chat.completions"];
  delete t.inputs.text;
  t.edges.push({
    source: { component: "thinker", port: "text" },
    target: { component: "thinker", port: "text" },
  });
  expect(() => parseModelPackageManifestV2(m)).toThrow(/cycle/);
  const dead = fixture("model");
  delete dead.tasks["chat.completions"].outputs.audio;
  expect(() => parseModelPackageManifestV2(dead)).toThrow(/dead graph branch/);
});
test("rejects compiled state exports and contract/resource mismatches", async () => {
  const c = fixture("compiled");
  delete c.components.thinker.exports.reset_state;
  expect(() => parseCompiledProgramArtifactV2(c)).toThrow(/missing export/);
  const mismatch = fixture("compiled");
  mismatch.components.thinker.spec.contract = "other.v1";
  await expect(
    validateGraphArtifactPair(fixture("model"), mismatch),
  ).rejects.toThrow(/contract mismatch/);
  const resource = fixture("compiled");
  resource.components.thinker.resources.estimated_device_memory_bytes = false;
  expect(() => parseCompiledProgramArtifactV2(resource)).toThrow();
});

jest.mock("@mlc-ai/web-runtime", () => ({
  createArtifactCache: jest.fn(),
}));

test("engine refuses v2 before config, WASM, or weight fetch", async () => {
  Object.defineProperty(globalThis, "location", {
    value: { origin: "https://example.com" },
    configurable: true,
  });
  const runtime = await import("@mlc-ai/web-runtime");
  const { MLCEngine } = await import("../src/engine");
  const bytes = new TextEncoder().encode(JSON.stringify(fixture("model")));
  const fetchWithCache = jest.fn().mockResolvedValue(bytes.buffer);
  (runtime.createArtifactCache as jest.Mock).mockReturnValue({
    fetchWithCache,
  });
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model_id: "omni",
          model: "https://example.com/model/",
          model_lib: "https://example.com/model.wasm",
          model_manifest: "mlc-model-manifest.json",
        },
      ],
    },
  });
  await expect(engine.reload("omni")).rejects.toThrow(
    /graph execution is not supported/,
  );
  expect(fetchWithCache).toHaveBeenCalledTimes(1);
  expect(fetchWithCache.mock.calls[0][0]).toBe(
    "https://example.com/model/resolve/main/mlc-model-manifest.json",
  );
});

test("pair, planner, and loader reject a stale claimed identity", async () => {
  const m = fixture("model"),
    c = fixture("compiled");
  m.components.thinker.weights.manifest = "changed/tensor-cache.json";
  c.components.thinker.spec = m.components.thinker;
  expect(() => parseModelPackageManifestV2(m)).not.toThrow();
  await expect(validateGraphArtifactPair(m, c)).rejects.toThrow(
    /hash mismatch/,
  );
  await expect(planTaskOutputs(m, "chat.completions")).rejects.toThrow(
    /hash mismatch/,
  );
  const bytes = new TextEncoder().encode(JSON.stringify(m));
  await expect(
    loadExecutableModelPackageManifest(bytes.buffer as ArrayBuffer),
  ).rejects.toThrow(/hash mismatch/);
});

test("authoring identity permits an absent or stale claimed ID", async () => {
  const m = fixture("model"),
    expected = m.interface_id;
  delete m.interface_id;
  expect(await computeGraphInterfaceId(m)).toBe(expected);
  m.interface_id = "not-an-id";
  expect(await computeGraphInterfaceId(m)).toBe(expected);
  expect(() => parseModelPackageManifestV2(m)).toThrow();
});

test.each([
  (m: any) => {
    m.interface_id += "\n";
  },
  (m: any) => {
    m.components.thinker.contract += "\n";
  },
  (m: any) => {
    m.components.thinker.weights.manifest += "\n";
  },
  (m: any) => {
    m.components.thinker.inputs.text.shape = ["tokens\n"];
  },
])("rejects newline suffixes in lexical fields %i", (mutate) => {
  const m = fixture("model");
  mutate(m);
  expect(() => parseModelPackageManifestV2(m)).toThrow(/Artifact v2/);
});
test.each(["toString", "valueOf"])(
  "rejects inherited endpoint/task %s",
  async (name) => {
    const m = fixture("model");
    m.tasks["chat.completions"].edges[0].source.component = name;
    expect(() => parseModelPackageManifestV2(m)).toThrow(
      /Artifact v2: unresolved endpoint/,
    );
    await expect(planTaskOutputs(fixture("model"), name)).rejects.toThrow(
      /Artifact v2: unknown task/,
    );
  },
);
