import {
  computeInterfaceId,
  fetchOptionalModelPackageManifest,
  parseCompiledProgramArtifact,
  parseModelPackageManifest,
  resolveChatCompletionArtifact,
} from "../src/artifact_manifest";
import { ArtifactManifestError } from "../src/error";

const interfaceId = `sha256:${"1".repeat(64)}`;
const parameterId = `sha256:${"2".repeat(64)}`;

function modelPackage() {
  return {
    schema: "mlc.model-package",
    schema_version: 1,
    chat_config: "mlc-chat-config.json",
    interface_id: interfaceId,
    weights: {
      manifest: "ndarray-cache.json",
      parameter_schema_id: parameterId,
    },
    tasks: {
      "chat.completions": {
        executor: "generation",
        inputs: {
          text: { processor: "tokenizer" },
          audio: {
            processor: {
              kind: "audio_decode",
              format: "pcm_f32",
              sample_rate_hz: 16000,
              channels: 1,
              min_samples: 161,
              max_samples: 480000,
            },
            adapter: "audio",
            prompt: {
              prefix_token_ids: [256000],
              placeholder_token_id: 258881,
              suffix_token_ids: [258883],
            },
          },
        },
        output: "text",
      },
    },
  };
}

function compiledProgram() {
  return {
    schema: "mlc.compiled-program",
    schema_version: 1,
    interface_id: interfaceId,
    parameter_schema_id: parameterId,
    programs: {
      generation: {
        kind: "token_generation",
        exports: {
          embed_tokens: "embed",
          prefill_prompt: "prefill_prompt",
          decode_tokens: "decode_tokens",
          create_kv_cache: "create_tir_paged_kv_cache",
        },
        adapters: { audio: "audio_embed" },
      },
    },
    resources: {
      required_features: ["shader-f16"],
      max_storage_buffer_binding_size: 1024,
      estimated_device_memory_bytes: 2048,
    },
  };
}

test("strictly parses and resolves the model artifact pair", () => {
  const resolved = resolveChatCompletionArtifact(
    parseModelPackageManifest(modelPackage()),
    parseCompiledProgramArtifact(compiledProgram()),
  );
  expect(resolved.program.exports.prefill_prompt).toBe("prefill_prompt");
  expect(resolved.audioInput?.processor.sample_rate_hz).toBe(16000);
  expect(resolved.audioInput?.prompt.placeholder_token_id).toBe(258881);
});

test("rejects unknown fields and unsupported schema versions", () => {
  const manifest = { ...modelPackage(), unexpected: true };
  expect(() => parseModelPackageManifest(manifest)).toThrow(
    ArtifactManifestError,
  );

  const compiled = { ...compiledProgram(), schema_version: 2 };
  expect(() => parseCompiledProgramArtifact(compiled)).toThrow(
    /unsupported version 2/,
  );
});

test("rejects interface, parameter, and adapter mismatches", () => {
  const packageValue = parseModelPackageManifest(modelPackage());
  const wrongInterface = compiledProgram();
  wrongInterface.interface_id = `sha256:${"3".repeat(64)}`;
  expect(() =>
    resolveChatCompletionArtifact(
      packageValue,
      parseCompiledProgramArtifact(wrongInterface),
    ),
  ).toThrow(/interface IDs differ/);

  const missingAdapter = compiledProgram();
  (
    missingAdapter.programs.generation as {
      adapters: Record<string, string>;
    }
  ).adapters = {};
  expect(() =>
    resolveChatCompletionArtifact(
      packageValue,
      parseCompiledProgramArtifact(missingAdapter),
    ),
  ).toThrow(/missing adapter/);
});

test("applies only schema-defined defaults", () => {
  const manifest = modelPackage();
  delete (manifest as Partial<typeof manifest>).chat_config;
  delete (
    manifest.tasks["chat.completions"].inputs.audio.processor as Partial<
      (typeof manifest.tasks)["chat.completions"]["inputs"]["audio"]["processor"]
    >
  ).min_samples;
  const parsed = parseModelPackageManifest(manifest);
  expect(parsed.chat_config).toBe("mlc-chat-config.json");
  const processor = parsed.tasks["chat.completions"].inputs.audio.processor;
  expect(typeof processor === "string" ? -1 : processor.min_samples).toBe(1);
});

test("computes the same interface identity as MLC", async () => {
  const parsed = parseModelPackageManifest(modelPackage());
  await expect(computeInterfaceId(parsed.tasks)).resolves.toBe(
    "sha256:6453d39d6c1a05b41e3d10ac1547892e2fde2ae228c6705b122ffde5e4c9c490",
  );
});

test("only a missing sidecar selects legacy behavior", async () => {
  const fetchSpy = jest.spyOn(globalThis, "fetch");
  fetchSpy.mockResolvedValueOnce(new Response("", { status: 404 }));
  await expect(
    fetchOptionalModelPackageManifest("https://example.test/missing.json"),
  ).resolves.toBeUndefined();

  const valid = modelPackage();
  valid.interface_id = await computeInterfaceId(
    parseModelPackageManifest(valid).tasks,
  );
  fetchSpy.mockResolvedValueOnce(
    new Response(JSON.stringify(valid), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );
  await expect(
    fetchOptionalModelPackageManifest("https://example.test/valid.json"),
  ).resolves.toMatchObject({ interface_id: valid.interface_id });

  valid.tasks["chat.completions"].inputs.audio.prompt.placeholder_token_id += 1;
  fetchSpy.mockResolvedValueOnce(
    new Response(JSON.stringify(valid), { status: 200 }),
  );
  await expect(
    fetchOptionalModelPackageManifest("https://example.test/stale.json"),
  ).rejects.toThrow(/interface_id does not match its tasks/);
  fetchSpy.mockRestore();
});
