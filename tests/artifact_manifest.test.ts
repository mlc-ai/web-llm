import {
  computeInterfaceId,
  loadModelPackageManifest,
  parseCompiledProgramArtifact,
  parseModelPackageManifest,
  resolveChatCompletionArtifact,
  resolveModelPackageResourceURLs,
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
      manifest: "tensor-cache.json",
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
  const adapters: Record<string, string> = { audio: "audio_embed" };
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
          prefill_tokens: "prefill_tokens",
          decode_tokens: "decode_tokens",
          create_kv_cache: "create_tir_paged_kv_cache",
        },
        adapters,
      },
    },
    resources: {
      required_features: ["shader-f16"],
      max_storage_buffer_binding_size: 1024,
      estimated_device_memory_bytes: 2048,
    },
  };
}

function imageInput() {
  return {
    processor: {
      kind: "image_decode",
      format: "rgb_u8",
      layout: "nhwc",
      resize: { mode: "center_crop", height: 336, width: 336 },
      num_embeddings: 576,
    },
    adapter: "image",
    prompt: {
      prefix_token_ids: [],
      placeholder_token_id: 32000,
      suffix_token_ids: [],
    },
  };
}

function imageModelPackage() {
  const manifest = modelPackage();
  const inputs = manifest.tasks["chat.completions"].inputs as Record<
    string,
    unknown
  >;
  delete inputs.audio;
  inputs.image = imageInput();
  return manifest;
}

function imageCompiledProgram() {
  const compiled = compiledProgram();
  compiled.programs.generation.adapters = { image: "image_embed" };
  return compiled;
}

test("strictly parses and resolves the model artifact pair", () => {
  const resolved = resolveChatCompletionArtifact(
    parseModelPackageManifest(modelPackage()),
    parseCompiledProgramArtifact(compiledProgram()),
  );
  expect(resolved.generation).toEqual({
    inputs: "tokens",
    prefill: "prefill_tokens",
    decode: "decode_tokens",
  });
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

test("parses and resolves an image input", () => {
  const resolved = resolveChatCompletionArtifact(
    parseModelPackageManifest(imageModelPackage()),
    parseCompiledProgramArtifact(imageCompiledProgram()),
  );
  expect(resolved.audioInput).toBeUndefined();
  expect(resolved.imageInput?.processor).toEqual(imageInput().processor);
  expect(resolved.imageInput?.adapter).toBe("image");
  expect(resolved.imageInput?.prompt.placeholder_token_id).toBe(32000);
});

test.each([
  [{ kind: "video_decode" }, /expected "audio_decode" or "image_decode"/],
  [{ format: "rgba_u8" }, /expected "rgb_u8"/],
  [{ layout: "nchw" }, /expected "nhwc"/],
  [{ num_embeddings: 0 }, /num_embeddings/],
  [{ padding: true }, /unknown field "padding"/],
  [{ resize: { mode: "pad", height: 1, width: 1 } }, /expected "stretch"/],
  [{ resize: { mode: "stretch", height: 0, width: 1 } }, /resize.height/],
  [{ resize: { mode: "stretch", height: 1 } }, /missing field "width"/],
])("rejects an invalid image processor %j", (override, message) => {
  const manifest = imageModelPackage();
  const inputs = manifest.tasks["chat.completions"].inputs as Record<
    string,
    Record<string, unknown>
  >;
  Object.assign(inputs.image.processor as Record<string, unknown>, override);
  expect(() => parseModelPackageManifest(manifest)).toThrow(message);
});

test("an image input needs an adapter, a prompt and a compiled adapter", () => {
  const noPrompt = imageModelPackage();
  const inputs = noPrompt.tasks["chat.completions"].inputs as Record<
    string,
    Record<string, unknown>
  >;
  delete inputs.image.prompt;
  expect(() => parseModelPackageManifest(noPrompt)).toThrow(
    /declared together/,
  );
  delete inputs.image.adapter;
  expect(() =>
    resolveChatCompletionArtifact(
      parseModelPackageManifest(noPrompt),
      parseCompiledProgramArtifact(imageCompiledProgram()),
    ),
  ).toThrow(/image input "image" has no adapter prompt/);

  const compiled = imageCompiledProgram();
  compiled.programs.generation.adapters = { audio: "audio_embed" };
  expect(() =>
    resolveChatCompletionArtifact(
      parseModelPackageManifest(imageModelPackage()),
      parseCompiledProgramArtifact(compiled),
    ),
  ).toThrow(/image input "image" references a missing adapter/);
});

test("adapter_dtypes picks the pixel tensor dtype for a declared adapter", () => {
  const resolve = (compiled: ReturnType<typeof compiledProgram>) =>
    resolveChatCompletionArtifact(
      parseModelPackageManifest(imageModelPackage()),
      parseCompiledProgramArtifact(compiled),
    );
  expect(resolve(imageCompiledProgram()).imageInput?.dtype).toBe("uint8");
  expect(resolve(imageCompiledProgram()).program.adapter_dtypes).toEqual({});

  const wide = imageCompiledProgram() as Record<string, any>;
  wide.programs.generation.adapter_dtypes = { image: "uint32" };
  expect(resolve(wide as any).imageInput?.dtype).toBe("uint32");

  const unknown = imageCompiledProgram() as Record<string, any>;
  unknown.programs.generation.adapter_dtypes = { audio: "uint32" };
  expect(() => parseCompiledProgramArtifact(unknown)).toThrow(
    /adapter_dtypes: "audio" is not an adapter/,
  );

  const wrong = imageCompiledProgram() as Record<string, any>;
  wrong.programs.generation.adapter_dtypes = { image: "float16" };
  expect(() => resolve(wrong as any)).toThrow(/must take uint8 or uint32/);
});

test("resolves audio and image inputs of the same task", () => {
  const manifest = imageModelPackage();
  const inputs = manifest.tasks["chat.completions"].inputs as Record<
    string,
    unknown
  >;
  inputs.audio = modelPackage().tasks["chat.completions"].inputs.audio;
  inputs.picture = imageInput();
  const compiled = imageCompiledProgram();
  compiled.programs.generation.adapters = {
    audio: "audio_embed",
    image: "image_embed",
  };
  expect(() =>
    resolveChatCompletionArtifact(
      parseModelPackageManifest(manifest),
      parseCompiledProgramArtifact(compiled),
    ),
  ).toThrow(/one image input per task/);
  delete inputs.picture;
  const resolved = resolveChatCompletionArtifact(
    parseModelPackageManifest(manifest),
    parseCompiledProgramArtifact(compiled),
  );
  expect(resolved.audioInput?.adapter).toBe("audio");
  expect(resolved.imageInput?.adapter).toBe("image");
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
  expect(
    typeof processor !== "string" && processor.kind === "audio_decode"
      ? processor.min_samples
      : -1,
  ).toBe(1);
});

test("resolves declared resources relative to the model artifact URL", () => {
  expect(
    resolveModelPackageResourceURLs(
      "https://models.example/gemma4/",
      parseModelPackageManifest(modelPackage()),
    ),
  ).toEqual({
    chatConfigUrl: "https://models.example/gemma4/mlc-chat-config.json",
    tensorCacheBaseUrl: "https://models.example/gemma4/",
    tensorCacheManifestUrl: "https://models.example/gemma4/tensor-cache.json",
  });

  expect(
    resolveModelPackageResourceURLs("https://models.example/legacy/"),
  ).toEqual({
    chatConfigUrl: "https://models.example/legacy/mlc-chat-config.json",
    tensorCacheBaseUrl: "https://models.example/legacy/",
    tensorCacheManifestUrl: "https://models.example/legacy/tensor-cache.json",
  });
});

test("rejects the obsolete ndarray cache filename", () => {
  const manifest = modelPackage();
  manifest.weights.manifest = "ndarray-cache.json" as "tensor-cache.json";
  expect(() => parseModelPackageManifest(manifest)).toThrow(
    /expected "tensor-cache.json"/,
  );
});

test("computes the same interface identity as MLC", async () => {
  const parsed = parseModelPackageManifest(modelPackage());
  await expect(computeInterfaceId(parsed.tasks)).resolves.toBe(
    "sha256:6453d39d6c1a05b41e3d10ac1547892e2fde2ae228c6705b122ffde5e4c9c490",
  );
});

test("loads a downloaded manifest and checks its interface_id", async () => {
  const encode = (value: unknown) =>
    new TextEncoder().encode(JSON.stringify(value)).buffer as ArrayBuffer;
  const valid = modelPackage();
  valid.interface_id = await computeInterfaceId(
    parseModelPackageManifest(valid).tasks,
  );
  await expect(loadModelPackageManifest(encode(valid))).resolves.toMatchObject({
    interface_id: valid.interface_id,
  });

  valid.tasks["chat.completions"].inputs.audio.prompt.placeholder_token_id += 1;
  await expect(loadModelPackageManifest(encode(valid))).rejects.toThrow(
    /interface_id does not match its tasks/,
  );
  await expect(
    loadModelPackageManifest(
      new TextEncoder().encode("{").buffer as ArrayBuffer,
    ),
  ).rejects.toThrow(/invalid JSON/);
});

test("resolves whichever pair of generation roles the library declares", () => {
  const compiled = compiledProgram();
  compiled.programs.generation.exports = {
    embed_tokens: "embed",
    prefill_embeds: "prefill",
    decode_embeds: "decode",
    create_kv_cache: "create_tir_paged_kv_cache",
  } as any;
  const resolved = resolveChatCompletionArtifact(
    parseModelPackageManifest(modelPackage()),
    parseCompiledProgramArtifact(compiled),
  );
  expect(resolved.generation).toEqual({
    inputs: "embeds",
    prefill: "prefill",
    decode: "decode",
  });
});

test("prefers the token roles when a library declares both pairs", () => {
  const compiled = compiledProgram();
  compiled.programs.generation.exports = {
    ...compiled.programs.generation.exports,
    prefill_embeds: "prefill",
    decode_embeds: "decode",
  } as any;
  const resolved = resolveChatCompletionArtifact(
    parseModelPackageManifest(modelPackage()),
    parseCompiledProgramArtifact(compiled),
  );
  expect(resolved.generation.inputs).toBe("tokens");
});

test.each([
  [{}, /complete pair/],
  [{ prefill_tokens: "prefill_tokens" }, /only one of prefill_tokens/],
  [
    { prefill_tokens: "prefill_tokens", decode_embeds: "decode" },
    /only one of prefill_tokens/,
  ],
  [
    {
      prefill_tokens: "prefill_tokens",
      decode_tokens: "decode_tokens",
      prefill_embeds: "prefill",
    },
    /only one of prefill_embeds/,
  ],
])("rejects generation roles without a complete pair", (roles, message) => {
  const compiled = compiledProgram();
  compiled.programs.generation.exports = {
    embed_tokens: "embed",
    create_kv_cache: "create_tir_paged_kv_cache",
    ...roles,
  } as any;
  expect(() =>
    resolveChatCompletionArtifact(
      parseModelPackageManifest(modelPackage()),
      parseCompiledProgramArtifact(compiled),
    ),
  ).toThrow(message);
});
