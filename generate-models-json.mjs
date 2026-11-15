import { writeFileSync } from "node:fs";
import { ModelType, prebuiltAppConfig } from "./lib/index.js";

// Run after building WebLLM so the page uses the same registry as the package.
const entries = prebuiltAppConfig.model_list.map((model) => {
  const vram = model.vram_required_MB;
  const context = model.overrides?.context_window_size;
  return {
    model_id: model.model_id,
    hf_url: model.model,
    vram:
      vram == null
        ? null
        : vram >= 1024
          ? `${(vram / 1024).toFixed(1)} GB`
          : `${Math.round(vram)} MB`,
    low_resource: model.low_resource_required ?? false,
    context_window:
      context > 0
        ? context % 1024 === 0
          ? `${context / 1024}K`
          : `${context}`
        : null,
    model_type:
      model.model_type === ModelType.embedding
        ? "embedding"
        : model.model_type === ModelType.VLM
          ? "vlm"
          : "llm",
    required_features: model.required_features ?? [],
  };
});

const FAMILIES = [
  {
    id: "deepseek",
    label: "DeepSeek",
    emoji: "🧠",
    match: (id) => id.startsWith("DeepSeek"),
  },
  {
    id: "llama3",
    label: "Llama 3",
    emoji: "🦙",
    match: (id) => /^(Llama-3|Hermes-[23]|Hermes-2-Theta)/.test(id),
  },
  {
    id: "llama2",
    label: "Llama 2",
    emoji: "🦙",
    match: (id) => id.startsWith("Llama-2"),
  },
  {
    id: "qwen3",
    label: "Qwen3",
    emoji: "🌊",
    match: (id) => id.startsWith("Qwen3"),
  },
  {
    id: "qwen25",
    label: "Qwen 2.5",
    emoji: "🌊",
    match: (id) => id.startsWith("Qwen2.5"),
  },
  {
    id: "qwen2",
    label: "Qwen 2",
    emoji: "🌊",
    match: (id) => id.startsWith("Qwen2-") || id.startsWith("Qwen2."),
  },
  { id: "phi", label: "Phi", emoji: "🔷", match: (id) => /^[Pp]hi/.test(id) },
  {
    id: "gemma",
    label: "Gemma",
    emoji: "💎",
    match: (id) => /^gemma/i.test(id),
  },
  {
    id: "mistral",
    label: "Mistral & Friends",
    emoji: "💨",
    match: (id) =>
      ["Mistral", "OpenHermes", "NeuralHermes", "WizardMath", "Ministral"].some(
        (p) => id.startsWith(p),
      ),
  },
  {
    id: "smollm",
    label: "SmolLM",
    emoji: "🔬",
    match: (id) => id.startsWith("SmolLM"),
  },
  {
    id: "stablelm",
    label: "StableLM",
    emoji: "🔩",
    match: (id) => id.startsWith("stablelm"),
  },
  {
    id: "tinyllama",
    label: "TinyLlama",
    emoji: "🐣",
    match: (id) => id.startsWith("TinyLlama"),
  },
  {
    id: "redpajama",
    label: "RedPajama",
    emoji: "🟥",
    match: (id) => id.startsWith("RedPajama"),
  },
  {
    id: "embedding",
    label: "Embedding",
    emoji: "📐",
    match: (id) => id.startsWith("snowflake"),
  },
];

const familyMap = new Map(FAMILIES.map((f) => [f.id, { ...f, models: [] }]));
const other = { id: "other", label: "Other", emoji: "📦", models: [] };

for (const entry of entries) {
  const family = FAMILIES.find((f) => f.match(entry.model_id));
  if (family) {
    familyMap.get(family.id).models.push(entry);
  } else {
    other.models.push(entry);
  }
}

const families = [
  ...familyMap.values(),
  ...(other.models.length ? [other] : []),
].filter((f) => f.models.length > 0);

const output = { total_models: entries.length, families };
writeFileSync(
  new URL("./site/models-data.json", import.meta.url),
  JSON.stringify(output, null, 2) + "\n",
);
console.log(`Wrote ${entries.length} models to site/models-data.json`);
