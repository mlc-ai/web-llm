import { mkdirSync, writeFileSync } from "node:fs";
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

const families = [
  ["deepseek", "DeepSeek", /^DeepSeek/],
  ["llama3", "Llama 3", /^(Llama-3|Hermes-[23]|Hermes-2-Theta)/],
  ["llama2", "Llama 2", /^Llama-2/],
  ["qwen3", "Qwen3", /^Qwen3/],
  ["qwen25", "Qwen 2.5", /^Qwen2\.5/],
  ["qwen2", "Qwen 2", /^Qwen2[-.]/],
  ["phi", "Phi", /^[Pp]hi/],
  ["gemma", "Gemma", /^gemma/i],
  [
    "mistral",
    "Mistral",
    /^(Mistral|OpenHermes|NeuralHermes|WizardMath|Ministral)/,
  ],
  ["smollm", "SmolLM", /^SmolLM/],
  ["stablelm", "StableLM", /^stablelm/],
  ["tinyllama", "TinyLlama", /^TinyLlama/],
  ["redpajama", "RedPajama", /^RedPajama/],
  ["embedding", "Embedding", /^snowflake/],
  ["other", "Other", /.*/],
].map(([id, label, pattern]) => ({ id, label, pattern, models: [] }));

for (const entry of entries) {
  const family = families.find((family) => family.pattern.test(entry.model_id));
  family.models.push(entry);
}

const output = {
  total_models: entries.length,
  families: families
    .filter((family) => family.models.length > 0)
    .map(({ id, label, models }) => ({ id, label, models })),
};
mkdirSync(new URL("./site/_data/", import.meta.url), { recursive: true });
writeFileSync(
  new URL("./site/_data/models.json", import.meta.url),
  JSON.stringify(output, null, 2) + "\n",
);
console.log(`Wrote ${entries.length} models to site/_data/models.json`);
