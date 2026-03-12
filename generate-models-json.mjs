/**
 * Generates models list in json,,
 * Run from web-llm repo root (no build needed — reads src/config.ts directly):
 *   node generate-models-json.mjs
 *
 * Outputs: site/models-data.json
 */

import { readFileSync, writeFileSync } from "fs";

const src = readFileSync("./src/config.ts", "utf8");

const listStart = src.indexOf("model_list: [");
const listEnd = src.indexOf("\n  ],\n};");
const listSrc = src.slice(listStart, listEnd);

const entries = [];
const modelListContent = listSrc.substring(listSrc.indexOf("[") + 1);

let braceLevel = 0;
let currentBlockStart = -1;
const modelBlocks = [];

for (let i = 0; i < modelListContent.length; i++) {
  if (modelListContent[i] === "{") {
    if (braceLevel === 0) currentBlockStart = i;
    braceLevel++;
  } else if (modelListContent[i] === "}") {
    braceLevel--;
    if (braceLevel === 0 && currentBlockStart !== -1) {
      modelBlocks.push(modelListContent.slice(currentBlockStart, i + 1));
      currentBlockStart = -1;
    }
  }
}

for (const block of modelBlocks) {
  const modelMatch = block.match(/model:\s*"([^"]+)"/);
  const modelIdMatch = block.match(/model_id:\s*"([^"]+)"/);
  if (!modelMatch || !modelIdMatch) continue;

  const hf_url = modelMatch[1];
  const model_id = modelIdMatch[1];

  const vramMatch = block.match(/vram_required_MB:\s*([\d.]+)/);
  const lowMatch = block.match(/low_resource_required:\s*(true|false)/);
  const ctxMatch = block.match(/context_window_size:\s*(\d+)/);
  const typeMatch = block.match(/model_type:\s*ModelType\.(\w+)/);
  const featMatch = block.match(/required_features:\s*\[([^\]]+)\]/);

  const vramMB = vramMatch ? parseFloat(vramMatch[1]) : null;
  const ctx = ctxMatch ? parseInt(ctxMatch[1]) : null;

  entries.push({
    model_id,
    hf_url,
    vram: vramMB
      ? vramMB >= 1024
        ? `${(vramMB / 1024).toFixed(1)} GB`
        : `${Math.round(vramMB)} MB`
      : null,
    low_resource: lowMatch ? lowMatch[1] === "true" : false,
    context_window: ctx
      ? `${ctx >= 1024 ? (ctx / 1024).toFixed(0) : ctx}K`
      : null,
    model_type: typeMatch
      ? typeMatch[1] === "embedding"
        ? "embedding"
        : typeMatch[1] === "VLM"
          ? "vlm"
          : "llm"
      : "llm",
    required_features: featMatch
      ? featMatch[1]
          .split(",")
          .map((s) => s.trim().replace(/"/g, ""))
          .filter(Boolean)
      : [],
  });
}

console.log(`Parsed ${entries.length} model entries from src/config.ts`);

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
    match: (id) => id.startsWith("gemma"),
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
    label: "SmolLM2",
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
    console.warn(`  ⚠️  No family matched: ${entry.model_id}`);
  }
}

const families = [
  ...familyMap.values(),
  ...(other.models.length ? [other] : []),
].filter((f) => f.models.length > 0);

const output = {
  generated_at: new Date().toISOString(),
  total_models: entries.length,
  families,
};

writeFileSync("site/models-data.json", JSON.stringify(output, null, 2));
console.log(
  `✅ Wrote ${entries.length} models in ${families.length} families → site/models-data.json`,
);
