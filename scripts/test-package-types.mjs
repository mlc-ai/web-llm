import { execFileSync } from "node:child_process";
import { log } from "node:console";
import { copyFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const consumer = mkdtempSync(join(tmpdir(), "webllm-package-types-"));
const npm = process.env.npm_execpath;
if (!npm) {
  throw new Error("Run this check with npm run test:package-types.");
}

log(`Checking the published package in ${consumer}`);
const [{ filename }] = JSON.parse(
  execFileSync(
    process.execPath,
    [npm, "pack", "--ignore-scripts", "--json", "--pack-destination", consumer],
    { cwd: root, encoding: "utf8" },
  ),
);
writeFileSync(
  join(consumer, "package.json"),
  JSON.stringify({
    private: true,
    type: "module",
    dependencies: { "@mlc-ai/web-llm": `file:./${filename}` },
  }),
);
execFileSync(
  process.execPath,
  [npm, "install", "--omit=dev", "--ignore-scripts", "--no-audit", "--no-fund"],
  { cwd: consumer, stdio: "inherit" },
);
copyFileSync(
  join(root, "tests/package-types/consumer.ts"),
  join(consumer, "consumer.ts"),
);
writeFileSync(
  join(consumer, "tsconfig.json"),
  JSON.stringify({
    compilerOptions: {
      target: "ES2022",
      module: "ESNext",
      moduleResolution: "Bundler",
      lib: ["ES2022", "DOM", "DOM.Iterable"],
      types: [],
      strict: true,
      skipLibCheck: false,
      noEmit: true,
    },
    files: ["consumer.ts"],
  }),
);
execFileSync(
  process.execPath,
  [join(root, "node_modules/typescript/bin/tsc"), "-p", consumer],
  { cwd: consumer, stdio: "inherit" },
);
