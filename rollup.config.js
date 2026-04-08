import { nodeResolve } from "@rollup/plugin-node-resolve";
import ignore from "rollup-plugin-ignore";
import commonjs from "@rollup/plugin-commonjs";
import typescript from "@rollup/plugin-typescript";

export default {
  input: "src/index.ts",
  output: [
    {
      file: "lib/index.js",
      exports: "named",
      format: "es",
      sourcemap: true,
      globals: { ws: "ws", perf_hooks: "perf_hooks" },
    },
  ],
  plugins: [
    ignore(["fs", "path", "crypto", "node:fs", "node:path", "node:crypto"]),
    nodeResolve({ browser: true }),
    commonjs({
      ignoreDynamicRequires: true,
    }),
    typescript({
      tsconfig: "./tsconfig.json",
    }),
  ],
};
