import commonjs from "@rollup/plugin-commonjs";
import { nodeResolve } from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";
import ignore from "rollup-plugin-ignore";
import { resolve } from "node:path";

const runtimePath = globalThis.process.env.WEBLLM_TEST_RUNTIME_PATH;

function stubNodePerformanceImport() {
  return {
    name: "stub-node-performance-import",
    renderChunk(code) {
      return code.replace(
        /import (require\$\$\d+) from '(?:perf_hooks|ws)';/g,
        'const $1 = "MLC_DUMMY_REQUIRE_VAR";',
      );
    },
  };
}

export default {
  input: "tests/browser/harness.js",
  output: {
    file: ".browser-test/harness.js",
    format: "es",
    sourcemap: false,
  },
  plugins: [
    {
      name: "local-tvm-runtime",
      resolveId(source) {
        if (runtimePath && source === "@mlc-ai/web-runtime") {
          return resolve(runtimePath, "lib/index.js");
        }
        return null;
      },
    },
    ignore(["fs", "path", "crypto", "node:fs", "node:path", "node:crypto"]),
    nodeResolve({ browser: true }),
    commonjs({ ignoreDynamicRequires: true }),
    typescript({
      tsconfig: "./tsconfig.json",
      compilerOptions: {
        declaration: false,
        declarationMap: false,
        sourceMap: false,
        outDir: ".browser-test",
      },
    }),
    stubNodePerformanceImport(),
  ],
};
