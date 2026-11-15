import path from "node:path";
import { defineConfig } from "vite";

const wsStub = path.resolve(__dirname, "src/ws-browser-stub.ts");

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  resolve: {
    alias: {
      ws: wsStub,
    },
  },
  optimizeDeps: {
    exclude: ["@mlc-ai/web-llm"],
    esbuildOptions: {
      alias: {
        ws: wsStub,
      },
    },
  },
  worker: {
    format: "es",
  },
});
