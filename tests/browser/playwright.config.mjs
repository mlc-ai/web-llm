import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.js",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  webServer: {
    command: "node server.mjs",
    port: 4178,
    reuseExistingServer: false,
  },
  use: {
    baseURL: "http://127.0.0.1:4178",
    browserName: "chromium",
    launchOptions: {
      executablePath: globalThis.process.env.WEBLLM_TEST_BROWSER_EXECUTABLE,
    },
  },
});
