import { test as base, expect } from "@playwright/test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// A fresh profile gives every test cold storage while preserving it across page
// crashes. Closing the browser also avoids macOS Chromium's incognito-context
// teardown crash, which reproduces with blank pages and no WebLLM loaded.
const test = base.extend({
  browserProfile: async ({ browserName }, use) => {
    const profile = await mkdtemp(
      join(
        globalThis.process.env.WEBLLM_TEST_PROFILE_ROOT ?? tmpdir(),
        `webllm-${browserName}-`,
      ),
    );
    try {
      await use(profile);
    } finally {
      await rm(profile, { recursive: true, force: true });
    }
  },
  // A crash test can relaunch the same profile after its browser process exits.
  launchTestContext: async (
    {
      playwright,
      browserName,
      launchOptions,
      headless,
      baseURL,
      browserProfile,
    },
    use,
  ) => {
    const contexts = [];
    const launch = async () => {
      const context = await playwright[browserName].launchPersistentContext(
        browserProfile,
        {
          ...launchOptions,
          headless,
          baseURL,
          ignoreDefaultArgs: ["--no-startup-window"],
        },
      );
      contexts.push(context);
      return context;
    };
    try {
      await use(launch);
    } finally {
      for (const context of contexts) await context.close();
    }
  },
  context: async ({ launchTestContext }, use) => {
    await use(await launchTestContext());
  },
});

export { test, expect };
