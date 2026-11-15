import { test, expect } from "./fixtures.mjs";

for (const operation of ["write", "append"]) {
  test(`failed OPFS ${operation} aborts without committing partial data`, async ({
    page,
  }) => {
    await page.goto("/");
    await page.waitForFunction(
      () => globalThis.webllmBrowserHarness !== undefined,
    );
    const result = await page.evaluate(async (operation) => {
      const { BrowserOPFSFileStore } = globalThis.webllmBrowserHarness;
      const files = new BrowserOPFSFileStore();
      const encode = (value) => new globalThis.TextEncoder().encode(value);
      await files.write("partial.bin", encode("committed"));
      const prototype = globalThis.FileSystemWritableFileStream.prototype;
      const write = prototype.write;
      const failure = new globalThis.DOMException(
        "injected after staging bytes",
        "QuotaExceededError",
      );
      prototype.write = async function (...args) {
        await write.apply(this, args);
        throw failure;
      };
      let originalError = false;
      try {
        await files[operation]("partial.bin", encode("partial"));
      } catch (err) {
        originalError = err === failure;
      } finally {
        prototype.write = write;
      }
      return {
        originalError,
        saved: new globalThis.TextDecoder().decode(
          await files.read("partial.bin"),
        ),
      };
    }, operation);
    expect(result).toEqual({ originalError: true, saved: "committed" });
  });
}
