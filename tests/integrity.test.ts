import { verifyIntegrity, isValidSRI } from "../src/integrity";
import { IntegrityError } from "../src/error";

// Helper: compute an SRI hash for a given string
async function computeSRI(
  data: string,
  algorithm: "SHA-256" | "SHA-384" | "SHA-512" = "SHA-256",
): Promise<string> {
  const encoder = new TextEncoder();
  const buffer = encoder.encode(data);
  const hashBuffer = await crypto.subtle.digest(algorithm, buffer);
  const hashArray = new Uint8Array(hashBuffer);
  let binary = "";
  for (let i = 0; i < hashArray.length; i++) {
    binary += String.fromCharCode(hashArray[i]);
  }
  const base64 = btoa(binary);
  const algoPrefix = algorithm.replace("-", "").toLowerCase();
  return `${algoPrefix}-${base64}`;
}

describe("isValidSRI", () => {
  test("accepts valid sha256 SRI", () => {
    expect(
      isValidSRI("sha256-MV9b23bQeMQ7isAGTkoBZGErH853yGk0W/yUx1iU7dM="),
    ).toBe(true);
  });

  test("accepts valid sha384 SRI", () => {
    expect(
      isValidSRI(
        "sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC",
      ),
    ).toBe(true);
  });

  test("accepts valid sha512 SRI", () => {
    expect(
      isValidSRI(
        "sha512-wVJ82JPBJHc9gRkRlwyP5uhX1t9dySJr2KFgYUwM2WOk3eorlLt9NgIe+dhl1c2goJO2nQE1hOwRs0AN/Y30Q==",
      ),
    ).toBe(true);
  });

  test("rejects invalid algorithm", () => {
    expect(isValidSRI("sha1-abc123")).toBe(false);
    expect(isValidSRI("md5-abc123")).toBe(false);
  });

  test("rejects missing algorithm prefix", () => {
    expect(isValidSRI("abc123")).toBe(false);
  });

  test("rejects empty string", () => {
    expect(isValidSRI("")).toBe(false);
  });

  test("rejects malformed SRI", () => {
    expect(isValidSRI("sha256-")).toBe(false);
    expect(isValidSRI("sha256-!!!invalid!!!")).toBe(false);
  });

  test("rejects excessive base64 padding", () => {
    // Valid base64 has at most 2 padding characters
    expect(isValidSRI("sha256-abc===")).toBe(false);
    expect(isValidSRI("sha256-abc====")).toBe(false);
  });
});

describe("verifyIntegrity", () => {
  const testData = "Hello, WebLLM!";
  const testBuffer = new TextEncoder().encode(testData).buffer;

  test("passes with correct SHA-256 hash", async () => {
    const sri = await computeSRI(testData, "SHA-256");
    await expect(
      verifyIntegrity(testBuffer, sri, "https://example.com/test.json"),
    ).resolves.toBeUndefined();
  });

  test("passes with correct SHA-384 hash", async () => {
    const sri = await computeSRI(testData, "SHA-384");
    await expect(
      verifyIntegrity(testBuffer, sri, "https://example.com/test.json"),
    ).resolves.toBeUndefined();
  });

  test("passes with correct SHA-512 hash", async () => {
    const sri = await computeSRI(testData, "SHA-512");
    await expect(
      verifyIntegrity(testBuffer, sri, "https://example.com/test.json"),
    ).resolves.toBeUndefined();
  });

  // Test with a well-known hardcoded hash to ensure the implementation
  // produces correct results, not just self-consistent ones.
  // SHA-256 of empty string = 47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=
  test("produces correct SHA-256 for known empty-string hash", async () => {
    const emptyBuffer = new ArrayBuffer(0);
    const knownEmptySHA256 =
      "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=";
    await expect(
      verifyIntegrity(emptyBuffer, knownEmptySHA256, "https://example.com/e"),
    ).resolves.toBeUndefined();
  });

  // SHA-256 of "abc" = ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=
  test("produces correct SHA-256 for known 'abc' hash", async () => {
    const abcBuffer = new TextEncoder().encode("abc").buffer;
    const knownAbcSHA256 =
      "sha256-ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=";
    await expect(
      verifyIntegrity(abcBuffer, knownAbcSHA256, "https://example.com/abc"),
    ).resolves.toBeUndefined();
  });

  test("throws IntegrityError on hash mismatch", async () => {
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    await expect(
      verifyIntegrity(
        testBuffer,
        wrongSRI,
        "https://example.com/config.json",
        "error",
      ),
    ).rejects.toThrow(IntegrityError);
  });

  test("IntegrityError contains expected and actual hashes", async () => {
    expect.assertions(5);
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    try {
      await verifyIntegrity(
        testBuffer,
        wrongSRI,
        "https://example.com/config.json",
        "error",
      );
    } catch (err) {
      expect(err).toBeInstanceOf(IntegrityError);
      const integrityErr = err as IntegrityError;
      expect(integrityErr.url).toBe("https://example.com/config.json");
      expect(integrityErr.expected).toBe(wrongSRI);
      expect(integrityErr.actual).toMatch(/^sha256-/);
      expect(integrityErr.message).toContain("Integrity verification failed");
    }
  });

  test("IntegrityError has correct name property", async () => {
    expect.assertions(1);
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    try {
      await verifyIntegrity(
        testBuffer,
        wrongSRI,
        "https://example.com/file",
        "error",
      );
    } catch (err) {
      expect((err as Error).name).toBe("IntegrityError");
    }
  });

  test("default onFailure throws on mismatch", async () => {
    // When onFailure is not specified, it should default to "error" and throw
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    await expect(
      verifyIntegrity(testBuffer, wrongSRI, "https://example.com/file"),
    ).rejects.toThrow(IntegrityError);
  });

  test("warns instead of throwing when onFailure is 'warn'", async () => {
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

    const logModule = await import("loglevel");
    const logWarnSpy = jest
      .spyOn(logModule.default, "warn")
      .mockImplementation(() => {});

    await expect(
      verifyIntegrity(
        testBuffer,
        wrongSRI,
        "https://example.com/model.wasm",
        "warn",
      ),
    ).resolves.toBeUndefined();

    expect(logWarnSpy).toHaveBeenCalledWith(
      expect.stringContaining("Integrity check failed"),
    );

    logWarnSpy.mockRestore();
  });

  test("warn message includes URL and both hashes", async () => {
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

    const logModule = await import("loglevel");
    const logWarnSpy = jest
      .spyOn(logModule.default, "warn")
      .mockImplementation(() => {});

    await verifyIntegrity(
      testBuffer,
      wrongSRI,
      "https://example.com/model.wasm",
      "warn",
    );

    const warnMessage = logWarnSpy.mock.calls[0][0] as string;
    expect(warnMessage).toContain("https://example.com/model.wasm");
    expect(warnMessage).toContain(wrongSRI);
    expect(warnMessage).toMatch(/sha256-[A-Za-z0-9+/]+=*/); // actual hash

    logWarnSpy.mockRestore();
  });

  test("throws on invalid SRI format", async () => {
    await expect(
      verifyIntegrity(testBuffer, "invalid-format", "https://example.com/file"),
    ).rejects.toThrow("Invalid SRI hash format");
  });

  test("throws on sha1 (unsupported algorithm)", async () => {
    await expect(
      verifyIntegrity(testBuffer, "sha1-abc123", "https://example.com/file"),
    ).rejects.toThrow("Invalid SRI hash format");
  });

  test("handles empty ArrayBuffer", async () => {
    const emptyBuffer = new ArrayBuffer(0);
    const sri = await computeSRI("", "SHA-256");
    await expect(
      verifyIntegrity(emptyBuffer, sri, "https://example.com/empty"),
    ).resolves.toBeUndefined();
  });

  test("handles large data (1MB)", async () => {
    const largeData = new Uint8Array(1024 * 1024);
    // Fill with deterministic data
    for (let i = 0; i < largeData.length; i++) {
      largeData[i] = i % 256;
    }
    const hashBuffer = await crypto.subtle.digest("SHA-256", largeData);
    const hashArray = new Uint8Array(hashBuffer);
    let binary = "";
    for (let i = 0; i < hashArray.length; i++) {
      binary += String.fromCharCode(hashArray[i]);
    }
    const sri = `sha256-${btoa(binary)}`;

    await expect(
      verifyIntegrity(largeData.buffer, sri, "https://example.com/large.bin"),
    ).resolves.toBeUndefined();
  });

  test("rejects when data is modified after hash computation", async () => {
    // Compute hash of original data, then modify data and verify it fails
    const originalData = new Uint8Array([1, 2, 3, 4, 5]);
    const sri = await computeSRI(
      String.fromCharCode(...originalData),
      "SHA-256",
    );

    // Modify the data
    const modifiedData = new Uint8Array([1, 2, 3, 4, 6]); // changed last byte
    await expect(
      verifyIntegrity(
        modifiedData.buffer,
        sri,
        "https://example.com/tampered",
        "error",
      ),
    ).rejects.toThrow(IntegrityError);
  });

  test("different data produces different actual hashes in error", async () => {
    expect.assertions(2);
    const wrongSRI = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";

    const data1 = new TextEncoder().encode("data1").buffer;
    const data2 = new TextEncoder().encode("data2").buffer;

    let actual1 = "";
    let actual2 = "";

    try {
      await verifyIntegrity(data1, wrongSRI, "https://example.com/1", "error");
    } catch (err) {
      actual1 = (err as IntegrityError).actual;
    }

    try {
      await verifyIntegrity(data2, wrongSRI, "https://example.com/2", "error");
    } catch (err) {
      actual2 = (err as IntegrityError).actual;
    }

    // The actual hashes should be different because the data is different
    expect(actual1).not.toBe(actual2);
    // Both should be valid sha256 SRIs
    expect(isValidSRI(actual1) && isValidSRI(actual2)).toBe(true);
  });
});
