import log from "loglevel";
import { IntegrityError } from "./error";

/**
 * SRI (Subresource Integrity) hash string.
 * Format: "sha256-BASE64", "sha384-BASE64", or "sha512-BASE64".
 * See https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity
 */
export type SRIString = string;

/** Map of filename to SRI hash for per-file verification. */
export type FileIntegrityMap = Record<string, SRIString>;

/**
 * Integrity configuration for a model's artifacts.
 * All fields are optional — only specified artifacts will be verified.
 *
 * @param config SRI hash for the model's `mlc-chat-config.json`.
 * @param model_lib SRI hash for the WASM model library file.
 * @param tokenizer SRI hashes for tokenizer files, keyed by filename
 *   (e.g. `"tokenizer.json"` or `"tokenizer.model"`).
 * @param onFailure Behavior on verification failure:
 *   `"error"` (default) throws an `IntegrityError`;
 *   `"warn"` logs a warning and continues.
 */
export interface ModelIntegrity {
  config?: SRIString;
  model_lib?: SRIString;
  tokenizer?: FileIntegrityMap;
  onFailure?: "error" | "warn";
}

const SRI_REGEX = /^(sha256|sha384|sha512)-([A-Za-z0-9+/]+={0,2})$/;

const ALGO_MAP: Record<string, string> = {
  sha256: "SHA-256",
  sha384: "SHA-384",
  sha512: "SHA-512",
};

/**
 * Verify an ArrayBuffer against an SRI hash using the Web Crypto API.
 *
 * @param data The raw bytes to verify.
 * @param expectedSRI The expected SRI hash (e.g. `"sha256-abc123..."`).
 * @param url The URL of the artifact, used for error messages.
 * @param onFailure `"error"` to throw on mismatch, `"warn"` to log and continue.
 * @throws {IntegrityError} When the hash does not match and `onFailure` is `"error"`.
 */
export async function verifyIntegrity(
  data: ArrayBuffer,
  expectedSRI: SRIString,
  url: string,
  onFailure: "error" | "warn" = "error",
): Promise<void> {
  const match = expectedSRI.match(SRI_REGEX);
  if (!match) {
    throw new Error(
      `Invalid SRI hash format: "${expectedSRI}". ` +
        `Expected format: "sha256-BASE64", "sha384-BASE64", or "sha512-BASE64".`,
    );
  }

  const [, algo, expectedHash] = match;
  const hashBuffer = await crypto.subtle.digest(ALGO_MAP[algo], data);
  const hashArray = new Uint8Array(hashBuffer);

  // Convert to base64
  let binary = "";
  for (let i = 0; i < hashArray.length; i++) {
    binary += String.fromCharCode(hashArray[i]);
  }
  const actualHash = btoa(binary);

  if (actualHash !== expectedHash) {
    const actualSRI = `${algo}-${actualHash}`;
    if (onFailure === "warn") {
      log.warn(
        `Integrity check failed for ${url}. ` +
          `Expected: ${expectedSRI}, Got: ${actualSRI}`,
      );
      return;
    }
    throw new IntegrityError(url, expectedSRI, actualSRI);
  }
}

/**
 * Validate that a string is a well-formed SRI hash.
 *
 * @param sri The string to validate.
 * @returns `true` if `sri` matches the SRI format.
 */
export function isValidSRI(sri: string): boolean {
  return SRI_REGEX.test(sri);
}
