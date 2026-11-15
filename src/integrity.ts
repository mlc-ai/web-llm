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

type SRIAlgorithm = "sha256" | "sha384" | "sha512";

const SRI_REGEX = /^(sha256|sha384|sha512)-([A-Za-z0-9+/]+={0,2})$/;

const SRI_HASH_BYTE_LENGTH: Record<SRIAlgorithm, number> = {
  sha256: 32,
  sha384: 48,
  sha512: 64,
};

const ALGO_MAP: Record<SRIAlgorithm, string> = {
  sha256: "SHA-256",
  sha384: "SHA-384",
  sha512: "SHA-512",
};

function getDecodedBase64ByteLength(base64: string): number | null {
  const length = base64.length;
  const remainder = length % 4;
  if (remainder === 1) {
    return null;
  }

  const paddingMatch = base64.match(/=+$/);
  const padding = paddingMatch ? paddingMatch[0].length : 0;
  if (padding > 2) {
    return null;
  }
  if (padding > 0 && remainder !== 0) {
    return null;
  }

  const fullQuartets = Math.floor(length / 4);
  let decodedLength = fullQuartets * 3;

  if (padding > 0) {
    decodedLength -= padding;
  } else if (remainder === 2) {
    decodedLength += 1;
  } else if (remainder === 3) {
    decodedLength += 2;
  }

  return decodedLength;
}

function parseSRI(sri: string): { algo: SRIAlgorithm; hash: string } | null {
  const match = sri.match(SRI_REGEX);
  if (!match) {
    return null;
  }

  const algo = match[1] as SRIAlgorithm;
  const hash = match[2];
  const decodedLength = getDecodedBase64ByteLength(hash);
  if (decodedLength !== SRI_HASH_BYTE_LENGTH[algo]) {
    return null;
  }

  return { algo, hash };
}

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
  const parsed = parseSRI(expectedSRI);
  if (!parsed) {
    throw new Error(
      `Invalid SRI hash format: "${expectedSRI}". ` +
        `Expected format: "sha256-BASE64", "sha384-BASE64", or "sha512-BASE64".`,
    );
  }

  const { algo, hash: expectedHash } = parsed;
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
  return parseSRI(sri) !== null;
}
