import { expect, test } from "@jest/globals";
import { crc32c } from "../src/resumable/crc32c";

test.each([
  ["", 0],
  ["123456789", 0xe3069283],
] as const)("CRC32C preserves the standard vector %s", (input, expected) => {
  expect(crc32c(new TextEncoder().encode(input))).toBe(expected);
});

test("CRC32C respects the byte offset and length of a payload view", () => {
  const bytes = new TextEncoder().encode("x123456789y");
  expect(crc32c(bytes.subarray(1, 10))).toBe(0xe3069283);
});
