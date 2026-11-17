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

test("CRC32C slice-by-8 matches the bytewise definition at every length and alignment", () => {
  const reference = (data: Uint8Array) => {
    let crc = 0xffffffff;
    for (const value of data) {
      crc ^= value;
      for (let bit = 0; bit < 8; bit++) {
        crc = crc & 1 ? (crc >>> 1) ^ 0x82f63b78 : crc >>> 1;
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  };
  const bytes = Uint8Array.from(
    { length: 80 },
    (_, i) => (i * 167 + 13) & 0xff,
  );
  for (let offset = 0; offset < 8; offset++) {
    for (let length = 0; offset + length <= bytes.length; length++) {
      const view = bytes.subarray(offset, offset + length);
      expect(crc32c(view)).toBe(reference(view));
    }
  }
});

// Guards streamed checkpoint writes, which checksum each page group batch by
// batch: a wrong continuation would make every streamed checkpoint fail its
// restore CRC check.
test("CRC32C continues across split payloads", () => {
  const bytes = new TextEncoder().encode("123456789");
  for (let split = 0; split <= bytes.length; split++) {
    const head = crc32c(bytes.subarray(0, split));
    expect(crc32c(bytes.subarray(split), head)).toBe(0xe3069283);
  }
});
