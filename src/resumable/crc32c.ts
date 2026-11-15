const table = new Uint32Array(256);
for (let i = 0; i < table.length; i++) {
  let crc = i;
  for (let bit = 0; bit < 8; bit++) {
    crc = (crc & 1) === 1 ? (crc >>> 1) ^ 0x82f63b78 : crc >>> 1;
  }
  table[i] = crc >>> 0;
}

/** CRC32C shared by journal records and checkpoint payloads. */
export function crc32c(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const value of data) {
    crc = table[(crc ^ value) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
