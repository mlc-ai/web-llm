// Slice-by-16 tables: table[k * 256 + b] is the CRC of byte b followed by k
// zero bytes, so sixteen table lookups advance the CRC by sixteen input bytes.
const table = new Uint32Array(16 * 256);
for (let i = 0; i < 256; i++) {
  let crc = i;
  for (let bit = 0; bit < 8; bit++) {
    crc = (crc & 1) === 1 ? (crc >>> 1) ^ 0x82f63b78 : crc >>> 1;
  }
  table[i] = crc >>> 0;
}
for (let i = 256; i < table.length; i++) {
  const previous = table[i - 256];
  table[i] = (previous >>> 8) ^ table[previous & 0xff];
}

// Input is read four bytes at a time through an Int32Array, which yields
// little-endian words only on a little-endian host.
const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/**
 * CRC32C shared by journal records and checkpoint payloads. Pass the CRC of
 * the bytes before `data` as `previous` to continue it.
 */
export function crc32c(data: Uint8Array, previous = 0): number {
  let crc = (previous ^ 0xffffffff) >>> 0;
  let i = 0;
  // Words start at the first 4-byte aligned byte.
  const head = (4 - data.byteOffset) & 3;
  if (littleEndian && data.length >= head + 16) {
    for (; i < head; i++) {
      crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
    }
    const words = new Int32Array(
      data.buffer,
      data.byteOffset + i,
      ((data.length - i) >>> 4) << 2,
    );
    for (let w = 0; w < words.length; w += 4) {
      const a = crc ^ words[w];
      const b = words[w + 1];
      const c = words[w + 2];
      const d = words[w + 3];
      crc =
        table[3840 + (a & 0xff)] ^
        table[3584 + ((a >>> 8) & 0xff)] ^
        table[3328 + ((a >>> 16) & 0xff)] ^
        table[3072 + (a >>> 24)] ^
        table[2816 + (b & 0xff)] ^
        table[2560 + ((b >>> 8) & 0xff)] ^
        table[2304 + ((b >>> 16) & 0xff)] ^
        table[2048 + (b >>> 24)] ^
        table[1792 + (c & 0xff)] ^
        table[1536 + ((c >>> 8) & 0xff)] ^
        table[1280 + ((c >>> 16) & 0xff)] ^
        table[1024 + (c >>> 24)] ^
        table[768 + (d & 0xff)] ^
        table[512 + ((d >>> 8) & 0xff)] ^
        table[256 + ((d >>> 16) & 0xff)] ^
        table[d >>> 24];
    }
    i += words.length * 4;
  }
  for (; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
