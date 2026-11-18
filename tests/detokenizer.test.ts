import { describe, expect, test } from "@jest/globals";
import { IncrementalDetokenizer } from "../src/detokenizer";

// The incremental text must equal a full decode after every token. The
// decoders below follow the HF tokenizers decoders WebLLM models use:
// byte-level BPE (Llama 3, Qwen; UTF-8 decoded lossily) and SentencePiece BPE
// with byte fallback (Llama 2, Mistral, Gemma: a run of byte tokens decodes as
// one group that becomes one U+FFFD per byte if it is not valid UTF-8, then
// "▁" becomes a space and the first leading space is stripped).

const utf8 = (text: string) => Array.from(new TextEncoder().encode(text));

// Byte b has id b; ids from 256 on are multi-byte pieces.
const byteLevelPieces = [
  " the",
  " hello",
  "é",
  "中",
  "文",
  "中文",
  "🎉",
  "👨‍👩",
  "❤️",
  " <",
  "st",
  "op",
  ">",
  "\n\n",
  "<think>",
  "</think>",
].map(utf8);
// Tokens that end inside a character: the first bytes of 🎉 and of 中.
byteLevelPieces.push(utf8("🎉").slice(0, 2), utf8("中").slice(0, 2));
const partialPartyId = 256 + byteLevelPieces.length - 2;
const partialZhongId = 256 + byteLevelPieces.length - 1;
const byteLevelId = (piece: string) =>
  256 + byteLevelPieces.findIndex((p) => p.join() === utf8(piece).join());

function byteLevelDecode(ids: Int32Array): string {
  const bytes = Array.from(ids).flatMap((id) =>
    id < 256 ? [id] : byteLevelPieces[id - 256],
  );
  return new TextDecoder().decode(Uint8Array.from(bytes));
}

// Ids 0-255 are the byte tokens <0x00>-<0xFF>; ids from 256 on are pieces.
const spmPieces = [
  "▁",
  "▁the",
  "▁hello",
  "é",
  "中",
  "文",
  "▁<",
  "st",
  "op",
  ">",
  "▁中",
];

function byteFallbackDecode(ids: Int32Array): string {
  let text = "";
  let run: number[] = [];
  const flush = () => {
    try {
      text += new TextDecoder("utf-8", { fatal: true }).decode(
        Uint8Array.from(run),
      );
    } catch {
      text += "�".repeat(run.length);
    }
    run = [];
  };
  for (const id of ids) {
    if (id < 256) {
      run.push(id);
      continue;
    }
    flush();
    text += spmPieces[id - 256].replaceAll("▁", " ");
  }
  flush();
  return text.startsWith(" ") ? text.slice(1) : text;
}

// Pushes ids one at a time and compares with a full decode after every token.
// unchangedLength must cover a prefix the previous and new text share.
function expectMatchesFullDecode(
  decode: (ids: Int32Array) => string,
  ids: number[],
) {
  const detokenizer = new IncrementalDetokenizer(decode);
  let previous = "";
  ids.forEach((id, i) => {
    const text = detokenizer.push(id);
    expect(text).toBe(decode(Int32Array.from(ids.slice(0, i + 1))));
    const unchanged = detokenizer.unchangedLength;
    expect(text.slice(0, unchanged)).toBe(previous.slice(0, unchanged));
    expect(unchanged).toBeLessThanOrEqual(previous.length);
    previous = text;
  });
}

// Seeded token sequences that draw from the given id ranges.
function randomSequences(ranges: Array<[number, number]>, count: number) {
  let seed = 20261007;
  const next = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  return Array.from({ length: count }, () =>
    Array.from({ length: 1 + Math.floor(next() * 48) }, () => {
      const [lo, hi] = ranges[Math.floor(next() * ranges.length)];
      return lo + Math.floor(next() * (hi - lo));
    }),
  );
}

describe("IncrementalDetokenizer, byte-level BPE", () => {
  test.each([
    ["multi-byte and CJK", [257, 258, 259, 262, 256]],
    ["emoji as bytes", utf8("a🎉b👨‍👩c❤️")],
    ["token ending inside a character", [partialPartyId, 0x8e, 0x89, 256]],
    ["CJK split across tokens", [partialZhongId, 0xad, byteLevelId("文")]],
    ["invalid bytes", [0x80, ...utf8("x"), 0xe4, ...utf8("y"), 0xe4]],
    [
      "think block",
      [
        byteLevelId("<think>"),
        byteLevelId("\n\n"),
        byteLevelId("</think>"),
        256,
      ],
    ],
  ])("%s", (_name, ids) => {
    expectMatchesFullDecode(byteLevelDecode, ids);
  });

  test("random sequences", () => {
    const ranges: Array<[number, number]> = [
      [0, 256],
      [0x80, 0xc0],
      [0xe0, 0xf5],
      [256, 256 + byteLevelPieces.length],
    ];
    for (const ids of randomSequences(ranges, 500)) {
      expectMatchesFullDecode(byteLevelDecode, ids);
    }
  });
});

describe("IncrementalDetokenizer, byte-fallback SentencePiece", () => {
  test.each([
    ["leading-space strip", [257, 258, 256, 257]],
    ["emoji as byte tokens", [257, ...utf8("🎉"), 258]],
    // An incomplete character turns the whole byte run into U+FFFD, including
    // characters before it that were already complete.
    ["pending character in a byte run", utf8("中文中")],
    ["invalid byte in a byte run", [...utf8("中"), 0x80, ...utf8("文"), 258]],
    ["CJK pieces", [266, 260, 261, 257]],
  ])("%s", (_name, ids) => {
    expectMatchesFullDecode(byteFallbackDecode, ids);
  });

  test("random sequences", () => {
    const ranges: Array<[number, number]> = [
      [0, 256],
      [0x80, 0xc0],
      [0xe0, 0xf5],
      [256, 256 + spmPieces.length],
    ];
    for (const ids of randomSequences(ranges, 500)) {
      expectMatchesFullDecode(byteFallbackDecode, ids);
    }
  });
});

// A token that rewrites text decoded before the window falls back to a full
// decode: here "b" merges with any number of preceding "a"s.
test("IncrementalDetokenizer falls back when a token rewrites earlier text", () => {
  const decode = (ids: Int32Array) =>
    Array.from(ids, (id) => "ab"[id])
      .join("")
      .replace(/a+b/g, "X");
  expectMatchesFullDecode(decode, [0, 0, 0, 0, 1, 0, 1, 1, 0, 0]);
});

test("IncrementalDetokenizer reset starts a new sequence", () => {
  const detokenizer = new IncrementalDetokenizer(byteFallbackDecode);
  detokenizer.push(257);
  detokenizer.reset();
  expect(detokenizer.text).toBe("");
  expect(detokenizer.push(258)).toBe("hello");
});
