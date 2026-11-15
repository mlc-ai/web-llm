import {
  sampleTalkerToken,
  TalkerSamplingOptions,
} from "../src/artifact_graph_talker_sampling";

// Independent Transformers 5.17.0 repetition/temperature/top-k/top-p processor outputs.
// Inputs use integer-derived scores; draws use inverse CDF in vocabulary order.
const expected = [
  4, 1, 22, 4, 10, 22, 7, 20, 22, 19, 16, 21, 20, 20, 18, 17, 16, 8294, 8292,
  22, 6, 8292, 8296, 16,
];
const defaults: TalkerSamplingOptions = {
  temperature: 1,
  topK: 0,
  topP: 1,
  repetitionPenalty: 1,
  random: () => 0,
};
const sparse = (values: Record<number, number>) => {
  const result = new Float32Array(8448).fill(-1000);
  for (const [id, value] of Object.entries(values)) result[Number(id)] = value;
  return result;
};

describe("Talker sampling", () => {
  test.each(expected.map((token, index) => ({ token, index })))(
    "matches HF case $index",
    ({ token, index: i }) => {
      const ids = [
        ...Array.from({ length: 24 }, (_, j) => j),
        8292,
        8293,
        8294,
        8296,
      ];
      const logits = new Float32Array(8448).fill(-1000);
      ids.forEach((id, j) => {
        logits[id] = ((((i + 3) * (j + 7) * 17) % 113) - 56) / 7;
      });
      const before = new Float32Array(logits);
      expect(
        sampleTalkerToken(
          logits,
          [8296, 8296, 8292, 8293, i % 24, (i + 3) % 24],
          {
            temperature: [0.9, 1, 0.5][i % 3],
            topK: [0, 7, 12][i % 3],
            topP: [0.8, 1, 0.4][i % 3],
            repetitionPenalty: [1.05, 0.8, 1.2][i % 3],
            random: () => (i + 0.5) / 24,
          },
        ),
      ).toBe(token);
      expect(logits).toEqual(before);
    },
  );
  test("suppresses BOS but permits both natural stop IDs", () => {
    const logits = sparse({ 8293: 100, 8292: 2, 8294: 2 });
    expect(sampleTalkerToken(logits, [], defaults)).toBe(8292);
    expect(
      sampleTalkerToken(logits, [], { ...defaults, random: () => 0.999999 }),
    ).toBe(8294);
  });
  test("suppresses BOS before temperature overflow can occur", () => {
    expect(
      sampleTalkerToken(sparse({ 0: 1, 8293: 3e38 }), [], {
        ...defaults,
        temperature: 0.5,
        topK: 1,
      }),
    ).toBe(0);
  });
  test("top-k keeps boundary ties and top-p retains at least one token", () => {
    const logits = sparse({ 0: 2, 1: 2, 2: 1 });
    expect(
      sampleTalkerToken(logits, [], {
        ...defaults,
        topK: 1,
        random: () => 0.99,
      }),
    ).toBe(1);
    expect(
      sampleTalkerToken(sparse({ 0: 2, 1: 3 }), [], {
        ...defaults,
        topP: 0.00001,
      }),
    ).toBe(1);
  });
  test("uses float32 cumulative thresholds at the top-p boundary", () => {
    const logits = sparse({ 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 });
    expect(
      sampleTalkerToken(logits, [], {
        ...defaults,
        topP: 0.8,
        random: () => 0,
      }),
    ).toBe(1);
    expect(
      sampleTalkerToken(logits, [], {
        ...defaults,
        topP: 0.8,
        random: () => 0.999999,
      }),
    ).toBe(4);
  });
  test("applies penalty once per distinct history ID with sign-aware scaling", () => {
    const logits = sparse({ 0: -2, 1: -2.5 });
    expect(
      sampleTalkerToken(logits, [0, 0], {
        ...defaults,
        topK: 1,
        repetitionPenalty: 2,
      }),
    ).toBe(1);
    expect(
      sampleTalkerToken(sparse({ 0: 4, 1: 3 }), [0, 0], {
        ...defaults,
        topK: 1,
        repetitionPenalty: 2,
      }),
    ).toBe(1);
  });
  test.each([-1, 1, NaN, Infinity])(
    "rejects invalid random draw %s",
    (draw) => {
      expect(() =>
        sampleTalkerToken(sparse({ 0: 1 }), [], {
          ...defaults,
          random: () => draw,
        }),
      ).toThrow("random draw");
    },
  );
  test.each([
    { temperature: 0 },
    { temperature: 1e-100 },
    { repetitionPenalty: 0 },
    { topP: 0 },
    { topP: 2 },
    { topK: -1 },
    { topK: 1.5 },
    { topK: 8449 },
  ])("rejects invalid options %j", (options) => {
    expect(() =>
      sampleTalkerToken(sparse({ 0: 1 }), [], { ...defaults, ...options }),
    ).toThrow("sampling inputs");
  });
  test("rejects invalid logits/history and overflow before consuming randomness", () => {
    const random = jest.fn(() => 0);
    const options = { ...defaults, random };
    expect(() => sampleTalkerToken(new Float32Array(2), [], options)).toThrow();
    expect(() => sampleTalkerToken(sparse({ 0: NaN }), [], options)).toThrow();
    expect(() => sampleTalkerToken(sparse({ 0: 1 }), [8448], options)).toThrow(
      "history",
    );
    expect(() =>
      sampleTalkerToken(sparse({ 0: 3e38 }), [], {
        ...options,
        temperature: 0.01,
      }),
    ).toThrow("overflow");
    expect(random).not.toHaveBeenCalled();
  });
});
