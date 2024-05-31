import { cleanModelUrl, getTopProbs } from "../src/support";

describe("Check getTopLogprobs correctness", () => {
  test("Correctness test 1", () => {
    const logitsOnCPUArray = new Float32Array([
      0.05, 0.15, 0.3, 0.16, 0.04, 0.2, 0.1,
    ]);
    const actual = getTopProbs(3, logitsOnCPUArray);
    const expected: Array<[number, number]> = [
      [2, 0.3],
      [5, 0.2],
      [3, 0.16],
    ];
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++) {
      expect(actual[i][0]).toBe(expected[i][0]);
      expect(actual[i][1]).toBeCloseTo(expected[i][1], 4);
    }
  });

  test("Zero top_logprobs", () => {
    const logitsOnCPUArray = new Float32Array([
      0.05, 0.15, 0.3, 0.16, 0.04, 0.2, 0.1,
    ]);
    const topLogProbs = getTopProbs(0, logitsOnCPUArray);
    expect(topLogProbs).toEqual([]);
  });
});

describe("Test clean model URL", () => {
  test("Input does not have branch or trailing /", () => {
    const input = "https://huggingface.co/mlc-ai/model";
    const output = cleanModelUrl(input);
    const expected = "https://huggingface.co/mlc-ai/model/resolve/main/";
    expect(output).toEqual(expected);
  });

  test("Input does not have branch but has trailing /", () => {
    const input = "https://huggingface.co/mlc-ai/model/";
    const output = cleanModelUrl(input);
    const expected = "https://huggingface.co/mlc-ai/model/resolve/main/";
    expect(output).toEqual(expected);
  });

  test("Input has branch but does not have trailing /", () => {
    const input = "https://huggingface.co/mlc-ai/model/resolve/main";
    const output = cleanModelUrl(input);
    const expected = "https://huggingface.co/mlc-ai/model/resolve/main/";
    expect(output).toEqual(expected);
  });

  test("Input has branch and trailing /", () => {
    const input = "https://huggingface.co/mlc-ai/model/resolve/main/";
    const output = cleanModelUrl(input);
    const expected = "https://huggingface.co/mlc-ai/model/resolve/main/";
    expect(output).toEqual(expected);
  });
});
