import {
  ModelNotLoadedError,
  SpecifiedModelNotFoundError,
  UnclearModelToUseError,
} from "../src/error";
import { cleanModelUrl, getModelIdToUse, getTopProbs } from "../src/support";

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

describe("Test getModelIdToUse", () => {
  test("Specified model not found", () => {
    const loadedModelIds = ["a", "b", "c"];
    const requestModel = "d";
    const requestName = "ChatCompletionRequest";
    expect(() => {
      getModelIdToUse(loadedModelIds, requestModel, requestName);
    }).toThrow(
      new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestModel,
        requestName,
      ),
    );
  });

  test("No model loaded", () => {
    const loadedModelIds: string[] = [];
    const requestModel = "d";
    const requestName = "ChatCompletionRequest";
    expect(() => {
      getModelIdToUse(loadedModelIds, requestModel, requestName);
    }).toThrow(new ModelNotLoadedError(requestName));
  });

  test("Unclear what model to use, undefined", () => {
    const loadedModelIds = ["a", "b", "c"];
    const requestModel = undefined;
    const requestName = "ChatCompletionRequest";
    expect(() => {
      getModelIdToUse(loadedModelIds, requestModel, requestName);
    }).toThrow(new UnclearModelToUseError(loadedModelIds, requestName));
  });

  test("Unclear what model to use, null", () => {
    const loadedModelIds = ["a", "b", "c"];
    const requestModel = null;
    const requestName = "ChatCompletionRequest";
    expect(() => {
      getModelIdToUse(loadedModelIds, requestModel, requestName);
    }).toThrow(new UnclearModelToUseError(loadedModelIds, requestName));
  });

  test("Valid config, unspecified request model", () => {
    const loadedModelIds = ["a"];
    const requestModel = null;
    const requestName = "ChatCompletionRequest";
    const selectedModelId = getModelIdToUse(
      loadedModelIds,
      requestModel,
      requestName,
    );
    expect(selectedModelId).toEqual("a");
  });

  test("Valid config, specified request model", () => {
    const loadedModelIds = ["a"];
    const requestModel = "a";
    const requestName = "ChatCompletionRequest";
    const selectedModelId = getModelIdToUse(
      loadedModelIds,
      requestModel,
      requestName,
    );
    expect(selectedModelId).toEqual("a");
  });

  test("Valid config, specified request model, multi models loaded", () => {
    const loadedModelIds = ["a", "b", "c"];
    const requestModel = "c";
    const requestName = "ChatCompletionRequest";
    const selectedModelId = getModelIdToUse(
      loadedModelIds,
      requestModel,
      requestName,
    );
    expect(selectedModelId).toEqual("c");
  });
});
