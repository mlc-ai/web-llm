import { ChatOptions } from "../src/config";
import {
  ModelNotLoadedError,
  SpecifiedModelNotFoundError,
  UnclearModelToUseError,
} from "../src/error";
import { cleanModelUrl, getModelIdToUse, getTopProbs } from "../src/support";
import { areChatOptionsListEqual } from "../src/utils";
import { MLCEngine } from "../src/engine";

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

  // Cannot test MLCEngine.getLLMStates E2E because `instanceof LLMChatPipeline` would not pass
  // with dummy pipeline variables
  test("E2E test with MLCEngine not loading a model for APIs", () => {
    const engine = new MLCEngine();
    expect(async () => {
      await engine.chatCompletion({
        messages: [{ role: "user", content: "hi" }],
      });
    }).rejects.toThrow(new ModelNotLoadedError("ChatCompletionRequest"));
    expect(async () => {
      await engine.getMessage();
    }).rejects.toThrow(new ModelNotLoadedError("getMessage"));

    // resetChat should not throw error because it is allowed to resetChat before pipeline
    // established, as a no-op
    expect(async () => {
      await engine.resetChat();
    }).not.toThrow(new ModelNotLoadedError("resetChat"));
  });

  test("E2E test with MLCEngine with two models without specifying a model", () => {
    const engine = new MLCEngine() as any;
    engine.loadedModelIdToPipeline = new Map<string, any>();
    engine.loadedModelIdToPipeline.set("model1", "dummyLLMChatPipeline");
    engine.loadedModelIdToPipeline.set("model2", "dummyLLMChatPipeline");
    const loadedModelIds = ["model1", "model2"];

    expect(async () => {
      await engine.chatCompletion({
        messages: [{ role: "user", content: "hi" }],
      });
    }).rejects.toThrow(
      new UnclearModelToUseError(loadedModelIds, "ChatCompletionRequest"),
    );
    expect(async () => {
      await engine.getMessage();
    }).rejects.toThrow(
      new UnclearModelToUseError(loadedModelIds, "getMessage"),
    );
    expect(async () => {
      await engine.resetChat();
    }).rejects.toThrow(new UnclearModelToUseError(loadedModelIds, "resetChat"));
  });

  test("E2E test with MLCEngine with two models specifying wrong model", () => {
    const engine = new MLCEngine() as any;
    engine.loadedModelIdToPipeline = new Map<string, any>();
    engine.loadedModelIdToPipeline.set("model1", "dummyLLMChatPipeline");
    engine.loadedModelIdToPipeline.set("model2", "dummyLLMChatPipeline");
    const loadedModelIds = ["model1", "model2"];
    const requestedModelId = "model3";

    expect(async () => {
      await engine.chatCompletion({
        messages: [{ role: "user", content: "hi" }],
        model: requestedModelId,
      });
    }).rejects.toThrow(
      new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestedModelId,
        "ChatCompletionRequest",
      ),
    );
    expect(async () => {
      await engine.getMessage(requestedModelId);
    }).rejects.toThrow(
      new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestedModelId,
        "getMessage",
      ),
    );
    expect(async () => {
      await engine.runtimeStatsText(requestedModelId);
    }).rejects.toThrow(
      new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestedModelId,
        "runtimeStatsText",
      ),
    );

    // resetChat should not throw error because it is allowed to resetChat before pipeline
    // established, as a no-op
    expect(async () => {
      await engine.resetChat(false, requestedModelId);
    }).not.toThrow(
      new SpecifiedModelNotFoundError(
        loadedModelIds,
        requestedModelId,
        "resetChat",
      ),
    );
  });
});

describe("Test areChatOptionsListEqual", () => {
  const dummyChatOpts1: ChatOptions = { tokenizer_files: ["a", "b"] };
  const dummyChatOpts2: ChatOptions = {};
  const dummyChatOpts3: ChatOptions = { tokenizer_files: ["a", "b"] };
  const dummyChatOpts4: ChatOptions = {
    tokenizer_files: ["a", "b"],
    top_p: 0.5,
  };

  test("Two undefined", () => {
    const options1: ChatOptions[] | undefined = undefined;
    const options2: ChatOptions[] | undefined = undefined;
    expect(areChatOptionsListEqual(options1, options2)).toEqual(true);
  });

  test("One undefined", () => {
    const options1: ChatOptions[] | undefined = [dummyChatOpts1];
    const options2: ChatOptions[] | undefined = undefined;
    expect(areChatOptionsListEqual(options1, options2)).toEqual(false);
  });

  test("Both defined, not equal", () => {
    const options1: ChatOptions[] | undefined = [dummyChatOpts1];
    const options2: ChatOptions[] | undefined = [dummyChatOpts2];
    expect(areChatOptionsListEqual(options1, options2)).toEqual(false);
  });

  test("Different size", () => {
    const options1: ChatOptions[] | undefined = [
      dummyChatOpts1,
      dummyChatOpts3,
    ];
    const options2: ChatOptions[] | undefined = [dummyChatOpts2];
    expect(areChatOptionsListEqual(options1, options2)).toEqual(false);
  });

  test("Same size, not equal 1", () => {
    const options1: ChatOptions[] | undefined = [
      dummyChatOpts1,
      dummyChatOpts3,
    ];
    const options2: ChatOptions[] | undefined = [
      dummyChatOpts1,
      dummyChatOpts2,
    ];
    expect(areChatOptionsListEqual(options1, options2)).toEqual(false);
  });

  test("Same size, not equal 2", () => {
    const options1: ChatOptions[] | undefined = [
      dummyChatOpts1,
      dummyChatOpts3,
    ];
    const options2: ChatOptions[] | undefined = [
      dummyChatOpts1,
      dummyChatOpts4,
    ];
    expect(areChatOptionsListEqual(options1, options2)).toEqual(false);
  });

  test("Same size, equal", () => {
    const options1: ChatOptions[] | undefined = [
      dummyChatOpts1,
      dummyChatOpts3,
    ];
    const options2: ChatOptions[] | undefined = [
      dummyChatOpts3,
      dummyChatOpts1,
    ];
    expect(areChatOptionsListEqual(options1, options2)).toEqual(true);
  });
});
