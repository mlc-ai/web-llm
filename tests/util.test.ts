import { ChatOptions } from "../src/config";
import {
  ModelNotLoadedError,
  SpecifiedModelNotFoundError,
  UnclearModelToUseError,
} from "../src/error";
import {
  cleanModelUrl,
  CustomLock,
  getModelIdToUse,
  getChunkedPrefillInputData,
  getTopProbs,
} from "../src/support";
import { areChatOptionsListEqual } from "../src/utils";
import { MLCEngine } from "../src/engine";
import { ChatCompletionContentPartImage } from "../src/openai_api_protocols";

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

describe("Test getChunkedPrefillInputData", () => {
  const rangeArr = (start: number, end: number) =>
    Array.from({ length: end - start }, (v, k) => k + start);
  type ImageURL = ChatCompletionContentPartImage.ImageURL;
  const prefillChunkSize = 2048;
  const image1 = { url: "url1" } as ImageURL;
  const image2 = { url: "url2" } as ImageURL;

  test("With image data", async () => {
    const inputData = [
      rangeArr(0, 200),
      image1, // 1921 size
      rangeArr(0, 10),
    ];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [[rangeArr(0, 200)], [image1, rangeArr(0, 10)]];
    const expectedChunkLens = [200, 1931];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });

  test("Single image data", async () => {
    const inputData = [image1];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [[image1]];
    const expectedChunkLens = [1921];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });

  test("Two images", async () => {
    const inputData = [image1, image2];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [[image1], [image2]];
    const expectedChunkLens = [1921, 1921];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });

  test("Single token array that needs to be chunked", async () => {
    const inputData = [rangeArr(0, 4097)];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [
      [rangeArr(0, 2048)],
      [rangeArr(2048, 4096)],
      [rangeArr(4096, 4097)],
    ];
    const expectedChunkLens = [2048, 2048, 1];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });

  test("Single token array that does not need to be chunked", async () => {
    const inputData = [rangeArr(0, 2048)];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [[rangeArr(0, 2048)]];
    const expectedChunkLens = [2048];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });

  test("Token array that needs to be chunked, grouped with others", async () => {
    const inputData = [
      image1, // 1921
      rangeArr(0, 2300),
      image2,
    ];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [
      [image1, rangeArr(0, 127)], // 127 = 2048 - 1921
      [rangeArr(127, 2175)], // 2175 = 127 + 2048
      [rangeArr(2175, 2300), image2],
    ];
    const expectedChunkLens = [2048, 2048, 2046];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });

  test("Image followed by token that fits just well.", async () => {
    const inputData = [
      image1, // 1921
      rangeArr(0, 127),
      image2,
    ];
    const chunks = getChunkedPrefillInputData(inputData, prefillChunkSize);
    const expectedChunks = [[image1, rangeArr(0, 127)], [image2]];
    const expectedChunkLens = [2048, 1921];
    expect(chunks).toEqual([expectedChunks, expectedChunkLens]);
  });
});

// Refers to https://jackpordi.com/posts/locks-in-js-because-why-not
describe("Test CustomLock", () => {
  test("Ensure five +1's give 5 with sleep between read/write", async () => {
    let value = 0;
    const lock = new CustomLock();

    async function addOne() {
      await lock.acquire();
      const readValue = value;
      await new Promise((r) => setTimeout(r, 100));
      value = readValue + 1;
      await lock.release();
    }
    await Promise.all([addOne(), addOne(), addOne(), addOne(), addOne()]);
    expect(value).toEqual(5); // without a lock, most likely less than 5
  });
});
