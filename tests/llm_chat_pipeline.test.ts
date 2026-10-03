import { LLMChatPipeline } from "../src/llm_chat";
import { MinValueError } from "../src/error";
import { Role } from "../src/config";
import { jest, test, expect, beforeEach } from "@jest/globals";
import log from "loglevel";

jest.mock("../src/support", () => ({
  ...(jest.requireActual("../src/support") as object),
  getResizedRGBArrayFromURL: jest.fn(),
}));

jest.mock("@mlc-ai/web-xgrammar", () => {
  const grammarMatcherInstances: any[] = [];
  const compileBuiltinJSONGrammar = jest
    .fn()
    .mockImplementation(async () => ({ dispose: jest.fn() }));
  const compileJSONSchema = jest
    .fn()
    .mockImplementation(async () => ({ dispose: jest.fn() }));
  const compileGrammar = jest
    .fn()
    .mockImplementation(async () => ({ dispose: jest.fn() }));
  const compileStructuralTag = jest
    .fn()
    .mockImplementation(async () => ({ dispose: jest.fn() }));
  return {
    TokenizerInfo: {
      createTokenizerInfo: jest.fn(async () => "tokenInfo"),
    },
    GrammarCompiler: {
      createGrammarCompiler: jest.fn(async () => ({
        compileBuiltinJSONGrammar,
        compileJSONSchema,
        compileGrammar,
        compileStructuralTag,
      })),
      __compileBuiltinJSONGrammar: compileBuiltinJSONGrammar,
      __compileJSONSchema: compileJSONSchema,
      __compileGrammar: compileGrammar,
      __compileStructuralTag: compileStructuralTag,
    },
    GrammarMatcher: {
      createGrammarMatcher: jest.fn(async () => {
        const matcher = {
          acceptToken: jest.fn(() => true),
          dispose: jest.fn(),
          getNextTokenBitmask: jest.fn(async () => new Int32Array()),
          reset: jest.fn(),
        };
        grammarMatcherInstances.push(matcher);
        return matcher;
      }),
      __instances: grammarMatcherInstances,
    },
  };
});

type XGrammarMock = {
  TokenizerInfo: {
    createTokenizerInfo: jest.Mock;
  };
  GrammarCompiler: {
    createGrammarCompiler: jest.Mock;
    __compileBuiltinJSONGrammar: jest.Mock;
    __compileJSONSchema: jest.Mock;
    __compileGrammar: jest.Mock;
    __compileStructuralTag: jest.Mock;
  };
  GrammarMatcher: {
    createGrammarMatcher: jest.Mock;
    __instances: any[];
  };
};

const xgrammar = jest.requireMock<XGrammarMock>("@mlc-ai/web-xgrammar");
const grammarMatcherInstances = xgrammar.GrammarMatcher.__instances;
const compileGrammarMock = xgrammar.GrammarCompiler.__compileGrammar;
const compileJSONSchemaMock = xgrammar.GrammarCompiler.__compileJSONSchema;
const compileStructuralTagMock =
  xgrammar.GrammarCompiler.__compileStructuralTag;

beforeEach(() => {
  grammarMatcherInstances.length = 0;
  compileGrammarMock.mockClear();
  compileJSONSchemaMock.mockClear();
  compileStructuralTagMock.mockClear();
});

type PipelineLike = LLMChatPipeline & Record<string, any>;

function createPipeline(): PipelineLike {
  const pipeline = Object.create(LLMChatPipeline.prototype) as PipelineLike;
  pipeline["stopTriggered"] = false;
  pipeline["finishReason"] = undefined;
  pipeline["conversation"] = {
    isTextCompletion: false,
    finishReply: jest.fn(),
    appendMessage: jest.fn(),
    appendEmptyThinkingReplyHeader: jest.fn(),
    appendReplyHeader: jest.fn(),
    config: {},
    getPromptArray: jest.fn(() => ["prompt"]),
    getPromptArrayLastRound: jest.fn(() => ["last"]),
    getPromptArrayTextCompletion: jest.fn(() => ["text"]),
  } as any;
  pipeline["config"] = {} as any;
  pipeline["outputIds"] = [];
  pipeline["appearedTokensFreq"] = new Map<number, number>();
  pipeline["stopTokens"] = [];
  pipeline["stopStr"] = [];
  pipeline["tokenizer"] = {
    decode: jest.fn((ids: Int32Array) =>
      Array.from(ids)
        .map((id) => `t${id}`)
        .join(" "),
    ),
    encode: jest.fn(() => Int32Array.from([1])),
    getVocabSize: jest.fn(() => 1),
    idToToken: jest.fn(() => "<tok>"),
  } as any;
  pipeline["contextWindowSize"] = 16;
  pipeline["slidingWindowSize"] = -1;
  pipeline["filledKVCacheLength"] = 0;
  pipeline["outputMessage"] = "";
  pipeline["curRoundLatencyBreakdown"] = {
    logitProcessorTime: [],
    logitBiasTime: [],
    penaltyTime: [],
    sampleTime: [],
    totalTime: [],
    grammarBitmaskTime: [],
  };
  pipeline["prefillChunkSize"] = 8;
  pipeline["tvm"] = {
    beginScope: jest.fn(),
    endScope: jest.fn(),
    detachFromCurrentScope: jest.fn((x: any) => x),
  } as any;
  pipeline["kvCheckpointFuncs"] = new Map();
  pipeline["device"] = {
    sync: jest.fn(async () => undefined),
  } as any;
  pipeline["embedAndForward"] = jest.fn(
    async (_chunk: any, chunkLen: number) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return {
        dispose: jest.fn(),
        shape: [],
        dtype: "float32",
        device: {},
        ndim: 0,
      };
    },
  ) as any;
  pipeline["sampleFromRawLogits"] = jest.fn(async () => 2);
  pipeline["resetRuntimeStats"] = jest.fn();
  pipeline["resetStatsPerPrefill"] = false;
  pipeline["prefillTotalTime"] = 0;
  pipeline["prefillTotalTokens"] = 0;
  pipeline["decodingTotalTime"] = 0;
  pipeline["decodingTotalTokens"] = 0;
  pipeline["curRoundPrefillTotalTokens"] = 0;
  pipeline["curRoundPrefillTotalTime"] = 0;
  pipeline["curRoundGrammarInitTotalTime"] = 0;
  pipeline["curRoundGrammarPerTokenTotalTime"] = 0;
  pipeline["tokenLogprobArray"] = [];
  pipeline["curRoundDecodingTotalTokens"] = 0;
  pipeline["curRoundDecodingTotalTime"] = 0;
  pipeline["imageDataCache"] = new Map();
  pipeline["artifactImagePixelCache"] = new Map();
  return pipeline;
}

test.each([
  ["frequency_penalty", "Make sure -2 < frequency_penalty <= 2."],
  ["presence_penalty", "Make sure -2 < presence_penalty <= 2."],
  ["repetition_penalty", "Make sure `repetition_penalty` > 0."],
  ["top_p", "Make sure 0 < top_p <= 1."],
  ["temperature", "Make sure `temperature` > 0."],
])("rejects a NaN model default for %s", async (field, message) => {
  const pipeline = createPipeline();
  pipeline["config"] = {
    frequency_penalty: 0,
    presence_penalty: 0,
    repetition_penalty: 1,
    top_p: 1,
    temperature: 1,
    [field]: Number.NaN,
  } as any;

  await expect(
    (LLMChatPipeline.prototype as any).sampleFromRawLogits.call(
      pipeline,
      {} as any,
    ),
  ).rejects.toThrow(message);
});

test("processNextToken stops on stop token and updates conversation", () => {
  const pipeline = createPipeline();
  pipeline["stopTokens"] = [42];
  (pipeline as any).processNextToken(42);
  expect(pipeline["stopTriggered"]).toBe(true);
  expect(pipeline["finishReason"]).toBe("stop");
  expect(pipeline["conversation"].finishReply).toHaveBeenCalledWith("");
});

test("processNextToken appends tokens until stop string reached", () => {
  const pipeline = createPipeline();
  pipeline["stopStr"] = ["<stop>"];
  pipeline["tokenizer"].decode = jest
    .fn<(ids: Int32Array) => string>()
    .mockReturnValueOnce("partial")
    .mockReturnValueOnce("partial<stop>");
  (pipeline as any).processNextToken(1, {
    max_tokens: 5,
  });
  expect(pipeline["stopTriggered"]).toBe(false);
  (pipeline as any).processNextToken(2, {
    max_tokens: 5,
  });
  expect(pipeline["stopTriggered"]).toBe(true);
  expect(pipeline["finishReason"]).toBe("stop");
  expect(pipeline["outputMessage"]).toBe("partial");
});

test("commitSampledStep records a reversible text rewrite", () => {
  const pipeline = createPipeline();
  pipeline["outputMessage"] = "caf\ufffd";
  pipeline["commitSampledToken"] = jest.fn(() => {
    pipeline["outputMessage"] = "caf\u00e9";
  });

  const committed = pipeline.commitSampledStep({
    source: "decode",
    tokenId: 2,
    globalTokenPos: 10,
  });

  expect(committed.textPrefixLength).toBe(3);
  expect(committed.textDelta).toBe("\u00e9");
  expect(committed.outputMessage).toBe("caf\u00e9");
});

test("processNextToken respects max_tokens and updates token frequency", () => {
  const pipeline = createPipeline();
  (pipeline as any).processNextToken(7, { max_tokens: 1 });
  expect(pipeline["appearedTokensFreq"].get(7)).toBe(1);
  expect(pipeline["finishReason"]).toBe("length");
});

test.each([
  ["zero", 0],
  ["below zero", -1],
  ["NaN", Number.NaN],
])("processNextToken rejects max_tokens when it is %s", (_name, value) => {
  const pipeline = createPipeline();
  expect(() =>
    (pipeline as any).processNextToken(1, { max_tokens: value }),
  ).toThrow(MinValueError);
});

test("triggerStop converts conversation reply to finished state", () => {
  const pipeline = createPipeline();
  pipeline["outputMessage"] = "final";
  pipeline["conversation"].isTextCompletion = false;
  pipeline.triggerStop();
  expect(pipeline["stopTriggered"]).toBe(true);
  expect(pipeline["finishReason"]).toBe("abort");
  expect(pipeline["conversation"].finishReply).toHaveBeenCalledWith("final");
});

function preparePrefillPipeline(): PipelineLike {
  const pipeline = createPipeline();
  pipeline["prefillTotalTime"] = 0;
  pipeline["prefillTotalTokens"] = 0;
  pipeline["getInputData"] = jest.fn(
    async (): Promise<[any[], number, any]> => [[[0]], 1, () => 0],
  );
  pipeline["processNextToken"] = jest.fn();
  return pipeline;
}

test("prefillStep adds thinking reply header when thinking disabled", async () => {
  const pipeline = preparePrefillPipeline();
  pipeline["tokenizer"].encode = jest.fn(() => Int32Array.from([9, 9]));
  await pipeline.prefillStep("hello", Role.user, undefined, {
    enable_thinking: false,
  });
  expect(
    pipeline["conversation"].appendEmptyThinkingReplyHeader,
  ).toHaveBeenCalled();
  expect(pipeline["conversation"].appendReplyHeader).not.toHaveBeenCalled();
  expect(pipeline["outputIds"].length).toBeGreaterThan(0);
  expect(pipeline["processNextToken"]).toHaveBeenCalled();
});

test("prefillStep appends standard reply header when thinking enabled", async () => {
  const pipeline = preparePrefillPipeline();
  pipeline["tokenizer"].encode = jest.fn(() => Int32Array.from([2]));
  await pipeline.prefillStep("hi", Role.user);
  expect(pipeline["conversation"].appendReplyHeader).toHaveBeenCalledWith(
    Role.assistant,
  );
  expect(
    pipeline["conversation"].appendEmptyThinkingReplyHeader,
  ).not.toHaveBeenCalled();
});

test("forwardPrefill returns raw logits and assistant prefix metadata", async () => {
  const pipeline = preparePrefillPipeline();
  const rawLogits = {
    dispose: jest.fn(),
    shape: [],
    dtype: "float32",
    device: {},
    ndim: 0,
  } as any;
  pipeline["tokenizer"].encode = jest.fn(() => Int32Array.from([9, 9]));
  pipeline["embedAndForward"] = jest.fn(
    async (_chunk: any, chunkLen: number) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return rawLogits;
    },
  ) as any;

  const result = await pipeline["forwardPrefill"](
    "hello",
    Role.user,
    undefined,
    {
      enable_thinking: false,
    },
  );

  expect(result.logits).toBe(rawLogits);
  expect(result.promptLen).toBe(1);
  expect(result.assistantPrefixTokenIds).toEqual([9, 9]);
  expect(pipeline["sampleFromRawLogits"]).not.toHaveBeenCalled();
  expect(
    pipeline["conversation"].appendEmptyThinkingReplyHeader,
  ).toHaveBeenCalled();
});

test("prefillStep samples raw prefill logits before committing token", async () => {
  const pipeline = preparePrefillPipeline();
  const rawLogits = {
    dispose: jest.fn(),
    shape: [],
    dtype: "float32",
    device: {},
    ndim: 0,
  } as any;
  const genConfig = { max_tokens: 5 };
  pipeline["embedAndForward"] = jest.fn(
    async (_chunk: any, chunkLen: number) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return rawLogits;
    },
  ) as any;
  pipeline["sampleFromRawLogits"] = jest.fn(async () => 4);

  await pipeline.prefillStep("hello", Role.user, undefined, genConfig);

  expect(pipeline["sampleFromRawLogits"]).toHaveBeenCalledWith(
    rawLogits,
    genConfig,
  );
  expect(rawLogits.dispose).toHaveBeenCalled();
  expect(pipeline["processNextToken"]).toHaveBeenCalledWith(4, genConfig);
});

test("decodeStep forwards last committed token and commits sampled token", async () => {
  const pipeline = createPipeline();
  const rawLogits = {
    dispose: jest.fn(),
    shape: [],
    dtype: "float32",
    device: {},
    ndim: 0,
  } as any;
  const genConfig = { max_tokens: 5 };
  pipeline["outputIds"] = [7];
  pipeline["processNextToken"] = jest.fn();
  pipeline["embedAndForward"] = jest.fn(
    async (_chunk: any, chunkLen: number) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return rawLogits;
    },
  ) as any;
  pipeline["sampleFromRawLogits"] = jest.fn(async () => 8);

  await pipeline.decodeStep(genConfig);

  expect(pipeline["embedAndForward"]).toHaveBeenCalledWith([[7]], 1);
  expect(pipeline["sampleFromRawLogits"]).toHaveBeenCalledWith(
    rawLogits,
    genConfig,
  );
  expect(rawLogits.dispose).toHaveBeenCalled();
  expect(pipeline["processNextToken"]).toHaveBeenCalledWith(8, genConfig);
  expect(pipeline["curRoundDecodingTotalTokens"]).toBe(1);
});

function prepareReplayPipeline(): PipelineLike {
  const pipeline = createPipeline();
  pipeline.resetChat = jest.fn();
  pipeline.setConversation = jest.fn();
  pipeline["resetGenerationRoundState"] = jest.fn();
  pipeline["prepareGrammarMatcherForSampling"] = jest.fn(async () => undefined);
  pipeline["outputIds"] = [];
  return pipeline;
}

test("token replay samples and returns the first token when no token was journaled", async () => {
  const pipeline = prepareReplayPipeline();
  const promptLogits = { dispose: jest.fn() } as any;
  const committed = {
    source: "prefill",
    tokenId: 17,
    globalTokenPos: 3,
    textDelta: "first",
    textPrefixLength: 0,
    outputMessage: "first",
    stopped: false,
  } as any;
  pipeline["forwardKnownTokens"] = jest.fn(async () => promptLogits);
  pipeline["sampleFromRawLogits"] = jest.fn(async () => 17);
  pipeline.commitSampledStep = jest.fn(() => committed);

  const result = await pipeline.replayGenerationTokens([1, 2, 3], [9], [], {
    temperature: 0.5,
  });

  expect(pipeline["outputIds"]).toEqual([9]);
  expect(pipeline["sampleFromRawLogits"]).toHaveBeenCalledWith(promptLogits, {
    temperature: 0.5,
  });
  expect(pipeline.commitSampledStep).toHaveBeenCalledWith(
    {
      source: "prefill",
      tokenId: 17,
      globalTokenPos: 3,
    },
    { temperature: 0.5 },
  );
  expect(result).toEqual({
    replayedTokens: 0,
    sampledFromCheckpointLogits: false,
    sampledToken: {
      source: "prefill",
      tokenId: 17,
      globalTokenPos: 3,
    },
    committedToken: committed,
  });
  expect(promptLogits.dispose).toHaveBeenCalled();
});

test("token replay forwards all but the final known generated token", async () => {
  const pipeline = prepareReplayPipeline();
  const promptLogits = { dispose: jest.fn() } as any;
  const decodeLogits = { dispose: jest.fn() } as any;
  pipeline["forwardKnownTokens"] = jest.fn(async () => promptLogits);
  pipeline["forwardDecodeToken"] = jest.fn(async () => decodeLogits);
  pipeline["commitSampledToken"] = jest.fn();

  const result = await pipeline.replayGenerationTokens(
    [1, 2],
    [8],
    [
      { globalTokenPos: 2, tokenId: 10, textDelta: "a" },
      { globalTokenPos: 3, tokenId: 11, textDelta: "b" },
    ],
    { max_tokens: 4 },
  );

  expect(pipeline["forwardDecodeToken"]).toHaveBeenCalledTimes(1);
  expect(pipeline["forwardDecodeToken"]).toHaveBeenCalledWith(10);
  expect(decodeLogits.dispose).toHaveBeenCalled();
  expect(pipeline["commitSampledToken"]).toHaveBeenNthCalledWith(
    1,
    10,
    { max_tokens: 4 },
    "decode",
  );
  expect(pipeline["commitSampledToken"]).toHaveBeenNthCalledWith(
    2,
    11,
    { max_tokens: 4 },
    "decode",
  );
  expect(result).toEqual({
    replayedTokens: 2,
    sampledFromCheckpointLogits: false,
  });
  expect(promptLogits.dispose).toHaveBeenCalled();
});

test("known-token forwarding detaches only the final chunk logits", async () => {
  const pipeline = createPipeline();
  const firstLogits = { dispose: jest.fn() } as any;
  const finalLogits = { dispose: jest.fn() } as any;
  pipeline["embedAndForward"] = jest
    .fn<(...args: any[]) => Promise<any>>()
    .mockImplementationOnce(async (_chunk, chunkLen) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return firstLogits;
    })
    .mockImplementationOnce(async (_chunk, chunkLen) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return finalLogits;
    });

  const result = await pipeline["forwardKnownTokens"](
    Array.from({ length: 10 }, (_, index) => index),
    true,
  );

  expect(pipeline["embedAndForward"]).toHaveBeenCalledTimes(2);
  expect(pipeline["tvm"].detachFromCurrentScope).toHaveBeenCalledTimes(1);
  expect(pipeline["tvm"].detachFromCurrentScope).toHaveBeenCalledWith(
    finalLogits,
  );
  expect(result).toBe(finalLogits);
  expect(pipeline["filledKVCacheLength"]).toBe(10);
  expect(pipeline["tvm"].endScope).toHaveBeenCalled();
});

test.each([false, true])(
  "prefill disposes intermediate logits (forward failure: %s)",
  async (fail) => {
    const pipeline = preparePrefillPipeline() as any;
    pipeline["prefillChunkSize"] = 2;
    pipeline["getInputData"] = jest.fn(async () => [
      [[1, 2, 3, 4]],
      4,
      () => 0,
    ]);
    const first = { dispose: jest.fn() };
    const last = { dispose: jest.fn() };
    pipeline["embedAndForward"] = jest
      .fn()
      .mockImplementationOnce(async () => {
        pipeline["filledKVCacheLength"] += 2;
        return first;
      })
      .mockImplementationOnce(async () => {
        if (fail) throw new Error("forward failed");
        pipeline["filledKVCacheLength"] += 2;
        return last;
      });

    const request = pipeline.samplePrefillStep("prompt", Role.user);
    if (fail) {
      await expect(request).rejects.toThrow("forward failed");
    } else {
      await request;
      expect(last.dispose).toHaveBeenCalledTimes(1);
    }
    expect(pipeline["tvm"].detachFromCurrentScope).not.toHaveBeenCalledWith(
      first,
    );
    expect(pipeline["tvm"].endScope).toHaveBeenCalledTimes(1);
  },
);

test("decode releases logits when sampling fails", async () => {
  const pipeline = createPipeline() as any;
  const logits = { dispose: jest.fn() };
  pipeline["outputIds"] = [1];
  pipeline["embedAndForward"] = jest.fn(async () => {
    pipeline["filledKVCacheLength"]++;
    return logits;
  });
  pipeline["sampleFromRawLogits"] = jest.fn(async () => {
    throw new Error("sample failed");
  });
  await expect(pipeline.sampleDecodeStep()).rejects.toThrow("sample failed");
  expect(logits.dispose).toHaveBeenCalledTimes(1);
});

test("decode closes its scope on forward failure", async () => {
  const pipeline = createPipeline();
  pipeline["outputIds"] = [1];
  pipeline["embedAndForward"] = jest.fn(async () => {
    throw new Error("forward failed");
  });
  await expect(pipeline.sampleDecodeStep()).rejects.toThrow("forward failed");
  expect(pipeline["tvm"].endScope).toHaveBeenCalledTimes(1);
});

test.each([false, true])(
  "checkpoint replay at the context limit preserves all covered tokens (tail: %s)",
  async (hasTail) => {
    const pipeline = prepareReplayPipeline() as any;
    pipeline["conversation"].isTextCompletion = true;
    pipeline["contextWindowSize"] = 4;
    pipeline["importPromptCheckpoint"] = jest.fn(async () => {
      pipeline["filledKVCacheLength"] = 4;
    });
    pipeline["tvm"].empty = jest.fn(() => ({ copyFromRawBytes: jest.fn() }));
    pipeline["sampleFromRawLogits"] = jest.fn(async () => 12);
    const result = await pipeline.replayFromPromptCheckpoint(
      {
        processedSeqLen: 4,
        metadata: {},
        pageGroups: [],
        nextLogits: { shape: [1], dtype: "float32", data: new Uint8Array(4) },
      },
      [],
      [
        { globalTokenPos: 2, tokenId: 10, textDelta: "t10" },
        { globalTokenPos: 3, tokenId: 11, textDelta: " t11" },
      ],
      hasTail ? [{ globalTokenPos: 4, tokenId: 12, textDelta: " t12" }] : [],
      { max_tokens: 10 },
    );
    expect(pipeline.getMessage()).toBe("t10 t11 t12");
    expect(pipeline.getFinishReason()).toBe("length");
    expect(pipeline["filledKVCacheLength"]).toBe(4);
    expect(result.sampledFromCheckpointLogits).toBe(!hasTail);
    expect(pipeline["sampleFromRawLogits"]).toHaveBeenCalledTimes(
      hasTail ? 0 : 1,
    );
  },
);

test("checkpoint replay exposes the token sampled from persisted logits", async () => {
  const pipeline = prepareReplayPipeline();
  const logits = {
    copyFromRawBytes: jest.fn(),
  } as any;
  pipeline["kvCache"] = {} as any;
  pipeline["importPromptCheckpoint"] = jest.fn(async () => {
    pipeline["filledKVCacheLength"] = 4;
  });
  pipeline["tvm"].empty = jest.fn(() => logits);
  pipeline["sampleFromRawLogits"] = jest.fn(async () => 21);
  pipeline.commitSampledStep = jest.fn((sampled: any) => ({
    ...sampled,
    textDelta: "new",
    textPrefixLength: 3,
    outputMessage: "oldnew",
    stopped: false,
  }));

  const result = await pipeline.replayFromPromptCheckpoint(
    {
      processedSeqLen: 4,
      metadata: {},
      pageGroups: [],
      nextLogits: {
        shape: [1, 4],
        dtype: "float32",
        data: new Uint8Array([1, 2, 3, 4]),
      },
    },
    [],
    [],
    [],
    { max_tokens: 5 },
  );

  expect(logits.copyFromRawBytes).toHaveBeenCalledWith(
    new Uint8Array([1, 2, 3, 4]),
  );
  expect(result.sampledToken).toEqual({
    source: "prefill",
    tokenId: 21,
    globalTokenPos: 4,
  });
  expect(result.committedToken?.textPrefixLength).toBe(3);
  expect(result.sampledFromCheckpointLogits).toBe(true);
});

test.each(["copy", "export", "import", "logits replay"])(
  "checkpoint %s closes its scope when allocation fails",
  async (operation) => {
    const pipeline = prepareReplayPipeline() as any;
    pipeline.kvCache = {};
    pipeline.kvStateKind = "kv_cache";
    pipeline.tvm.cpu = jest.fn();
    pipeline.tvm.empty = jest.fn(() => {
      throw new Error("allocation failed");
    });
    const checkpoint = {
      processedSeqLen: 4,
      metadata: {
        groups: [
          {
            group_index: 0,
            layer_begin: 0,
            layer_end: 1,
            shape: [1],
            dtype: "float32",
          },
        ],
      },
      pageGroups: [
        { groupId: 0, layerStart: 0, layerEnd: 1, data: new Uint8Array(4) },
      ],
      nextLogits: { shape: [1], dtype: "float32", data: new Uint8Array(4) },
    };
    pipeline.getKVCheckpointFunc = jest.fn(
      () => () => JSON.stringify(checkpoint.metadata),
    );
    let promise;
    if (operation === "copy")
      promise = pipeline.copyTensorToCPUBytes({ shape: [1], dtype: "float32" });
    if (operation === "export")
      promise = pipeline.exportPromptCheckpoint({}, false);
    if (operation === "import")
      promise = pipeline.importPromptCheckpoint(checkpoint);
    if (operation === "logits replay") {
      pipeline.importPromptCheckpoint = jest.fn(async () => {
        pipeline.filledKVCacheLength = 4;
      });
      promise = pipeline.replayFromPromptCheckpoint(checkpoint, [], [], []);
    }
    await expect(promise).rejects.toThrow("allocation failed");
    expect(pipeline.tvm.beginScope).toHaveBeenCalledTimes(1);
    expect(pipeline.tvm.endScope).toHaveBeenCalledTimes(1);
  },
);

test("KV checkpoint import refuses hybrid state", async () => {
  const pipeline = createPipeline() as any;
  pipeline.kvCache = {};
  pipeline.kvStateKind = "hybrid";
  await expect(pipeline.importPromptCheckpoint({})).rejects.toThrow(
    "requires a pure KV cache",
  );
  expect(pipeline.tvm.beginScope).not.toHaveBeenCalled();
});

test("getKVCheckpointFunc uses a scope and caches packed functions", () => {
  const pipeline = createPipeline();
  const func = jest.fn() as any;
  func.dispose = jest.fn();
  pipeline["tvm"].getGlobalFunc = jest.fn((name: string) => {
    expect(name).toBe("vm.builtin.attention_kv_cache_get_checkpoint_metadata");
    return func;
  });

  const first = pipeline["getKVCheckpointFunc"](
    "vm.builtin.attention_kv_cache_get_checkpoint_metadata",
  );
  const second = pipeline["getKVCheckpointFunc"](
    "vm.builtin.attention_kv_cache_get_checkpoint_metadata",
  );

  expect(first).toBe(func);
  expect(second).toBe(func);
  expect(pipeline["tvm"].beginScope).toHaveBeenCalledTimes(1);
  expect(pipeline["tvm"].detachFromCurrentScope).toHaveBeenCalledWith(func);
  expect(pipeline["tvm"].endScope).toHaveBeenCalledTimes(1);
  expect(pipeline["tvm"].getGlobalFunc).toHaveBeenCalledTimes(1);
});

test("checkpoint capture disables itself when runtime globals are missing", async () => {
  const pipeline = createPipeline();
  const logits = {} as any;
  const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);
  pipeline["kvCache"] = {} as any;
  pipeline["kvStateKind"] = "kv_cache";
  pipeline["tvm"].getGlobalFunc = jest.fn(() => {
    throw new Error("checkpoint global is missing");
  });

  await expect(
    pipeline["tryExportPromptCheckpoint"](logits, true),
  ).resolves.toBeUndefined();
  await expect(
    pipeline["tryExportPromptCheckpoint"](logits, true),
  ).resolves.toBeUndefined();

  expect(pipeline["tvm"].getGlobalFunc).toHaveBeenCalledTimes(1);
  expect(pipeline["kvCheckpointUnavailableReason"]).toBe(
    "checkpoint global is missing",
  );
  expect(warn).toHaveBeenCalledWith(
    "KV checkpoint capture disabled for this model: checkpoint global is missing",
  );
  warn.mockRestore();
});

test("checkpoint capture disables itself when the runtime rejects the cache layout", async () => {
  const pipeline = createPipeline();
  const logits = {} as any;
  const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);
  pipeline["kvCache"] = {} as any;
  pipeline["kvStateKind"] = "kv_cache";
  pipeline["tvm"].getGlobalFunc = jest.fn((name: string) => {
    const func = jest.fn(() => {
      if (name === "vm.builtin.attention_kv_cache_get_checkpoint_metadata") {
        throw new Error("cache layout does not support checkpoint export");
      }
    }) as any;
    func.dispose = jest.fn();
    return func;
  });

  await expect(
    pipeline["tryExportPromptCheckpoint"](logits, true),
  ).resolves.toBeUndefined();
  await expect(
    pipeline["tryExportPromptCheckpoint"](logits, true),
  ).resolves.toBeUndefined();

  expect(pipeline["tvm"].getGlobalFunc).toHaveBeenCalledTimes(6);
  expect(pipeline["kvCheckpointUnavailableReason"]).toBe(
    "cache layout does not support checkpoint export",
  );
  expect(warn).toHaveBeenCalledWith(
    "KV checkpoint capture disabled for this model: cache layout does not support checkpoint export",
  );
  warn.mockRestore();
});

test("prefillStep reuses grammar matcher when schema unchanged", async () => {
  const pipeline = preparePrefillPipeline();
  const matcher = { acceptToken: jest.fn(() => true), reset: jest.fn() };
  pipeline["grammarMatcher"] = matcher as any;
  pipeline["responseFormatCacheKey"] = "schema_v1";
  await pipeline.prefillStep("hello", Role.user, undefined, {
    response_format: { type: "grammar", grammar: "schema_v1" },
  });
  expect(matcher.reset).toHaveBeenCalled();
});

test("prefillStep instantiates new grammar matcher when schema changes", async () => {
  const pipeline = preparePrefillPipeline();
  pipeline["grammarMatcher"] = undefined;
  pipeline["responseFormatCacheKey"] = undefined;
  pipeline["xgTokenizerInfo"] = undefined;
  pipeline["grammarCompiler"] = undefined;
  await pipeline.prefillStep("hello", Role.user, undefined, {
    response_format: { type: "json_object", schema: "{}" },
  });
  expect(xgrammar.TokenizerInfo.createTokenizerInfo).toHaveBeenCalled();
  expect(xgrammar.GrammarMatcher.createGrammarMatcher).toHaveBeenCalled();
  expect(pipeline["responseFormatCacheKey"]).toBe("{}");
});

test("prefillStep compiles custom grammar when response type is grammar", async () => {
  const pipeline = preparePrefillPipeline();
  pipeline["grammarMatcher"] = undefined;
  pipeline["responseFormatCacheKey"] = undefined;
  pipeline["xgTokenizerInfo"] = undefined;
  pipeline["grammarCompiler"] = undefined;
  await pipeline.prefillStep("hello", Role.user, undefined, {
    response_format: { type: "grammar", grammar: "root ::= WORD" },
  });
  expect(compileGrammarMock).toHaveBeenCalledWith("root ::= WORD");
});

test("prefillStep compiles structural tag response format", async () => {
  const pipeline = preparePrefillPipeline();
  pipeline["grammarMatcher"] = undefined;
  pipeline["responseFormatCacheKey"] = undefined;
  pipeline["xgTokenizerInfo"] = undefined;
  pipeline["grammarCompiler"] = undefined;
  const structuralTag = {
    type: "structural_tag",
    format: { type: "any_text" },
  } as const;
  await pipeline.prefillStep("hello", Role.user, undefined, {
    response_format: {
      type: "structural_tag",
      structural_tag: structuralTag,
    },
  });
  expect(compileStructuralTagMock).toHaveBeenCalledWith(structuralTag);
});

test("prefillStep rejects when structural tag compilation fails", async () => {
  const pipeline = preparePrefillPipeline();
  const logits = {
    dispose: jest.fn(),
    shape: [],
    dtype: "float32",
    device: {},
    ndim: 0,
  };
  pipeline["embedAndForward"] = jest.fn(
    async (_chunk: any, chunkLen: number) => {
      pipeline["filledKVCacheLength"] += chunkLen;
      return logits;
    },
  ) as any;
  pipeline["grammarMatcher"] = undefined;
  pipeline["responseFormatCacheKey"] = undefined;
  pipeline["xgTokenizerInfo"] = undefined;
  pipeline["grammarCompiler"] = undefined;
  compileStructuralTagMock.mockImplementationOnce(() =>
    Promise.reject(8476360),
  );

  await expect(
    pipeline.prefillStep("hello", Role.user, undefined, {
      response_format: {
        type: "structural_tag",
        structural_tag: {
          type: "structural_tag",
          format: { type: "any_text" },
        },
      },
    }),
  ).rejects.toThrow(
    "Failed to initialize the grammar matcher for response format `structural_tag`: 8476360",
  );
  expect(logits.dispose).toHaveBeenCalledTimes(1);
  expect(pipeline["processNextToken"]).not.toHaveBeenCalled();
});

test("getInputData uses cached prompts when KV cache filled", async () => {
  const pipeline = createPipeline();
  pipeline["tokenizer"].encode = jest.fn((prompt: string) =>
    Int32Array.from(prompt === "prompt" ? [1, 2, 3] : [4]),
  );
  pipeline["conversation"].config.system_prefix_token_ids = undefined;
  pipeline["filledKVCacheLength"] = 0;
  const [fullPrompt] = await (pipeline as any).getInputData();
  expect(fullPrompt).toEqual([[1, 2, 3]]);
  expect(pipeline["conversation"].getPromptArray).toHaveBeenCalled();
  pipeline["filledKVCacheLength"] = 1;
  const [lastRoundPrompt] = await (pipeline as any).getInputData();
  expect(lastRoundPrompt).toEqual([[4]]);
  expect(pipeline["conversation"].getPromptArrayLastRound).toHaveBeenCalled();
});

test("adapter embeddings that fit one chunk are reused without allocation", () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const tensor = { shape: [68, 1536], dtype: "float16" };
  raw.fTensorCreateView = jest.fn();
  raw.tvm.empty = jest.fn();

  expect(raw.sliceTensorRows(tensor, 0, 68)).toBe(tensor);
  expect(raw.fTensorCreateView).not.toHaveBeenCalled();
  expect(raw.tvm.empty).not.toHaveBeenCalled();
});

test.each([
  ["float16", 2, 0, 128],
  ["float16", 2, 128, 128],
  ["float16", 2, 256, 3],
  ["float32", 4, 128, 128],
  ["bfloat16", 2, 256, 3],
])(
  "adapter chunk (%s, %i bytes/element, start=%i, rows=%i) has zero offset",
  (dtype, bytes, start, count) => {
    const pipeline = createPipeline();
    const raw = pipeline as any;
    const tensor = { shape: [259, 1536], dtype };
    const byteOffset = start * 1536 * bytes;
    const view = { shape: [count, 1536], dtype, byteOffset };
    const contiguous = { byteOffset: 0, copyFrom: jest.fn() };
    contiguous.copyFrom.mockReturnValue(contiguous);
    raw.tvm.makeShapeTuple = jest.fn((shape: number[]) => shape);
    raw.fTensorCreateView = jest.fn(() => view);
    raw.tvm.empty = jest.fn(() => contiguous);

    const chunk = raw.sliceTensorRows(tensor, start, count);

    expect(raw.fTensorCreateView).toHaveBeenCalledWith(
      tensor,
      [count, 1536],
      dtype,
      expect.objectContaining({ value: byteOffset }),
    );
    expect(chunk.byteOffset).toBe(0);
    if (start === 0) {
      expect(chunk).toBe(view);
      expect(raw.tvm.empty).not.toHaveBeenCalled();
    } else {
      expect(chunk).toBe(contiguous);
      expect(raw.tvm.empty).toHaveBeenCalledTimes(1);
      expect(raw.tvm.empty).toHaveBeenCalledWith(
        [count, 1536],
        dtype,
        raw.device,
      );
      expect(contiguous.copyFrom).toHaveBeenCalledWith(view);
    }
  },
);

test("artifact prompt assembly chunks dynamic audio embeddings", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const audioPart = {
    type: "input_audio" as const,
    input_audio: {
      format: "pcm_f32" as const,
      data: new Float32Array([0, 1]),
      sample_rate: 16000,
    },
  };
  raw["artifact"] = {
    audioInput: {
      processor: {
        kind: "audio_decode",
        format: "pcm_f32",
        sample_rate_hz: 16000,
        channels: 1,
        min_samples: 1,
        max_samples: 10,
      },
      adapter: "audio",
      prompt: {
        prefix_token_ids: [10],
        placeholder_token_id: 99,
        suffix_token_ids: [11],
      },
    },
  } as any;
  pipeline["prefillChunkSize"] = 4;
  pipeline["conversation"].config.system_prefix_token_ids = [9];
  pipeline["conversation"].getArtifactPromptSegments = jest.fn(() => [
    "before",
    audioPart,
    "after",
  ]);
  pipeline["tokenizer"].encode = jest.fn((text: string) =>
    text === "before" ? Int32Array.from([1, 2]) : Int32Array.from([3]),
  );
  const audioEmbeddings = { shape: [6, 8], dtype: "float16" };
  raw["getArtifactAudioEmbeddings"] = jest.fn(() => audioEmbeddings);
  raw["sliceTensorRows"] = jest.fn(
    (_tensor: unknown, start: number, count: number) => ({
      shape: [count, 8],
      start,
    }),
  );

  const [chunks, promptLength] = await raw["getArtifactPrefillChunks"]();
  expect(promptLength).toBe(12);
  expect(chunks.map((chunk: any) => chunk.tokenIds)).toEqual([
    [9, 1, 2, 10],
    [99, 99, 99, 99],
    [99, 99],
    [11, 3],
  ]);
  expect(chunks[1].modalityIds).toEqual([1, 1, 1, 1]);
  expect(raw["sliceTensorRows"]).toHaveBeenCalledTimes(2);
});

test("artifact prompt assembly splices image embeddings at the placeholder span", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const imagePart = {
    type: "image_url" as const,
    image_url: { url: "data:image/png;base64,AAAA" },
  };
  raw["artifact"] = {
    imageInput: {
      processor: {
        kind: "image_decode",
        format: "rgb_u8",
        layout: "nhwc",
        resize: { mode: "center_crop", height: 2, width: 2 },
        num_embeddings: 5,
      },
      adapter: "image",
      prompt: {
        prefix_token_ids: [20],
        placeholder_token_id: 32000,
        suffix_token_ids: [21],
      },
    },
  } as any;
  pipeline["prefillChunkSize"] = 4;
  pipeline["conversation"].config.system_prefix_token_ids = [1];
  pipeline["conversation"].getArtifactPromptSegments = jest.fn(() => [
    "USER: ",
    imagePart,
    "\nwhat? ASSISTANT:",
  ]);
  pipeline["tokenizer"].encode = jest.fn((text: string) =>
    text === "USER: " ? Int32Array.from([2, 3]) : Int32Array.from([4, 5]),
  );
  const imageEmbeddings = { shape: [5, 8], dtype: "float16" };
  raw["getArtifactImageEmbeddings"] = jest.fn(async () => imageEmbeddings);
  raw["getArtifactAudioEmbeddings"] = jest.fn();
  raw["sliceTensorRows"] = jest.fn(
    (_tensor: unknown, start: number, count: number) => ({
      shape: [count, 8],
      start,
    }),
  );

  const [chunks, promptLength] = await raw["getArtifactPrefillChunks"]();
  expect(promptLength).toBe(12);
  expect(chunks.map((chunk: any) => chunk.tokenIds)).toEqual([
    [1, 2, 3, 20],
    [32000, 32000, 32000, 32000],
    [32000],
    [21, 4, 5],
  ]);
  expect(chunks[1].modalityIds).toEqual([1, 1, 1, 1]);
  expect(chunks[2].embeddings).toEqual({ shape: [1, 8], start: 4 });
  expect(raw["getArtifactImageEmbeddings"]).toHaveBeenCalledWith(imagePart);
  expect(raw["getArtifactAudioEmbeddings"]).not.toHaveBeenCalled();
});

test("an image part is rejected when the artifact declares no image input", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  raw["artifact"] = { audioInput: { prompt: {} } };
  raw["conversation"].config.system_prefix_token_ids = null;
  pipeline["conversation"].getArtifactPromptSegments = jest.fn(() => [
    {
      type: "image_url" as const,
      image_url: { url: "data:image/png;base64,AAAA" },
    },
  ]);
  await expect(raw["getArtifactPrefillChunks"]()).rejects.toThrow(
    /does not declare an image input/,
  );
});

test("image embeddings are built from a uint8 NHWC tensor of the declared size", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const resize = { mode: "stretch", height: 2, width: 3 };
  raw["artifact"] = {
    imageInput: { processor: { resize, num_embeddings: 4 }, dtype: "uint8" },
  };
  const pixels = new Uint8ClampedArray(2 * 3 * 3).fill(7);
  const support = jest.requireMock("../src/support") as any;
  support.getResizedRGBArrayFromURL.mockResolvedValue(pixels);
  const pixelTensor = { copyFrom: jest.fn(), copyFromRawBytes: jest.fn() };
  raw["tvm"].empty = jest.fn(() => pixelTensor);
  raw["tvm"].attachToCurrentScope = jest.fn();
  raw["params"] = "params";
  const embeddings = { shape: [4, 8] };
  raw["artifactImageAdapter"] = jest.fn(() => embeddings);

  const imagePart = {
    type: "image_url" as const,
    image_url: { url: "data:image/png;base64,AAAA" },
  };
  await expect(raw["getArtifactImageEmbeddings"](imagePart)).resolves.toBe(
    embeddings,
  );
  expect(support.getResizedRGBArrayFromURL).toHaveBeenCalledWith(
    imagePart.image_url.url,
    resize,
  );
  expect(raw["tvm"].empty).toHaveBeenCalledWith(
    [1, 2, 3, 3],
    "uint8",
    raw["device"],
  );
  expect(pixelTensor.copyFrom).toHaveBeenCalledWith(pixels);
  expect(raw["artifactImageAdapter"]).toHaveBeenCalledWith(
    pixelTensor,
    "params",
  );

  raw["artifactImageAdapter"] = jest.fn(() => ({ shape: [3, 8] }));
  await expect(raw["getArtifactImageEmbeddings"](imagePart)).rejects.toThrow(
    /must return \[4, hidden_size\], got \[3, 8\]/,
  );

  // A uint32 adapter gets the same byte values in a wider tensor.
  raw["artifact"].imageInput.dtype = "uint32";
  raw["artifactImageAdapter"] = jest.fn(() => embeddings);
  await raw["getArtifactImageEmbeddings"](imagePart);
  expect(raw["tvm"].empty).toHaveBeenLastCalledWith(
    [1, 2, 3, 3],
    "uint32",
    raw["device"],
  );
  const bytes = pixelTensor.copyFromRawBytes.mock.calls.at(-1)?.[0];
  expect(bytes).toBeInstanceOf(Uint8Array);
  expect(Array.from(new Uint32Array((bytes as Uint8Array).buffer))).toEqual(
    Array.from(pixels),
  );
});

test("a text prompt is chunked when the config has a null system prefix", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  raw["artifact"] = {
    generation: { inputs: "embeds", prefill: "prefill", decode: "decode" },
  };
  pipeline["prefillChunkSize"] = 4;
  raw["conversation"].config.system_prefix_token_ids = null;
  pipeline["conversation"].getArtifactPromptSegments = jest.fn(() => ["text"]);
  pipeline["tokenizer"].encode = jest.fn(() =>
    Int32Array.from([1, 2, 3, 4, 5]),
  );

  const [chunks, promptLength] = await raw["getArtifactPrefillChunks"]();
  expect(promptLength).toBe(5);
  expect(chunks.map((chunk: any) => chunk.tokenIds)).toEqual([
    [1, 2, 3, 4],
    [5],
  ]);
});

test("a recording that cannot fit is rejected before its chunks are sliced", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const audioPart = {
    type: "input_audio" as const,
    input_audio: { format: "wav" as const, data: "" },
  };
  raw["artifact"] = {
    generation: { inputs: "tokens", prefill: "p", decode: "d" },
    audioInput: {
      adapter: "audio",
      prompt: {
        prefix_token_ids: [],
        placeholder_token_id: 99,
        suffix_token_ids: [],
      },
    },
  };
  pipeline["prefillChunkSize"] = 4;
  pipeline["contextWindowSize"] = 8;
  pipeline["slidingWindowSize"] = -1;
  raw["conversation"].config.system_prefix_token_ids = null;
  pipeline["conversation"].getArtifactPromptSegments = jest.fn(() => [
    "text",
    audioPart,
  ]);
  pipeline["tokenizer"].encode = jest.fn(() => Int32Array.from([1, 2, 3]));
  const embeddings = { shape: [6, 8], dispose: jest.fn() };
  raw["getArtifactAudioEmbeddings"] = jest.fn(() => embeddings);
  raw["sliceTensorRows"] = jest.fn();

  await expect(raw["getArtifactPrefillChunks"]()).rejects.toThrow(
    /exceed context window size/,
  );
  expect(embeddings.dispose).toHaveBeenCalled();
  expect(raw["sliceTensorRows"]).not.toHaveBeenCalled();
});

// The prefill scope disposes what is attached to it when it closes.
function trackScopes(raw: any): Array<Array<{ dispose: () => void }>> {
  const scopes: Array<Array<{ dispose: () => void }>> = [];
  raw["tvm"].beginScope = jest.fn(() => scopes.push([]));
  raw["tvm"].endScope = jest.fn(() =>
    scopes.pop()!.forEach((obj) => obj.dispose()),
  );
  raw["tvm"].attachToCurrentScope = jest.fn((obj: any) => {
    scopes.at(-1)!.push(obj);
    return obj;
  });
  raw["tvm"].detachFromCurrentScope = jest.fn((obj: any) => obj);
  return scopes;
}

// Guards the order that frees a rejected adapter output: it must be attached
// to the caller's scope before the shape check, and the adapter scope closed.
test.each([
  [
    "audio",
    (raw: any) => {
      raw["artifact"] = {
        audioInput: {
          processor: {
            kind: "audio_decode",
            format: "pcm_f32",
            sample_rate_hz: 16000,
            channels: 1,
            min_samples: 1,
            max_samples: 16,
          },
        },
      };
      raw["tvm"].empty = jest.fn(() => ({ copyFrom: jest.fn(() => ({})) }));
      return () =>
        raw["getArtifactAudioEmbeddings"]({
          type: "input_audio",
          input_audio: {
            format: "pcm_f32",
            sample_rate: 16000,
            data: new Float32Array(4),
          },
        });
    },
  ],
  [
    "image",
    (raw: any) => {
      raw["artifact"] = {
        imageInput: {
          processor: {
            resize: { mode: "stretch", height: 2, width: 3 },
            num_embeddings: 4,
          },
          dtype: "uint8",
        },
      };
      (
        jest.requireMock("../src/support") as any
      ).getResizedRGBArrayFromURL.mockResolvedValue(new Uint8ClampedArray(18));
      raw["tvm"].empty = jest.fn(() => ({ copyFrom: jest.fn() }));
      return () =>
        raw["getArtifactImageEmbeddings"]({
          type: "image_url",
          image_url: { url: "data:image/png;base64,AAAA" },
        });
    },
  ],
] as const)(
  "a rejected %s adapter output is freed with the caller's scope",
  async (_kind, arrange) => {
    const pipeline = createPipeline();
    const raw = pipeline as any;
    const scopes = trackScopes(raw);
    const rejected = { shape: [0, 8], dispose: jest.fn() };
    raw["artifactAudioAdapter"] = jest.fn(() => rejected);
    raw["artifactImageAdapter"] = jest.fn(() => rejected);
    const callAdapter = arrange(raw);

    raw["tvm"].beginScope(); // the prefill's scope
    await expect((async () => callAdapter())()).rejects.toThrow(
      /adapter must return/,
    );
    raw["tvm"].endScope();

    expect(rejected.dispose).toHaveBeenCalledTimes(1);
    expect(scopes).toHaveLength(0);
  },
);

test("audio prefill uses the sampled-step flow without text replay metadata", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  raw.artifact = {};
  const embeddings = { kind: "audio" };
  const logits = { dispose: jest.fn() };
  raw.getArtifactPrefillChunks = jest.fn(async () => [
    [{ tokenIds: [99, 99], modalityIds: [1, 1], embeddings }],
    2,
  ]);
  raw.artifactPrefillAndForward = jest.fn(() => {
    raw.filledKVCacheLength += 2;
    return logits;
  });

  const step = await pipeline.samplePrefillStep(
    [{ type: "input_audio", input_audio: { format: "wav", data: "UklGRg==" } }],
    Role.user,
  );
  expect(step.promptLen).toBe(2);
  expect(step.promptTokenIds).toBeUndefined();
  expect(step.tokenId).toBe(2);
  expect(raw.artifactPrefillAndForward).toHaveBeenCalledWith(
    embeddings,
    [99, 99],
    [1, 1],
  );
  expect(raw.embedAndForward).not.toHaveBeenCalled();
  expect(logits.dispose).toHaveBeenCalledTimes(1);
  expect(raw.tvm.endScope).toHaveBeenCalledTimes(1);
});

test("manifest text replay forwards known tokens through the declared ABI", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  raw.artifact = {};
  raw.prefillChunkSize = 2;
  raw.getArtifactTextEmbeddings = jest.fn((ids: number[]) => ({ ids }));
  const logits = { dispose: jest.fn() };
  raw.artifactPrefillAndForward = jest.fn(
    (_embeddings: unknown, ids: number[]) => {
      raw.filledKVCacheLength += ids.length;
      return logits;
    },
  );

  await expect(raw.forwardKnownTokens([4, 5, 6], true)).resolves.toBe(logits);
  expect(raw.artifactPrefillAndForward.mock.calls).toEqual([
    [{ ids: [4, 5] }, [4, 5], [0, 0]],
    [{ ids: [6] }, [6], [0]],
  ]);
  expect(raw.embedAndForward).not.toHaveBeenCalled();
  expect(raw.filledKVCacheLength).toBe(3);
  expect(raw.tvm.endScope).toHaveBeenCalledTimes(1);
});

test("manifest decode uses the sampled-step flow and closes its scope on failure", async () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  raw.artifact = {};
  raw.outputIds = [7];
  const logits = { dispose: jest.fn() };
  raw.artifactDecodeAndForward = jest.fn(() => {
    raw.filledKVCacheLength++;
    return logits;
  });
  await expect(pipeline.sampleDecodeStep()).resolves.toMatchObject({
    tokenId: 2,
  });
  expect(raw.artifactDecodeAndForward).toHaveBeenCalledWith(7);
  expect(raw.embedAndForward).not.toHaveBeenCalled();
  expect(logits.dispose).toHaveBeenCalledTimes(1);

  raw.artifactDecodeAndForward.mockImplementation(() => {
    throw new Error("decode failed");
  });
  await expect(pipeline.sampleDecodeStep()).rejects.toThrow("decode failed");
  expect(raw.tvm.endScope).toHaveBeenCalledTimes(2);
});

test("artifact prefill forwards the canonical prompt bundle", () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const logits = { kind: "logits" };
  const embeddings = {
    shape: [2, 8],
    view: jest.fn(() => ({ shape: [1, 2, 8] })),
  };
  const tensors: any[] = [];
  raw["tvm"].empty = jest.fn((shape: number[]) => {
    const tensor = { shape, copyFrom: jest.fn() };
    tensor.copyFrom.mockReturnValue(tensor);
    tensors.push(tensor);
    return tensor;
  });
  raw["tvm"].makeShapeTuple = jest.fn((shape: number[]) => shape);
  raw["tvm"].attachToCurrentScope = jest.fn();
  raw["prefill"] = jest.fn(() => ({
    get: jest.fn(() => logits),
  }));
  raw["kvCache"] = { kind: "kv" };
  raw["params"] = { kind: "params" };
  raw["fKVCacheBeginForward"] = jest.fn();
  raw["fKVCacheEndForward"] = jest.fn();

  const result = raw["artifactPrefillAndForward"](embeddings, [7, 99], [0, 1]);
  expect(result).toBe(logits);
  expect(tensors[0].copyFrom).toHaveBeenCalledWith([7, 99]);
  expect(tensors[1].copyFrom).toHaveBeenCalledWith([0, 1]);
  expect(raw["prefill"]).toHaveBeenCalledWith(
    { shape: [1, 2, 8] },
    tensors[0],
    tensors[1],
    pipeline["kvCache"],
    pipeline["params"],
  );
  expect(pipeline["filledKVCacheLength"]).toBe(2);
});

test("a library with the embedding roles is called without token IDs", () => {
  const pipeline = createPipeline();
  const raw = pipeline as any;
  const logits = { kind: "logits" };
  const prompt = { shape: [2, 8], view: jest.fn(() => ({ shape: [1, 2, 8] })) };
  const token = { shape: [1, 8], view: jest.fn(() => ({ shape: [1, 1, 8] })) };
  raw["artifact"] = {
    generation: { inputs: "embeds", prefill: "prefill", decode: "decode" },
  };
  raw["tvm"].empty = jest.fn();
  raw["tvm"].makeShapeTuple = jest.fn((shape: number[]) => shape);
  raw["tvm"].attachToCurrentScope = jest.fn();
  raw["prefill"] = jest.fn(() => ({ get: jest.fn(() => logits) }));
  raw["decoding"] = jest.fn(() => ({ get: jest.fn(() => logits) }));
  raw["getArtifactTextEmbeddings"] = jest.fn(() => token);
  raw["kvCache"] = { kind: "kv" };
  raw["params"] = { kind: "params" };
  raw["fKVCacheBeginForward"] = jest.fn();
  raw["fKVCacheEndForward"] = jest.fn();

  expect(raw["artifactPrefillAndForward"](prompt, [7, 99], [0, 1])).toBe(
    logits,
  );
  expect(raw["prefill"]).toHaveBeenCalledWith(
    { shape: [1, 2, 8] },
    pipeline["kvCache"],
    pipeline["params"],
  );

  expect(raw["artifactDecodeAndForward"](7)).toBe(logits);
  expect(raw["getArtifactTextEmbeddings"]).toHaveBeenCalledWith([7]);
  expect(raw["decoding"]).toHaveBeenCalledWith(
    { shape: [1, 1, 8] },
    pipeline["kvCache"],
    pipeline["params"],
  );
  expect(raw["tvm"].empty).not.toHaveBeenCalled();
  expect(pipeline["filledKVCacheLength"]).toBe(3);
});

test("processNextToken ignores eos when requested", () => {
  const pipeline = createPipeline();
  pipeline["stopTokens"] = [1];
  (pipeline as any).processNextToken(1, { ignore_eos: true });
  expect(pipeline["stopTriggered"]).toBe(false);
  expect(pipeline["finishReason"]).toBeUndefined();
  expect(pipeline["outputIds"]).toContain(1);
});

describe("calculateResizeShape", () => {
  test("phi3_v square image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["calculateResizeShape"](336, 336)).toEqual([1344, 1344]);
  });

  test("phi3_v landscape image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["calculateResizeShape"](1080, 1920)).toEqual([945, 1680]);
  });

  test("phi3_v portrait image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["calculateResizeShape"](1920, 1080)).toEqual([1194, 672]);
  });
});

describe("calculateCropShape", () => {
  test("phi3_v square image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["calculateCropShape"](336, 336)).toEqual([4, 4]);
  });

  test("phi3_v landscape image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["calculateCropShape"](1080, 1920)).toEqual([3, 5]);
  });

  test("phi3_v portrait image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["calculateCropShape"](1920, 1080)).toEqual([4, 2]);
  });
});

describe("computeImageEmbedSize", () => {
  test("phi3_v square image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["computeImageEmbedSize"](336, 336)).toBe(2509);
  });

  test("phi3_v landscape image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["computeImageEmbedSize"](1080, 1920)).toBe(2353);
  });

  test("phi3_v portrait image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "phi3_v" } as any;
    expect(pipeline["computeImageEmbedSize"](1920, 1080)).toBe(1357);
  });

  test("model with mm_tokens_per_image", () => {
    const pipeline = createPipeline();
    pipeline["config"] = {
      model_type: "gemma3_v",
      model_config: { mm_tokens_per_image: 256 },
    } as any;
    expect(pipeline["computeImageEmbedSize"](1080, 1920)).toBe(256);
  });

  test("unknown model without mm_tokens throws", () => {
    const pipeline = createPipeline();
    pipeline["config"] = { model_type: "unknown_model" } as any;
    expect(() => pipeline["computeImageEmbedSize"](336, 336)).toThrow(
      "Cannot determine image embed size",
    );
  });
});
