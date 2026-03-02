import { LLMChatPipeline } from "../src/llm_chat";
import { MinValueError } from "../src/error";
import { Role } from "../src/config";
import { jest, test, expect, beforeEach } from "@jest/globals";

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
  return {
    TokenizerInfo: {
      createTokenizerInfo: jest.fn(async () => "tokenInfo"),
    },
    GrammarCompiler: {
      createGrammarCompiler: jest.fn(async () => ({
        compileBuiltinJSONGrammar,
        compileJSONSchema,
        compileGrammar,
      })),
      __compileBuiltinJSONGrammar: compileBuiltinJSONGrammar,
      __compileJSONSchema: compileJSONSchema,
      __compileGrammar: compileGrammar,
    },
    GrammarMatcher: {
      createGrammarMatcher: jest.fn(async () => {
        const matcher = { dispose: jest.fn(), reset: jest.fn() };
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

beforeEach(() => {
  grammarMatcherInstances.length = 0;
  compileGrammarMock.mockClear();
  compileJSONSchemaMock.mockClear();
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
  pipeline["sampleTokenFromLogits"] = jest.fn(async () => 2);
  pipeline["resetRuntimeStats"] = jest.fn();
  pipeline["resetStatsPerPrefill"] = false;
  pipeline["prefillTotalTime"] = 0;
  pipeline["prefillTotalTokens"] = 0;
  pipeline["curRoundPrefillTotalTokens"] = 0;
  pipeline["curRoundPrefillTotalTime"] = 0;
  pipeline["curRoundGrammarInitTotalTime"] = 0;
  pipeline["curRoundGrammarPerTokenTotalTime"] = 0;
  pipeline["tokenLogprobArray"] = [];
  pipeline["curRoundDecodingTotalTokens"] = 0;
  pipeline["curRoundDecodingTotalTime"] = 0;
  return pipeline;
}

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

test("processNextToken respects max_tokens and updates token frequency", () => {
  const pipeline = createPipeline();
  (pipeline as any).processNextToken(7, { max_tokens: 1 });
  expect(pipeline["appearedTokensFreq"].get(7)).toBe(1);
  expect(pipeline["finishReason"]).toBe("length");
});

test("processNextToken throws when max_tokens is below zero", () => {
  const pipeline = createPipeline();
  expect(() =>
    (pipeline as any).processNextToken(1, { max_tokens: -1 }),
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
  pipeline["getInputData"] = jest.fn<() => [number[][], number]>(() => [
    [[0]],
    1,
  ]);
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

test("prefillStep reuses grammar matcher when schema unchanged", async () => {
  const pipeline = preparePrefillPipeline();
  const matcher = { reset: jest.fn(), dispose: jest.fn() };
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

test("getInputData uses cached prompts when KV cache filled", () => {
  const pipeline = createPipeline();
  pipeline["tokenizer"].encode = jest.fn(() => Int32Array.from([1]));
  pipeline["conversation"].config.system_prefix_token_ids = undefined;
  pipeline["filledKVCacheLength"] = 0;
  (pipeline as any).getInputData();
  expect(pipeline["conversation"].getPromptArray).toHaveBeenCalled();
  pipeline["filledKVCacheLength"] = 1;
  (pipeline as any).getInputData();
  expect(pipeline["conversation"].getPromptArrayLastRound).toHaveBeenCalled();
});

test("processNextToken ignores eos when requested", () => {
  const pipeline = createPipeline();
  pipeline["stopTokens"] = [1];
  (pipeline as any).processNextToken(1, { ignore_eos: true });
  expect(pipeline["stopTriggered"]).toBe(false);
  expect(pipeline["finishReason"]).toBeUndefined();
  expect(pipeline["outputIds"]).toContain(1);
});
