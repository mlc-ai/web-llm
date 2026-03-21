import { EmbeddingPipeline } from "../src/embedding";
import {
  EmbeddingExceedContextWindowSizeError,
  EmbeddingInputEmptyError,
} from "../src/error";
import { jest, test, expect } from "@jest/globals";

type EmbeddingLike = EmbeddingPipeline & Record<string, any>;

test("embedding pipeline performance getters", () => {
  const pipeline = Object.create(EmbeddingPipeline.prototype) as EmbeddingLike;
  pipeline["curRoundEmbedTotalTime"] = 0.5;
  pipeline["curRoundEmbedTotalTokens"] = 4;
  expect(pipeline.getCurRoundEmbedTotalTime()).toBe(0.5);
  expect(pipeline.getCurRoundEmbedTotalTokens()).toBe(4);
  expect(pipeline.getCurRoundEmbedTokensPerSec()).toBe(8);
});

test("sync and asyncLoadWebGPUPipelines delegate to tvm/device", async () => {
  const pipeline = Object.create(EmbeddingPipeline.prototype) as EmbeddingLike;
  const internalModule = { tag: "module" } as any;
  pipeline["device"] = {
    sync: jest.fn(async () => undefined),
  } as any;
  pipeline["tvm"] = {
    asyncLoadWebGPUPipelines: jest.fn(),
  } as any;
  pipeline["vm"] = {
    getInternalModule: jest.fn(() => internalModule),
  } as any;
  await pipeline.sync();
  expect(pipeline["device"].sync).toHaveBeenCalled();
  await pipeline.asyncLoadWebGPUPipelines();
  expect(pipeline["tvm"].asyncLoadWebGPUPipelines).toHaveBeenCalledWith(
    internalModule,
  );
});

function createEmbeddingPipelineBase(): EmbeddingLike {
  const pipeline = Object.create(EmbeddingPipeline.prototype) as EmbeddingLike;
  pipeline["tokenizer"] = {
    encode: jest.fn(
      (input: string) => new Int32Array(Math.max(1, input.length)),
    ),
    decode: jest.fn(),
    dispose: jest.fn(),
    getVocabSize: jest.fn(() => 1),
    idToToken: jest.fn(() => "<tok>"),
    handle: 0,
  } as any;
  pipeline["contextWindowSize"] = 8;
  pipeline["prefillChunkSize"] = 8;
  pipeline["maxBatchSize"] = 2;
  pipeline["device"] = {
    sync: jest.fn(async () => undefined),
    deviceType: "cpu",
    deviceId: 0,
    lib: {},
  } as any;
  pipeline["tvm"] = {
    beginScope: jest.fn(),
    endScope: jest.fn(),
    empty: jest.fn(() => createNDArray()),
    cpu: jest.fn(() => ({ deviceType: "cpu", deviceId: 0, lib: {} })),
    detachFromCurrentScope: jest.fn((x: any) => x),
  } as any;
  const packedFunc: any = jest.fn(() => ({
    shape: [1, 1, 1],
    dtype: "float32",
    dispose: jest.fn(),
    device: {},
    ndim: 3,
  }));
  packedFunc.dispose = jest.fn();
  pipeline["prefill"] = packedFunc;
  pipeline["params"] = {} as any;
  return pipeline;
}

function createNDArray() {
  const tensor: any = { dispose: jest.fn(), dtype: "int32", shape: [1, 1, 1] };
  tensor.copyFrom = jest.fn();
  tensor.view = jest.fn(() => tensor);
  tensor.toArray = jest.fn(() => new Float32Array([0.1]));
  return tensor;
}

test("embedStep throws when input is empty", async () => {
  const pipeline = createEmbeddingPipelineBase();
  await expect(pipeline.embedStep("")).rejects.toThrow(
    EmbeddingInputEmptyError,
  );
});

test("embedStep validates context window size", async () => {
  const pipeline = createEmbeddingPipelineBase();
  pipeline["contextWindowSize"] = 1;
  pipeline["tokenizer"].encode = jest.fn(() => new Int32Array([1, 2]));
  await expect(pipeline.embedStep("toolong")).rejects.toThrow(
    EmbeddingExceedContextWindowSizeError,
  );
});

test("embedStep returns mocked embeddings without WebGPU", async () => {
  const pipeline = createEmbeddingPipelineBase();
  const result = await pipeline.embedStep("ok");
  expect(result[0][0]).toBeCloseTo(0.1);
  expect(pipeline.getCurRoundEmbedTotalTokens()).toBeGreaterThan(0);
});
