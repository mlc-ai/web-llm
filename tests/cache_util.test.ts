import {
  asyncLoadTokenizer,
  deleteModelInCache,
  hasModelInCache,
} from "../src/cache_util";
import { AppConfig } from "../src/config";
import * as tvmMockImport from "@mlc-ai/web-runtime";
import * as tokenizerMockImport from "@mlc-ai/web-tokenizers";
import { jest, test, expect, beforeEach } from "@jest/globals";

jest.mock("@mlc-ai/web-runtime", () => {
  const state = {
    hasTensorInCache: jest
      .fn<() => Promise<boolean>>()
      .mockResolvedValue(false),
    deleteTensorCache: jest.fn(),
    deletes: [] as Array<{ cache: string; url: string }>,
    fetches: [] as Array<{ cache: string; url: string; format: string }>,
  };
  class BaseCache {
    constructor(private name: string) {}
    async deleteInCache(url: string) {
      state.deletes.push({ cache: this.name, url });
    }
    async fetchWithCache(url: string, format: string) {
      state.fetches.push({ cache: this.name, url, format });
      return new ArrayBuffer(4);
    }
  }
  return {
    hasTensorInCache: state.hasTensorInCache,
    deleteTensorCache: state.deleteTensorCache,
    ArtifactCache: BaseCache,
    ArtifactIndexedDBCache: BaseCache,
    __cacheState: state,
  };
});

jest.mock("@mlc-ai/web-tokenizers", () => {
  return {
    Tokenizer: {
      fromJSON: jest.fn(() => ({ kind: "json" })),
      fromSentencePiece: jest.fn(() => ({ kind: "sp" })),
    },
  };
});

const tvmMock = tvmMockImport as any;
const tokenizerMock = tokenizerMockImport as any;

const baseAppConfig: AppConfig = {
  useIndexedDBCache: false,
  model_list: [
    {
      model: "https://huggingface.co/mlc-ai/demo-model",
      model_id: "demo-model",
      model_lib: "https://example.com/model.wasm",
    },
  ],
};

beforeEach(() => {
  tvmMock.__cacheState.deletes.length = 0;
  tvmMock.__cacheState.fetches.length = 0;
  tvmMock.__cacheState.hasTensorInCache.mockClear();
  tvmMock.__cacheState.deleteTensorCache.mockClear();
  tokenizerMock.Tokenizer.fromJSON.mockClear();
  tokenizerMock.Tokenizer.fromSentencePiece.mockClear();
});

test("hasModelInCache delegates to tvm cache helpers", async () => {
  tvmMock.__cacheState.hasTensorInCache.mockResolvedValueOnce(true);
  const result = await hasModelInCache("demo-model", baseAppConfig);
  expect(result).toBe(true);
  expect(tvmMock.__cacheState.hasTensorInCache).toHaveBeenCalledWith(
    "https://huggingface.co/mlc-ai/demo-model/resolve/main/",
    "webllm/model",
    "cache",
  );
});

test("deleteModelInCache clears tensors and tokenizer assets for indexeddb cache", async () => {
  const indexedConfig: AppConfig = {
    ...baseAppConfig,
    useIndexedDBCache: true,
  };
  await deleteModelInCache("demo-model", indexedConfig);
  expect(tvmMock.__cacheState.deleteTensorCache).toHaveBeenCalledWith(
    "https://huggingface.co/mlc-ai/demo-model/resolve/main/",
    "webllm/model",
    "indexeddb",
  );
  expect(tvmMock.__cacheState.deletes).toEqual(
    expect.arrayContaining([
      {
        cache: "webllm/model",
        url: "https://huggingface.co/mlc-ai/demo-model/resolve/main/tokenizer.model",
      },
      {
        cache: "webllm/model",
        url: "https://huggingface.co/mlc-ai/demo-model/resolve/main/tokenizer.json",
      },
    ]),
  );
});

test("asyncLoadTokenizer prefers tokenizer.json and falls back to sentencepiece", async () => {
  const makeChatConfig = (files: string[]) =>
    ({
      tokenizer_files: files,
    }) as unknown as import("../src/config").ChatConfig;

  const configJson = makeChatConfig(["tokenizer.json"]);
  await asyncLoadTokenizer(
    baseAppConfig.model_list[0].model,
    configJson,
    baseAppConfig,
  );
  expect(tokenizerMock.Tokenizer.fromJSON).toHaveBeenCalled();
  expect(tokenizerMock.Tokenizer.fromSentencePiece).not.toHaveBeenCalled();
  expect(tvmMock.__cacheState.fetches[0]).toEqual({
    cache: "webllm/model",
    url: "https://huggingface.co/mlc-ai/tokenizer.json",
    format: "arraybuffer",
  });

  const configSp = makeChatConfig(["tokenizer.model"]);
  await asyncLoadTokenizer(
    baseAppConfig.model_list[0].model,
    configSp,
    baseAppConfig,
  );
  expect(tokenizerMock.Tokenizer.fromSentencePiece).toHaveBeenCalled();
});
