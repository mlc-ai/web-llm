import {
  MODEL_ID,
  SECOND_MODEL_ID,
  createEngineWithPipeline,
  createEngineWithMultiplePipelines,
  createEngineWithEmbeddingPipeline,
} from "./helpers/engine_fixture";
import {
  ModelNotLoadedError,
  SpecifiedModelNotFoundError,
  UnclearModelToUseError,
} from "../src/error";
import {
  WebWorkerMLCEngine,
  WebWorkerMLCEngineHandler,
} from "../src/web_worker";
import { expect, jest, test } from "@jest/globals";

function tokenizer(offset: number) {
  return {
    encode: jest.fn((text: string) =>
      Int32Array.from(
        Array.from(text),
        (char) => char.codePointAt(0)! + offset,
      ),
    ),
    decode: jest.fn((ids: Int32Array) => {
      expect(ids).toBeInstanceOf(Int32Array);
      return Array.from(ids, (id) => String.fromCodePoint(id - offset)).join(
        "",
      );
    }),
  };
}

test("tokenization uses the selected model and rejects ambiguous or missing models", async () => {
  const engine = createEngineWithMultiplePipelines();
  const pipelines = (engine as any).loadedModelIdToPipeline;
  pipelines.get(MODEL_ID).getTokenizer = () => tokenizer(0);
  pipelines.get(SECOND_MODEL_ID).getTokenizer = () => tokenizer(100);
  const text = "Hello 世界 🦙";
  const first = await engine.tokenize(text, MODEL_ID);
  const second = await engine.tokenize(text, SECOND_MODEL_ID);
  expect(second).toEqual(first.map((id) => id + 100));
  expect(await engine.decodeTokens(first, MODEL_ID)).toBe(text);
  expect(await engine.decodeTokens(second, SECOND_MODEL_ID)).toBe(text);
  await expect(engine.tokenize(text)).rejects.toBeInstanceOf(
    UnclearModelToUseError,
  );
  await expect(engine.decodeTokens(first)).rejects.toBeInstanceOf(
    UnclearModelToUseError,
  );
  await expect(engine.tokenize(text, "missing")).rejects.toBeInstanceOf(
    SpecifiedModelNotFoundError,
  );
  await expect(engine.decodeTokens(first, "missing")).rejects.toBeInstanceOf(
    SpecifiedModelNotFoundError,
  );
  await engine.unload();
  await expect(engine.tokenize(text, MODEL_ID)).rejects.toBeInstanceOf(
    ModelNotLoadedError,
  );
  await expect(engine.decodeTokens(first, MODEL_ID)).rejects.toBeInstanceOf(
    ModelNotLoadedError,
  );
});

test.each([createEngineWithPipeline, createEngineWithEmbeddingPipeline])(
  "a single chat or embedding model can tokenize without a model ID",
  async (createEngine) => {
    const { engine, pipeline } = createEngine();
    pipeline.getTokenizer = () => tokenizer(0);
    const ids = await engine.tokenize("raw text");
    expect(await engine.decodeTokens(ids)).toBe("raw text");
    expect(await engine.tokenize("")).toEqual(new Int32Array());
    expect(await engine.decodeTokens(new Int32Array())).toBe("");
  },
);

test.each(["structured clone", "JSON"])(
  "tokenizer requests preserve model IDs and arrays over %s messaging",
  async (transport) => {
    const engine = createEngineWithMultiplePipelines();
    const pipelines = (engine as any).loadedModelIdToPipeline;
    pipelines.get(MODEL_ID).getTokenizer = () => tokenizer(0);
    pipelines.get(SECOND_MODEL_ID).getTokenizer = () => tokenizer(100);
    const copy = (value: any) =>
      transport === "JSON"
        ? JSON.parse(JSON.stringify(value))
        : structuredClone(value);
    const handler = new WebWorkerMLCEngineHandler();
    (handler as any).engine = engine;
    const worker = {
      onmessage: jest.fn<(event: any) => void>(),
      postMessage: (message: any) => handler.onmessage(copy(message)),
    };
    handler.postMessage = (message: any) =>
      worker.onmessage(
        transport === "JSON"
          ? copy(message)
          : new MessageEvent("message", { data: copy(message) }),
      );
    const client = new WebWorkerMLCEngine(worker);
    const text = "Hello 世界 🦙";
    for (const model of [MODEL_ID, SECOND_MODEL_ID]) {
      const ids = await client.tokenize(text, model);
      expect(ids).toBeInstanceOf(Int32Array);
      expect(ids).toEqual(await engine.tokenize(text, model));
      expect(await client.decodeTokens(ids, model)).toBe(text);
    }
    await expect(client.tokenize(text)).rejects.toBeDefined();
    await expect(
      client.decodeTokens(new Int32Array(), "missing"),
    ).rejects.toBeDefined();
    await engine.unload();
    await expect(client.tokenize(text, MODEL_ID)).rejects.toBeDefined();
  },
);
