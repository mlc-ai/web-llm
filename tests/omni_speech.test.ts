import { readFileSync } from "fs";
import { join } from "path";
import { webcrypto } from "crypto";
import { createOmniSpeechGraph, OmniSpeechParts } from "../src/omni_speech";
import {
  GraphComponentFactory,
  GraphTensor,
} from "../src/artifact_graph_executor";
beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  }),
);
const fixture = (name: string) =>
  JSON.parse(
    readFileSync(
      join(__dirname, "fixtures/omni_speech", name + ".json"),
      "utf8",
    ),
  );
async function setup(audio = false, embeddingSize = 2048) {
  const received: { name: string; values: Record<string, GraphTensor> }[] = [];
  const disposed: string[] = [],
    created: string[] = [];
  const faults = { flow: false, thinkerTokens: 2 };
  const parts = {} as OmniSpeechParts;
  for (const name of [
    "thinker",
    "talker",
    "reference",
    "flow",
    "vocoder",
    ...(audio ? ["features", "audio"] : []),
  ]) {
    const data =
      embeddingSize === 3584 && ["thinker", "talker", "audio"].includes(name)
        ? fixture(name + "-7b")
        : name === "thinker"
          ? Object.fromEntries(
              ["model", "compiled"].map((key) => [
                key,
                JSON.parse(
                  readFileSync(
                    join(
                      __dirname,
                      "fixtures/thinker_speech_tvm",
                      key + ".json",
                    ),
                    "utf8",
                  ),
                ),
              ]),
            )
          : fixture(name);
    const componentId = Object.keys(data.compiled.components)[0];
    const factory: GraphComponentFactory = {
      create(id, component, signal) {
        expect(id).toBe(componentId);
        expect(component).toEqual(data.compiled.components[componentId]);
        created.push(name);
        return {
          dispose() {
            disposed.push(name);
          },
          async *run(events) {
            const values: Record<string, GraphTensor> = {};
            for await (const event of events)
              if (event.kind === "data") values[event.port] = event.tensor;
            signal.throwIfAborted();
            received.push({ name, values });
            if (name === "thinker") {
              for (let i = 0; i < faults.thinkerTokens; i++)
                yield {
                  port: "token_ids",
                  tensor: { shape: [1], data: new Int32Array([7 + i]) },
                };
              yield {
                port: "prefix_embeddings",
                tensor: {
                  shape: [4, embeddingSize],
                  data: new Float32Array(4 * embeddingSize),
                },
              };
              yield {
                port: "reply_embeddings",
                tensor: {
                  shape: [3, embeddingSize],
                  data: new Float32Array(3 * embeddingSize),
                },
              };
              yield {
                port: "finish_reason",
                tensor: { shape: [1], data: new Int32Array([0]) },
              };
            } else if (name === "talker")
              yield {
                port: "code",
                tensor: { shape: [2], data: new Int32Array([3, 4]) },
              };
            else if (name === "reference")
              yield {
                port: "embedding",
                tensor: {
                  shape: [128],
                  data: new Float32Array(128).fill(values.mel.data[0]),
                },
              };
            else if (name === "flow") {
              if (faults.flow) throw Error("original flow failure");
              yield {
                port: "velocity",
                tensor: { shape: [4, 80], data: new Float32Array(320).fill(1) },
              };
            } else if (name === "vocoder")
              yield {
                port: "pcm",
                tensor: {
                  shape: [960],
                  data: new Float32Array(960).fill(0.25),
                },
              };
            else if (name === "features")
              yield {
                port: "mel_features",
                tensor: { shape: [128, 4], data: new Float32Array(512) },
              };
            else if (name === "audio")
              yield {
                port: "embeddings",
                tensor: {
                  shape: [1, embeddingSize],
                  data: new Float32Array(embeddingSize),
                },
              };
          },
        };
      },
    };
    (parts as any)[name] = { ...data, componentId, factory };
  }
  const options = {
    parts,
    maxFrames: 4,
    timePoints: new Float32Array([0, 1]),
    guidanceScale: 0.5,
  };
  const graph = await createOmniSpeechGraph(options);
  return { graph, parts, options, received, created, disposed, faults };
}
const stream = (tensor: GraphTensor) =>
  (async function* () {
    yield tensor;
  })();
function inputs(audio = false) {
  return {
    token_ids: stream({
      shape: [2],
      data: new Int32Array(audio ? [3, 151646] : [3, 4]),
    }),
    reference_mel: stream({
      shape: [5, 80],
      data: new Float32Array(400).fill(0.5),
    }),
    speaker: stream({ shape: [192], data: new Float32Array(192) }),
    noise: stream({ shape: [4, 80], data: new Float32Array(320) }),
    ...(audio
      ? { pcm: stream({ shape: [640], data: new Float32Array(640) }) }
      : {}),
  };
}
async function consume(
  execution: Awaited<
    ReturnType<Awaited<ReturnType<typeof setup>>["graph"]["execute"]>
  >,
) {
  const done = execution.done.catch((error) => {
    throw error;
  });
  return Promise.all([
    Promise.all(
      Object.entries(execution.outputs).map(async ([name, stream]) => {
        const values = [];
        for await (const value of stream) values.push(value);
        return [name, values] as const;
      }),
    ).then((x) => Object.fromEntries(x)),
    done,
  ]).then(([values]) => values);
}
test("public text task runs actual graph routes and sequential reference/flow calls", async () => {
  const s = await setup();
  const result = await consume(
    await s.graph.execute({ mode: "text", inputs: inputs(), queueCapacity: 1 }),
  );
  expect(result.token_ids.map((x) => x.data[0])).toEqual([7, 8]);
  expect(result.finish_reason[0].data[0]).toBe(0);
  expect(result.pcm[0].data.length).toBe(960);
  expect(
    s.received.find((x) => x.name === "thinker")!.values.audio_embeddings.shape,
  ).toEqual([0, 2048]);
  expect(
    s.received
      .filter((x) => x.name === "reference")
      .map((x) => x.values.mel.data[0]),
  ).toEqual([0.5, 0]);
  expect(s.received.filter((x) => x.name === "flow")).toHaveLength(8);
  expect(
    s.received.find((x) => x.name === "vocoder")!.values.log_mel.shape,
  ).toEqual([80, 4]);
  expect(
    s.received
      .find((x) => x.name === "vocoder")!
      .values.log_mel.data.every((x) => x === 1),
  ).toBe(true);
  expect(s.created.slice().sort()).toEqual(s.disposed.slice().sort());
});
test("audio task binds frontend and encoder outputs rather than empty embeddings", async () => {
  const s = await setup(true);
  await consume(await s.graph.execute({ mode: "audio", inputs: inputs(true) }));
  expect(
    s.received.find((x) => x.name === "thinker")!.values.audio_embeddings.shape,
  ).toEqual([1, 2048]);
  expect(s.created).toContain("features");
  expect(s.created).toContain("audio");
});
test("3B and 7B manifests run concurrently with their own embedding widths", async () => {
  const [small, large] = await Promise.all([setup(true), setup(true, 3584)]);
  await Promise.all(
    [small, large].map(async (s) =>
      consume(await s.graph.execute({ mode: "audio", inputs: inputs(true) })),
    ),
  );
  for (const [s, width] of [
    [small, 2048],
    [large, 3584],
  ] as const) {
    const thinker = s.received.find((x) => x.name === "thinker")!;
    const talker = s.received.find((x) => x.name === "talker")!;
    expect(thinker.values.audio_embeddings.shape).toEqual([1, width]);
    expect(talker.values.prefix_embeddings.shape).toEqual([4, width]);
    expect(talker.values.reply_embeddings.shape).toEqual([3, width]);
  }
});
test("rejects mismatched Omni profile manifests before opening components", async () => {
  const [small, large] = await Promise.all([setup(true), setup(true, 3584)]);
  await expect(
    createOmniSpeechGraph({
      ...large.options,
      parts: { ...large.parts, talker: small.parts.talker },
    }),
  ).rejects.toThrow(/incompatible|shape/i);
  expect(small.created).toHaveLength(0);
  expect(large.created).toHaveLength(0);
});
test("streams more text than queue capacity while speech completes", async () => {
  const s = await setup();
  s.faults.thinkerTokens = 35;
  const result = await consume(
    await s.graph.execute({ mode: "text", inputs: inputs(), queueCapacity: 1 }),
  );
  expect(result.token_ids).toHaveLength(35);
});
test("nested flow failure preserves the original error and disposes all created state", async () => {
  const s = await setup();
  s.faults.flow = true;
  await expect(
    consume(await s.graph.execute({ mode: "text", inputs: inputs() })),
  ).rejects.toThrow("original flow failure");
  expect(s.created.slice().sort()).toEqual(s.disposed.slice().sort());
});
test("aborted outer request cannot start synthesis", async () => {
  const s = await setup();
  const controller = new AbortController();
  controller.abort();
  await expect(
    s.graph.execute({
      mode: "text",
      inputs: inputs(),
      signal: controller.signal,
    }),
  ).rejects.toThrow();
  expect(s.created).toHaveLength(0);
});
test("snapshots synthesis policy before callers mutate it", async () => {
  const s = await setup();
  s.options.timePoints[1] = NaN;
  s.options.guidanceScale = -1;
  await consume(await s.graph.execute({ mode: "text", inputs: inputs() }));
});
test("text task rejects injected audio embeddings", async () => {
  const s = await setup();
  expect(() =>
    s.graph.execute({
      mode: "text",
      inputs: {
        ...inputs(),
        audio_embeddings: stream({
          shape: [1, 2048],
          data: new Float32Array(2048),
        }),
      },
    }),
  ).toThrow("own empty");
});
