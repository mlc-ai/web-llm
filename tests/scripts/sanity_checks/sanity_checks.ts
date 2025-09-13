import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) return;
  label.innerText = text;
}

async function createEngine(
  modelId: string,
  appConfig: webllm.AppConfig,
  logitProcessorRegistry?: Map<string, webllm.LogitProcessor>,
) {
  return await webllm.CreateMLCEngine(modelId, {
    appConfig,
    logLevel: "ERROR",
    logitProcessorRegistry,
  });
}

async function deleteModel(modelId: string, appConfig: webllm.AppConfig) {
  await webllm.deleteModelAllInfoInCache(modelId, appConfig);
}

async function testLogitProcessor(
  modelId: string,
  appConfig: webllm.AppConfig,
) {
  // Set up a logit processor that sets logits[0] = 100.0, rest -100.0
  const logitProcessor = {
    processLogits: (logits: Float32Array) => {
      logits.fill(-100.0);
      logits[0] = 100.0;
      return logits;
    },
    processSampledToken: () => {},
    resetState: () => {},
  };
  const logitProcessorRegistry: Map<string, webllm.LogitProcessor> = new Map();
  logitProcessorRegistry.set(modelId, logitProcessor);
  const engine: webllm.MLCEngineInterface = await createEngine(
    modelId,
    appConfig,
    logitProcessorRegistry,
  );

  const prompt = "Test logit processor.";
  const reply: webllm.ChatCompletion = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 1.0,
    max_tokens: 20,
    logprobs: true,
    top_logprobs: 1,
  });
  const logprobs = reply.choices[0]?.logprobs;
  const logprobsAllZero = !!(
    logprobs &&
    Array.isArray(logprobs.content) &&
    logprobs.content.every(
      (lp: webllm.ChatCompletionTokenLogprob) =>
        lp.top_logprobs[0].logprob === 0,
    )
  );

  console.log(`[LogitProcessor] Logprobs all zero: ${logprobsAllZero}`);
  setLabel("logit-processor-label", `Logprobs all zero: ${logprobsAllZero}`);
  await deleteModel(modelId, appConfig);
  return logprobsAllZero;
}

async function testLogitBias(modelId: string, appConfig: webllm.AppConfig) {
  // Set logit_bias to strongly favor token 0
  const prompt = "Test logit bias.";
  const engine: webllm.MLCEngineInterface = await createEngine(
    modelId,
    appConfig,
  );
  const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 1.0,
    max_tokens: 20,
    logprobs: true,
    top_logprobs: 1,
    logit_bias: { "0": 100.0 },
  });
  const logprobs = reply.choices[0]?.logprobs;
  const logprobsAllZero = !!(
    logprobs &&
    Array.isArray(logprobs.content) &&
    logprobs.content.every(
      (lp: webllm.ChatCompletionTokenLogprob) =>
        lp.top_logprobs[0].logprob === 0,
    )
  );

  console.log(`[LogitBias] Logprobs all zero: ${logprobsAllZero}`);
  setLabel("logit-bias-label", `Logprobs all zero: ${logprobsAllZero}`);
  await deleteModel(modelId, appConfig);
  return logprobsAllZero;
}

async function testPenalties(modelId: string, appConfig: webllm.AppConfig) {
  const prompt = "Test presence and frequency penalties.";
  const engine: webllm.MLCEngineInterface = await createEngine(
    modelId,
    appConfig,
  );
  const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 1.0,
    max_tokens: 256,
    presence_penalty: 2.0,
    frequency_penalty: 2.0,
    logit_bias: { "0": 100.0 },
    logprobs: true,
  });
  const logprobs = reply.choices[0]?.logprobs;
  const logprobsNotAllZero = !logprobs?.content?.every(
    (lp: webllm.ChatCompletionTokenLogprob) => lp.logprob === 0,
  );
  console.log(`[Penalties] Logprobs not all zero: ${logprobsNotAllZero}`);
  setLabel("penalty-label", `Logprobs not all zero: ${logprobsNotAllZero}`);
  await deleteModel(modelId, appConfig);
  return logprobsNotAllZero;
}

async function testLogprobs(modelId: string, appConfig: webllm.AppConfig) {
  // Test logprobs: check that logprobs are returned and sum to ~1 after exp
  const prompt = "Test logprobs.";
  const engine: webllm.MLCEngineInterface = await createEngine(
    modelId,
    appConfig,
  );
  const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    temperature: 1.0,
    max_tokens: 20,
    logprobs: true,
    top_logprobs: 5,
  });
  const logprobs = reply.choices[0]?.logprobs;

  let logprobsAllCloseTo1 = true;
  for (const lp of logprobs?.content || []) {
    const expSum = lp.top_logprobs?.reduce(
      (acc: number, val: webllm.TopLogprob) => acc + Math.exp(val.logprob),
      0,
    );
    logprobsAllCloseTo1 &&= Math.abs(expSum - 1.0) < 0.1;
  }
  console.log(`[Logprobs] Logprobs all close to 1: ${logprobsAllCloseTo1}`);
  setLabel("logprobs-label", `Logprobs all close to 1: ${logprobsAllCloseTo1}`);
  await deleteModel(modelId, appConfig);
  return logprobsAllCloseTo1;
}

async function main() {
  const modelId = "Qwen3-0.6B-q0f32-MLC";
  const appConfig = webllm.prebuiltAppConfig;
  appConfig.useIndexedDBCache = true;
  setLabel("gpu-test-label", "Running tests...");
  let passed = 0,
    total = 0;

  if (await testLogitProcessor(modelId, appConfig)) passed++;
  total++;
  if (await testLogitBias(modelId, appConfig)) passed++;
  total++;
  if (await testPenalties(modelId, appConfig)) passed++;
  total++;
  if (await testLogprobs(modelId, appConfig)) passed++;
  total++;

  setLabel(
    "gpu-test-label",
    `GPU sampleTokenFromLogits tests: ${passed}/${total} passed.`,
  );
  setLabel(
    "gpu-test-label",
    `Tests complete. Model deleted. ${passed}/${total} passed.`,
  );
}

main();
