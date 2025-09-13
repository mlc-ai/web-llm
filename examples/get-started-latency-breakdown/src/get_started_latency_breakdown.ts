import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

type LatencyBreakdown = {
  logitProcessorTime: number[];
  logitBiasTime: number[];
  penaltyTime: number[];
  sampleTime: number[];
  totalTime: number[];
  grammarBitmaskTime: number[];
};
function computeStats(
  latency_breakdown: LatencyBreakdown,
): Record<string, any> {
  function _computeStats(arr: number[]) {
    if (!arr.length) return undefined;
    const sorted = [...arr].sort((a, b) => a - b);
    const sum = arr.reduce((a, b) => a + b, 0);
    const avg = sum / arr.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const p99 = sorted[Math.floor(0.99 * (sorted.length - 1))];
    return { avg, min, max, p99 };
  }

  const latencyStats: Record<string, any> = {};
  for (const key of Object.keys(latency_breakdown)) {
    const arr = (latency_breakdown as any)[key];
    if (Array.isArray(arr) && arr.length > 0) {
      latencyStats[key] = _computeStats(arr);
    }
  }
  return latencyStats;
}

async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  // Option 1: If we do not specify appConfig, we use `prebuiltAppConfig` defined in `config.ts`
  const selectedModel = "Qwen3-0.6B-q0f32-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
      logLevel: "INFO", // specify the log level
    },
    // customize kv cache, use either context_window_size or sliding_window_size (with attention sink)
    {
      context_window_size: 2048,
      // sliding_window_size: 1024,
      // attention_sink_size: 4,
    },
  );

  const latencyBreakdown: LatencyBreakdown = {
    logitProcessorTime: [],
    logitBiasTime: [],
    penaltyTime: [],
    sampleTime: [],
    totalTime: [],
    grammarBitmaskTime: [],
  };

  const decodeTokensPerS: number[] = [];
  const completionTokens: number[] = [];
  const e2eLatencyS: number[] = [];
  const timePerOutputTokenS: number[] = [];

  const numTrials = 20;
  for (let i = 0; i < numTrials; i++) {
    console.log(`Trial ${i + 1} / ${numTrials}`);
    const reply0 = await engine.chat.completions.create({
      messages: [{ role: "user", content: "List twenty US states." }],
      // below configurations are all optional
      n: 1,
      temperature: 0,
      max_tokens: 2048,
      // 46510 and 7188 are "California", and 8421 and 51325 are "Texas" in Llama-3.1-8B-Instruct
      // So we would have a higher chance of seeing the latter two, but never the first in the answer
      // logit_bias: {
      //   "46510": -100,
      //   "7188": -100,
      //   "8421": 5,
      //   "41325": 5,
      // },
      top_p: 0.8,
      logprobs: true,
      top_logprobs: 2,
      frequency_penalty: 1.2,
      presence_penalty: 1.0,
      repetition_penalty: 1.1,
    });

    const logitProcessorTime =
      reply0.usage?.extra.latencyBreakdown?.logitProcessorTime;
    const logitBiasTime = reply0.usage?.extra.latencyBreakdown?.logitBiasTime;
    const penaltyTime = reply0.usage?.extra.latencyBreakdown?.penaltyTime;
    const sampleTime = reply0.usage?.extra.latencyBreakdown?.sampleTime;
    const totalTime = reply0.usage?.extra.latencyBreakdown?.totalTime;
    const grammarBitmaskTime =
      reply0.usage?.extra.latencyBreakdown?.grammarBitmaskTime;

    latencyBreakdown.logitProcessorTime.push(...(logitProcessorTime || []));
    latencyBreakdown.logitBiasTime.push(...(logitBiasTime || []));
    latencyBreakdown.penaltyTime.push(...(penaltyTime || []));
    latencyBreakdown.sampleTime.push(...(sampleTime || []));
    latencyBreakdown.totalTime.push(...(totalTime || []));
    latencyBreakdown.grammarBitmaskTime.push(...(grammarBitmaskTime || []));

    decodeTokensPerS.push(reply0.usage?.extra.decode_tokens_per_s || 0);
    e2eLatencyS.push(reply0.usage?.extra.e2e_latency_s || 0);
    timePerOutputTokenS.push(reply0.usage?.extra.time_per_output_token_s || 0);
    completionTokens.push(reply0.usage?.completion_tokens || 0);
  }

  const latencyStats: { [key: string]: number } =
    computeStats(latencyBreakdown);
  console.log("Latency stats: ", latencyStats);
  console.log("Decode tokens per second: ", decodeTokensPerS);
  console.log("Completion tokens: ", completionTokens);
  console.log("E2E latency (s): ", e2eLatencyS);
  console.log("Time per output token (s): ", timePerOutputTokenS);

  // To change model, either create a new engine via `CreateMLCEngine()`, or call `engine.reload(modelId)`
}

main();
