import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

// Models to test: uncomment the specific ones you want to test
const TEST_MODELS = [
  // Llama 2 7B
  // "Llama-2-7b-chat-hf-q4f16_1-MLC",
  // "Llama-2-7b-chat-hf-q4f32_1-MLC",

  // // Llama 3 8B
  // "Llama-3-8B-Instruct-q4f16_1-MLC",
  // "Llama-3-8B-Instruct-q4f32_1-MLC",

  // // Llama 3.1 8B
  // "Llama-3.1-8B-Instruct-q4f16_1-MLC",
  // "Llama-3.1-8B-Instruct-q4f32_1-MLC",

  // // Llama 3.2 1B, 3B
  // "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  // "Llama-3.2-1B-Instruct-q4f32_1-MLC",
  // "Llama-3.2-3B-Instruct-q4f16_1-MLC",
  // "Llama-3.2-3B-Instruct-q4f32_1-MLC",

  // // Mistral 7B v0.3
  // "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
  // "Mistral-7B-Instruct-v0.3-q4f32_1-MLC",

  // // Phi models
  // "phi-1_5-q4f16_1-MLC",
  // "phi-1_5-q4f32_1-MLC",
  // "phi-2-q4f16_1-MLC",
  // "phi-2-q4f32_1-MLC",
  // "Phi-3-mini-4k-instruct-q4f16_1-MLC",
  // "Phi-3-mini-4k-instruct-q4f32_1-MLC",
  // "Phi-3.5-mini-instruct-q4f16_1-MLC",
  // "Phi-3.5-mini-instruct-q4f32_1-MLC",

  // // Qwen2
  "Qwen2-0.5B-Instruct-q4f16_1-MLC",
  // "Qwen2-0.5B-Instruct-q4f32_1-MLC",
  // "Qwen2-1.5B-Instruct-q4f16_1-MLC",
  // "Qwen2-1.5B-Instruct-q4f32_1-MLC",

  // // Qwen2.5
  // "Qwen2.5-3B-Instruct-q4f16_1-MLC",
  // "Qwen2.5-3B-Instruct-q4f32_1-MLC",

  // // Qwen3 (including q0 for 0.6B)
  // "Qwen3-0.6B-q4f16_1-MLC",
  // "Qwen3-0.6B-q4f32_1-MLC",
  // "Qwen3-0.6B-q0f32-MLC",
  // "Qwen3-1.7B-q4f16_1-MLC",
  // "Qwen3-1.7B-q4f32_1-MLC",
  // "Qwen3-4B-q4f16_1-MLC",
  // "Qwen3-4B-q4f32_1-MLC",
  // "Qwen3-8B-q4f16_1-MLC",
  // "Qwen3-8B-q4f32_1-MLC",

  // // RedPajama
  // "RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
  // "RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",

  // // SmolLM2 (including q0 for smaller ones)
  // "SmolLM2-135M-Instruct-q0f16-MLC",
  // "SmolLM2-135M-Instruct-q0f32-MLC",
  // "SmolLM2-360M-Instruct-q0f16-MLC",
  // "SmolLM2-360M-Instruct-q0f32-MLC",
  // "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
  // "SmolLM2-1.7B-Instruct-q4f32_1-MLC",

  // // TinyLlama v1.0
  // "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
  // "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",

  // // Gemma models
  // "gemma-2b-it-q4f16_1-MLC",
  // "gemma-2b-it-q4f32_1-MLC",
  // "gemma-2-2b-it-q4f16_1-MLC",
  // "gemma-2-2b-it-q4f32_1-MLC",
  // "gemma-2-9b-it-q4f16_1-MLC",
  // "gemma-2-9b-it-q4f32_1-MLC",

  // // StableLM
  // "stablelm-2-zephyr-1_6b-q4f16_1-MLC",
  // "stablelm-2-zephyr-1_6b-q4f32_1-MLC",
];

const TEST_PROMPT = "Tell me a joke.";

const initProgressCallback = (report: webllm.InitProgressReport) => {
  setLabel("init-label", report.text);
};

async function testModel(
  modelId: string,
  modelIndex: number,
  totalModels: number,
): Promise<boolean> {
  try {
    // print output into console
    console.log(
      `\n=== Testing Model ${modelIndex + 1}/${totalModels}: ${modelId} ===`,
    );
    setLabel(
      "current-model-label",
      `${modelId} (${modelIndex + 1}/${totalModels})`,
    );
    setLabel("progress-label", `Loading model...`);
    setLabel("response-label", "");

    const startTime = Date.now();

    const appConfig = webllm.prebuiltAppConfig;
    appConfig.useIndexedDBCache = true;

    const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
      modelId,
      {
        initProgressCallback: initProgressCallback,
        appConfig: appConfig,
        logLevel: "ERROR",
      },
    );

    const loadTime = Date.now() - startTime;
    console.log(`Model loaded in ${(loadTime / 1000).toFixed(1)}s`);
    setLabel(
      "progress-label",
      `Model loaded in ${(loadTime / 1000).toFixed(1)}s. Generating...`,
    );

    // Test chat completion
    const generateStart = Date.now();
    const reply = await engine.chat.completions.create({
      messages: [{ role: "user", content: TEST_PROMPT }],
      temperature: 0.1,
      max_tokens: 500,
    });

    const generateTime = Date.now() - generateStart;
    const response = reply.choices[0]?.message?.content || "No response";

    console.log(`Generated response in ${(generateTime / 1000).toFixed(1)}s`);
    console.log(`Response: "${response}"`);

    setLabel(
      "response-label",
      response.substring(0, 200) + (response.length > 200 ? "..." : ""),
    );
    setLabel(
      "stats-label",
      `Load: ${(loadTime / 1000).toFixed(1)}s, Generate: ${(generateTime / 1000).toFixed(1)}s, Tokens: ${reply.usage?.completion_tokens || "?"}`,
    );

    // Clear cache for this model
    setLabel("progress-label", `Clearing cache...`);
    await webllm.deleteModelAllInfoInCache(modelId, appConfig);
    console.log(`Cleared cache for ${modelId}`);

    return true;
  } catch (error) {
    console.error(`Error testing ${modelId}:`, error);
    setLabel("response-label", `Error: ${error.message}`);
    setLabel("progress-label", `Error with ${modelId}`);

    // Still try to clear cache even if test failed
    try {
      const appConfig = webllm.prebuiltAppConfig;
      appConfig.useIndexedDBCache = true;
      await webllm.deleteModelAllInfoInCache(modelId, appConfig);
      console.log(`Cleared cache for ${modelId} (after error)`);
    } catch (clearError) {
      console.error(`Failed to clear cache for ${modelId}:`, clearError);
    }

    return false;
  }
}

async function main() {
  console.log("Starting WebLLM Model Testing");
  console.log(`Testing ${TEST_MODELS.length} chat models`);

  const results = {
    passed: 0,
    failed: 0,
    total: TEST_MODELS.length,
  };

  setLabel("current-model-label", "Starting tests...");
  setLabel("progress-label", `0/${TEST_MODELS.length} models tested`);

  for (let i = 0; i < TEST_MODELS.length; i++) {
    const modelId = TEST_MODELS[i];
    const success = await testModel(modelId, i, TEST_MODELS.length);

    if (success) {
      results.passed++;
    } else {
      results.failed++;
    }

    setLabel(
      "progress-label",
      `${i + 1}/${TEST_MODELS.length} models tested (${results.passed} passed, ${results.failed} failed)`,
    );

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  console.log(`\nTesting completed!`);
  console.log(
    `Results: ${results.passed}/${results.total} models passed (${Math.round((results.passed / results.total) * 100)}%)`,
  );
  console.log(`Passed: ${results.passed}`);
  console.log(`Failed: ${results.failed}`);

  setLabel("current-model-label", "All tests completed!");
  setLabel(
    "progress-label",
    `Final: ${results.passed}/${results.total} passed (${Math.round((results.passed / results.total) * 100)}%)`,
  );
  setLabel("response-label", "Check console for full results");
  setLabel("stats-label", `${results.passed} passed, ${results.failed} failed`);
}

main();
