import {
  GenerationConfig,
  postInitAndCheckGenerationConfigValues,
} from "../src/config";
import { describe, expect, test } from "@jest/globals";

describe("Check generation config illegal values", () => {
  test("High-level unsupported fields", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        max_tokens: 0,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("Make sure `max_tokens` > 0");
  });

  test("logit_bias exceeds range", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        max_tokens: 10,
        logit_bias: {
          "1355": 155,
        },
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("Make sure -100 < logit_bias <= 100.");
  });

  test("logit_bias invalid key", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        max_tokens: 10,
        logit_bias: {
          thisRaisesError: 50,
        },
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow(
      "Make sure logit_bias's keys to be number represented in string.",
    );
  });

  test("top_logprobs out of range", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        logprobs: true,
        top_logprobs: 6,
        max_tokens: 10,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("Make sure 0 < top_logprobs <= 5.");
  });

  test("top_logprobs set without setting logprobs", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        top_logprobs: 3,
        max_tokens: 10,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("top_logprobs requires logprobs to be true");
  });

  test("top_logprobs set though logprobs is false", () => {
    expect(() => {
      const genConfig: GenerationConfig = {
        logprobs: false,
        top_logprobs: 3,
        max_tokens: 10,
      };
      postInitAndCheckGenerationConfigValues(genConfig);
    }).toThrow("top_logprobs requires logprobs to be true");
  });
});

describe("Check generation post init", () => {
  test("Only set one of presence or frequency penalty", () => {
    const genConfig: GenerationConfig = {
      frequency_penalty: 1.5,
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig.presence_penalty).toBe(0.0);
  });

  test("Set logprobs without setting top_logprobs", () => {
    const genConfig: GenerationConfig = {
      logprobs: true,
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig.top_logprobs).toBe(0);
  });

  test("Set both logprobs and top_logprobs", () => {
    const genConfig: GenerationConfig = {
      logprobs: true,
      top_logprobs: 2,
    };
    postInitAndCheckGenerationConfigValues(genConfig);
    expect(genConfig.top_logprobs).toBe(2);
  });
});
