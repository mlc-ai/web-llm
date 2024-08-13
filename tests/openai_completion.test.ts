import { getConversation } from "../src/conversation";
import {
  TextCompletionConversationError,
  TextCompletionConversationExpectsPrompt,
} from "../src/error";
import {
  CompletionCreateParams,
  postInitAndCheckFields,
} from "../src/openai_api_protocols/completion";
import { llama3_1ChatConfig } from "./constants";
import { describe, expect, test } from "@jest/globals";

describe("Conversation object with text completion", () => {
  test("Conversation checks ", () => {
    const conv = getConversation(
      llama3_1ChatConfig.conv_template,
      llama3_1ChatConfig.conv_config,
      /*isTextCompletion=*/ true,
    );
    expect(() => {
      conv.getPromptArrayTextCompletion();
    }).toThrow(new TextCompletionConversationExpectsPrompt());
    expect(() => {
      conv.getPromptArray();
    }).toThrow(new TextCompletionConversationError("getPromptArray"));

    conv.prompt = "Hi";
    expect(conv.getPromptArrayTextCompletion()).toEqual(["Hi"]);

    conv.reset();
    expect(conv.prompt === undefined).toEqual(true);
  });
});

describe("Check completion unsupported requests", () => {
  test("stream_options without stream specified", () => {
    expect(() => {
      const request: CompletionCreateParams = {
        prompt: "Hello, ",
        stream_options: { include_usage: true },
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow("Only specify stream_options when stream=True.");
  });

  test("stream_options with stream=false", () => {
    expect(() => {
      const request: CompletionCreateParams = {
        stream: false,
        prompt: "Hello, ",
        stream_options: { include_usage: true },
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow("Only specify stream_options when stream=True.");
  });

  test("High-level unsupported fields", () => {
    expect(() => {
      const request: CompletionCreateParams = {
        prompt: "Hello, ",
        suffix: "this is suffix", // this raises error
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow(
      "The following fields in CompletionCreateParams are not yet supported",
    );

    expect(() => {
      const request: CompletionCreateParams = {
        prompt: "Hello, ",
        best_of: 3, // this raises error
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow(
      "The following fields in CompletionCreateParams are not yet supported",
    );

    expect(() => {
      const request: CompletionCreateParams = {
        prompt: "Hello, ",
        user: "Bob", // this raises error
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow(
      "The following fields in CompletionCreateParams are not yet supported",
    );
  });

  test("When streaming `n` needs to be 1", () => {
    expect(() => {
      const request: CompletionCreateParams = {
        stream: true,
        n: 2,
        prompt: "Hello, ",
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow("When streaming, `n` cannot be > 1.");
  });

  test("Non-integer seed", () => {
    expect(() => {
      const request: CompletionCreateParams = {
        prompt: "Hello, ",
        max_tokens: 10,
        seed: 42.2, // Note that Number.isInteger(42.0) is true
      };
      postInitAndCheckFields(request, "Llama-3.1-8B-Instruct-q4f32_1-MLC");
    }).toThrow("`seed` should be an integer, but got");
  });
});
