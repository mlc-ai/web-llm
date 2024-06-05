import {
  postInitAndCheckFields,
  ChatCompletionRequest,
  ChatCompletionTool,
} from "../src/openai_api_protocols/chat_completion";
import {
  hermes2FunctionCallingSystemPrompt,
  officialHermes2FunctionCallSchemaArray,
} from "../src/support";
import { MessagePlaceholders } from "../src/config";
import { describe, expect, test } from "@jest/globals";

describe("Check chat completion unsupported requests", () => {
  test("stream_options without stream specified", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        stream_options: { include_usage: true },
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("Only specify stream_options when stream=True.");
  });

  test("stream_options with stream=false", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        stream: false,
        messages: [{ role: "user", content: "Hello! " }],
        stream_options: { include_usage: true },
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("Only specify stream_options when stream=True.");
  });

  test("High-level unsupported fields", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        model: "phi-2-q4f32_1-MLC", // this raises error
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello! " },
        ],
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow(
      "The following fields in ChatCompletionRequest are not yet supported",
    );
  });

  test("Last message should be from user", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello! " },
          { role: "assistant", content: "Hello! How may I help you today?" },
        ],
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("Last message should be from `user`.");
  });

  test("System prompt should always be the first one in `messages`", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          { role: "user", content: "Hello! " },
          { role: "assistant", content: "Hello! How may I help you today?" },
          { role: "user", content: "Tell me about Pittsburgh" },
          { role: "system", content: "You are a helpful assistant." },
        ],
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("System prompt should always be the first one in `messages`.");
  });

  test("When streaming `n` needs to be 1", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        stream: true,
        n: 2,
        messages: [{ role: "user", content: "Hello! " }],
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("When streaming, `n` cannot be > 1.");
  });

  test("Non-integer seed", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        max_tokens: 10,
        seed: 42.2, // Note that Number.isInteger(42.0) is true
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("`seed` should be an integer, but got");
  });

  test("Schema without type json object", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        response_format: { schema: "some json schema" },
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow(
      "JSON schema is only supported with `json_object` response format.",
    );
  });

  // Remove when we support image input (e.g. LlaVA model)
  test("Image input is unsupported", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is in this image?" },
              {
                type: "image_url",
                image_url: { url: "https://url_here.jpg" },
              },
            ],
          },
        ],
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow("User message only supports string `content` for now");
  });
});

describe("Supported requests", () => {
  test("Supproted chat completion request", () => {
    const request: ChatCompletionRequest = {
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello! " },
        { role: "assistant", content: "How can I help you? " },
        { role: "user", content: "Give me 5 US states. " },
      ],
      n: 3,
      temperature: 1.5,
      max_tokens: 25,
      frequency_penalty: 0.2,
      seed: 42,
      logprobs: true,
      top_logprobs: 2,
      logit_bias: {
        "13813": -100,
        "10319": 5,
        "7660": 5,
      },
    };
    postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
  });
});

describe("Function calling", () => {
  const tools: Array<ChatCompletionTool> = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: ["location"],
        },
      },
    },
  ];

  test("Unsupported model", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        tools: tools,
        messages: [
          {
            role: "user",
            content: "Get weather of Tokyo",
          },
        ],
      };
      postInitAndCheckFields(request, "Llama-3-8B-Instruct-q4f32_1-MLC");
    }).toThrow(
      "Llama-3-8B-Instruct-q4f32_1-MLC is not supported for ChatCompletionRequest.tools.",
    );
  });

  test("Should not specify response format", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        tools: tools,
        messages: [
          {
            role: "user",
            content: "Get weather of Tokyo",
          },
        ],
        response_format: { type: "json_object" },
      };
      postInitAndCheckFields(request, "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC");
    }).toThrow(
      "When using Hermes-2-Pro function calling via ChatCompletionRequest.tools, " +
        "cannot specify customized response_format. We will set it for you internally.",
    );
  });

  test("Should not specify system prompt", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        tools: tools,
        messages: [
          {
            role: "system",
            content: "Write a function.",
          },
          {
            role: "user",
            content: "Get weather of Tokyo",
          },
        ],
      };
      postInitAndCheckFields(request, "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC");
    }).toThrow(
      "When using Hermes-2-Pro function calling via ChatCompletionRequest.tools, cannot " +
        "specify customized system prompt.",
    );
  });

  test("Should not specify system prompt", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        tools: tools,
        messages: [
          {
            role: "system",
            content: "Write a function.",
          },
          {
            role: "user",
            content: "Get weather of Tokyo",
          },
        ],
      };
      postInitAndCheckFields(request, "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC");
    }).toThrow(
      "When using Hermes-2-Pro function calling via ChatCompletionRequest.tools, cannot " +
        "specify customized system prompt.",
    );
  });

  test("Check system prompt and response format post init", () => {
    const request: ChatCompletionRequest = {
      tools: tools,
      messages: [
        {
          role: "user",
          content: "Get weather of Tokyo",
        },
      ],
    };
    postInitAndCheckFields(request, "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC");
    expect(request.messages[0].role).toEqual("system");
    expect(request.messages[0].content).toEqual(
      hermes2FunctionCallingSystemPrompt.replace(
        MessagePlaceholders.hermes_tools,
        JSON.stringify(request.tools),
      ),
    );
    expect(request.response_format!.type).toEqual("json_object");
    expect(request.response_format!.schema).toEqual(
      officialHermes2FunctionCallSchemaArray,
    );
  });
});
