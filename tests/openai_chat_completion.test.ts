/* eslint-disable no-useless-escape */
import {
  postInitAndCheckFields,
  ChatCompletionRequest,
  ChatCompletionTool,
} from "../src/openai_api_protocols/chat_completion";
import {
  hermes2FunctionCallingSystemPrompt,
  officialHermes2FunctionCallSchemaArray,
} from "../src/support";
import { MessagePlaceholders, ModelType } from "../src/config";
import { describe, expect, test } from "@jest/globals";

describe("Check chat completion unsupported requests", () => {
  test("stream_options without stream specified", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        stream_options: { include_usage: true },
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("Only specify stream_options when stream=True.");
  });

  test("stream_options with stream=false", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        stream: false,
        messages: [{ role: "user", content: "Hello! " }],
        stream_options: { include_usage: true },
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("Only specify stream_options when stream=True.");
  });

  test("Last message should be from user or tool", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello! " },
          { role: "assistant", content: "Hello! How may I help you today?" },
        ],
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("Last message should be from either `user` or `tool`.");
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
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow(
      "System prompt should always be the first message in `messages`.",
    );
  });

  test("When streaming `n` needs to be 1", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        stream: true,
        n: 2,
        messages: [{ role: "user", content: "Hello! " }],
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("When streaming, `n` cannot be > 1.");
  });

  test("Non-integer seed", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        max_tokens: 10,
        seed: 42.2, // Note that Number.isInteger(42.0) is true
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("`seed` should be an integer, but got");
  });

  test("Schema without type json object", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        response_format: { schema: "some json schema" },
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow(
      "JSON schema is only supported with `json_object` response format.",
    );
  });

  test("Grammar string without grammar type", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        response_format: { grammar: "some grammar string" },
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("When ResponseFormat.type is `grammar`,");
  });

  test("Grammar type without grammar string", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [{ role: "user", content: "Hello! " }],
        response_format: { type: "grammar" },
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow("When ResponseFormat.type is `grammar`,");
  });

  test("Valid: Grammar type with grammar string", () => {
    const request: ChatCompletionRequest = {
      messages: [{ role: "user", content: "Hello! " }],
      response_format: { type: "grammar", grammar: "some grammar string" },
    };
    postInitAndCheckFields(
      request,
      "Llama-3.1-8B-Instruct-q4f32_1-MLC",
      ModelType.LLM,
    );
  });

  test("image_url.detail is unsupported", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is in this image?" },
              {
                type: "image_url",
                image_url: {
                  url: "https://url_here.jpg",
                  detail: "high",
                },
              },
            ],
          },
        ],
      };
      postInitAndCheckFields(
        request,
        "Phi-3.5-vision-instruct-q4f16_1-MLC",
        ModelType.VLM,
      );
    }).toThrow(
      "Currently do not support field image_url.detail, but received: high",
    );
  });

  test("User content cannot have multiple text content parts", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is in this image?" },
              {
                type: "image_url",
                image_url: {
                  url: "https://url_here.jpg",
                },
              },
              { type: "text", text: "Thank you." },
            ],
          },
        ],
      };
      postInitAndCheckFields(
        request,
        "Phi-3.5-vision-instruct-q4f16_1-MLC",
        ModelType.VLM,
      );
    }).toThrow(
      "Each message can have at most one text contentPart, but received more than 1.",
    );
  });

  test("Non-VLM cannot support non-string content", () => {
    expect(() => {
      const request: ChatCompletionRequest = {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is in this image?" },
              {
                type: "image_url",
                image_url: {
                  url: "https://url_here.jpg",
                },
              },
            ],
          },
        ],
      };
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow(
      "The model loaded is not of type ModelType.VLM (vision-language model).",
    );
  });
});

describe("Supported requests", () => {
  test("Supported chat completion request", () => {
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
    postInitAndCheckFields(
      request,
      "Llama-3.1-8B-Instruct-q4f32_1-MLC",
      ModelType.LLM,
    );
  });

  test("Support image input, single or multiple images", () => {
    const request: ChatCompletionRequest = {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is in this image?" },
            {
              type: "image_url",
              image_url: { url: "https://url_here1.jpg" },
            },
            {
              type: "image_url",
              image_url: { url: "https://url_here2.jpg" },
            },
          ],
        },
      ],
    };
    postInitAndCheckFields(
      request,
      "Phi-3.5-vision-instruct-q4f16_1-MLC",
      ModelType.VLM,
    );
  });
});

describe("Manual function calling", () => {
  test("Hermes2 style function calling", () => {
    const system_prompt = `You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call>`;
    const request: ChatCompletionRequest = {
      messages: [
        { role: "system", content: system_prompt },
        {
          role: "user",
          content: "Fetch the stock fundamentals data for Tesla (TSLA)",
        },
        {
          role: "assistant",
          content: `<tool_call>\n{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}\n</tool_call>`,
        },
        {
          role: "tool",
          tool_call_id: "0",
          content: `<tool_response>\n{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}\n</tool_response>`,
        },
      ],
    };
    postInitAndCheckFields(
      request,
      "Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC",
      ModelType.LLM,
    );
  });
});

describe("OpenAI API function calling", () => {
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
      postInitAndCheckFields(
        request,
        "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        ModelType.LLM,
      );
    }).toThrow(
      "Llama-3.1-8B-Instruct-q4f32_1-MLC is not supported for ChatCompletionRequest.tools.",
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
      postInitAndCheckFields(
        request,
        "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
        ModelType.LLM,
      );
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
      postInitAndCheckFields(
        request,
        "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
        ModelType.LLM,
      );
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
      postInitAndCheckFields(
        request,
        "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
        ModelType.LLM,
      );
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
    postInitAndCheckFields(
      request,
      "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
      ModelType.LLM,
    );
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
