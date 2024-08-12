/* eslint-disable no-useless-escape */
import {
  Role,
  MessagePlaceholders,
  ConvTemplateConfig,
  ChatConfig,
} from "../src/config";
import {
  getConversation,
  getConversationFromChatCompletionRequest,
  getFunctionCallUsage,
} from "../src/conversation";
import { ChatCompletionRequest } from "../src/openai_api_protocols/chat_completion";

import { describe, expect, test } from "@jest/globals";
import { llama3_1ChatConfig } from "./constants";

describe("Test gorilla conversation template", () => {
  const gorillaConv: ConvTemplateConfig = {
    system_template: `${MessagePlaceholders.system}\n`,
    system_message:
      "A chat between a curious user and an artificial intelligence assistant. " +
      "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles: {
      [Role.user]: "USER",
      [Role.assistant]: "ASSISTANT",
      [Role.tool]: "TOOL",
    },
    role_templates: {
      [Role.user]: `<<question>> ${MessagePlaceholders.user} <<function>> ${MessagePlaceholders.function}`,
    },
    seps: ["\n", "<|EOT|>"],
    stop_str: ["<|EOT|>"],
    system_prefix_token_ids: [1],
    stop_token_ids: [2],
  };

  test("Test getPromptArrayInternal", () => {
    const conv = getConversation(gorillaConv);
    conv.appendMessage(
      Role.user,
      'Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes',
      "Tom",
    );
    const prompt_array = conv.getPromptArray();

    expect(prompt_array).toEqual([
      "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n",
      'Tom: <<question>> Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes <<function>> \n',
    ]);
  });

  test("Test getPromptArrayInternal function call", () => {
    const conv = getConversation(gorillaConv);
    conv.appendMessage(
      Role.user,
      'Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes',
    );
    conv.use_function_calling = true;
    conv.function_string = JSON.stringify([
      {
        name: "Uber Carpool",
        api_name: "uber.ride",
        description:
          "Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters",
        parameters: [
          {
            name: "loc",
            description: "Location of the starting place of the Uber ride",
          },
          {
            name: "type",
            enum: ["plus", "comfort", "black"],
            description: "Types of Uber ride user is ordering",
          },
          {
            name: "time",
            description:
              "The amount of time in minutes the customer is willing to wait",
          },
        ],
      },
    ]);
    const prompt_array = conv.getPromptArray();

    expect(prompt_array).toEqual([
      "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n",
      'USER: <<question>> Call me an Uber ride type "Plus" in Berkeley at zipcode 94704 in 10 minutes <<function>> [{"name":"Uber Carpool","api_name":"uber.ride","description":"Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters","parameters":[{"name":"loc","description":"Location of the starting place of the Uber ride"},{"name":"type","enum":["plus","comfort","black"],"description":"Types of Uber ride user is ordering"},{"name":"time","description":"The amount of time in minutes the customer is willing to wait"}]}]\n',
    ]);
  });
});

describe("Test gorilla MLCEngine", () => {
  test("Test getFunctionCallUsage none", () => {
    const request: ChatCompletionRequest = {
      model: "gorilla-openfunctions-v1-q4f16_1_MLC",
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
      ],
      tool_choice: "none",
      tools: [
        {
          type: "function",
          function: {
            description: "A",
            name: "fn_A",
            parameters: { foo: "bar" },
          },
        },
        {
          type: "function",
          function: {
            description: "B",
            name: "fn_B",
            parameters: { foo: "bar" },
          },
        },
        {
          type: "function",
          function: {
            description: "C",
            name: "fn_C",
            parameters: { foo: "bar" },
          },
        },
      ],
    };

    expect(getFunctionCallUsage(request)).toEqual("");
  });

  test("Test getFunctionCallUsage auto", () => {
    const request: ChatCompletionRequest = {
      model: "gorilla-openfunctions-v1-q4f16_1_MLC",
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
      ],
      tool_choice: "auto",
      tools: [
        {
          type: "function",
          function: {
            description: "A",
            name: "fn_A",
            parameters: { foo: "bar" },
          },
        },
        {
          type: "function",
          function: {
            description: "B",
            name: "fn_B",
            parameters: { foo: "bar" },
          },
        },
        {
          type: "function",
          function: {
            description: "C",
            name: "fn_C",
            parameters: { foo: "bar" },
          },
        },
      ],
    };
    expect(getFunctionCallUsage(request)).toEqual(
      '[{"description":"A","name":"fn_A","parameters":{"foo":"bar"}},{"description":"B","name":"fn_B","parameters":{"foo":"bar"}},{"description":"C","name":"fn_C","parameters":{"foo":"bar"}}]',
    );
  });

  test("Test getFunctionCallUsage function", () => {
    const request: ChatCompletionRequest = {
      model: "gorilla-openfunctions-v1-q4f16_1_MLC",
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
      ],
      tool_choice: {
        type: "function",
        function: {
          name: "fn_B",
        },
      },
      tools: [
        {
          type: "function",
          function: {
            description: "A",
            name: "fn_A",
            parameters: { foo: "bar" },
          },
        },
        {
          type: "function",
          function: {
            description: "B",
            name: "fn_B",
            parameters: { foo: "bar" },
          },
        },
        {
          type: "function",
          function: {
            description: "C",
            name: "fn_C",
            parameters: { foo: "bar" },
          },
        },
      ],
    };
    expect(getFunctionCallUsage(request)).toEqual(
      '[{"description":"B","name":"fn_B","parameters":{"foo":"bar"}}]',
    );
  });
});

describe("Test Hermes2 formatting", () => {
  const hermes2LlamaChatConfig: ChatConfig = {
    vocab_size: 128288,
    context_window_size: 8192,
    sliding_window_size: -1,
    attention_sink_size: -1,
    temperature: 1.0,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    repetition_penalty: 1.0,
    top_p: 1.0,
    tokenizer_files: ["tokenizer.json", "tokenizer_config.json"],
    tokenizer_info: {
      token_postproc_method: "byte_level",
      prepend_space_in_encode: false,
      strip_space_in_decode: false,
    },
    conv_template: {
      system_template: "<|im_start|>system\n{system_message}<|im_end|>\n",
      system_message:
        'You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.',
      add_role_after_system_message: true,
      roles: {
        user: "<|im_start|>user",
        assistant: "<|im_start|>assistant",
        tool: "<|im_start|>tool",
      },
      role_templates: {
        user: "{user_message}",
        assistant: "{assistant_message}",
        tool: "{tool_message}",
      },
      seps: ["<|im_end|>\n"],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|im_end|>"],
      stop_token_ids: [128001, 128009, 128003],
    },
    bos_token_id: 128000,
  };

  // Follows https://github.com/NousResearch/Hermes-Function-Calling/blob/96ebfd7c903216b05e1eb7b155f7d5842b0fbce8/README.md#prompt-format
  test("Test formatting", () => {
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
          content: `<tool_response>\n{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}\n</tool_response>\n`,
        },
        { role: "assistant", content: "Some replies here" },
        { role: "user", content: "Thank you." },
      ],
    };
    // Since we treat last input as PrefillStep input, last message is not included in `conv`
    const conv = getConversationFromChatCompletionRequest(
      request,
      hermes2LlamaChatConfig,
    );
    const promptArray = conv.getPromptArray();
    let finalMessage = "";
    for (const msg of promptArray) {
      finalMessage += msg;
    }
    const expected =
      `<|im_start|>system\n` +
      system_prompt +
      `<|im_end|>\n` +
      `<|im_start|>user\nFetch the stock fundamentals data for Tesla (TSLA)<|im_end|>\n` +
      `<|im_start|>assistant\n<tool_call>\n{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}\n</tool_call><|im_end|>\n` +
      `<|im_start|>tool\n<tool_response>\n{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}\n</tool_response>\n<|im_end|>\n` +
      `<|im_start|>assistant\nSome replies here<|im_end|>\n`;
    expect(finalMessage).toEqual(expected);
  });
});

describe("Test Llama3.1 formatting", () => {
  // Follows https://github.com/NousResearch/Hermes-Function-Calling/blob/96ebfd7c903216b05e1eb7b155f7d5842b0fbce8/README.md#prompt-format
  test("Test formatting", () => {
    const system_prompt = `Cutting Knowledge Date: December 2023
    Today Date: 23 Jul 2024
    # Tool Instructions
    - When looking for real time information use relevant functions if available
    You have access to the following functions:
    
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the temperature for, in the format \"City, Country\""
                    }
                },
                "required": [
                    "location"
                ]
            },
            "return": {
                "type": "number",
                "description": "The current temperature at the specified location in the specified units, as a float."
            }
        }
    }
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Name of the recipient of the message"
                    }
                    "content": {
                        "type": "string",
                        "description": "Content of the message"
                    }
                },
                "required": [
                    "recipient",
                    "content"
                ]
            },
            "return": {
                "type": "None"
            }
        }
    }
    If a you choose to call a function ONLY reply in the following format:
        <function>{"name": function name, "parameters": dictionary of argument name and its value}</function>
    Here is an example,
        <function>{"name": "example_function_name", "parameters": {"example_name": "example_value"}}</function>
    Reminder:
    - Function calls MUST follow the specified format and use BOTH <function> and </function>
    - Required parameters MUST be specified
    - Only call one function at a time
    - When calling a function, do NOT add any other words, ONLY the function calling
    - Put the entire function call reply on one line
    - Always add your sources when using search results to answer the user query
    You are a helpful Assistant.`;
    const user1 = "Hey, what's the temperature in Paris right now?";
    const assistant1 = `<function>{"name": "get_current_temperature", "parameters": {"location": "Paris, France"}}</function>`;
    const tool1 = `{"output": 22.5}`;
    const assistant2 = `The current temperature in Paris is 22.5°C.`;
    const user2 = "Send a message to Tom to tell him this information.";
    const assistant3 = `<function>{"name": "send_message", "parameters": {"recipient": "Tom", "content": "The current temperature in Paris is 22.5°C."}}</function>`;
    const tool2 = `{"output": None}`;
    const assistant4 = `The message has been sent to Tom.`;

    const request: ChatCompletionRequest = {
      messages: [
        { role: "system", content: system_prompt },
        { role: "user", content: user1 },
        { role: "assistant", content: assistant1 },
        { role: "tool", tool_call_id: "0", content: tool1 },
        { role: "assistant", content: assistant2 },
        { role: "user", content: user2 },
        { role: "assistant", content: assistant3 },
        { role: "tool", tool_call_id: "1", content: tool2 },
        { role: "assistant", content: assistant4 },
        { role: "user", content: "Thank you." },
      ],
    };
    // Since we treat last input as PrefillStep input, last message is not included in `conv`
    const conv = getConversationFromChatCompletionRequest(
      request,
      llama3_1ChatConfig,
    );
    const promptArray = conv.getPromptArray();
    let finalMessage = "";
    for (const msg of promptArray) {
      finalMessage += msg;
    }
    // Expected is generated with transformers in Python `tokenizer.apply_chat_template()`
    const expected = `<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n    Today Date: 23 Jul 2024\n    # Tool Instructions\n    - When looking for real time information use relevant functions if available\n    You have access to the following functions:\n    \n    {\n        "type": "function",\n        "function": {\n            "name": "get_current_temperature",\n            "description": "Get the current temperature at a location.",\n            "parameters": {\n                "type": "object",\n                "properties": {\n                    "location": {\n                        "type": "string",\n                        "description": "The location to get the temperature for, in the format "City, Country""\n                    }\n                },\n                "required": [\n                    "location"\n                ]\n            },\n            "return": {\n                "type": "number",\n                "description": "The current temperature at the specified location in the specified units, as a float."\n            }\n        }\n    }\n    {\n        "type": "function",\n        "function": {\n            "name": "send_message",\n            "description": "Send a message to a recipient.",\n            "parameters": {\n                "type": "object",\n                "properties": {\n                    "recipient": {\n                        "type": "string",\n                        "description": "Name of the recipient of the message"\n                    }\n                    "content": {\n                        "type": "string",\n                        "description": "Content of the message"\n                    }\n                },\n                "required": [\n                    "recipient",\n                    "content"\n                ]\n            },\n            "return": {\n                "type": "None"\n            }\n        }\n    }\n    If a you choose to call a function ONLY reply in the following format:\n        <function>{"name": function name, "parameters": dictionary of argument name and its value}</function>\n    Here is an example,\n        <function>{"name": "example_function_name", "parameters": {"example_name": "example_value"}}</function>\n    Reminder:\n    - Function calls MUST follow the specified format and use BOTH <function> and </function>\n    - Required parameters MUST be specified\n    - Only call one function at a time\n    - When calling a function, do NOT add any other words, ONLY the function calling\n    - Put the entire function call reply on one line\n    - Always add your sources when using search results to answer the user query\n    You are a helpful Assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHey, what\'s the temperature in Paris right now?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<function>{"name": "get_current_temperature", "parameters": {"location": "Paris, France"}}</function><|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n{"output": 22.5}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe current temperature in Paris is 22.5°C.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSend a message to Tom to tell him this information.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<function>{"name": "send_message", "parameters": {"recipient": "Tom", "content": "The current temperature in Paris is 22.5°C."}}</function><|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n{"output": None}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe message has been sent to Tom.<|eot_id|>`;
    expect(finalMessage).toEqual(expected);
  });
});
