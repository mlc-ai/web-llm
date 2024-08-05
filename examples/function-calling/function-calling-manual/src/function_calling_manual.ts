/* eslint-disable no-useless-escape */
import * as webllm from "@mlc-ai/web-llm";

// Common helper methods
function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

const initProgressCallback = (report: webllm.InitProgressReport) => {
  setLabel("init-label", report.text);
};

// Same example as https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B#prompt-format-for-function-calling
async function hermes2_example() {
  // 0. Setups
  // Most manual function calling models specify the tools inside the system prompt
  const system_prompt = `You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> {"type": "function", "function": {"name": "get_stock_fundamentals", "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\\n\\n    Args:\\n        symbol (str): The stock symbol.\\n\\n    Returns:\\n        dict: A dictionary containing fundamental data.\\n            Keys:\\n                - \'symbol\': The stock symbol.\\n                - \'company_name\': The long name of the company.\\n                - \'sector\': The sector to which the company belongs.\\n                - \'industry\': The industry to which the company belongs.\\n                - \'market_cap\': The market capitalization of the company.\\n                - \'pe_ratio\': The forward price-to-earnings ratio.\\n                - \'pb_ratio\': The price-to-book ratio.\\n                - \'dividend_yield\': The dividend yield.\\n                - \'eps\': The trailing earnings per share.\\n                - \'beta\': The beta value of the stock.\\n                - \'52_week_high\': The 52-week high price of the stock.\\n                - \'52_week_low\': The 52-week low price of the stock.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}}  </tools> Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call>`;
  // Same formatting for Hermes-2-Pro-Llama-3, Hermes-2-Theta-Llama-3
  // const selectedModel = "Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC";
  const selectedModel = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback, logLevel: "INFO" },
  );
  const seed = 0;

  // 1. First request, expect to generate tool call
  const messages: webllm.ChatCompletionMessageParam[] = [
    { role: "system", content: system_prompt },
    {
      role: "user",
      content: "Fetch the stock fundamentals data for Tesla (TSLA)",
    },
  ];
  const request1: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply1 = await engine.chat.completions.create(request1);
  const response1 = reply1.choices[0].message.content;
  console.log(reply1.usage);
  console.log("Response 1: " + response1);
  messages.push({ role: "assistant", content: response1 });
  // <tool_call>\n{"arguments": {"symbol": "TSLA"}, "name": "get_stock_fundamentals"}\n</tool_call>

  // 2. Call function on your own to get tool response
  const tool_response = `<tool_response>\n{"name": "get_stock_fundamentals", "content": {'symbol': 'TSLA', 'company_name': 'Tesla, Inc.', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'market_cap': 611384164352, 'pe_ratio': 49.604652, 'pb_ratio': 9.762013, 'dividend_yield': None, 'eps': 4.3, 'beta': 2.427, '52_week_high': 299.29, '52_week_low': 152.37}}\n</tool_response>`;
  messages.push({ role: "tool", content: tool_response, tool_call_id: "0" });

  // 3. Get natural language response
  const request2: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply2 = await engine.chat.completions.create(request2);
  const response2 = reply2.choices[0].message.content;
  messages.push({ role: "assistant", content: response2 });
  console.log(reply2.usage);
  console.log("Response 2: " + response2);

  // 4. Another function call
  messages.push({
    role: "user",
    content: "Now do another one with NVIDIA, symbol being NVDA.",
  });
  const request3: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply3 = await engine.chat.completions.create(request3);
  const response3 = reply3.choices[0].message.content;
  messages.push({ role: "assistant", content: response3 });
  console.log(reply3.usage);
  console.log("Response 3: " + response3);
  // <tool_call>\n{"arguments": {"symbol": "NVDA"}, "name": "get_stock_fundamentals"}\n</tool_call>
}

// Similar example to https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#user-defined-custom-tool-calling
async function llama3_1_example() {
  // Follows example, but tweaks the formatting with <function>
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

  const selectedModel = "Llama-3.1-8B-Instruct-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback, logLevel: "INFO" },
  );
  const seed = 0;

  // 1. First request, expect to generate tool call to get temperature of Paris
  const messages: webllm.ChatCompletionMessageParam[] = [
    { role: "system", content: system_prompt },
    {
      role: "user",
      content: "Hey, what's the temperature in Paris right now?",
    },
  ];
  const request1: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply1 = await engine.chat.completions.create(request1);
  const response1 = reply1.choices[0].message.content;
  console.log(reply1.usage);
  console.log("Response 1: " + response1);
  messages.push({ role: "assistant", content: response1 });
  // <function>{"name": "get_current_temperature", "parameters": {"location": "Paris, France"}}</function>

  // 2. Call function on your own to get tool response
  const tool_response = `{"output": 22.5}`;
  messages.push({ role: "tool", content: tool_response, tool_call_id: "0" });

  // 3. Get natural language response
  const request2: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply2 = await engine.chat.completions.create(request2);
  const response2 = reply2.choices[0].message.content;
  messages.push({ role: "assistant", content: response2 });
  console.log(reply2.usage);
  console.log("Response 2: " + response2);
  // The current temperature in Paris is 22.5°C.

  // 4. Make another request, expect model to call `send_message`
  messages.push({
    role: "user",
    content: "Send a message to Tom to tell him this information.",
  });
  const request3: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply3 = await engine.chat.completions.create(request3);
  const response3 = reply3.choices[0].message.content;
  messages.push({ role: "assistant", content: response3 });
  console.log(reply3.usage);
  console.log("Response 3: " + response3);
  // <function>{"name": "send_message", "parameters": {"recipient": "Tom", "content": "The current temperature in Paris is 22.5°C."}}</function>

  // 5. Call API, which has no return value, so simply prompt model again
  const tool_response2 = `{"output": None}`;
  messages.push({ role: "tool", content: tool_response2, tool_call_id: "1" });
  const request4: webllm.ChatCompletionRequest = {
    stream: false, // works with either streaming or non-streaming; code below assumes non-streaming
    messages: messages,
    seed: seed,
  };
  const reply4 = await engine.chat.completions.create(request4);
  const response4 = reply4.choices[0].message.content;
  console.log(reply4.usage);
  console.log("Response 4: " + response4);
  // The message has been sent to Tom.
}

// Pick one to run
// hermes2_example();
llama3_1_example();
