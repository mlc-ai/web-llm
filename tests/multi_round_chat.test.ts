import { describe, expect, test } from "@jest/globals";

import {
  ChatCompletionMessageParam,
  ChatCompletionRequest,
  ChatCompletionUserMessageParam,
} from "../src/openai_api_protocols/chat_completion";
import {
  Conversation,
  compareConversationObject,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import { ChatConfig, Role } from "../src/config";

const configStr =
  "{" +
  '  "conv_template": {' +
  '    "name": "llama-2",' +
  '    "system_template": "[INST] <<SYS>>\\n{system_message}\\n<</SYS>>\\n\\n",' +
  '    "system_message": "You are a helpful, respectful and honest assistant.",' +
  '    "system_prefix_token_ids": [' +
  "      1" +
  "    ]," +
  '    "add_role_after_system_message": false,' +
  '    "roles": {' +
  '      "user": "[INST]",' +
  '      "assistant": "[/INST]",' +
  '      "tool": "[INST]"' +
  "    }," +
  '    "role_templates": {' +
  '      "user": "{user_message}",' +
  '      "assistant": "{assistant_message}",' +
  '      "tool": "{tool_message}"' +
  "    }," +
  '    "messages": [],' +
  '    "seps": [' +
  '      " "' +
  "    ]," +
  '    "role_content_sep": " ",' +
  '    "role_empty_sep": " ",' +
  '    "stop_str": [' +
  '      "[INST]"' +
  "    ]," +
  '    "stop_token_ids": [' +
  "      2" +
  "    ]," +
  '    "function_string": "",' +
  '    "use_function_calling": false' +
  "  }" +
  "}";

describe("Test multi-round chatting", () => {
  test("Test is multi-round", () => {
    // Setups
    const config_json = JSON.parse(configStr);
    const chatConfig = { ...config_json } as ChatConfig;

    // Simulate request0
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content:
          "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n ",
      },
      { role: "user", content: "Provide me three US states." },
    ];
    const request0: ChatCompletionRequest = {
      messages: messages,
    };

    // Simulate processing of request0, appending response to convA (done by LLMChatPipeline)
    const conv0: Conversation = getConversationFromChatCompletionRequest(
      request0,
      chatConfig,
    );
    conv0.appendMessage(Role.user, "Provide me three US states.");
    const reply0 = "California, New York, Nevada.";
    conv0.appendMessage(Role.assistant, reply0); // simulated response

    // Simulate request1, where user maintain the chat history, appending the resposne
    const newMessages = [...messages];
    newMessages.push({ role: "assistant", content: reply0 });
    newMessages.push({ role: "user", content: "Two more please" }); // next input

    const request1: ChatCompletionRequest = {
      messages: newMessages,
    };
    const conv1: Conversation = getConversationFromChatCompletionRequest(
      request1,
      chatConfig,
    );

    expect(compareConversationObject(conv0, conv1)).toBe(true);
  });

  test("Test is NOT multi-round due to multiple new inputs", () => {
    // Setups
    const config_json = JSON.parse(configStr);
    const chatConfig = { ...config_json } as ChatConfig;

    // Simulate request0
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content:
          "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n ",
      },
      { role: "user", content: "Provide me three US states." },
    ];
    const request0: ChatCompletionRequest = {
      messages: messages,
    };

    // Simulate processing of request0, appending response to convA (done by LLMChatPipeline)
    const conv0: Conversation = getConversationFromChatCompletionRequest(
      request0,
      chatConfig,
    );
    conv0.appendMessage(Role.user, "Provide me three US states.");
    const reply0 = "California, New York, Nevada.";
    conv0.appendMessage(Role.assistant, reply0); // simulated response

    // Simulate request1, where user maintain the chat history, appending the resposne
    const newMessages = [...messages];
    newMessages.push({ role: "assistant", content: reply0 });
    newMessages.push({ role: "user", content: "Two more please" }); // next input

    // Code above same as previous tests
    // Add one more round of chat history
    newMessages.push({ role: "assistant", content: "Pennsylvania, Florida" }); // next response
    newMessages.push({ role: "user", content: "Thank you!" }); // next input

    const request1: ChatCompletionRequest = {
      messages: newMessages,
    };
    const conv1: Conversation = getConversationFromChatCompletionRequest(
      request1,
      chatConfig,
    );

    expect(compareConversationObject(conv0, conv1)).toBe(false);
  });

  test("Test is NOT multi-round due to change in system prompt", () => {
    // Setups
    const config_json = JSON.parse(configStr);
    const chatConfig = { ...config_json } as ChatConfig;

    // Simulate request0
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content:
          "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n ",
      },
      { role: "user", content: "Provide me three US states." },
    ];
    const request0: ChatCompletionRequest = {
      messages: messages,
    };

    // Simulate processing of request0, appending response to convA (done by LLMChatPipeline)
    const conv0: Conversation = getConversationFromChatCompletionRequest(
      request0,
      chatConfig,
    );
    conv0.appendMessage(Role.user, "Provide me three US states.");
    const reply0 = "California, New York, Nevada.";
    conv0.appendMessage(Role.assistant, reply0); // simulated response

    // Simulate request1, where user maintain the chat history, appending the resposne
    const newMessages = [...messages];
    newMessages.push({ role: "assistant", content: reply0 });
    newMessages.push({ role: "user", content: "Two more please" }); // next input

    // Code above same as previous tests
    // Changed system prompt, should be false
    newMessages[0].content = "No system prompt";

    const request1: ChatCompletionRequest = {
      messages: newMessages,
    };
    const conv1: Conversation = getConversationFromChatCompletionRequest(
      request1,
      chatConfig,
    );

    expect(compareConversationObject(conv0, conv1)).toBe(false);
  });

  test("Test is NOT multi-round due to change in role name", () => {
    // Setups
    const config_json = JSON.parse(configStr);
    const chatConfig = { ...config_json } as ChatConfig;

    // Simulate request0
    const messages: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content:
          "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
          "Be as happy as you can when speaking please.\n<</SYS>>\n\n ",
      },
      { role: "user", content: "Provide me three US states." },
    ];
    const request0: ChatCompletionRequest = {
      messages: messages,
    };

    // Simulate processing of request0, appending response to convA (done by LLMChatPipeline)
    const conv0: Conversation = getConversationFromChatCompletionRequest(
      request0,
      chatConfig,
    );
    conv0.appendMessage(Role.user, "Provide me three US states.");
    const reply0 = "California, New York, Nevada.";
    conv0.appendMessage(Role.assistant, reply0); // simulated response

    // Simulate request1, where user maintain the chat history, appending the resposne
    const newMessages = [...messages];
    newMessages.push({ role: "assistant", content: reply0 });
    newMessages.push({ role: "user", content: "Two more please" }); // next input

    // Code above same as previous tests
    // Changed system prompt, should be false
    (newMessages[1] as ChatCompletionUserMessageParam).name = "Bob";

    const request1: ChatCompletionRequest = {
      messages: newMessages,
    };
    const conv1: Conversation = getConversationFromChatCompletionRequest(
      request1,
      chatConfig,
    );

    expect(compareConversationObject(conv0, conv1)).toBe(false);
  });
});
