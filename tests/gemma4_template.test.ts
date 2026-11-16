import { readFileSync } from "node:fs";
import { ChatConfig, Role } from "../src/config";
import {
  compareConversationObject,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import type { ChatCompletionMessageParam } from "../src/openai_api_protocols";
import { llama2ChatConfigJSONString } from "./constants";

const { cases } = JSON.parse(
  readFileSync("tests/fixtures/gemma4/large_text_chat.json", "utf8"),
) as {
  cases: {
    name: string;
    thinking: boolean;
    messages: ChatCompletionMessageParam[];
    prompt: string;
    ids: number[];
  }[];
};

function config(thinking: boolean): ChatConfig {
  const base = JSON.parse(llama2ChatConfigJSONString) as ChatConfig;
  base.conv_template = {
    system_template:
      "<|turn>system\n" +
      (thinking ? "<|think|>\n" : "") +
      "{system_message}<turn|>\n",
    system_message: "",
    render_empty_system_message: thinking,
    roles: { user: "<|turn>user", assistant: "<|turn>model", tool: "" },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
    },
    seps: ["<turn|>\n"],
    role_content_sep: "\n",
    role_empty_sep: thinking ? "\n" : "\n<|channel>thought\n<channel|>",
    stop_str: ["<turn|>"],
    stop_token_ids: [1, 106, 50],
    system_prefix_token_ids: [2],
    reasoning_history_style: "gemma4",
  };
  return base;
}

test.each(cases)("$name with thinking=$thinking", (fixture) => {
  const request = { messages: fixture.messages };
  const chatConfig = config(fixture.thinking);
  const conv = getConversationFromChatCompletionRequest(request, chatConfig);
  const last = fixture.messages.at(-1)!;
  conv.appendMessage(Role.user, last.content as string);
  conv.appendReplyHeader(Role.assistant);
  expect(conv.getPromptArray()).toEqual([fixture.prompt]);
  expect(conv.getArtifactPromptSegments()).toEqual([fixture.prompt]);
  expect(conv.config.system_prefix_token_ids).toEqual([fixture.ids[0]]);
  expect(conv.getStopTokens()).toEqual([1, 106, 50]);
  expect(compareConversationObject(conv, conv)).toBe(false);
});

test("Gemma 4 selects thinking mode in the template", () => {
  expect(() =>
    getConversationFromChatCompletionRequest(
      {
        messages: [{ role: "user", content: "Hello" }],
        extra_body: { enable_thinking: false },
      },
      config(false),
    ),
  ).toThrow("static thinking mode");
});
