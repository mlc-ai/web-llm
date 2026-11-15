import { readFileSync } from "fs";
import { join } from "path";
import { ChatConfig, Role } from "../src/config";
import {
  compareConversationObject,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import { ChatCompletionRequest } from "../src/openai_api_protocols";
import { qwen3ChatConfigJSONString } from "./constants";

const templates = JSON.parse(
  readFileSync(join(__dirname, "fixtures/qwen36/mlc-templates.json"), "utf8"),
);
const fixture = JSON.parse(
  readFileSync(join(__dirname, "fixtures/qwen36/chat-template.json"), "utf8"),
);
function conversation(request: ChatCompletionRequest, template = "qwen3_6") {
  const config: ChatConfig = {
    ...JSON.parse(qwen3ChatConfigJSONString),
    conv_template: templates[template],
  };
  return getConversationFromChatCompletionRequest(request, config, true);
}
test.each(
  fixture.cases as {
    name: string;
    template: string;
    messages: ChatCompletionRequest["messages"];
    rendered: string;
  }[],
)("pinned HF $template/$name", (item) => {
  const request = { messages: item.messages };
  const snapshot = JSON.stringify(request);
  const conv = conversation(request, item.template);
  conv.appendReplyHeader(Role.assistant);
  expect(conv.getPromptArray().join("")).toBe(item.rendered);
  expect(conv.config.stop_token_ids).toEqual(fixture.stop_token_ids);
  expect(JSON.stringify(request)).toBe(snapshot);
});
test("normalized history cannot reuse the raw generated cache", () => {
  const request: ChatCompletionRequest = {
    messages: [{ role: "user", content: "hi" }],
  };
  expect(
    compareConversationObject(conversation(request), conversation(request)),
  ).toBe(false);
});
test.each(["enable_thinking", "preserve_thinking", "reasoning_effort"])(
  "rejects dynamic %s including null",
  (key) => {
    const messages = [{ role: "user", content: "hi" }];
    for (const option of [{ [key]: null }, { extra_body: { [key]: null } }]) {
      expect(() =>
        conversation({ messages, ...option } as ChatCompletionRequest),
      ).toThrow("static templates");
    }
  },
);
test.each(
  [
    [{ role: "assistant", content: "partial" }],
    [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    [{ role: "user", content: "<tool_response>x</tool_response>" }],
    [{ role: "user", content: "hi", name: null }],
    [{ role: "user", content: "hi", reasoning_content: "x" }],
    [
      { role: "assistant", content: "hi", reasoning_content: null },
      { role: "user", content: "next" },
    ],
  ].map((messages) => ({ messages })),
)("rejects unsupported messages %j", ({ messages }) => {
  expect(() =>
    conversation({ messages } as unknown as ChatCompletionRequest),
  ).toThrow();
});
test.each(["auto", "none"] as const)(
  "rejects explicit tool_choice %s",
  (tool_choice) => {
    expect(() =>
      conversation({
        messages: [{ role: "user", content: "hi" }],
        tool_choice,
      }),
    ).toThrow("plain text");
  },
);
