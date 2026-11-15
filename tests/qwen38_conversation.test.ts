import { readFileSync } from "fs";
import { join } from "path";
import { ChatConfig, Role } from "../src/config";
import {
  compareConversationObject,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import { ChatCompletionRequest } from "../src/openai_api_protocols";
import { qwen3ChatConfigJSONString } from "./constants";

type Effort = "xhigh" | "medium" | "low" | "none";
const serializedTemplates = JSON.parse(
  readFileSync(join(__dirname, "fixtures/qwen38/mlc-templates.json"), "utf8"),
);
export function qwenConfig(effort: Effort): ChatConfig {
  const name = {
    xhigh: "qwen3_8",
    medium: "qwen3_8_medium",
    low: "qwen3_8_low",
    none: "qwen3_8_nothink",
  }[effort];
  return {
    ...JSON.parse(qwen3ChatConfigJSONString),
    conv_template: serializedTemplates[name],
  };
}
export function render(request: ChatCompletionRequest, effort: Effort): string {
  const c = getConversationFromChatCompletionRequest(
    request,
    qwenConfig(effort),
    true,
  );
  c.appendReplyHeader(Role.assistant);
  return c.getPromptArray().join("");
}
test("structured reasoning stays separate from literal inline tags; Python trim", () => {
  const request: ChatCompletionRequest = {
    messages: [
      { role: "user", content: "\u0085 first \u001c" },
      {
        role: "assistant",
        content: "\u2003<think>literal</think> answer\u00a0",
        reasoning_content: "\u0085 actual reasoning \u001c",
      },
      { role: "user", content: "\ufeffnext\ufeff" },
    ],
  };
  const snapshot = JSON.stringify(request);
  expect(render(request, "none")).toBe(
    "<|im_start|>user\nfirst<|im_end|>\n" +
      "<|im_start|>assistant\n<think>\nactual reasoning\n</think>\n\n<think>literal</think> answer<|im_end|>\n" +
      "<|im_start|>user\n\ufeffnext\ufeff<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  );
  expect(JSON.stringify(request)).toBe(snapshot);
});
test.each(["medium", "none"] as const)(
  "%s omits whitespace system",
  (effort) => {
    expect(
      render(
        {
          messages: [
            { role: "system", content: "\u0085\u001c" },
            { role: "user", content: "hi" },
          ],
        },
        effort,
      ),
    ).toBe(
      "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n" +
        (effort === "none" ? "\n</think>\n\n" : ""),
    );
  },
);
test.each(["enable_thinking", "reasoning_effort", "preserve_thinking"])(
  "rejects dynamic %s",
  (key) => {
    for (const value of [false, true, null, "medium"])
      for (const extra of [false, true]) {
        const request = {
          messages: [{ role: "user", content: "hi" }],
          ...(extra ? { extra_body: { [key]: value } } : { [key]: value }),
        } as ChatCompletionRequest;
        expect(() => render(request, "xhigh")).toThrow("static templates");
      }
  },
);
test.each(
  [
    [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    [{ role: "user", content: "hi", name: "Ada" }],
    [{ role: "assistant", content: "hi" }],
    [{ role: "user", content: "<tool_response>x</tool_response>" }],
    [{ role: "user", content: "hi", reasoning_content: "wrongrole" }],
    [
      { role: "assistant", content: "hi", reasoning_content: 3 },
      { role: "user", content: "next" },
    ],
  ].map((messages) => ({ messages })),
)("rejects unsupported messages %j", ({ messages }) => {
  expect(() =>
    render({ messages } as unknown as ChatCompletionRequest, "xhigh"),
  ).toThrow("plain text");
});
test("rebuilds caller history instead of reusing raw generated reasoning", () => {
  const request: ChatCompletionRequest = {
    messages: [
      { role: "user", content: "first" },
      { role: "assistant", content: "answer", reasoning_content: "reason" },
      { role: "user", content: "next" },
    ],
  };
  const a = getConversationFromChatCompletionRequest(
    request,
    qwenConfig("medium"),
  );
  const b = getConversationFromChatCompletionRequest(
    request,
    qwenConfig("medium"),
  );
  expect(compareConversationObject(a, b)).toBe(false);
});

// Generated independently with the pinned HF tokenizer's apply_chat_template.
const goldens = JSON.parse(
  readFileSync(join(__dirname, "fixtures/qwen38/chat-template.json"), "utf8"),
) as {
  stop_token_ids: number[];
  cases: {
    name: string;
    template: string;
    messages: ChatCompletionRequest["messages"];
    rendered: string;
  }[];
};
const efforts: Record<string, Effort> = {
  qwen3_8: "xhigh",
  qwen3_8_medium: "medium",
  qwen3_8_low: "low",
  qwen3_8_nothink: "none",
};
test.each(goldens.cases)("HF parity $template / $name", (row) => {
  const request = { messages: row.messages } as ChatCompletionRequest;
  expect(render(request, efforts[row.template])).toBe(row.rendered);
  expect(
    qwenConfig(efforts[row.template]).conv_template.stop_token_ids,
  ).toEqual(goldens.stop_token_ids);
});

test("Qwen3.8 system text does not interpret replacement patterns", () => {
  expect(
    render(
      {
        messages: [
          { role: "system", content: "literal $& $$ $` $'" },
          { role: "user", content: "hi" },
        ],
      },
      "medium",
    ),
  ).toContain("<|im_start|>system\nliteral $& $$ $` $'<|im_end|>\n");
});

test("serialized Qwen3.8 role templates preserve literal replacement syntax", () => {
  const literal =
    "literal $& $$ $` $' {function_string} {user_message} {assistant_message} {system_message} {tool_message}";
  expect(
    render(
      {
        messages: [
          { role: "user", content: literal },
          { role: "assistant", content: literal, reasoning_content: literal },
          { role: "user", content: literal },
        ],
      },
      "medium",
    ),
  ).toBe(
    `<|im_start|>user\n${literal}<|im_end|>\n<|im_start|>assistant\n<think>\n${literal}\n</think>\n\n${literal}<|im_end|>\n<|im_start|>user\n${literal}<|im_end|>\n<|im_start|>assistant\n<think>\n`,
  );
});

test.each([[], "invalid", 1].map((extra_body) => ({ extra_body })))(
  "Qwen3.8 rejects malformed extra_body %j",
  ({ extra_body }) => {
    expect(() =>
      render(
        {
          messages: [{ role: "user", content: "hi" }],
          extra_body,
        } as unknown as ChatCompletionRequest,
        "medium",
      ),
    ).toThrow("extra_body");
  },
);
test("Qwen3.8 accepts null extra_body without dynamic options", () => {
  expect(
    render(
      {
        messages: [{ role: "user", content: "hi" }],
        extra_body: null,
      } as unknown as ChatCompletionRequest,
      "medium",
    ),
  ).toBe(render({ messages: [{ role: "user", content: "hi" }] }, "medium"));
});
