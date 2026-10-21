import { describe, expect, test, jest } from "@jest/globals";
import type { ChatConfig, ConvTemplateConfig } from "../src/config";
import {
  compareConversationObject,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionRequest,
} from "../src/openai_api_protocols";
import { qwen3ChatConfigJSONString } from "./constants";
import { createEngineWithPipeline, MODEL_ID } from "./helpers/engine_fixture";

type Profile = ConvTemplateConfig["reasoning"];
const call: ChatCompletionAssistantMessageParam = {
  role: "assistant",
  content: "model-formatted-call",
  reasoning_content: "Read the code.",
  tool_calls: [
    {
      id: "call",
      type: "function",
      function: { name: "lookup", arguments: '{"code":"A7"}' },
    },
  ],
};
const profiles = [
  {
    profile: "qwen3",
    thought: "<think>\nPlan\n</think>\n\n",
    historical: false,
  },
  {
    profile: "qwen3_5",
    thought: "<think>\nPlan\n</think>\n\n",
    historical: false,
  },
  {
    profile: "gemma4",
    thought: "<|channel>thought\nPlan\n<channel|>",
    historical: false,
  },
  { profile: "ministral3", thought: "[THINK]Plan[/THINK]", historical: true },
] as const;
function config(profile?: Profile): ChatConfig {
  const config = JSON.parse(qwen3ChatConfigJSONString) as ChatConfig;
  config.conv_template.roles.tool = "<|im_start|>user";
  if (profile) config.conv_config = { reasoning: profile };
  return config;
}
function conversation(
  request: ChatCompletionRequest,
  profile: Profile = "qwen3",
) {
  return getConversationFromChatCompletionRequest(
    request,
    config(profile),
    true,
  );
}
function history(
  reasoning = "Plan",
  content: string | null = "Answer",
): ChatCompletionRequest {
  return {
    messages: [
      { role: "user", content: "Find A7." },
      { ...call, content, reasoning_content: reasoning, tool_calls: undefined },
      { role: "tool", tool_call_id: "call", content: "42" },
    ],
  };
}

describe("separated reasoning history", () => {
  test("matches Qwen profiles when history has no genuine user query", () => {
    const request = history();
    request.messages.shift();
    expect(conversation(request, "qwen3").messages[0][2]).toBe("Answer");
    expect(() => conversation(request, "qwen3_5")).toThrow(
      "No user query found",
    );
  });
  test.each(["\u001c", "\u0085", "\ufeff"])(
    "uses canonical whitespace rules for %j",
    (space) => {
      const thought = space + "Plan" + space;
      const content = space + "Answer" + space;
      const stripped = space === "\ufeff" ? content : "Answer";
      const strippedThought = space === "\ufeff" ? thought : "Plan";
      expect(
        conversation(history(thought, content), "qwen3_5").messages[1][2],
      ).toBe("<think>\n" + strippedThought + "\n</think>\n\n" + stripped);
      expect(
        conversation(history(thought, content), "gemma4").messages[1][2],
      ).toBe("<|channel>thought\n" + thought + "\n<channel|>" + stripped);
      expect(
        conversation(history(thought, content), "ministral3").messages[1][2],
      ).toBe("[THINK]" + thought + "[/THINK]" + content);
    },
  );
  test.each([
    "A <|channel> B",
    "A <channel|> B",
    "<|channel>thought\ntext<channel|>",
  ])(
    "keeps literal Gemma channel markers inside a formatted tool argument: %s",
    (value) => {
      const content =
        '<|tool_call>call:lookup{code:<|"|>' + value + '<|"|>}<tool_call|>';
      const request = history();
      request.messages[1] = { ...call, content, reasoning_content: "Plan" };
      expect(conversation(request, "gemma4").messages[1][2]).toBe(
        "<|channel>thought\nPlan\n<channel|>" + content,
      );
      expect(request.messages[1].content).toBe(content);
    },
  );
  test("does not broaden unsupported empty-reasoning content validation", () => {
    const request = history("", null);
    expect(() =>
      getConversationFromChatCompletionRequest(request, config(), true),
    ).toThrow("assistant's message");
  });
  test.each(profiles)(
    "$profile preserves each assistant/tool step in the current turn",
    ({ profile, thought }) => {
      const request = history();
      request.messages.push(
        { role: "assistant", content: "Answer", reasoning_content: "Plan" },
        { role: "tool", tool_call_id: "next", content: "43" },
      );
      const snapshot = structuredClone(request);
      request.messages.forEach(Object.freeze);
      Object.freeze(request.messages);
      Object.freeze(request);
      const output = conversation(request, profile);
      expect(output.messages[1][2]).toBe(thought + "Answer");
      expect(output.messages[3][2]).toBe(thought + "Answer");
      expect(request).toEqual(snapshot);
    },
  );
  test.each(profiles)(
    "$profile uses its own historical default",
    ({ profile, thought, historical }) => {
      const request = history();
      request.messages.push({ role: "user", content: "Next." });
      expect(conversation(request, profile).messages[1][2]).toBe(
        (historical ? thought : "") + "Answer",
      );
    },
  );
  test.each([undefined, null, false, true])(
    "Gemma keeps historical tool thoughts only when enabled: %s",
    (preserve) => {
      const request: ChatCompletionRequest = {
        messages: [
          { role: "user", content: "Find A7." },
          call,
          { role: "tool", tool_call_id: "call", content: "42" },
          { role: "assistant", content: "42", reasoning_content: "Summarize." },
          { role: "user", content: "Next." },
        ],
        extra_body: { preserve_thinking: preserve },
      };
      const output = conversation(request, "gemma4");
      expect(output.messages[1][2]).toBe(
        (preserve ? "<|channel>thought\nRead the code.\n<channel|>" : "") +
          call.content,
      );
      expect(output.messages[3][2]).toBe("42");
    },
  );
  test.each([undefined, "qwen3", "qwen3_5", "ministral3"] as const)(
    "rejects a preservation option unsupported by %s",
    (profile) => {
      for (const preserve of [true, false])
        expect(() =>
          getConversationFromChatCompletionRequest(
            {
              messages: [{ role: "user", content: "Hi" }],
              extra_body: { preserve_thinking: preserve },
            },
            config(profile),
          ),
        ).toThrow("does not support preserve_thinking");
    },
  );
  test("rejects unsupported reasoning instead of losing it", () => {
    expect(() =>
      getConversationFromChatCompletionRequest(history(), config(), true),
    ).toThrow("does not support reasoning_content");
  });
  test.each([false, 1, {}])("rejects invalid reasoning: %j", (value) => {
    const request = history();
    (
      request.messages[1] as ChatCompletionAssistantMessageParam
    ).reasoning_content = value as unknown as string;
    expect(() => conversation(request)).toThrow(
      "reasoning_content must be a string",
    );
  });
  test.each([false, 0, {}])(
    "preserves native content validation: %j",
    (value) => {
      const request = history();
      (request.messages[1] as ChatCompletionAssistantMessageParam).content =
        value as unknown as string;
      expect(() => conversation(request)).toThrow("assistant's message");
    },
  );
  test.each(["true", 1, {}])(
    "rejects invalid preservation flags: %j",
    (value) => {
      expect(() =>
        conversation(
          {
            messages: [{ role: "user", content: "Hi" }],
            extra_body: { preserve_thinking: value as unknown as boolean },
          },
          "gemma4",
        ),
      ).toThrow("preserve_thinking must be a boolean");
    },
  );
  test.each([undefined, null, ""])(
    "does not silently omit tool calls without content: %s",
    (content) => {
      for (const reasoning of ["", "Plan"]) {
        const request = history();
        request.messages[1] = {
          ...call,
          content,
          reasoning_content: reasoning,
        };
        expect(() => conversation(request, "gemma4")).toThrow(
          "model-formatted tool-call content",
        );
      }
    },
  );
  test.each([undefined, null, ""])(
    "keeps old raw strings unchanged when no separated reasoning is rendered: %s",
    (reasoning) => {
      const content = "<think>\nRaw thought\n</think>\n\nAnswer";
      const request: ChatCompletionRequest = {
        messages: [
          { role: "assistant", content, reasoning_content: reasoning },
          { role: "user", content: "Next" },
        ],
      };
      expect(conversation(request).messages[0][2]).toBe(content);
    },
  );
  test.each([null, ""])(
    "supports a reasoning-only assistant message (content=%s)",
    (content) => {
      expect(conversation(history("Plan", content)).messages[1][2]).toBe(
        "<think>\nPlan\n</think>\n\n",
      );
    },
  );
  test.each([
    ["qwen3", "", "\n\nAnswer\n", "\n\nAnswer\n"],
    ["qwen3", "\n", "\n\nAnswer\n", "<think>\n\n</think>\n\nAnswer\n"],
    ["qwen3", "\n Plan \n", "Answer", "<think>\n Plan \n</think>\n\nAnswer"],
    ["qwen3_5", "", "\n Answer \n", "<think>\n\n</think>\n\nAnswer"],
    ["qwen3_5", " \n\t", "Answer", "<think>\n\n</think>\n\nAnswer"],
    ["qwen3_5", "\n Plan \n", "Answer", "<think>\nPlan\n</think>\n\nAnswer"],
    ["gemma4", "\n", " Answer ", "<|channel>thought\n\n\n<channel|>Answer"],
    ["ministral3", "\n", " Answer ", "[THINK]\n[/THINK] Answer "],
  ] as const)(
    "matches canonical separated-field boundaries: %s %j",
    (profile, reasoning, content, expected) => {
      expect(
        conversation(history(reasoning, content), profile).messages[1][2],
      ).toBe(expected);
    },
  );
  test.each(["qwen3", "qwen3_5"] as const)(
    "%s keeps reasoning through a model-formatted tool response",
    (profile) => {
      const request = history();
      request.messages[2] = {
        role: "user",
        content: "<tool_response>42</tool_response>",
      };
      expect(conversation(request, profile).messages[1][2]).toContain(
        "<think>\nPlan",
      );
      request.messages[2] = { role: "user", content: "New query" };
      expect(conversation(request, profile).messages[1][2]).not.toContain(
        "<think>",
      );
    },
  );
  test("rejects unknown profiles on a separated-reasoning request", () => {
    expect(() => conversation(history(), "unknown" as Profile)).toThrow(
      "Unknown conversation reasoning profile",
    );
  });
  test("rendered reasoning participates in conversation equality", () => {
    const request = history();
    const raw = structuredClone(request);
    raw.messages[1] = {
      role: "assistant",
      content: "<think>\nPlan\n</think>\n\nAnswer",
    };
    expect(
      compareConversationObject(conversation(request), conversation(raw)),
    ).toBe(true);
    (raw.messages[1] as ChatCompletionAssistantMessageParam).content =
      "Different";
    expect(
      compareConversationObject(conversation(request), conversation(raw)),
    ).toBe(false);
  });
  test.each(["same", "thought", "retention"] as const)(
    "engine cache equality includes %s reasoning state",
    async (change) => {
      const { engine, pipeline } = createEngineWithPipeline(1);
      const modelConfig = config("gemma4");
      const request: ChatCompletionRequest = {
        model: MODEL_ID,
        messages: [
          { role: "user", content: "Hi" },
          call,
          { role: "user", content: "Next" },
        ],
        extra_body: { preserve_thinking: true },
      };
      pipeline.setConversation(
        getConversationFromChatCompletionRequest(request, modelConfig),
      );
      jest.spyOn(pipeline, "getFinishReason").mockReturnValue("stop");
      const reset = jest.spyOn(pipeline, "resetChat");
      const next = structuredClone(request);
      if (change === "thought")
        (
          next.messages[1] as ChatCompletionAssistantMessageParam
        ).reasoning_content = "Different";
      if (change === "retention") next.extra_body!.preserve_thinking = false;
      await engine.prefill(next, pipeline, modelConfig, { max_tokens: 1 });
      expect(reset).toHaveBeenCalledTimes(change === "same" ? 0 : 1);
    },
  );
});
