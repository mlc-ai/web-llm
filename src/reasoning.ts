import type { ConvTemplateConfig } from "./config";
import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionRequest,
} from "./openai_api_protocols";
import { ContentTypeError } from "./error";

type ReasoningProfile = ConvTemplateConfig["reasoning"];

export function preserveThinking(
  request: ChatCompletionRequest,
  profile: ReasoningProfile,
) {
  const preserve = request.extra_body?.preserve_thinking;
  if (preserve == null) return undefined;
  if (typeof preserve !== "boolean") {
    throw new Error("preserve_thinking must be a boolean or null.");
  }
  if (profile !== "gemma4") {
    throw new Error(
      "This conversation template does not support preserve_thinking.",
    );
  }
  return preserve;
}

export function lastUserIndex(
  messages: readonly ChatCompletionMessageParam[],
  profile: ReasoningProfile,
) {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message.role !== "user") continue;
    if (profile !== "qwen3" && profile !== "qwen3_5") return index;
    let content = message.content;
    if (typeof content !== "string") {
      if (profile === "qwen3") continue;
      if (content.some((part) => part.type !== "text")) return index;
      content = content
        .map((part) => (part.type === "text" ? part.text : ""))
        .join("");
    }
    if (profile === "qwen3_5") content = trimSpace(content);
    if (
      !(
        content.startsWith("<tool_response>") &&
        content.endsWith("</tool_response>")
      )
    )
      return index;
  }
  if (profile === "qwen3_5")
    throw new Error("No user query found in messages.");
  return profile === "qwen3" ? messages.length - 1 : -1;
}

/** Render separated reasoning without changing legacy requests or generation. */
export function formatAssistantReasoning(
  message: ChatCompletionAssistantMessageParam,
  profile: ReasoningProfile,
  current: boolean,
  preserve: boolean | undefined,
) {
  const reasoning = message.reasoning_content;
  if (typeof reasoning !== "string") {
    throw new Error("assistant reasoning_content must be a string or null.");
  }
  if (message.content != null && typeof message.content !== "string") {
    throw new ContentTypeError("assistant's message");
  }
  if (profile === undefined && reasoning === "") return message.content;
  if (message.tool_calls?.length && !message.content) {
    throw new Error(
      "Tool-call reasoning requires model-formatted tool-call content.",
    );
  }
  const content = message.content ?? "";
  switch (profile) {
    case "qwen3":
      if (!current || !reasoning) return content;
      return (
        "<think>\n" +
        trimNewlines(reasoning) +
        "\n</think>\n\n" +
        content.replace(/^\n+/, "")
      );
    case "qwen3_5":
      return current
        ? "<think>\n" +
            trimSpace(reasoning) +
            "\n</think>\n\n" +
            trimSpace(content)
        : trimSpace(content);
    case "gemma4": {
      // Calls are already formatted by the caller, including their literal argument values.
      const answer = message.tool_calls?.length
        ? content
        : trimSpace(
            content
              .split("<channel|>")
              .map((part) => part.split("<|channel>")[0])
              .join(""),
          );
      return reasoning && (current || (preserve && message.tool_calls?.length))
        ? "<|channel>thought\n" + reasoning + "\n<channel|>" + answer
        : answer;
    }
    case "ministral3":
      return reasoning ? "[THINK]" + reasoning + "[/THINK]" + content : content;
    case undefined:
      if (!reasoning) return content;
      throw new Error(
        "This conversation template does not support reasoning_content.",
      );
    default:
      throw new Error("Unknown conversation reasoning profile.");
  }
}

function trimNewlines(text: string) {
  let start = 0;
  let end = text.length;
  while (text[start] === "\n") start++;
  while (end > start && text[end - 1] === "\n") end--;
  return text.slice(start, end);
}

// Jinja uses Python's whitespace rules, which differ from JavaScript's trim().
function trimSpace(text: string) {
  const space =
    "\t\n\v\f\r\u001c\u001d\u001e\u001f \u0085\u00a0\u1680" +
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a" +
    "\u2028\u2029\u202f\u205f\u3000";
  let start = 0;
  let end = text.length;
  while (start < end && space.includes(text[start])) start++;
  while (end > start && space.includes(text[end - 1])) end--;
  return text.slice(start, end);
}
