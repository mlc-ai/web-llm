import { ChatConfig, Role } from "../src/config";
import {
  compareConversationObject,
  getConversation,
  getConversationFromChatCompletionRequest,
} from "../src/conversation";
import { describe, expect, test } from "@jest/globals";
import {
  llama2ChatConfigJSONString,
  phi3_5VisionChatConfigJSONString,
  qwen3ChatConfigJSONString,
} from "./constants";
import {
  ChatCompletionContentPartImage,
  ChatCompletionMessageParam,
  ChatCompletionRequest,
} from "../src/openai_api_protocols";

describe("Test basic conversation loading and getPromptArray", () => {
  test("Test from json", () => {
    const config_json = JSON.parse(llama2ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversation(config.conv_template);
    const config_obj = conversation.config;

    expect(config_obj.system_template).toEqual(
      "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
    );
    expect(config_obj.system_message).toEqual(
      "You are a helpful, respectful and honest assistant.",
    );
    expect(config_obj.roles.user).toEqual("[INST]");
    expect(config_obj.roles.assistant).toEqual("[/INST]");
    expect(config_obj.role_templates?.user).toEqual("{user_message}");
    expect(config_obj.role_templates?.assistant).toEqual("{assistant_message}");
    expect(config_obj.role_content_sep).toEqual(" ");
    expect(config_obj.role_empty_sep).toEqual(" ");
    expect(config_obj.seps).toEqual([" "]);
    expect(config_obj.stop_str).toEqual(["[INST]"]);
    expect(config_obj.stop_token_ids).toEqual([2]);
    expect(config_obj.system_prefix_token_ids).toEqual([1]);
    expect(config_obj.add_role_after_system_message).toBe(false);

    conversation.appendMessage(Role.user, "test1");
    conversation.appendMessage(Role.assistant, "test2");
    conversation.appendMessage(Role.user, "test3");
    conversation.appendReplyHeader(Role.assistant);
    const prompt = conversation.getPromptArray().join("");
    expect(prompt).toEqual(
      "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\ntest1 [/INST] test2 [INST] test3 [/INST] ",
    );
  });
});

describe("Test getConversationFromChatCompletionRequest with Qwen3", () => {
  test("Test Qwen3 appendEmptyThinkingReplyHeader", () => {
    const config_json = JSON.parse(qwen3ChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const conversation = getConversation(config.conv_template);

    conversation.appendMessage(Role.user, "test1");
    conversation.appendMessage(Role.assistant, "test2");
    const emptyThinkingBlockStr = "<think>\n\n</think>\n\n";
    conversation.appendEmptyThinkingReplyHeader(
      Role.user,
      emptyThinkingBlockStr,
    );
    const prompt = conversation.getPromptArray().join("");
    expect(prompt).toEqual(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
        "<|im_start|>user\n" +
        "test1<|im_end|>\n" +
        "<|im_start|>assistant\n" +
        "test2<|im_end|>\n" +
        "<|im_start|>user\n" +
        emptyThinkingBlockStr,
    );

    const message = emptyThinkingBlockStr + "test3";
    conversation.finishReply(message);
    expect(conversation.messages[conversation.messages.length - 1][2]).toEqual(
      message,
    );
  });
});

describe("Test getConversationFromChatCompletionRequest with image", () => {
  // Constants for testing
  type ImageURL = ChatCompletionContentPartImage.ImageURL;
  const dummySystemPromptStr = "dummy system prompt.";
  const dummyRequestStr = "dummy request.";
  const dummyResponseStr = "dummy response.";
  const imageUrl1 = "https://url1";
  const imageUrl2 = "https://url2";
  const imageUrl3 = "https://url3";
  const singleImageInputMessages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: dummySystemPromptStr,
    },
    {
      role: "user",
      content: [
        { type: "text", text: dummyRequestStr },
        {
          type: "image_url",
          image_url: {
            url: imageUrl1,
          },
        },
      ],
    },
  ];

  // system message, single-image user, assistant response, multi-image user
  const multiImageMultiRoundInputMessages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: dummySystemPromptStr,
    },
    {
      role: "user",
      content: [
        { type: "text", text: dummyRequestStr },
        {
          type: "image_url",
          image_url: {
            url: imageUrl1,
          },
        },
      ],
    },
    {
      role: "assistant",
      content: dummyResponseStr,
    },
    {
      role: "user",
      content: [
        { type: "text", text: dummyRequestStr },
        {
          type: "image_url",
          image_url: {
            url: imageUrl2,
          },
        },
        {
          type: "image_url",
          image_url: {
            url: imageUrl3,
          },
        },
      ],
    },
  ];

  test("Test compareConversationObject with different image input", () => {
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    // deep copy
    const messages1 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const messages2 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const messages3 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const messages4 = JSON.parse(JSON.stringify(singleImageInputMessages));
    messages3[1].content[1].image_url.url = "https://a_different_url";
    messages4[1].content[0].text = "a different text";
    const request1: ChatCompletionRequest = { messages: messages1 };
    const request2: ChatCompletionRequest = { messages: messages2 };
    const request3: ChatCompletionRequest = { messages: messages3 };
    const request4: ChatCompletionRequest = { messages: messages4 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    const conv2 = getConversationFromChatCompletionRequest(
      request2,
      config,
      true,
    );
    const conv3 = getConversationFromChatCompletionRequest(
      request3,
      config,
      true,
    );
    const conv4 = getConversationFromChatCompletionRequest(
      request4,
      config,
      true,
    );
    expect(compareConversationObject(conv1, conv2)).toEqual(true);
    expect(compareConversationObject(conv1, conv3)).toEqual(false);
    expect(compareConversationObject(conv2, conv3)).toEqual(false);
    expect(compareConversationObject(conv1, conv4)).toEqual(false);
  });

  test("Test getPromptArray with ContentPart array but only a single text", () => {
    // This should be equivalent to `content: dummyRequestStr`
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const messages1: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: dummySystemPromptStr,
      },
      {
        role: "user",
        content: [{ type: "text", text: dummyRequestStr }],
      },
    ];
    const messages2: ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: dummySystemPromptStr,
      },
      {
        role: "user",
        content: dummyRequestStr,
      },
    ];
    const request1: ChatCompletionRequest = { messages: messages1 };
    const request2: ChatCompletionRequest = { messages: messages2 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    const conv2 = getConversationFromChatCompletionRequest(
      request2,
      config,
      true,
    );
    expect(conv1.getPromptArray()).toEqual([
      dummySystemPromptStr,
      `<|user|>\n${dummyRequestStr}<|end|>\n`,
    ]);
    expect(conv1.getPromptArray()).toEqual(conv2.getPromptArray());
  });

  test("Test getPromptArray with single image input", () => {
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const messages1 = JSON.parse(JSON.stringify(singleImageInputMessages));
    const request1: ChatCompletionRequest = { messages: messages1 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    expect(conv1.getPromptArray()).toEqual([
      dummySystemPromptStr, // phi3_5-vision does not have system template
      [
        `<|user|>\n`,
        { url: imageUrl1 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
    ]);
  });

  test("Test multiple round with multiple image input, with reply header", () => {
    const config_json = JSON.parse(phi3_5VisionChatConfigJSONString);
    const config = { ...config_json } as ChatConfig;
    const messages1 = JSON.parse(
      JSON.stringify(multiImageMultiRoundInputMessages),
    );
    const request1: ChatCompletionRequest = { messages: messages1 };
    const conv1 = getConversationFromChatCompletionRequest(
      request1,
      config,
      true,
    );
    conv1.appendReplyHeader(Role.assistant);
    expect(conv1.getPromptArray()).toEqual([
      dummySystemPromptStr, // phi3_5-vision does not have system template
      [
        `<|user|>\n`,
        { url: imageUrl1 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
      `<|assistant|>\n${dummyResponseStr}<|end|>\n`,
      [
        `<|user|>\n`,
        { url: imageUrl2 } as ImageURL,
        `\n`,
        { url: imageUrl3 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
      `<|assistant|>\n`,
    ]);
    expect(conv1.getPromptArrayLastRound()).toEqual([
      [
        `<|user|>\n`,
        { url: imageUrl2 } as ImageURL,
        `\n`,
        { url: imageUrl3 } as ImageURL,
        `\n`,
        `${dummyRequestStr}<|end|>\n`,
      ],
      `<|assistant|>\n`,
    ]);
  });
});
