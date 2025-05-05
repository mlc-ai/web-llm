import {
  ChatConfig,
  ConvTemplateConfig,
  MessagePlaceholders,
  Role,
} from "./config";
import {
  ChatCompletionContentPart,
  ChatCompletionContentPartImage,
  ChatCompletionMessageParam,
  ChatCompletionRequest,
} from "./openai_api_protocols/index";
import {
  ContentTypeError,
  FunctionNotFoundError,
  InvalidToolChoiceError,
  MessageOrderError,
  MultipleTextContentError,
  SystemMessageOrderError,
  TextCompletionConversationError,
  TextCompletionConversationExpectsPrompt,
  UnsupportedRoleError,
  UnsupportedToolChoiceTypeError,
  UnsupportedToolTypeError,
} from "./error";

type ImageURL = ChatCompletionContentPartImage.ImageURL;

/**
 * Helper to keep track of history conversations.
 */
export class Conversation {
  // NOTE: Update `compareConversationObject()` whenever a new state is introduced.
  /** Each message is a tuple of (Role, role_name_str, message), where message can be either a
   *  string or an array of contentPart for possible image input.
   */
  public messages: Array<
    [Role, string, string | Array<ChatCompletionContentPart> | undefined]
  > = [];
  readonly config: ConvTemplateConfig;

  /** Whether the Conversation object is for text completion with no conversation-style formatting */
  public isTextCompletion: boolean;
  /** Used when isTextCompletion is true */
  public prompt: string | undefined;

  public function_string = "";
  public use_function_calling = false;
  public override_system_message?: string = undefined;

  /**
   * Tracks whether the last message is an empty thinking block. Should only
   * be true when we are in the middle of a generation. Will be set to
   * false when the reply is finished with `finishReply()`.
   */
  private isLastMessageEmptyThinkingReplyHeader = false;

  // TODO(tvm-team) confirm and remove
  // private contextWindowStart = 0;

  constructor(config: ConvTemplateConfig, isTextCompletion = false) {
    this.config = config;
    this.isTextCompletion = isTextCompletion;
  }

  // TODO: Consider rewriting this method, a bit messy.
  private getPromptArrayInternal(
    addSystem: boolean,
    startPos: number,
  ): Array<string | Array<string | ImageURL>> {
    if (this.config.seps.length == 0) {
      throw Error("Need seps to work");
    }

    // Prepare system message
    // Get overridden system message if exists, else use default one in config
    let system_message = this.config.system_message;
    if (this.override_system_message !== undefined) {
      system_message = this.override_system_message;
    }
    const system_prompt = this.config.system_template.replace(
      MessagePlaceholders.system,
      system_message,
    );
    const ret: Array<string | Array<string | ImageURL>> =
      addSystem && system_prompt !== "" ? [system_prompt] : [];

    // Process each message in this.messages
    for (let i = startPos; i < this.messages.length; ++i) {
      const item = this.messages[i];
      const role = item[0];
      const role_str = item[1];
      const messageContent = item[2];

      // 1. Message from `appendReplyHeader()`, message is empty; not much processing is needed.
      if (messageContent === undefined) {
        if (i !== this.messages.length - 1) {
          throw new Error(
            "InternalError: Only expect message to be undefined for last " +
              "message for a reply header.",
          );
        }
        // Add ": " if there is no such field. If "", do not add sep
        const empty_sep =
          this.config.role_empty_sep || this.config.role_empty_sep == ""
            ? this.config.role_empty_sep
            : ": ";
        ret.push(role_str + empty_sep);
        continue;
      }

      // 2. Message from `appendEmptyThinkingReplyHeader()`, message is an empty thinking block.
      if (
        this.isLastMessageEmptyThinkingReplyHeader &&
        i === this.messages.length - 1
      ) {
        // TODO(Charlie): content_sep or empty_sep? For Qwen3, both are "\n".
        const content_sep =
          this.config.role_content_sep || this.config.role_content_sep == ""
            ? this.config.role_content_sep
            : ": ";
        ret.push(role_str + content_sep + messageContent);
        continue;
      }

      // 3. Each messageContent consists of one textPart, and >= 0 imageParts, regardless whether
      // it is Array<ChatCompletionContentPart> or text message. So we extract out each.
      let textContentPart = ""; // if no textPart, use an empty string
      const imageContentParts: ImageURL[] = [];
      if (Array.isArray(messageContent)) {
        // 2.1 content is Array<ChatCompletionContentPart>
        // Iterate through the contentParts, get the text and list of images. There should
        // be only a single text. TODO: is it always the case the number of textContentPart <= 1?
        let seenText = false;
        for (let i = 0; i < messageContent.length; i++) {
          const curContentPart = messageContent[i];
          if (curContentPart.type === "text") {
            if (seenText) {
              throw new MultipleTextContentError();
            }
            textContentPart = curContentPart.text;
            seenText = true;
          } else {
            imageContentParts.push(curContentPart.image_url);
          }
        }
      } else {
        // 2.2 content is just a string
        textContentPart = messageContent;
      }

      // 3. Format textContentPart with role and sep to get message_str and role_prefix
      let message_str;
      let role_prefix;
      if (this.config.role_templates !== undefined) {
        message_str = this.config.role_templates[role]?.replace(
          MessagePlaceholders[Role[role] as keyof typeof MessagePlaceholders],
          textContentPart,
        );
        if (this.use_function_calling && this.function_string !== "") {
          message_str = message_str?.replace(
            MessagePlaceholders.function,
            this.function_string,
          );
        }
        message_str = message_str?.replace(MessagePlaceholders.function, "");
      }

      if (message_str == undefined) {
        message_str = textContentPart;
      }
      if (
        this.config.add_role_after_system_message === false &&
        system_prompt != "" &&
        i == 0
      ) {
        role_prefix = "";
      } else {
        // Add ": " if there is no such field. If "", do not add sep
        const content_sep =
          this.config.role_content_sep || this.config.role_content_sep == ""
            ? this.config.role_content_sep
            : ": ";
        role_prefix = role_str + content_sep;
      }

      // 4. Combine everything together
      if (imageContentParts.length === 0) {
        // If no image, just a single string to represent this message
        ret.push(
          role_prefix +
            message_str +
            this.config.seps[i % this.config.seps.length],
        );
      } else {
        // If has image input, currently we hard code it to Phi3.5-vision's format:
        // `<|user|>\n<|image_1|>\n<|image_2|>\n{prompt}<|end|>\n`
        // So we will return a list for this:
        // [`<|user|>\n`, imageUrl1, `\n`, imageUrl2, `\n`, `{prompt}<|end|>\n`]
        const curMessageList: Array<string | ImageURL> = [role_prefix];
        imageContentParts.forEach((curImage: ImageURL) => {
          curMessageList.push(curImage);
          curMessageList.push("\n");
        });
        curMessageList.push(
          message_str + this.config.seps[i % this.config.seps.length],
        );
        ret.push(curMessageList);
      }
    }
    return ret;
  }

  /**
   * Get prompt arrays with the first one as system.
   *
   * It is returned as an array of `string | Array<string | ImageURL>`, where each element of
   * the array represents the formatted message of a role/turn. If the message only contains text,
   * it will be a string that concatenates the role string, message, and separators. If the
   * message contains image(s), it will be an array of string and ImageURL in the order of which
   * they will be prefilled into the model. e.g. it can be something like
   * [
   *   "<|system|>\nSome system prompt\n",
   *   [
   *     "<|user|>\n",
   *     imageURL1,
   *     "\n",
   *     imageURL2,
   *     "\n",
   *     "Some user input<|end|>\n"
   *   ],
   * ]
   *
   * @returns The prompt array.
   */
  getPromptArray(): Array<string | Array<string | ImageURL>> {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError("getPromptArray");
    }
    return this.getPromptArrayInternal(true, 0);
  }

  /**
   * Get the last round of prompt has not been fed as input.
   *
   * @note This function needs to be used with the assumption that
   *       the caller call appendMessage then appendReplyHeader.
   *
   * @returns The prompt array.
   */
  getPromptArrayLastRound() {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError("getPromptArrayLastRound");
    }
    if (this.messages.length < 3) {
      throw Error("needs to call getPromptArray for the first message");
    }
    return this.getPromptArrayInternal(false, this.messages.length - 2);
  }

  /**
   * Return prompt in an array for non-conversation text completion.
   */
  getPromptArrayTextCompletion(): Array<string> {
    if (!this.isTextCompletion || this.prompt === undefined) {
      throw new TextCompletionConversationExpectsPrompt();
    }
    return [this.prompt];
  }

  /**
   * Resets all states for this.conversation.
   */
  reset() {
    // Note: Update this whenever we introduce a new state to conversation.
    this.messages = [];
    this.override_system_message = undefined;
    this.function_string = "";
    this.use_function_calling = false;
    this.isTextCompletion = false;
    this.prompt = undefined;
  }

  getStopStr(): string[] {
    // TODO(Charlie): Is this needed?
    // if (this.config.stop_str.length > 0) {
    //   return this.config.stop_str;
    // }
    // return [this.config.seps[this.config.seps.length - 1]];
    return this.config.stop_str;
  }

  getStopTokens() {
    return this.config.stop_token_ids;
  }

  appendMessage(
    role: Role,
    message: string | Array<ChatCompletionContentPart>,
    role_name?: string,
  ) {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError("appendMessage");
    }
    if (
      this.messages.length != 0 &&
      this.messages[this.messages.length - 1][2] == undefined
    ) {
      throw Error("Have unfinished reply");
    }
    if (!(role in this.config.roles)) {
      throw Error("Role is not supported: " + role);
    }
    const role_name_str = role_name ? role_name : this.config.roles[role];
    this.messages.push([role, role_name_str, message]);
  }

  appendReplyHeader(role: Role) {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError("appendReplyHeader");
    }
    if (!(role in this.config.roles)) {
      throw Error("Role is not supported: " + role);
    }
    this.messages.push([role, this.config.roles[role], undefined]);
  }

  appendEmptyThinkingReplyHeader(role: Role, emptyThinkingBlockStr: string) {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError(
        "appendEmptyThinkingReplyHeader",
      );
    }
    this.isLastMessageEmptyThinkingReplyHeader = true;
    this.messages.push([role, this.config.roles[role], emptyThinkingBlockStr]);
  }

  finishReply(message: string) {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError("finishReply");
    }
    if (this.messages.length == 0) {
      throw Error("Message error should not be 0");
    }
    if (
      this.messages[this.messages.length - 1][2] !== undefined &&
      // If the last message has an empty thinknig block, last message is expected
      // to be non-empty.
      this.isLastMessageEmptyThinkingReplyHeader === false
    ) {
      throw Error("Already assigned");
    }
    this.messages[this.messages.length - 1][2] = message;
    this.isLastMessageEmptyThinkingReplyHeader = false;
  }
}

export function getConversation(
  conv_template: ConvTemplateConfig,
  conv_config?: Partial<ConvTemplateConfig>,
  isTextCompletion = false,
): Conversation {
  // Update with conv_config
  return new Conversation(
    { ...conv_template, ...conv_config },
    isTextCompletion,
  );
}

/**
 * Compare the states of two conversation instances. Equality is defined as their getPromptArray()
 * should return the exact same things, which is determined by fields: messages, function_string,
 * use_function_calling, and override_system_message.
 *
 * @returns True if `convA` equals to `convB`
 * @note We assume convA and convB has the same `this.config`.
 */
export function compareConversationObject(
  convA: Conversation,
  convB: Conversation,
): boolean {
  // NOTE: Update this function whenever a new state is introduced to `Conversation`.
  // Check the easy ones first
  if (
    convA.function_string !== convB.function_string ||
    convA.use_function_calling !== convB.use_function_calling ||
    convA.override_system_message !== convB.override_system_message ||
    convA.messages.length !== convB.messages.length ||
    convA.isTextCompletion !== convB.isTextCompletion
  ) {
    return false;
  }

  // Then check message
  if (convA.messages.length === 0 && convB.messages.length === 0) {
    // both are empty
    return true;
  }
  if (convA.messages.length !== convB.messages.length) {
    // different number of messages
    return false;
  }

  const msgLen = convA.messages.length;
  const msgEntryLen = convA.messages[0].length; // always 3 for now
  for (let i = 0; i < msgLen; i++) {
    for (let j = 0; j < msgEntryLen; j++) {
      const entryA = convA.messages[i][j];
      const entryB = convB.messages[i][j];
      if (typeof entryA === "string" && typeof entryB === "string") {
        // Case 1: both are strings
        if (convA.messages[i][j] !== convB.messages[i][j]) {
          return false;
        }
      } else if (entryA === undefined && entryB === undefined) {
        // Case 2: both undefined
        continue;
      } else if (Array.isArray(entryA) && Array.isArray(entryB)) {
        // Case 3: both are ChatCompletionContentPart[]
        if (entryA.length !== entryB.length) {
          return false;
        }
        const numContentParts = entryA.length;
        for (let k = 0; k < numContentParts; k++) {
          const entryA_k = entryA[k];
          const entryB_k = entryB[k];
          if (entryA_k.type === "text" && entryB_k.type === "text") {
            // Case 3.1: both are text
            if (entryA_k.text !== entryB_k.text) {
              return false;
            }
          } else if (
            entryA_k.type === "image_url" &&
            entryB_k.type === "image_url"
          ) {
            // Case 3.2: both are image_url
            if (
              entryA_k.image_url.url !== entryB_k.image_url.url ||
              entryA_k.image_url.detail !== entryB_k.image_url.detail
            ) {
              return false;
            }
          } else {
            // Case 3.3: of different type
            return false;
          }
        }
      } else {
        // Case 4: two entries are of different types
        return false;
      }
    }
  }
  return true;
}

/**
 * Get a new Conversation object based on the chat completion request.
 *
 * @param request The incoming ChatCompletionRequest
 * @param includeLastMsg Include last message, by default is false. Set to true for testing only.
 * @note By default, `request.messages[-1]` is not included as it would be treated as a normal
 * input to `prefill()`.
 */
export function getConversationFromChatCompletionRequest(
  request: ChatCompletionRequest,
  config: ChatConfig,
  includeLastMsg = false,
): Conversation {
  // 0. Instantiate a new Conversation object
  const conversation = getConversation(
    config.conv_template,
    config.conv_config,
  );

  // 1. Populate function-calling-related fields
  // TODO: either remove these or support gorilla-like function calling models.
  // These commented code was used to support gorilla, but we could not use grammar to
  // guarantee its output, nor make it conform to OpenAI's function calling output. Kept for now.
  // const functionCallUsage = this.getFunctionCallUsage(request);
  // conversation.function_string = functionCallUsage;
  // conversation.use_function_calling = functionCallUsage !== "";

  // 2. Populate conversation.messages
  const input = request.messages;
  const lastId = input.length - 1;
  if (input[lastId].role !== "user" && input[lastId].role !== "tool") {
    throw new MessageOrderError(
      "The last message should be from the `user` or `tool`.",
    );
  }
  const iterEnd = includeLastMsg ? input.length : input.length - 1;
  for (let i = 0; i < iterEnd; i++) {
    const message: ChatCompletionMessageParam = input[i];
    if (message.role === "system") {
      if (i !== 0) {
        throw new SystemMessageOrderError();
      }
      conversation.override_system_message = message.content;
    } else if (message.role === "user") {
      conversation.appendMessage(Role.user, message.content, message.name);
    } else if (message.role === "assistant") {
      if (typeof message.content !== "string") {
        throw new ContentTypeError(message.role + "'s message");
      }
      conversation.appendMessage(Role.assistant, message.content, message.name);
    } else if (message.role === "tool") {
      conversation.appendMessage(Role.tool, message.content);
    } else {
      // Use `["role"]` instead of `.role` to suppress "Property does not exist on type 'never'"
      throw new UnsupportedRoleError(message["role"]);
    }
  }
  return conversation;
}

/**
 * Returns the function string based on the request.tools and request.tool_choice, raises erros if
 * encounter invalid request.
 *
 * @param request The chatCompletionRequest we are about to prefill for.
 * @returns The string used to set Conversation.function_string
 */
export function getFunctionCallUsage(request: ChatCompletionRequest): string {
  if (
    request.tools == undefined ||
    (typeof request.tool_choice == "string" && request.tool_choice == "none")
  ) {
    return "";
  }
  if (
    typeof request.tool_choice == "string" &&
    request.tool_choice !== "auto"
  ) {
    throw new InvalidToolChoiceError(request.tool_choice);
  }
  if (
    typeof request.tool_choice !== "string" &&
    request.tool_choice?.type !== "function"
  ) {
    throw new UnsupportedToolChoiceTypeError();
  }

  const singleFunctionToCall =
    typeof request.tool_choice !== "string" &&
    request.tool_choice?.function?.name;
  if (singleFunctionToCall) {
    for (const f of request.tools) {
      if (singleFunctionToCall == f.function.name) {
        return JSON.stringify([f.function]);
      }
    }
    throw new FunctionNotFoundError(singleFunctionToCall);
  }

  const function_list = [];
  for (const f of request.tools) {
    if (f.type !== "function") {
      throw new UnsupportedToolTypeError();
    }
    function_list.push(f.function);
  }
  return JSON.stringify(function_list);
}
