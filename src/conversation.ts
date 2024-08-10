import {
  ChatConfig,
  ConvTemplateConfig,
  MessagePlaceholders,
  Role,
} from "./config";
import {
  ChatCompletionMessageParam,
  ChatCompletionRequest,
} from "./openai_api_protocols/index";
import {
  ContentTypeError,
  FunctionNotFoundError,
  InvalidToolChoiceError,
  MessageOrderError,
  SystemMessageOrderError,
  TextCompletionConversationError,
  TextCompletionConversationExpectsPrompt,
  UnsupportedRoleError,
  UnsupportedToolChoiceTypeError,
  UnsupportedToolTypeError,
} from "./error";

/**
 * Helper to keep track of history conversations.
 */
export class Conversation {
  // NOTE: Update `compareConversationObject()` whenever a new state is introduced.
  public messages: Array<[Role, string, string | undefined]> = [];
  readonly config: ConvTemplateConfig;

  /** Whether the Conversation object is for text completion with no conversation-style formatting */
  public isTextCompletion: boolean;
  /** Used when isTextCompletion is true */
  public prompt: string | undefined;

  public function_string = "";
  public use_function_calling = false;
  public override_system_message?: string = undefined;

  // TODO(tvm-team) confirm and remove
  // private contextWindowStart = 0;

  constructor(config: ConvTemplateConfig, isTextCompletion = false) {
    this.config = config;
    this.isTextCompletion = isTextCompletion;
  }

  private getPromptArrayInternal(addSystem: boolean, startPos: number) {
    if (this.config.seps.length == 0) {
      throw Error("Need seps to work");
    }

    // Prepare system message
    // Get overrided system message if exists, else use default one in config
    let system_message = this.config.system_message;
    if (this.override_system_message !== undefined) {
      system_message = this.override_system_message;
    }
    const system_prompt = this.config.system_template.replace(
      MessagePlaceholders.system,
      system_message,
    );
    const ret = addSystem ? [system_prompt] : [];

    // Process each message in this.messages
    for (let i = startPos; i < this.messages.length; ++i) {
      const item = this.messages[i];
      const role = item[0];
      const role_str = item[1];
      const message = item[2];

      if (message !== undefined && message != "") {
        let message_str;
        if (this.config.role_templates !== undefined) {
          message_str = this.config.role_templates[role]?.replace(
            MessagePlaceholders[Role[role] as keyof typeof MessagePlaceholders],
            message,
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
          message_str = message;
        }
        let role_prefix;
        if (
          this.config.add_role_after_system_message === false &&
          system_prompt != "" &&
          i == 0
        ) {
          role_prefix = "";
        } else {
          const content_sep = this.config.role_content_sep
            ? this.config.role_content_sep
            : ": ";
          role_prefix = role_str + content_sep;
        }
        ret.push(
          role_prefix +
            message_str +
            this.config.seps[i % this.config.seps.length],
        );
      } else {
        const empty_sep = this.config.role_empty_sep
          ? this.config.role_empty_sep
          : ": ";
        ret.push(role_str + empty_sep);
      }
    }
    return ret;
  }

  /**
   * Get prompt arrays with the first one as system.
   *
   * @returns The prompt array.
   */
  getPromptArray(): Array<string> {
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
      throw new TextCompletionConversationError("getPromptyArrayLastRound");
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
    if (this.config.stop_str.length > 0) {
      return this.config.stop_str;
    }
    return [this.config.seps[this.config.seps.length - 1]];
  }

  getStopTokens() {
    return this.config.stop_token_ids;
  }

  appendMessage(role: Role, message: string, role_name?: string) {
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

  finishReply(message: string) {
    if (this.isTextCompletion) {
      throw new TextCompletionConversationError("finishReply");
    }
    if (this.messages.length == 0) {
      throw Error("Message error should not be 0");
    }
    if (this.messages[this.messages.length - 1][2] !== undefined) {
      throw Error("Already assigned");
    }
    this.messages[this.messages.length - 1][2] = message;
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

  const msgLen = convA.messages.length;
  const msgEntryLen = convA.messages[0].length;
  for (let i = 0; i < msgLen; i++) {
    for (let j = 0; j < msgEntryLen; j++) {
      if (convA.messages[i][j] !== convB.messages[i][j]) {
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
 * @note `request.messages[-1]` is not included as it would be treated as a normal input to
 * `prefill()`.
 */
export function getConversationFromChatCompletionRequest(
  request: ChatCompletionRequest,
  config: ChatConfig,
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
  if (
    (input[lastId].role !== "user" && input[lastId].role !== "tool") ||
    typeof input[lastId].content !== "string"
  ) {
    // TODO(Charlie): modify condition after we support multimodal inputs
    throw new MessageOrderError(
      "The last message should be a string from the `user` or `tool`.",
    );
  }
  for (let i = 0; i < input.length - 1; i++) {
    const message: ChatCompletionMessageParam = input[i];
    if (message.role === "system") {
      if (i !== 0) {
        throw new SystemMessageOrderError();
      }
      conversation.override_system_message = message.content;
    } else if (message.role === "user") {
      if (typeof message.content !== "string") {
        // TODO(Charlie): modify condition after we support multimodal inputs
        throw new ContentTypeError(message.role + "'s message");
      }
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
 * @returns The string used to set Conversatoin.function_string
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
