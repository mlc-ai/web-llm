import { ConvTemplateConfig, MessagePlaceholders, Role } from "./config";

/**
 * Helper to keep track of history conversations.
 */
export class Conversation {
  // NOTE: Update `compareConversationObject()` whenever a new state is introduced.
  public messages: Array<[Role, string, string | undefined]> = [];
  readonly config: ConvTemplateConfig;

  public function_string = "";
  public use_function_calling = false;
  public override_system_message?: string = undefined;

  // TODO(tvm-team) confirm and remove
  // private contextWindowStart = 0;

  constructor(config: ConvTemplateConfig) {
    this.config = config;
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
  getPrompArrayLastRound() {
    if (this.messages.length < 3) {
      throw Error("needs to call getPromptArray for the first message");
    }
    return this.getPromptArrayInternal(false, this.messages.length - 2);
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
    if (!(role in this.config.roles)) {
      throw Error("Role is not supported: " + role);
    }
    this.messages.push([role, this.config.roles[role], undefined]);
  }

  finishReply(message: string) {
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
): Conversation {
  // Update with conv_config
  return new Conversation({ ...conv_template, ...conv_config });
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
    convA.messages.length !== convB.messages.length
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
