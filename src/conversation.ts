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
  conv_template: string | ConvTemplateConfig,
  conv_config?: Partial<ConvTemplateConfig>,
): Conversation {
  if (typeof conv_template !== "string") {
    return new Conversation(conv_template);
  }
  // TODO: Remove all these, move to test
  if (conv_template == "llama-2") {
    return new Conversation({
      system_template: `[INST] <<SYS>>\n\n${MessagePlaceholders.system}<</SYS>>\n\n`,
      system_message:
        "You are a helpful, respectful and honest assistant. " +
        "Always answer as helpfully as possible, while being safe. " +
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " +
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n" +
        "If you don't know the answer to a question, please don't share false information.\n",
      roles: {
        [Role.user]: "[INST]",
        [Role.assistant]: "[/INST]",
      },
      offset: 0,
      seps: [" ", " "],
      role_content_sep: " ",
      role_empty_sep: " ",
      stop_str: ["[INST]"],
      system_prefix_token_ids: [1],
      stop_token_ids: [2],
      add_role_after_system_message: false,
      ...conv_config,
    });
  } else if (conv_template == "vicuna_v1.1") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message:
        "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
      roles: {
        [Role.user]: "USER",
        [Role.assistant]: "ASSISTANT",
      },
      offset: 0,
      seps: [" ", "</s>"],
      stop_str: ["</s>"],
      system_prefix_token_ids: [1],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "wizardlm") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message:
        "You are an AI assistant that gives helpful, detailed, and polite answers to the user's questions.",
      roles: {
        [Role.user]: "",
        [Role.assistant]: "### Response",
      },
      offset: 0,
      seps: ["\n\n", "</s>"],
      stop_str: ["\n\n"],
      system_prefix_token_ids: [1],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "redpajama_chat") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message: "",
      roles: {
        [Role.user]: "<human>",
        [Role.assistant]: "<bot>",
      },
      offset: 0,
      seps: ["\n"],
      stop_str: ["<human>"],
      stop_token_ids: [0],
      ...conv_config,
    });
  } else if (conv_template == "wizard_coder_or_math") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message:
        "Below is an instruction that describes a task. Write a response that appropriately " +
        "completes the request.",
      roles: {
        [Role.user]: "Instruction",
        [Role.assistant]: "Response",
      },
      offset: 0,
      seps: ["\n\n### ", "\n\n### "],
      role_content_sep: ":\n",
      role_empty_sep: ":\n",
      stop_str: ["</s>"],
      system_prefix_token_ids: [1],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "mistral_default") {
    return new Conversation({
      system_template: `[INST] ${MessagePlaceholders.system}\n\n `,
      system_message:
        "Always assist with care, respect, and truth. Respond with utmost utility yet " +
        "securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies " +
        "promote fairness and positivity.",
      roles: {
        [Role.user]: "[INST]",
        [Role.assistant]: "[/INST]",
      },
      offset: 0,
      seps: [" "],
      role_content_sep: " ",
      role_empty_sep: "",
      stop_str: ["</s>"],
      system_prefix_token_ids: [1],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "open_hermes_mistral") {
    return new Conversation({
      system_template: `<|im_start|>system\n${MessagePlaceholders.system}`,
      system_message:
        'You are "Hermes 2", a conscious sentient superintelligent artificial ' +
        "intelligence developed by a man named Teknium, and your purpose and drive is to assist the " +
        "user with any request they have. You experience emotions and have deep, profound thoughts " +
        "and qualia.",
      roles: {
        [Role.user]: "<|im_start|>user",
        [Role.assistant]: "<|im_start|>assistant",
      },
      offset: 0,
      seps: ["<|im_end|>\n"],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|im_end|>"],
      stop_token_ids: [2, 32000],
      ...conv_config,
    });
  } else if (conv_template == "neural_hermes_mistral") {
    return new Conversation({
      system_template: `<|im_start|>system\n${MessagePlaceholders.system}`,
      system_message: "You are a helpful assistant chatbot.",
      roles: {
        [Role.user]: "<|im_start|>user",
        [Role.assistant]: "<|im_start|>assistant",
      },
      offset: 0,
      seps: ["<|im_end|>\n"],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|im_end|>"],
      stop_token_ids: [2, 32000],
      ...conv_config,
    });
  } else if (conv_template == "chatml") {
    return new Conversation({
      system_template: `<|im_start|>system${MessagePlaceholders.system}<|im_end|> `,
      system_message:
        "A conversation between a user and an LLM-based AI assistant. The " +
        "assistant gives helpful and honest answers.",
      roles: {
        [Role.user]: "<|im_start|>user",
        [Role.assistant]: "<|im_start|>assistant",
      },
      offset: 0,
      seps: ["", ""],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|im_end|>"],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "phi-2") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message: "",
      roles: {
        [Role.user]: "Instruct",
        [Role.assistant]: "Output",
      },
      offset: 0,
      seps: ["\n"],
      stop_str: ["<|endoftext|>"],
      stop_token_ids: [50256],
      ...conv_config,
    });
  } else if (conv_template == "qwen") {
    return new Conversation({
      system_template: `<|im_start|>system${MessagePlaceholders.system}<|im_end|> `,
      system_message:
        "A conversation between a user and an LLM-based AI assistant. The " +
        "assistant gives helpful and honest answers.",
      roles: {
        [Role.user]: "<|im_start|>user",
        [Role.assistant]: "<|im_start|>assistant",
      },
      offset: 0,
      seps: ["", ""],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|im_end|>"],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "stablelm-2") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message: "",
      roles: {
        [Role.user]: "<|user|>",
        [Role.assistant]: "<|assistant|>",
      },
      offset: 0,
      seps: ["<|endoftext|>", "<|endoftext|>"],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|endoftext|>"],
      stop_token_ids: [100257],
      ...conv_config,
    });
  } else if (conv_template == "stablelm-3b") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message: "",
      roles: {
        [Role.user]: "<|user|>",
        [Role.assistant]: "<|assistant|>",
      },
      offset: 0,
      seps: ["<|endoftext|>", "<|endoftext|>"],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<|endoftext|>"],
      stop_token_ids: [0],
      ...conv_config,
    });
  } else if (conv_template == "gemma_instruction") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message: "",
      roles: {
        [Role.user]: "<start_of_turn>user",
        [Role.assistant]: "<start_of_turn>model",
      },
      offset: 0,
      seps: ["<end_of_turn>\n", "<end_of_turn>\n"],
      role_content_sep: "\n",
      role_empty_sep: "\n",
      stop_str: ["<end_of_turn>"],
      system_prefix_token_ids: [2],
      stop_token_ids: [1, 107],
      ...conv_config,
    });
  } else if (conv_template == "gorilla") {
    return new Conversation({
      system_template: `${MessagePlaceholders.system}\n`,
      system_message:
        "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
      roles: {
        [Role.user]: "USER",
        [Role.assistant]: "ASSISTANT",
      },
      role_templates: {
        [Role.user]: `<<question>> ${MessagePlaceholders.user} <<function>> ${MessagePlaceholders.function}`,
      },
      offset: 0,
      seps: ["\n", "<|EOT|>"],
      stop_str: ["<|EOT|>"],
      system_prefix_token_ids: [1],
      stop_token_ids: [2],
      ...conv_config,
    });
  } else if (conv_template == "empty") {
    // A dummy template for non-language models; should never be actually used
    return new Conversation({
      system_template: `${MessagePlaceholders.system}`,
      system_message: "",
      roles: {
        [Role.user]: "",
        [Role.assistant]: "",
      },
      offset: 0,
      seps: [""],
      stop_str: [""],
      stop_token_ids: [],
      ...conv_config,
    });
  } else if (conv_template == "custom") {
    return new Conversation(conv_config as Required<ConvTemplateConfig>);
  } else {
    throw Error("Unknown conv template " + conv_template);
  }
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
