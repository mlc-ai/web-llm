import { ConvTemplateConfig } from "./config";

/**
 * Helper to keep track of history conversations.
 */
export class Conversation {
  public messages: Array<[string, string | undefined]> = [];
  public config: ConvTemplateConfig;

  // TODO(tvm-team) confirm and remove
  // private contextWindowStart = 0;

  constructor(config: ConvTemplateConfig) {
    this.config = config;
  }

  private getPromptArrayInternal(
    addSystem: boolean,
    startPos: number
  ) {
    if (this.config.seps.length === 0) {
      throw Error("Need seps to work")
    }
    const ret = addSystem ? [this.config.system + this.config.seps[0]] : [];

    if (this.config.separator_style === "Two") {
      for (let i = startPos; i < this.messages.length; ++i) {
        const [role, message] = this.messages[i];
        if (message) {
          ret.push(role + ": " + message + this.config.seps[i % this.config.seps.length]);
        } else {
          ret.push(role + ":");
        }
      }
      return ret;
    } else if (this.config.separator_style === "RedPajamaChat") {
      for (let i = startPos; i < this.messages.length; ++i) {
        const [role, message] = this.messages[i];
        if (message) {
          ret.push(role + ": " + message + this.config.seps[i % this.config.seps.length] + "\n");
        } else {
          ret.push(role + ":");
        }
      }
      return ret;
    }
    throw Error("Unknown separator style " + this.config.separator_style);
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

  reset() {
    this.messages = [];
  }

  getStopStr() {
    if (this.config.stop_str !== "") {
      return this.config.stop_str;
    } else if (this.config.separator_style === "Two") {
      return this.config.seps[this.config.seps.length - 1];
    }
    throw Error("Unknown separator style " + this.config.separator_style);
  }

  appendMessage(role: string, message: string) {
    if (this.messages.length !== 0 &&
      this.messages[this.messages.length - 1][1] === undefined) {
      throw Error("Have unfinished reply");
    }
    this.messages.push([role, message]);
  }

  appendReplyHeader(role: string) {
    this.messages.push([role, undefined]);
  }

  finishReply(message: string) {
    if (this.messages.length === 0) {
      throw Error("Message error should not be 0");
    }
    if (this.messages[this.messages.length - 1][1] !== undefined) {
      throw Error("Already assigned");
    }
    this.messages[this.messages.length - 1][1] = message;
  }
}

export function getConversation(conv_template: string, conv_config?: Partial<ConvTemplateConfig>): Conversation {
  if (conv_template === "llama-2") {
    return new Conversation({
      system: "[INST] <<SYS>>\n\nYou are a helpful, respectful and honest assistant. " +
        "Always answer as helpfully as possible, while being safe. " +
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " +
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n" +
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n ",
      roles: ["[INST]", "[/INST]"],
      offset: 0,
      seps: [" ", " "],
      separator_style: "Two",
      stop_str: "[INST]",
      add_bos: true,
      ...conv_config,
    });
  } else if (conv_template === "vicuna_v1.1") {
    return new Conversation({
      system: "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
      roles: ["USER", "ASSISTANT"],
      offset: 0,
      seps: [" ", "</s>"],
      separator_style: "Two",
      stop_str: "</s>",
      add_bos: true,
      ...conv_config,
    });
  } else if (conv_template === "wizardlm") {
    return new Conversation({
      system: "You are an AI assistant that gives helpful, detailed, and polite answers to the user's questions.",
      roles: ["", "### Response"],
      offset: 0,
      seps: ["\n\n", "</s>"],
      separator_style: "Two",
      stop_str: "\n\n",
      add_bos: true,
      ...conv_config,
    })
  } else if (conv_template === "redpajama_chat") {
    return new Conversation({
      system: "",
      roles: ["<human>", "<bot>"],
      offset: 0,
      seps: ["", ""],
      separator_style: "RedPajamaChat",
      stop_str: "<human>",
      add_bos: false,
      ...conv_config,
    })
  } else {
    throw Error("Unknown conv template " + conv_template);
  }
}
