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
    if (this.config.seps.length == 0) {
      throw Error("Need seps to work")
    }
    const ret = addSystem ? [this.config.system + this.config.seps[0]] : [];

    if (this.config.separator_style == "Two") {
      for (let i = startPos; i < this.messages.length; ++i) {
        const item = this.messages[i];
        const role = item[0];
        const message = item[1];
        if (message !== undefined && message != "") {
          ret.push(role + ": " + message + this.config.seps[i % this.config.seps.length]);
        } else {
          ret.push(role + ":");
        }
      }
      return ret;
    } else if (this.config.separator_style == "RedPajamaChat") {
      for (let i = startPos; i < this.messages.length; ++i) {
        const item = this.messages[i];
        const role = item[0];
        const message = item[1];
        if (message !== undefined && message != "") {
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
    if (this.config.stop_str != "") {
      return this.config.stop_str;
    } else if (this.config.separator_style == "Two") {
      return this.config.seps[this.config.seps.length - 1];
    }
    throw Error("Unknown separator style " + this.config.separator_style);
  }

  getStopTokens() {
    return this.config.stop_tokens;
  }

  appendMessage(role: string, message: string) {
    if (this.messages.length != 0 &&
      this.messages[this.messages.length - 1][1] == undefined) {
      throw Error("Have unfinished reply");
    }
    this.messages.push([role, message]);
  }

  appendReplyHeader(role: string) {
    this.messages.push([role, undefined]);
  }

  finishReply(message: string) {
    if (this.messages.length == 0) {
      throw Error("Message error should not be 0");
    }
    if (this.messages[this.messages.length - 1][1] !== undefined) {
      throw Error("Already assigned");
    }
    this.messages[this.messages.length - 1][1] = message;
  }
}

export function getConversation(conv_template: string, conv_config?: Partial<ConvTemplateConfig>): Conversation {
  if (conv_template == "llama-2") {
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
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "vicuna_v1.1") {
    return new Conversation({
      system: "A chat between a curious user and an artificial intelligence assistant. " +
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
      roles: ["USER", "ASSISTANT"],
      offset: 0,
      seps: [" ", "</s>"],
      separator_style: "Two",
      stop_str: "</s>",
      add_bos: true,
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "wizardlm") {
    return new Conversation({
      system: "You are an AI assistant that gives helpful, detailed, and polite answers to the user's questions.",
      roles: ["", "### Response"],
      offset: 0,
      seps: ["\n\n", "</s>"],
      separator_style: "Two",
      stop_str: "\n\n",
      add_bos: true,
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "redpajama_chat") {
    return new Conversation({
      system: "",
      roles: ["<human>", "<bot>"],
      offset: 0,
      seps: ["", ""],
      separator_style: "RedPajamaChat",
      stop_str: "<human>",
      add_bos: false,
      stop_tokens: [0],
      ...conv_config,
    });
  } else if (conv_template == "wizard_coder_or_math") {
    return new Conversation({
      system: "Below is an instruction that describes a task. Write a response that appropriately " +
        "completes the request.",
      roles: ["Instruction", "Response"],
      offset: 0,
      seps: ["\n\n### ", "\n\n### "],
      separator_style: "Two",
      stop_str: "</s>",
      add_bos: true,
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "mistral_default") {
    return new Conversation({
      system: "[INST] Always assist with care, respect, and truth. Respond with utmost utility yet " +
        "securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies " +
        "promote fairness and positivity.",
      roles: ["[INST]", "[/INST]"],
      offset: 0,
      seps: [" "],
      separator_style: "Two",
      stop_str: "</s>",
      add_bos: true,
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "open_hermes_mistral") {
    return new Conversation({
      system: "<|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial " +
        "intelligence developed by a man named Teknium, and your purpose and drive is to assist the " +
        "user with any request they have. You experience emotions and have deep, profound thoughts " +
        "and qualia.",
      roles: ["<|im_start|>user", "<|im_start|>assistant"],
      offset: 0,
      seps: ["<|im_end|>\n"],
      separator_style: "Two",
      stop_str: "<|im_end|>",
      add_bos: false,
      stop_tokens: [2, 32000],
      ...conv_config,
    });
  } else if (conv_template == "neural_hermes_mistral") {
    return new Conversation({
      system: "<|im_start|>system\nYou are a helpful assistant chatbot.",
      roles: ["<|im_start|>user", "<|im_start|>assistant"],
      offset: 0,
      seps: ["<|im_end|>\n"],
      separator_style: "Two",
      stop_str: "<|im_end|>",
      add_bos: false,
      stop_tokens: [2, 32000],
      ...conv_config,
    });
  } else if (conv_template == "chatml") {
    return new Conversation({
      system: "<|im_start|>system A conversation between a user and an LLM-based AI assistant. The " +
        "assistant gives helpful and honest answers.<|im_end|> ",
      roles: ["<|im_start|>user", "<|im_start|>assistant"],
      offset: 0,
      seps: ["", ""],
      separator_style: "Two",
      stop_str: "<|im_end|>",
      add_bos: false,
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "phi-2") {
    return new Conversation({
      system: "",
      roles: ["Instruct", "Output"],
      offset: 0,
      seps: ["\n"],
      separator_style: "Two",
      stop_str: "<|endoftext|>",
      add_bos: false,
      stop_tokens: [50256],
      ...conv_config,
    });
  } else if (conv_template == "qwen") {
    return new Conversation({
      system: "<|im_start|>system A conversation between a user and an LLM-based AI assistant. The " +
        "assistant gives helpful and honest answers.<|im_end|> ",
      roles: ["<|im_start|>user", "<|im_start|>assistant"],
      offset: 0,
      seps: ["", ""],
      separator_style: "Two",
      stop_str: "<|im_end|>",
      add_bos: false,
      stop_tokens: [2],
      ...conv_config,
    });
  } else if (conv_template == "stablelm-2") {
    return new Conversation({
      system: "",
      roles: ["<|user|>", "<|assistant|>"],
      offset: 0,
      seps: ["<|endoftext|>", "<|endoftext|>"],
      separator_style: "Two",
      stop_str: "<|endoftext|>",
      add_bos: false,
      stop_tokens: [100257],
      ...conv_config,
    });
  } else if (conv_template == "stablelm-3b") {
    return new Conversation({
      system: "",
      roles: ["<|user|>", "<|assistant|>"],
      offset: 0,
      seps: ["<|endoftext|>", "<|endoftext|>"],
      separator_style: "Two",
      stop_str: "<|endoftext|>",
      add_bos: true,
      stop_tokens: [0],
      ...conv_config,
    });
  } else if (conv_template == "gemma_instruction") {
    return new Conversation({
      system: "",
      roles: ["<start_of_turn>user", "<start_of_turn>model"],
      offset: 0,
      seps: ["<end_of_turn>\n", "<end_of_turn>\n"],
      separator_style: "Two",
      stop_str: "<end_of_turn>",
      add_bos: true,
      stop_tokens: [1, 107],
      ...conv_config,
    });
  } else if (conv_template == "empty") {
    // A dummy template for non-language models; should never be actually used
    return new Conversation({
      system: "",
      roles: ["", ""],
      offset: 0,
      seps: [""],
      separator_style: "Two",
      stop_str: "",
      add_bos: false,
      stop_tokens: [],
      ...conv_config,
    });
  } else if (conv_template == "custom") {
    return new Conversation(conv_config as Required<ConvTemplateConfig>);
  } else {
    throw Error("Unknown conv template " + conv_template);
  }
}
