import { ChatConfig } from "../src/config";

export const llama3_1ChatConfig: ChatConfig = {
  vocab_size: 128256,
  context_window_size: 131072,
  sliding_window_size: -1,
  attention_sink_size: -1,
  temperature: 0.6,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  repetition_penalty: 1.0,
  top_p: 0.9,
  tokenizer_files: ["tokenizer.json", "tokenizer_config.json"],
  tokenizer_info: {
    token_postproc_method: "byte_level",
    prepend_space_in_encode: false,
    strip_space_in_decode: false,
  },
  conv_template: {
    system_template:
      "<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
    system_message: "You are a helpful, respectful and honest assistant.",
    system_prefix_token_ids: [128000],
    add_role_after_system_message: true,
    roles: {
      user: "<|start_header_id|>user",
      assistant: "<|start_header_id|>assistant",
      tool: "<|start_header_id|>ipython",
    },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
      tool: "{tool_message}",
    },
    seps: ["<|eot_id|>"],
    role_content_sep: "<|end_header_id|>\n\n",
    role_empty_sep: "<|end_header_id|>\n\n",
    stop_str: [],
    stop_token_ids: [128001, 128008, 128009],
  },
  bos_token_id: 128000,
};
