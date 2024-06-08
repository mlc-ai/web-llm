import { ChatConfig, Role } from "../src/config";
import { getConversation } from "../src/conversation";
import { describe, expect, test } from "@jest/globals";

describe("Test conversation template", () => {
  test("Test from json", () => {
    const config_str =
      "{" +
      '  "model_type": "llama",' +
      '  "quantization": "q4f16_1",' +
      '  "model_config": {' +
      '    "hidden_size": 4096,' +
      '    "intermediate_size": 11008,' +
      '    "num_attention_heads": 32,' +
      '    "num_hidden_layers": 32,' +
      '    "rms_norm_eps": 1e-05,' +
      '    "vocab_size": 32000,' +
      '    "position_embedding_base": 10000,' +
      '    "context_window_size": 4096,' +
      '    "prefill_chunk_size": 4096,' +
      '    "num_key_value_heads": 32,' +
      '    "head_dim": 128,' +
      '    "tensor_parallel_shards": 1,' +
      '    "max_batch_size": 80' +
      "  }," +
      '  "vocab_size": 32000,' +
      '  "context_window_size": 4096,' +
      '  "sliding_window_size": -1,' +
      '  "prefill_chunk_size": 4096,' +
      '  "attention_sink_size": -1,' +
      '  "tensor_parallel_shards": 1,' +
      '  "mean_gen_len": 128,' +
      '  "max_gen_len": 512,' +
      '  "shift_fill_factor": 0.3,' +
      '  "temperature": 0.6,' +
      '  "presence_penalty": 0.0,' +
      '  "frequency_penalty": 0.0,' +
      '  "repetition_penalty": 1.0,' +
      '  "top_p": 0.9,' +
      '  "conv_template": {' +
      '    "name": "llama-2",' +
      '    "system_template": "[INST] <<SYS>>\\n{system_message}\\n<</SYS>>\\n\\n",' +
      '    "system_message": "You are a helpful, respectful and honest assistant.",' +
      '    "system_prefix_token_ids": [' +
      "      1" +
      "    ]," +
      '    "add_role_after_system_message": false,' +
      '    "roles": {' +
      '      "user": "[INST]",' +
      '      "assistant": "[/INST]",' +
      '      "tool": "[INST]"' +
      "    }," +
      '    "role_templates": {' +
      '      "user": "{user_message}",' +
      '      "assistant": "{assistant_message}",' +
      '      "tool": "{tool_message}"' +
      "    }," +
      '    "messages": [],' +
      '    "seps": [' +
      '      " "' +
      "    ]," +
      '    "role_content_sep": " ",' +
      '    "role_empty_sep": " ",' +
      '    "stop_str": [' +
      '      "[INST]"' +
      "    ]," +
      '    "stop_token_ids": [' +
      "      2" +
      "    ]," +
      '    "function_string": "",' +
      '    "use_function_calling": false' +
      "  }," +
      '  "pad_token_id": 0,' +
      '  "bos_token_id": 1,' +
      '  "eos_token_id": 2,' +
      '  "tokenizer_files": [' +
      '    "tokenizer.model",' +
      '    "tokenizer.json",' +
      '    "tokenizer_config.json"' +
      "  ]," +
      '  "version": "0.1.0"' +
      "}";
    const config_json = JSON.parse(config_str);
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
    console.log(prompt);
  });
});
