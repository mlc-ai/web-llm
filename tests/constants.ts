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

export const llama2ChatConfigJSONString =
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

export const phi3_5VisionChatConfigJSONString = String.raw`{
  "version": "0.1.0",
  "model_type": "phi3_v",
  "quantization": "q4f16_1",
  "model_config": {
    "model_type": "phi3_v",
    "hidden_size": 3072,
    "vocab_size": 32064,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 8192,
    "rms_norm_eps": 1e-05,
    "num_key_value_heads": 32,
    "max_position_embeddings": 131072,
    "vision_config": {
      "hidden_size": 1024,
      "image_size": 336,
      "intermediate_size": 4096,
      "num_attention_heads": 16,
      "num_hidden_layers": 24,
      "patch_size": 14,
      "projection_dim": 768,
      "vocab_size": null,
      "num_channels": 3,
      "layer_norm_eps": 1e-05,
      "kwargs": {}
    },
    "img_processor": {
      "image_dim_out": 1024,
      "model_name": "openai/clip-vit-large-patch14-336",
      "name": "clip_vision_model",
      "num_img_tokens": 144
    },
    "position_embedding_base": 10000.0,
    "original_max_position_embeddings": 4096,
    "context_window_size": 131072,
    "prefill_chunk_size": 2048,
    "head_dim": 96,
    "tensor_parallel_shards": 1,
    "max_batch_size": 80
  },
  "vocab_size": 32064,
  "context_window_size": 131072,
  "sliding_window_size": -1,
  "prefill_chunk_size": 2048,
  "attention_sink_size": -1,
  "tensor_parallel_shards": 1,
  "pipeline_parallel_stages": 1,
  "temperature": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "top_p": 1.0,
  "tokenizer_files": [
    "tokenizer.json",
    "tokenizer_config.json"
  ],
  "tokenizer_info": {
    "token_postproc_method": "byte_fallback",
    "prepend_space_in_encode": true,
    "strip_space_in_decode": true
  },
  "conv_template": {
    "name": "phi-3-vision",
    "system_template": "{system_message}",
    "system_message": "\n",
    "system_prefix_token_ids": [
      1
    ],
    "add_role_after_system_message": true,
    "roles": {
      "user": "<|user|>",
      "assistant": "<|assistant|>"
    },
    "role_templates": {
      "user": "{user_message}",
      "assistant": "{assistant_message}",
      "tool": "{tool_message}"
    },
    "messages": [],
    "seps": [
      "<|end|>\n"
    ],
    "role_content_sep": "\n",
    "role_empty_sep": "\n",
    "stop_str": [
      "<|endoftext|>"
    ],
    "stop_token_ids": [
      2,
      32000,
      32001,
      32007
    ],
    "function_string": "",
    "use_function_calling": false
  },
  "pad_token_id": 32000,
  "bos_token_id": 1,
  "eos_token_id": 2
}`;

export const qwen3ChatConfigJSONString = String.raw`{
  "version": "0.1.0",
  "model_type": "qwen3",
  "quantization": "q0f32",
  "model_config": {
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "attention_bias": false,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "vocab_size": 151936,
    "tie_word_embeddings": true,
    "context_window_size": 40960,
    "prefill_chunk_size": 2048,
    "tensor_parallel_shards": 1,
    "head_dim": 128,
    "dtype": "float32",
    "max_batch_size": 128,
    "weight_block_size": null
  },
  "vocab_size": 151936,
  "context_window_size": 40960,
  "sliding_window_size": -1,
  "prefill_chunk_size": 2048,
  "attention_sink_size": -1,
  "tensor_parallel_shards": 1,
  "pipeline_parallel_stages": 1,
  "temperature": 0.6,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "top_p": 0.95,
  "tokenizer_files": [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json"
  ],
  "tokenizer_info": {
    "token_postproc_method": "byte_level",
    "prepend_space_in_encode": false,
    "strip_space_in_decode": false
  },
  "conv_template": {
    "name": "qwen2",
    "system_template": "<|im_start|>system\n{system_message}<|im_end|>\n",
    "system_message": "You are a helpful assistant.",
    "system_prefix_token_ids": null,
    "add_role_after_system_message": true,
    "roles": {
      "user": "<|im_start|>user",
      "assistant": "<|im_start|>assistant"
    },
    "role_templates": {
      "user": "{user_message}",
      "assistant": "{assistant_message}",
      "tool": "{tool_message}"
    },
    "messages": [],
    "seps": [
      "<|im_end|>\n"
    ],
    "role_content_sep": "\n",
    "role_empty_sep": "\n",
    "stop_str": [
      "<|endoftext|>",
      "<|im_end|>"
    ],
    "stop_token_ids": [
      151643,
      151645
    ],
    "function_string": "",
    "use_function_calling": false
  },
  "pad_token_id": 151643,
  "bos_token_id": 151643,
  "eos_token_id": [
    151645,
    151643
  ]
}`;
