/**
 * Conversation template config
 */
export interface ConvTemplateConfig {
  system: string;
  roles: Array<string>;
  seps: Array<string>;
  separator_style: string;
  offset: number;
  stop_str: string;
  add_bos: boolean;
  stop_tokens: Array<number>;
}

/**
 * Config of one chat model
 */
export interface ChatConfig {
  tokenizer_files: Array<string>;
  conv_config?: Partial<ConvTemplateConfig>;
  conv_template: string;
  // additional metadata
  mean_gen_len: number;
  shift_fill_factor: number;
  repetition_penalty: number;
  top_p: number;
  temperature: number;
}

/**
 * Information for a model.
 * @param model_url: the huggingface link to download the model weights.
 * @param local_id: what we call the model.
 * @param model_lib: the model library the model uses.
 * @param required_features: feature needed to run this model (e.g. shader-f16).
 */
export interface ModelRecord {
  model_url: string;
  local_id: string;
  model_lib: string;
  required_features?: Array<string>;
}

/**
 * Extra configuration that can be
 * passed to the load.
 * 
 * @param model_list: models to be used.
 * @param model_lib_map: maps each `ModelRecord`'s `model_lib` to a url that
 * can download the model library.
 */
export interface AppConfig {
  model_list: Array<ModelRecord>;
  model_lib_map: Record<string, string>;
}

/**
 * Default models and model library mapping to be used if unspecified.
 */
export const prebuiltAppConfig: AppConfig = {
  model_list: [
    {
      "model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_1/resolve/main/",
      "model_lib": "RedPajama-INCITE-Chat-3B-v1-q4f32_1",
      "local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1"
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f32_1/resolve/main/",
      "model_lib": "Llama-2-7b-chat-hf-q4f32_1",
      "local_id": "Llama-2-7b-chat-hf-q4f32_1"
    }
  ],
  model_lib_map: {
    "Llama-2-7b-chat-hf-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-webgpu.wasm",
    "RedPajama-INCITE-Chat-3B-v1-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_1-webgpu.wasm"
  }
}
