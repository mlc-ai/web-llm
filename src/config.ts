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
}

/**
 * Config of one chat model
 */
export interface ChatConfig {
  local_id: string;
  model_lib: string;
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

export interface ModelRecord {
  model_url: string;
  local_id: string;
  required_features?: Array<string>;
}
/**
 * Extra configuration that can be
 * passed to the load
 */
export interface AppConfig {
  model_list: Array<ModelRecord>;
  model_lib_map: Record<string, string>;
}

/**
 * default libmap used in prebuilt
 */
export const prebuiltAppConfig : AppConfig = {
	model_list: [
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_0"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0/resolve/main/",
			"local_id": "vicuna-v1-7b-q4f32_0"
		}
	],
  model_lib_map: {
    "vicuna-v1-7b-q4f32_0": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/vicuna-v1-7b-q4f32_0-webgpu-v1.wasm",
    "RedPajama-INCITE-Chat-3B-v1-q4f32_0": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_0-webgpu-v1.wasm"
  }
}
