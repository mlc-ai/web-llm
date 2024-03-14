/* eslint-disable @typescript-eslint/no-non-null-assertion */
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
 * Config of one chat model, a data structure representing `mlc-chat-config.json`.
 * This only corresponds to the chat-related fields and `tokenizer_files` of `mlc-chat-config.json`.
 * Only these fields affect the conversation in runtime.
 * i.e. The third part in https://llm.mlc.ai/docs/get_started/mlc_chat_config.html.
 * 
 * This is initialized in `ChatModule.reload()` with the model's `mlc-chat-config.json`.
 */
export interface ChatConfig {
  // First three fields affect the entire conversation, i.e. used in `ChatModule.reload()`
  tokenizer_files: Array<string>;
  conv_config?: Partial<ConvTemplateConfig>;
  conv_template: string;
  // Fields below can be swapped per-generation via `GenerationConfig`
  // Fields only used in MLC
  mean_gen_len: number;
  max_gen_len: number;
  shift_fill_factor: number;
  repetition_penalty: number;
  frequency_penalty: number;
  presence_penalty: number;
  // Fields shared by MLC and OpenAI APIs
  top_p: number;
  temperature: number;
  bos_token_id?: number;
}

/**
 * Custom options that can be used to override known config values.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface ChatOptions extends Partial<ChatConfig> { }

/**
 * Config for a single generation.
 * Essentially `ChatConfig` without `tokenizer_files`, `conv_config`, or `conv_template`.
 * We also support additional fields not present in `mlc-chat-config.json` due to OpenAI-like APIs.
 * 
 * Note that all values are optional. If unspecified, we use whatever values in `ChatConfig`
 * initialized during `ChatModule.reload()`.
 */
export interface GenerationConfig {
  // Only used in MLC
  mean_gen_len?: number;
  shift_fill_factor?: number;
  repetition_penalty?: number;
  // Shared by MLC and OpenAI APIs
  top_p?: number | null;
  temperature?: number | null;
  max_gen_len?: number | null;
  // Only in OpenAI APIs
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
  stop?: string | null | Array<string>;
  n?: number | null;
  logit_bias?: Record<string, number> | null;
  logprobs?: boolean | null;
  top_logprobs?: number | null;
}

export function postInitAndCheckGenerationConfigValues(config: GenerationConfig): void {
  function _hasValue(value: any): boolean {
    // if we use `if value` directly, `value` being 0 evaluates to false, violating semantics
    return value !== undefined && value !== null;
  }
  if (config.frequency_penalty && (config.frequency_penalty < -2.0 || config.frequency_penalty > 2.0)) {
    throw new Error("`frequency_penalty` should be between -2.0 and 2.0.");
  }
  if (config.presence_penalty && (config.presence_penalty < -2.0 || config.presence_penalty > 2.0)) {
    throw new Error("`presence_penalty` should be between -2.0 and 2.0.");
  }
  if (_hasValue(config.repetition_penalty) && config.repetition_penalty! <= 0) {
    throw new Error("Make sure `repetition_penalty` > 0.");
  }
  if (_hasValue(config.max_gen_len) && config.max_gen_len! <= 0) {
    throw new Error("`max_gen_len` should be greater than zero.");
  }
  if (_hasValue(config.mean_gen_len) && config.mean_gen_len! <= 0) {
    throw new Error("`mean_gen_len` should be greater than zero.");
  }
  if (_hasValue(config.shift_fill_factor) && config.shift_fill_factor! <= 0 || config.shift_fill_factor! > 1) {
    throw new Error("Make sure 0 < `shift_fill_factor` <= 1.");
  }
  if (_hasValue(config.top_p) && config.top_p! <= 0 || config.top_p! >= 1) {
    throw new Error("Make sure 0 < `top_p` < 1.");
  }
  if (_hasValue(config.temperature) && config.temperature! < 0) {
    throw new Error("Make sure `temperature` >= 0.");
  }
  // If only one of frequency or presence penatly is set, make the other one 0.0
  if (_hasValue(config.frequency_penalty) && !_hasValue(config.presence_penalty)) {
    config.presence_penalty = 0.0;
    console.log("Only frequency_penalty is set; we default presence_penaty to 0.")
  }
  if (_hasValue(config.presence_penalty) && !_hasValue(config.frequency_penalty)) {
    config.frequency_penalty = 0.0;
    console.log("Only presence_penalty is set; we default frequency_penalty to 0.")
  }
  // Check logit_bias range
  if (_hasValue(config.logit_bias)) {
    for (const tokenID in config.logit_bias) {
      const bias = config.logit_bias[tokenID];
      if (bias > 100 || bias < -100) {
        throw new Error(
          "logit_bias should be in range [-100, 100]; got " + bias + "for tokenID " + tokenID
        );
      }
      if (isNaN(parseInt(tokenID))) {
        throw new Error(
          "Expect logit_bias's keys to be number represented in string; got " + tokenID
        )
      }
    }
  }
  // logprobs and top_logprobs
  if (_hasValue(config.top_logprobs)) {
    // If top_logprobs is non-null, logprobs must be true
    if (!config.logprobs) {
      throw new Error("`logprobs` must be true if `top_logprobs` is set.");
    }
    // top_logprobs should be in range [0,5]
    if ((config.top_logprobs! < 0 || config.top_logprobs! > 5)) {
      throw new Error("`top_logprobs` should be in range [0,5]; got " + config.top_logprobs);
    }
  }
  // If defined logprobs but not top_logprobs, simply make it 0
  if (config.logprobs) {
    if (!_hasValue(config.top_logprobs)) {
      config.top_logprobs = 0;
    }
  }
}

/**
 * Information for a model.
 * @param model_url: the huggingface link to download the model weights.
 * @param local_id: what we call the model.
 * @param model_lib_url: link to the model library (wasm file) the model uses.
 * @param vram_required_MB: amount of vram in MB required to run the model (can use
 *    `utils/vram_requirements` to calculate).
 * @param low_resource_required: whether the model can run on limited devices (e.g. Android phone).
 * @param buffer_size_required_bytes: required `maxStorageBufferBindingSize`, different for each device.
 * @param required_features: feature needed to run this model (e.g. shader-f16).
 */
export interface ModelRecord {
  model_url: string;
  local_id: string;
  model_lib_url: string;
  vram_required_MB?: number;
  low_resource_required?: boolean;
  buffer_size_required_bytes?: number;
  required_features?: Array<string>;
}

/**
 * Extra configuration that can be
 * passed to the load.
 * 
 * @param model_list: models to be used.
 */
export interface AppConfig {
  model_list: Array<ModelRecord>;
}

/**
 * Default models and model library mapping to be used if unspecified.
 */
export const prebuiltAppConfig: AppConfig = {
  model_list: [
    {
      "model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC/resolve/main/",
      "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k-webgpu.wasm",
      "local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1",
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC/resolve/main/",
      "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      "local_id": "Llama-2-7b-chat-hf-q4f32_1"
    }
  ]
}
