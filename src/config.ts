/* eslint-disable @typescript-eslint/no-non-null-assertion */

import { ResponseFormat } from "./openai_api_protocols";

/**
 * Conversation template config
 */
export interface ConvTemplateConfig {
  system_template: string;
  system_message: string;
  roles: Record<Role, string>;
  role_templates?: Partial<Record<Role, string>>;
  seps: Array<string>;
  role_content_sep?: string;
  role_empty_sep?: string;
  offset: number;
  stop_str: Array<string>;
  system_prefix_token_ids?: Array<number>;
  stop_token_ids: Array<number>;
  add_role_after_system_message?: boolean;
}

export enum Role {
  user = "user",
  assistant = "assistant"
}

/**
 * Place holders that can be used in role templates.
 * For example, a role template of
 * `<<question>> ${MessagePlaceholders.USER} <<function>> ${MessagePlaceholders.FUNCTION}`
 * will insert the user message to ${MessagePlaceholders.USER}
 * and insert the function message to ${MessagePlaceholders.FUNCTION}
 * at run time.
 */
export enum MessagePlaceholders {
  system = "{system_message}",
  user = "{user_message}",
  assistant = "{assistant_message}",
  tool = "{tool_message}",
  function = "{function_string}"
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
  conv_template: string | ConvTemplateConfig;
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
  response_format?: ResponseFormat | null;
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
  if (_hasValue(config.top_p) && config.top_p! <= 0 || config.top_p! > 1) {
    throw new Error("Make sure 0 < `top_p` <= 1.");
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
 * @param model_id: what we call the model.
 * @param model_lib_url: link to the model library (wasm file) the model uses.
 * @param vram_required_MB: amount of vram in MB required to run the model (can use
 *    `utils/vram_requirements` to calculate).
 * @param low_resource_required: whether the model can run on limited devices (e.g. Android phone).
 * @param buffer_size_required_bytes: required `maxStorageBufferBindingSize`, different for each device.
 * @param required_features: feature needed to run this model (e.g. shader-f16).
 */
export interface ModelRecord {
  model_url: string;
  model_id: string;
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
 * modelVersion: the prebuilt model libraries that the current npm is compatible with, affects the
 * `model_lib_url`s in `prebuiltAppConfig`.
 * 
 * @note The model version does not have to match the npm version, since not each npm update
 * requires an update of the model libraries.
 */
export const modelVersion = "v0_2_30";
export const modelLibURLPrefix =
  "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/";

/**
 * Default models and model library mapping to be used if unspecified.
 * 
 * @note This is the only source of truth of which prebuilt model libraries are compatible with the
 * current WebLLM npm version.
 */
export const prebuiltAppConfig: AppConfig = {
  model_list: [
    // Llama-2
    {
      "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC/resolve/main/",
      "model_id": "Llama-2-7b-chat-hf-q4f32_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      "vram_required_MB": 9109.03,
      "low_resource_required": false,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC/resolve/main/",
      "model_id": "Llama-2-7b-chat-hf-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Llama-2-7b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      "vram_required_MB": 6749.02,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC/resolve/main/",
      "model_id": "Llama-2-7b-chat-hf-q4f16_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Llama-2-7b-chat-hf-q4f16_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 4618.52,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/Llama-2-13b-chat-hf-q4f16_1-MLC/resolve/main/",
      "model_id": "Llama-2-13b-chat-hf-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Llama-2-13b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      "vram_required_MB": 11814.09,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/Llama-2-70b-chat-hf-q4f16_1-MLC/resolve/main/",
      "model_id": "Llama-2-70b-chat-hf-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Llama-2-70b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      "vram_required_MB": 43729.05,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    // Mistral variants
    {
      "model_url": "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC/resolve/main/",
      "model_id": "WizardMath-7B-V1.1-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
      "vram_required_MB": 6079.02,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/resolve/main/",
      "model_id": "Mistral-7B-Instruct-v0.2-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
      "vram_required_MB": 6079.02,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/OpenHermes-2.5-Mistral-7B-q4f16_1-MLC/resolve/main/",
      "model_id": "OpenHermes-2.5-Mistral-7B-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
      "vram_required_MB": 6079.02,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC/resolve/main/",
      "model_id": "NeuralHermes-2.5-Mistral-7B-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
      "vram_required_MB": 6079.02,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    // Gemma-2B
    {
      "model_url": "https://huggingface.co/mlc-ai/gemma-2b-it-q4f16_1-MLC/resolve/main/",
      "model_id": "gemma-2b-it-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/gemma-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      "vram_required_MB": 1476.52,
      "low_resource_required": false,
      "buffer_size_required_bytes": 262144000,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/gemma-2b-it-q4f32_1-MLC/resolve/main/",
      "model_id": "gemma-2b-it-q4f32_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/gemma-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      "vram_required_MB": 1750.66,
      "low_resource_required": false,
      "buffer_size_required_bytes": 262144000,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/gemma-2b-it-q4f16_1-MLC/resolve/main/",
      "model_id": "gemma-2b-it-q4f16_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/gemma-2b-it-q4f16_1-ctx1k_cs1k-webgpu.wasm",
      "vram_required_MB": 1476.52,
      "low_resource_required": true,
      "buffer_size_required_bytes": 262144000,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/gemma-2b-it-q4f32_1-MLC/resolve/main/",
      "model_id": "gemma-2b-it-q4f32_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/gemma-2b-it-q4f32_1-ctx1k_cs1k-webgpu.wasm",
      "vram_required_MB": 1750.66,
      "low_resource_required": true,
      "buffer_size_required_bytes": 262144000,
    },
    // RedPajama
    {
      "model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/resolve/main/",
      "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx2k-webgpu.wasm",
      "vram_required_MB": 2972.09,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC/resolve/main/",
      "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k-webgpu.wasm",
      "vram_required_MB": 3928.09,
      "low_resource_required": false,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/resolve/main/",
      "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 2041.09,
      "low_resource_required": true,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC/resolve/main/",
      "model_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 2558.09,
      "low_resource_required": true,
    },
    // Phi-2
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-2-q0f16-MLC/resolve/main/",
      "model_id": "Phi2-q0f16",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-2-q0f16-ctx2k-webgpu.wasm",
      "vram_required_MB": 11079.47,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-2-q0f32-MLC/resolve/main/",
      "model_id": "Phi2-q0f32",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-2-q0f32-ctx2k-webgpu.wasm",
      "vram_required_MB": 12043.48,
      "low_resource_required": false,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC/resolve/main/",
      "model_id": "Phi2-q4f16_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-2-q4f16_1-ctx2k-webgpu.wasm",
      "vram_required_MB": 3053.97,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-2-q4f32_1-MLC/resolve/main/",
      "model_id": "Phi2-q4f32_1",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-2-q4f32_1-ctx2k-webgpu.wasm",
      "vram_required_MB": 4032.48,
      "low_resource_required": false,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC/resolve/main/",
      "model_id": "Phi2-q4f16_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-2-q4f16_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 2131.97,
      "low_resource_required": true,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-2-q4f32_1-MLC/resolve/main/",
      "model_id": "Phi2-q4f32_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-2-q4f32_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 2740.48,
      "low_resource_required": true,
    },
    // Phi-1.5
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-1_5-q0f16-MLC/resolve/main/",
      "model_id": "Phi1.5-q0f16",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-1_5-q0f16-ctx2k-webgpu.wasm",
      "vram_required_MB": 5818.09,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-1_5-q0f32-MLC/resolve/main/",
      "model_id": "Phi1.5-q0f32",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-1_5-q0f32-ctx2k-webgpu.wasm",
      "vram_required_MB": 6514.09,
      "low_resource_required": false,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-1_5-q4f16_1-MLC/resolve/main/",
      "model_id": "Phi1.5-q4f16_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-1_5-q4f16_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 1210.09,
      "low_resource_required": true,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/phi-1_5-q4f32_1-MLC/resolve/main/",
      "model_id": "Phi1.5-q4f32_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/phi-1_5-q4f32_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 1682.09,
      "low_resource_required": true,
    },
    // TinyLlama
    {
      "model_url": "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q0f16-MLC/resolve/main/",
      "model_id": "TinyLlama-1.1B-Chat-v0.4-q0f16",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/TinyLlama-1.1B-Chat-v0.4-q0f16-ctx2k-webgpu.wasm",
      "vram_required_MB": 5063.52,
      "low_resource_required": false,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q0f32-MLC/resolve/main/",
      "model_id": "TinyLlama-1.1B-Chat-v0.4-q0f32",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/TinyLlama-1.1B-Chat-v0.4-q0f32-ctx2k-webgpu.wasm",
      "vram_required_MB": 5394.53,
      "low_resource_required": false,
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC/resolve/main/",
      "model_id": "TinyLlama-1.1B-Chat-v0.4-q4f16_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/TinyLlama-1.1B-Chat-v0.4-q4f16_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 899.11,
      "low_resource_required": true,
      "required_features": ["shader-f16"],
    },
    {
      "model_url": "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC/resolve/main/",
      "model_id": "TinyLlama-1.1B-Chat-v0.4-q4f32_1-1k",
      "model_lib_url": modelLibURLPrefix + modelVersion + "/TinyLlama-1.1B-Chat-v0.4-q4f32_1-ctx1k-webgpu.wasm",
      "vram_required_MB": 992.11,
      "low_resource_required": true,
    },
  ]
}
