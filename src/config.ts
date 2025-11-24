import log from "loglevel";
import { ResponseFormat } from "./openai_api_protocols";
import { LogitProcessor, InitProgressCallback, LogLevel } from "./types";
import {
  DependencyError,
  InvalidNumberStringError,
  MinValueError,
  NonNegativeError,
  RangeError,
} from "./error";

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
  stop_str: Array<string>;
  system_prefix_token_ids?: Array<number>;
  stop_token_ids: Array<number>;
  add_role_after_system_message?: boolean;
}

export enum Role {
  user = "user",
  assistant = "assistant",
  tool = "tool",
}

export const DefaultLogLevel: LogLevel = "WARN";

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
  function = "{function_string}",
  hermes_tools = "{hermes_tools}",
}

/**
 * Information about the tokenizer. Currently, only `token_postproc_method` is used to
 * post process the token table when using grammar.
 */
export interface TokenizerInfo {
  token_postproc_method: string;
  prepend_space_in_encode: boolean;
  strip_space_in_decode: boolean;
}

/**
 * Config of one chat model, a data structure representing `mlc-chat-config.json`.
 * This only corresponds to the chat-related fields and `tokenizer_files` of `mlc-chat-config.json`.
 * Only these fields affect the conversation in runtime.
 * i.e. The third part in https://llm.mlc.ai/docs/get_started/mlc_chat_config.html.
 *
 * This is initialized in `MLCEngine.reload()` with the model's `mlc-chat-config.json`.
 */
export interface ChatConfig {
  // First three fields affect the entire conversation, i.e. used in `MLCEngine.reload()`
  tokenizer_files: Array<string>;
  tokenizer_info?: TokenizerInfo;
  token_table_postproc_method?: string; // TODO: backward compatibility, remove soon
  vocab_size: number;
  conv_config?: Partial<ConvTemplateConfig>;
  conv_template: ConvTemplateConfig;
  // KVCache settings
  context_window_size: number;
  sliding_window_size: number;
  attention_sink_size: number;
  // Fields below can be swapped per-generation via `GenerationConfig`
  // Fields only used in MLC
  repetition_penalty: number;
  // Fields shared by MLC and OpenAI APIs
  frequency_penalty: number;
  presence_penalty: number;
  top_p: number;
  temperature: number;
  bos_token_id?: number;
}

/**
 * Custom options that can be used to override known config values.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ChatOptions extends Partial<ChatConfig> {}

/**
 * Optional configurations for `CreateMLCEngine()` and `CreateWebWorkerMLCEngine()`.
 *
 * appConfig: Configure the app, including the list of models and whether to use IndexedDB cache.
 * initProgressCallback: A callback for showing the progress of loading the model.
 * logitProcessorRegistry: A register for stateful logit processors, see `webllm.LogitProcessor`.
 *
 * @note All fields are optional, and `logitProcessorRegistry` is only used for `MLCEngine` and not
 * other `MLCEngine`s.
 */
export interface MLCEngineConfig {
  appConfig?: AppConfig;
  initProgressCallback?: InitProgressCallback;
  logitProcessorRegistry?: Map<string, LogitProcessor>;
  logLevel?: LogLevel;
}

/**
 * Config for a single generation.
 * Essentially `ChatConfig` without `tokenizer_files`, `conv_config`, or `conv_template`.
 * We also support additional fields not present in `mlc-chat-config.json` due to OpenAI-like APIs.
 *
 * Note that all values are optional. If unspecified, we use whatever values in `ChatConfig`
 * initialized during `MLCEngine.reload()`.
 */
export interface GenerationConfig {
  // Only used in MLC
  repetition_penalty?: number | null;
  ignore_eos?: boolean;
  // Shared by MLC and OpenAI APIs
  top_p?: number | null;
  temperature?: number | null;
  // Only in OpenAI APIs
  max_tokens?: number | null;
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
  stop?: string | null | Array<string>;
  n?: number | null;
  logit_bias?: Record<string, number> | null;
  logprobs?: boolean | null;
  top_logprobs?: number | null;
  response_format?: ResponseFormat | null;
  // extra_body in ChatCompletionsRequest
  enable_thinking?: boolean | null;
  enable_latency_breakdown?: boolean | null;
}

export function postInitAndCheckGenerationConfigValues(
  config: GenerationConfig,
): void {
  function _hasValue(value: any): boolean {
    // if we use `if value` directly, `value` being 0 evaluates to false, violating semantics
    return value !== undefined && value !== null;
  }
  if (
    config.frequency_penalty &&
    (config.frequency_penalty < -2.0 || config.frequency_penalty > 2.0)
  ) {
    throw new RangeError("frequency_penalty", -2.0, 2.0);
  }
  if (
    config.presence_penalty &&
    (config.presence_penalty < -2.0 || config.presence_penalty > 2.0)
  ) {
    throw new RangeError("presence_penalty", -2.0, 2.0);
  }
  if (_hasValue(config.repetition_penalty) && config.repetition_penalty! <= 0) {
    throw new MinValueError("repetition_penalty", 0);
  }
  if (_hasValue(config.max_tokens) && config.max_tokens! <= 0) {
    throw new MinValueError("max_tokens", 0);
  }
  if ((_hasValue(config.top_p) && config.top_p! <= 0) || config.top_p! > 1) {
    throw new RangeError("top_p", 0, 1);
  }
  if (_hasValue(config.temperature) && config.temperature! < 0) {
    throw new NonNegativeError("temperature");
  }
  // If only one of frequency or presence penatly is set, make the other one 0.0
  if (
    _hasValue(config.frequency_penalty) &&
    !_hasValue(config.presence_penalty)
  ) {
    config.presence_penalty = 0.0;
    log.warn("Only frequency_penalty is set; we default presence_penaty to 0.");
  }
  if (
    _hasValue(config.presence_penalty) &&
    !_hasValue(config.frequency_penalty)
  ) {
    config.frequency_penalty = 0.0;
    log.warn(
      "Only presence_penalty is set; we default frequency_penalty to 0.",
    );
  }
  // Check logit_bias range
  if (_hasValue(config.logit_bias)) {
    for (const tokenID in config.logit_bias) {
      const bias = config.logit_bias[tokenID];
      if (bias > 100 || bias < -100) {
        throw new RangeError(
          "logit_bias",
          -100,
          100,
          "Got " + bias + " for tokenID " + tokenID,
        );
      }
      if (isNaN(parseInt(tokenID))) {
        throw new InvalidNumberStringError("logit_bias's keys", tokenID);
      }
    }
  }
  // logprobs and top_logprobs
  if (_hasValue(config.top_logprobs)) {
    // If top_logprobs is non-null, logprobs must be true
    if (!config.logprobs) {
      throw new DependencyError("top_logprobs", "logprobs", true);
    }
    // top_logprobs should be in range [0,5]
    if (config.top_logprobs! < 0 || config.top_logprobs! > 5) {
      throw new RangeError("top_logprobs", 0, 5, "Got " + config.top_logprobs);
    }
  }
  // If defined logprobs but not top_logprobs, simply make it 0
  if (config.logprobs) {
    if (!_hasValue(config.top_logprobs)) {
      config.top_logprobs = 0;
    }
  }
}

export enum ModelType {
  "LLM",
  "embedding",
  "VLM", // vision-language model
}

/**
 * Information for a model.
 * @param model: the huggingface link to download the model weights, accepting four formats:
 *    - https://huggingface.co/{USERNAME}/{MODEL}, which we automatically use the main branch
 *    - https://huggingface.co/{USERNAME}/{MODEL}/, which we automatically use the main branch
 *    - https://huggingface.co/{USERNAME}/{MODEL}/resolve/{BRANCH}
 *    - https://huggingface.co/{USERNAME}/{MODEL}/resolve/{BRANCH}/
 * @param model_id: what we call the model.
 * @param model_lib: link to the model library (wasm file) the model uses.
 * @param overrides: partial ChatConfig to override mlc-chat-config.json; can be used to change KVCache settings.
 * @param vram_required_MB: amount of vram in MB required to run the model (can use
 *    `utils/vram_requirements` to calculate).
 * @param low_resource_required: whether the model can run on limited devices (e.g. Android phone).
 * @param buffer_size_required_bytes: required `maxStorageBufferBindingSize`, different for each device.
 * @param required_features: feature needed to run this model (e.g. shader-f16).
 * @param model_type: the intended usecase for the model, if unspecified, default to LLM.
 */
export interface ModelRecord {
  model: string;
  model_id: string;
  model_lib: string;
  overrides?: ChatOptions;
  vram_required_MB?: number;
  low_resource_required?: boolean;
  buffer_size_required_bytes?: number;
  required_features?: Array<string>;
  model_type?: ModelType;
}

/**
 * Extra configuration that can be
 * passed to the load.
 *
 * @param model_list: models to be used.
 * @param useIndexedDBCache: if true, will use IndexedDBCache to cache models and other artifacts.
 * If false or unspecified, will use the Cache API. For more information of the two, see:
 * https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria#what_technologies_store_data_in_the_browser
 *
 * @note Note that the Cache API is more well-tested in WebLLM as of now.
 */
export interface AppConfig {
  model_list: Array<ModelRecord>;
  useIndexedDBCache?: boolean;
}

/**
 * modelVersion: the prebuilt model libraries that the current npm is compatible with, affects the
 * `model_lib`s in `prebuiltAppConfig`.
 *
 * @note The model version does not have to match the npm version, since not each npm update
 * requires an update of the model libraries.
 */
export const modelVersion = "v0_2_80";
export const modelLibURLPrefix =
  "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/";

/**
 * Models that support function calling (i.e. usage of `ChatCompletionRequest.tools`). More to come.
 */
export const functionCallingModelIds = [
  "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
  "Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC",
  "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
  "Hermes-3-Llama-3.1-8B-q4f32_1-MLC",
  "Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
];

/**
 * Default models and model library mapping to be used if unspecified.
 *
 * @note This is the only source of truth of which prebuilt model libraries are compatible with the
 * current WebLLM npm version.
 */
export const prebuiltAppConfig: AppConfig = {
  useIndexedDBCache: false,
  model_list: [
    // Llama-3.2
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f32_1-MLC",
      model_id: "Llama-3.2-1B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-1B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1128.82,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-1B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 879.04,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    // TODO: temporarily commenting out q0f32 models due to correctness issues
    // {
    //   model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q0f32-MLC",
    //   model_id: "Llama-3.2-1B-Instruct-q0f32-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Llama-3.2-1B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
    //   vram_required_MB: 5106.26,
    //   low_resource_required: true,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC",
      model_id: "Llama-3.2-1B-Instruct-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-1B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2573.13,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f32_1-MLC",
      model_id: "Llama-3.2-3B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2951.51,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2263.69,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Llama-3.1
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f32_1-MLC",
      model_id: "Llama-3.1-8B-Instruct-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5295.7,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3.1-8B-Instruct-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4598.34,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f32_1-MLC",
      model_id: "Llama-3.1-8B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6101.01,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3.1-8B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5001.0,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // DeepSeek-R1-Distill-Qwen
    // TODO(Charlie): Qwen2-1.5B is experiencing correctness issue, hence commented for now.
    // {
    //   model: "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC",
    //   model_id: "DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
    //   low_resource_required: true,
    //   vram_required_MB: 1629.75,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    // {
    //   model: "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-1.5B-q4f32_1-MLC",
    //   model_id: "DeepSeek-R1-Distill-Qwen-1.5B-q4f32_1-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
    //   low_resource_required: true,
    //   vram_required_MB: 1888.97,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    {
      model:
        "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC",
      model_id: "DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5106.67,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC",
      model_id: "DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5900.09,
      overrides: {
        context_window_size: 4096,
      },
    },
    // DeepSeek-R1-Distill-Llama
    {
      model:
        "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC",
      model_id: "DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6101.01,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
      model_id: "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5001.0,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Hermes-3 and Hermes-2
    {
      model:
        "https://huggingface.co/mlc-ai/Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC",
      model_id: "Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4976.13,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Hermes-2-Theta-Llama-3-8B-q4f32_1-MLC",
      model_id: "Hermes-2-Theta-Llama-3-8B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6051.27,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
      model_id: "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4976.13,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC",
      model_id: "Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6051.27,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f32_1-MLC",
      model_id: "Hermes-3-Llama-3.2-3B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2951.51,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
      model_id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3.2-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2263.69,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.1-8B-q4f32_1-MLC",
      model_id: "Hermes-3-Llama-3.1-8B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5779.27,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
      model_id: "Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4876.13,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
      model_id: "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4033.28,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    // Phi3.5-mini-instruct
    {
      model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC",
      model_id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3.5-mini-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 3672.07,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f32_1-MLC",
      model_id: "Phi-3.5-mini-instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3.5-mini-instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5483.12,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC",
      model_id: "Phi-3.5-mini-instruct-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3.5-mini-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2520.07,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f32_1-MLC",
      model_id: "Phi-3.5-mini-instruct-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3.5-mini-instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 3179.12,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // Phi-3.5-vision-instruct
    {
      model:
        "https://huggingface.co/mlc-ai/Phi-3.5-vision-instruct-q4f16_1-MLC",
      model_id: "Phi-3.5-vision-instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3.5-vision-instruct-q4f16_1-ctx4k_cs2k-webgpu.wasm",
      vram_required_MB: 3952.18,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
      model_type: ModelType.VLM,
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Phi-3.5-vision-instruct-q4f32_1-MLC",
      model_id: "Phi-3.5-vision-instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3.5-vision-instruct-q4f32_1-ctx4k_cs2k-webgpu.wasm",
      vram_required_MB: 5879.84,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
      model_type: ModelType.VLM,
    },
    // Mistral variants
    {
      model:
        "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
      model_id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4573.39,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f32_1-MLC",
      model_id: "Mistral-7B-Instruct-v0.3-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5619.27,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
      model_id: "Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4573.39,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/OpenHermes-2.5-Mistral-7B-q4f16_1-MLC",
      model_id: "OpenHermes-2.5-Mistral-7B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4573.39,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC",
      model_id: "NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4573.39,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC",
      model_id: "WizardMath-7B-V1.1-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4573.39,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
        sliding_window_size: -1,
      },
    },
    // SmolLM2
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f16_1-MLC",
      model_id: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-1.7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1774.19,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f32_1-MLC",
      model_id: "SmolLM2-1.7B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-1.7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2692.38,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },

    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q0f16-MLC",
      model_id: "SmolLM2-360M-Instruct-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-360M-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 871.99,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q0f32-MLC",
      model_id: "SmolLM2-360M-Instruct-q0f32-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-360M-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1743.99,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f16_1-MLC",
      model_id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-360M-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 376.06,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f32_1-MLC",
      model_id: "SmolLM2-360M-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-360M-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 579.61,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC",
      model_id: "SmolLM2-135M-Instruct-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-135M-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 359.69,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f32-MLC",
      model_id: "SmolLM2-135M-Instruct-q0f32-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/SmolLM2-135M-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 719.38,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Gemma2
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC",
      model_id: "gemma-2-2b-it-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1895.3,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f32_1-MLC",
      model_id: "gemma-2-2b-it-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2508.75,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC",
      model_id: "gemma-2-2b-it-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1583.3,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f32_1-MLC",
      model_id: "gemma-2-2b-it-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1884.75,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f16_1-MLC",
      model_id: "gemma-2-9b-it-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-9b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6422.01,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f32_1-MLC",
      model_id: "gemma-2-9b-it-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-9b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 8383.33,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Gemma2-2b-jpn
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-2b-jpn-it-q4f16_1-MLC",
      model_id: "gemma-2-2b-jpn-it-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-2b-jpn-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1895.3,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-2b-jpn-it-q4f32_1-MLC",
      model_id: "gemma-2-2b-jpn-it-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2-2b-jpn-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2508.75,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Qwen-3
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC",
      model_id: "Qwen3-0.6B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1403.34,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f32_1-MLC",
      model_id: "Qwen3-0.6B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-0.6B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1924.98,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q0f16-MLC",
      model_id: "Qwen3-0.6B-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-0.6B-q0f16-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2220.38,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    // TODO: temporarily commenting out q0f32 models due to correctness issues
    // {
    //   model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q0f32-MLC",
    //   model_id: "Qwen3-0.6B-q0f32-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Qwen3-0.6B-q0f32-ctx4k_cs1k-webgpu.wasm",
    //   vram_required_MB: 3843.25,
    //   low_resource_required: true,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC",
      model_id: "Qwen3-1.7B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-1.7B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2036.66,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f32_1-MLC",
      model_id: "Qwen3-1.7B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-1.7B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2635.44,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC",
      model_id: "Qwen3-4B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-4B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 3431.59,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-4B-q4f32_1-MLC",
      model_id: "Qwen3-4B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-4B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4327.71,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-8B-q4f16_1-MLC",
      model_id: "Qwen3-8B-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-8B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5695.78,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen3-8B-q4f32_1-MLC",
      model_id: "Qwen3-8B-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen3-8B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6852.55,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Qwen-2
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 944.62,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1060.2,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q0f16-MLC",
      model_id: "Qwen2.5-0.5B-Instruct-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1624.12,
      overrides: {
        context_window_size: 4096,
      },
    },
    // TODO: temporarily commenting out q0f32 models due to correctness issues
    // {
    //   model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q0f32-MLC",
    //   model_id: "Qwen2.5-0.5B-Instruct-q0f32-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Qwen2-0.5B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
    //   low_resource_required: true,
    //   vram_required_MB: 2654.75,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1629.75,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-1.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1888.97,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-3B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-3B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2.5-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 2504.76,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-3B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-3B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2.5-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 2893.64,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-7B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5106.67,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2.5-7B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-7B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5900.09,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Qwen2.5-Coder
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-Coder-0.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 944.62,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-Coder-0.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1060.2,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
      model_id: "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1624.12,
      overrides: {
        context_window_size: 4096,
      },
    },
    // TODO: temporarily commenting out q0f32 models due to correctness issues
    // {
    //   model:
    //     "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q0f32-MLC",
    //   model_id: "Qwen2.5-Coder-0.5B-Instruct-q0f32-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Qwen2-0.5B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
    //   low_resource_required: true,
    //   vram_required_MB: 2654.75,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 1629.75,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 1888.97,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2.5-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 2504.76,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2.5-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 2893.64,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5106.67,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5900.09,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Qwen2.5-Math
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Math-1.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2.5-Math-1.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1629.75,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2.5-Math-1.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2.5-Math-1.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1888.97,
      overrides: {
        context_window_size: 4096,
      },
    },
    // StableLM-zephyr-1.6B
    {
      model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f16_1-MLC",
      model_id: "stablelm-2-zephyr-1_6b-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/stablelm-2-zephyr-1_6b-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2087.66,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f32_1-MLC",
      model_id: "stablelm-2-zephyr-1_6b-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/stablelm-2-zephyr-1_6b-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2999.33,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f16_1-MLC",
      model_id: "stablelm-2-zephyr-1_6b-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/stablelm-2-zephyr-1_6b-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1511.66,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f32_1-MLC",
      model_id: "stablelm-2-zephyr-1_6b-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/stablelm-2-zephyr-1_6b-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1847.33,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // RedPajama
    {
      model:
        "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
      model_id: "RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 2972.09,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",
      model_id: "RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 3928.09,
      low_resource_required: false,
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
      model_id: "RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 2041.09,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",
      model_id: "RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 2558.09,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // TinyLlama v1.0
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v1.0-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 697.24,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v1.0-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 839.98,
      low_resource_required: true,
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v1.0-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 675.24,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v1.0-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 795.98,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // BELOW ARE MODELS OF OLDER VERSIONS OR NOT AS PRACTICAL
    // Llama-3.1 70B
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-70B-Instruct-q3f16_1-MLC",
      model_id: "Llama-3.1-70B-Instruct-q3f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3_1-70B-Instruct-q3f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 31153.13,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Qwen-2
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-0.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2-0.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 944.62,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-0.5B-Instruct-q0f16-MLC",
      model_id: "Qwen2-0.5B-Instruct-q0f16-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-0.5B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1624.12,
      overrides: {
        context_window_size: 4096,
      },
    },
    // TODO: temporarily commenting out q0f32 models due to correctness issues
    // {
    //   model: "https://huggingface.co/mlc-ai/Qwen2-0.5B-Instruct-q0f32-MLC",
    //   model_id: "Qwen2-0.5B-Instruct-q0f32-MLC",
    //   model_lib:
    //     modelLibURLPrefix +
    //     modelVersion +
    //     "/Qwen2-0.5B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
    //   low_resource_required: true,
    //   vram_required_MB: 2654.75,
    //   overrides: {
    //     context_window_size: 4096,
    //   },
    // },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-1.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2-1.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1629.75,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-1.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2-1.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1888.97,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-7B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2-7B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5106.67,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-7B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2-7B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5900.09,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Qwen2-Math
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2-Math-1.5B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2-Math-1.5B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1629.75,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/Qwen2-Math-1.5B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2-Math-1.5B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: true,
      vram_required_MB: 1888.97,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-Math-7B-Instruct-q4f16_1-MLC",
      model_id: "Qwen2-Math-7B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5106.67,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Qwen2-Math-7B-Instruct-q4f32_1-MLC",
      model_id: "Qwen2-Math-7B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      low_resource_required: false,
      vram_required_MB: 5900.09,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Llama-3
    {
      model: "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f32_1-MLC",
      model_id: "Llama-3-8B-Instruct-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5295.7,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3-8B-Instruct-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4598.34,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f32_1-MLC",
      model_id: "Llama-3-8B-Instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6101.01,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3-8B-Instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5001.0,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3-70B-Instruct-q3f16_1-MLC",
      model_id: "Llama-3-70B-Instruct-q3f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-3-70B-Instruct-q3f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 31153.13,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Phi3-mini-instruct
    {
      model: "https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC",
      model_id: "Phi-3-mini-4k-instruct-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3-mini-4k-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 3672.07,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f32_1-MLC",
      model_id: "Phi-3-mini-4k-instruct-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3-mini-4k-instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5483.12,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC",
      model_id: "Phi-3-mini-4k-instruct-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3-mini-4k-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 2520.07,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f32_1-MLC",
      model_id: "Phi-3-mini-4k-instruct-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Phi-3-mini-4k-instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 3179.12,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // Llama-2
    {
      model: "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC",
      model_id: "Llama-2-7b-chat-hf-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 5284.01,
      low_resource_required: false,
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_id: "Llama-2-7b-chat-hf-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-2-7b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 4618.52,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC",
      model_id: "Llama-2-7b-chat-hf-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 9109.03,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_id: "Llama-2-7b-chat-hf-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-2-7b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 6749.02,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-2-13b-chat-hf-q4f16_1-MLC",
      model_id: "Llama-2-13b-chat-hf-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/Llama-2-13b-chat-hf-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 11814.09,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    // Gemma-2B
    {
      model: "https://huggingface.co/mlc-ai/gemma-2b-it-q4f16_1-MLC",
      model_id: "gemma-2b-it-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1476.52,
      low_resource_required: false,
      buffer_size_required_bytes: 262144000,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2b-it-q4f32_1-MLC",
      model_id: "gemma-2b-it-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1750.66,
      low_resource_required: false,
      buffer_size_required_bytes: 262144000,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2b-it-q4f16_1-MLC",
      model_id: "gemma-2b-it-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1476.52,
      low_resource_required: true,
      buffer_size_required_bytes: 262144000,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2b-it-q4f32_1-MLC",
      model_id: "gemma-2b-it-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/gemma-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1750.66,
      low_resource_required: true,
      buffer_size_required_bytes: 262144000,
      overrides: {
        context_window_size: 1024,
      },
    },
    // Phi-2
    {
      model: "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC",
      model_id: "phi-2-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-2-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 3053.97,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/phi-2-q4f32_1-MLC",
      model_id: "phi-2-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-2-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 4032.48,
      low_resource_required: false,
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/phi-2-q4f16_1-MLC",
      model_id: "phi-2-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-2-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 2131.97,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/phi-2-q4f32_1-MLC",
      model_id: "phi-2-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-2-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 2740.48,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // Phi-1.5
    {
      model: "https://huggingface.co/mlc-ai/phi-1_5-q4f16_1-MLC",
      model_id: "phi-1_5-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-1_5-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 1210.09,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/phi-1_5-q4f32_1-MLC",
      model_id: "phi-1_5-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-1_5-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 1682.09,
      low_resource_required: true,
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/phi-1_5-q4f16_1-MLC",
      model_id: "phi-1_5-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-1_5-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 1210.09,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model: "https://huggingface.co/mlc-ai/phi-1_5-q4f32_1-MLC",
      model_id: "phi-1_5-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/phi-1_5-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 1682.09,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // TinyLlama v0.4
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v0.4-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 697.24,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v0.4-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 839.98,
      low_resource_required: true,
      overrides: {
        context_window_size: 2048,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v0.4-q4f16_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 675.24,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: {
        context_window_size: 1024,
      },
    },
    {
      model:
        "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC",
      model_id: "TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC-1k",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/TinyLlama-1.1B-Chat-v0.4-q4f32_1-ctx2k_cs1k-webgpu.wasm",
      vram_required_MB: 795.98,
      low_resource_required: true,
      overrides: {
        context_window_size: 1024,
      },
    },
    // Embedding models
    // -b means max_batch_size this model allows. The smaller it is, the less memory the model consumes.
    {
      model: "https://huggingface.co/mlc-ai/snowflake-arctic-embed-m-q0f32-MLC",
      model_id: "snowflake-arctic-embed-m-q0f32-MLC-b32",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/snowflake-arctic-embed-m-q0f32-ctx512_cs512_batch32-webgpu.wasm",
      vram_required_MB: 1407.51,
      model_type: ModelType.embedding,
    },
    {
      model: "https://huggingface.co/mlc-ai/snowflake-arctic-embed-m-q0f32-MLC",
      model_id: "snowflake-arctic-embed-m-q0f32-MLC-b4",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/snowflake-arctic-embed-m-q0f32-ctx512_cs512_batch4-webgpu.wasm",
      vram_required_MB: 539.4,
      model_type: ModelType.embedding,
    },
    {
      model: "https://huggingface.co/mlc-ai/snowflake-arctic-embed-s-q0f32-MLC",
      model_id: "snowflake-arctic-embed-s-q0f32-MLC-b32",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/snowflake-arctic-embed-s-q0f32-ctx512_cs512_batch32-webgpu.wasm",
      vram_required_MB: 1022.82,
      model_type: ModelType.embedding,
    },
    {
      model: "https://huggingface.co/mlc-ai/snowflake-arctic-embed-s-q0f32-MLC",
      model_id: "snowflake-arctic-embed-s-q0f32-MLC-b4",
      model_lib:
        modelLibURLPrefix +
        modelVersion +
        "/snowflake-arctic-embed-s-q0f32-ctx512_cs512_batch4-webgpu.wasm",
      vram_required_MB: 238.71,
      model_type: ModelType.embedding,
    },
  ],
};
