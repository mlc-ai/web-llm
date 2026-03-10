import * as tvmjs from "@mlc-ai/web-runtime";
import {
  AppConfig,
  ChatConfig,
  ModelRecord,
  prebuiltAppConfig,
  getCacheBackend,
} from "./config";
import { cleanModelUrl } from "./support";
import { ModelNotFoundError, UnsupportedTokenizerFilesError } from "./error";
import { Tokenizer } from "@mlc-ai/web-tokenizers";

type CacheScope = "webllm/model" | "webllm/config" | "webllm/wasm";

function getCacheAccessOptions(
  scope: CacheScope,
  appConfig: AppConfig,
): tvmjs.TensorCacheAccessOptions {
  return {
    cacheScope: scope,
    cacheType: getCacheBackend(appConfig),
  };
}

function createScopedArtifactCache(
  scope: CacheScope,
  appConfig: AppConfig,
): tvmjs.ArtifactCacheTemplate {
  return tvmjs.createArtifactCache(
    scope,
    getCacheAccessOptions(scope, appConfig),
  );
}

function findModelRecord(modelId: string, appConfig?: AppConfig): ModelRecord {
  const matchedItem = appConfig?.model_list.find(
    (item) => item.model_id == modelId,
  );
  if (matchedItem !== undefined) {
    return matchedItem;
  }
  throw new ModelNotFoundError(modelId);
}

export async function hasModelInCache(
  modelId: string,
  appConfig?: AppConfig,
): Promise<boolean> {
  if (appConfig === undefined) {
    appConfig = prebuiltAppConfig;
  }
  const modelRecord = findModelRecord(modelId, appConfig);
  const modelUrl = cleanModelUrl(modelRecord.model);
  return tvmjs.hasTensorInCache(
    modelUrl,
    getCacheAccessOptions("webllm/model", appConfig),
  );
}

export async function deleteModelAllInfoInCache(
  modelId: string,
  appConfig?: AppConfig,
) {
  // function to delete model all information in cache
  if (appConfig === undefined) {
    appConfig = prebuiltAppConfig;
  }
  // delete model and tokenizer in Cache
  await deleteModelInCache(modelId, appConfig);
  // delete wasm in cache
  await deleteModelWasmInCache(modelId, appConfig);
  // delete chat config
  await deleteChatConfigInCache(modelId, appConfig);
}

export async function deleteModelInCache(
  modelId: string,
  appConfig?: AppConfig,
) {
  // delete the model NDArray In Cache
  if (appConfig === undefined) {
    appConfig = prebuiltAppConfig;
  }
  const modelRecord = findModelRecord(modelId, appConfig);
  const modelUrl = cleanModelUrl(modelRecord.model);
  const modelCache = createScopedArtifactCache("webllm/model", appConfig);
  await tvmjs.deleteTensorCache(
    modelUrl,
    getCacheAccessOptions("webllm/model", appConfig),
  );
  await modelCache.deleteInCache(new URL("tokenizer.model", modelUrl).href);
  await modelCache.deleteInCache(new URL("tokenizer.json", modelUrl).href);
}

export async function deleteChatConfigInCache(
  modelId: string,
  appConfig?: AppConfig,
) {
  // delete the chat configuration in Cache
  if (appConfig === undefined) {
    appConfig = prebuiltAppConfig;
  }
  const modelRecord = findModelRecord(modelId, appConfig);
  const configCache = createScopedArtifactCache("webllm/config", appConfig);
  const modelUrl = cleanModelUrl(modelRecord.model);
  const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
  await configCache.deleteInCache(configUrl);
}

export async function deleteModelWasmInCache(
  modelId: string,
  appConfig?: AppConfig,
) {
  // delete the wasm in Cache
  if (appConfig === undefined) {
    appConfig = prebuiltAppConfig;
  }
  const modelRecord = findModelRecord(modelId, appConfig);
  const wasmCache = createScopedArtifactCache("webllm/wasm", appConfig);
  await wasmCache.deleteInCache(modelRecord.model_lib);
}

/**
 *
 * @param baseUrl The link to which we can find tokenizer files, usually is a `ModelRecord.model`.
 * @param config A ChatConfig, usually loaded from `mlc-chat-config.json` in `baseUrl`.
 * @param appConfig An AppConfig, usually `webllm.prebuiltAppConfig` if not defined by user.
 * @param logger Logging function, console.log by default.
 * @returns
 */
export async function asyncLoadTokenizer(
  baseUrl: string,
  config: ChatConfig,
  appConfig: AppConfig,
  logger: (msg: string) => void = console.log,
): Promise<Tokenizer> {
  const modelCache = createScopedArtifactCache("webllm/model", appConfig);

  if (config.tokenizer_files.includes("tokenizer.json")) {
    const url = new URL("tokenizer.json", baseUrl).href;
    const model = await modelCache.fetchWithCache(url, "arraybuffer");
    return Tokenizer.fromJSON(model);
  } else if (config.tokenizer_files.includes("tokenizer.model")) {
    logger(
      "Using `tokenizer.model` since we cannot locate `tokenizer.json`.\n" +
        "It is recommended to use `tokenizer.json` to ensure all token mappings are included, " +
        "since currently, files like `added_tokens.json`, `tokenizer_config.json` are ignored.\n" +
        "Consider converting `tokenizer.model` to `tokenizer.json` by compiling the model " +
        "with MLC again, or see if MLC's huggingface provides this file.",
    );
    const url = new URL("tokenizer.model", baseUrl).href;
    const model = await modelCache.fetchWithCache(url, "arraybuffer");
    return Tokenizer.fromSentencePiece(model);
  }
  throw new UnsupportedTokenizerFilesError(config.tokenizer_files);
}
