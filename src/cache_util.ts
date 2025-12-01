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
import CrossOriginStorage from "./cross_origin_storage";
import CrossOriginStorageCache from "./cross_origin_storage_cache";

type CacheScope = "webllm/model" | "webllm/config" | "webllm/wasm";

let crossOriginUnavailableLogged = false;
let crossOriginAvailabilityWait: Promise<void> | null = null;

function scheduleCrossOriginFallbackWarning(
  logger: (msg: string) => void,
): void {
  if (crossOriginUnavailableLogged || crossOriginAvailabilityWait) {
    return;
  }
  crossOriginAvailabilityWait = (async () => {
    const available = CrossOriginStorage.isAvailable();
    crossOriginAvailabilityWait = null;
    if (available || crossOriginUnavailableLogged) {
      return;
    }
    logger(
      "Cross-origin storage backend is not yet available; temporarily falling back to the Cache API.",
    );
    crossOriginUnavailableLogged = true;
  })();
}

function useCrossOrigin(appConfig: AppConfig): boolean {
  return (
    getCacheBackend(appConfig) === "cross-origin" &&
    CrossOriginStorage.isAvailable()
  );
}

export function getArtifactCache(
  scope: CacheScope,
  appConfig: AppConfig,
  logger: (msg: string) => void = console.warn,
): tvmjs.ArtifactCacheTemplate {
  const backend = getCacheBackend(appConfig);
  if (backend === "cross-origin") {
    if (CrossOriginStorage.isAvailable()) {
      return new CrossOriginStorageCache(scope);
    }
    scheduleCrossOriginFallbackWarning(logger);
  }
  if (backend === "indexeddb") {
    return new tvmjs.ArtifactIndexedDBCache(scope);
  }
  return new tvmjs.ArtifactCache(scope);
}

async function hasTensorCache(
  cache: tvmjs.ArtifactCacheTemplate,
  tensorCacheUrl: string,
): Promise<boolean> {
  const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
  const hasManifest = await cache.hasAllKeys([jsonUrl]);
  if (!hasManifest) {
    return false;
  }
  const manifest = await cache.fetchWithCache(jsonUrl, "json");
  const records = manifest?.records ?? [];
  if (!Array.isArray(records) || records.length === 0) {
    return false;
  }
  const shardUrls = records.map(
    (entry: { dataPath: string }) =>
      new URL(entry.dataPath, tensorCacheUrl).href,
  );
  return cache.hasAllKeys(shardUrls);
}

async function deleteTensorCacheEntries(
  cache: tvmjs.ArtifactCacheTemplate,
  tensorCacheUrl: string,
): Promise<void> {
  const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
  const hasManifest = await cache.hasAllKeys([jsonUrl]);
  if (!hasManifest) {
    return;
  }
  let manifest: { records?: Array<{ dataPath: string }> };
  try {
    manifest = await cache.fetchWithCache(jsonUrl, "json");
  } catch (err) {
    console.warn(
      `Failed to load tensor cache manifest at ${jsonUrl}; skipping deletion.`,
      err,
    );
    return;
  }
  const records = manifest?.records ?? [];
  await Promise.all(
    records.map(async (entry) => {
      if (!entry?.dataPath) {
        return;
      }
      const dataUrl = new URL(entry.dataPath, tensorCacheUrl).href;
      await cache.deleteInCache(dataUrl);
    }),
  );
  await cache.deleteInCache(jsonUrl);
}

export async function fetchModelArtifacts(
  tvm: tvmjs.Instance,
  tensorCacheUrl: string,
  device: tvmjs.DLDevice,
  appConfig: AppConfig,
  signal?: AbortSignal,
): Promise<any> {
  if (!useCrossOrigin(appConfig)) {
    const backend = getCacheBackend(appConfig);
    const cacheType = backend === "indexeddb" ? "indexeddb" : "cache";
    return tvm.fetchTensorCache(
      tensorCacheUrl,
      device,
      "webllm/model",
      cacheType,
      signal,
    );
  }

  const artifactCache = getArtifactCache("webllm/model", appConfig);
  const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
  const manifest = await artifactCache.fetchWithCache(jsonUrl, "json", signal);
  const records = (
    Array.isArray(manifest?.records) ? manifest.records : []
  ) as Array<any>;
  await (tvm as any).fetchTensorCacheInternal(
    tensorCacheUrl,
    records,
    device,
    artifactCache,
    signal,
  );
  if (manifest?.metadata !== undefined) {
    const runtime = tvm as any;
    runtime.cacheMetadata = {
      ...runtime.cacheMetadata,
      ...(manifest.metadata as Record<string, unknown>),
    };
  }
  return manifest;
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
  if (useCrossOrigin(appConfig)) {
    const cache = getArtifactCache("webllm/model", appConfig);
    return hasTensorCache(cache, modelUrl);
  }
  const backend = getCacheBackend(appConfig);
  const cacheType = backend === "indexeddb" ? "indexeddb" : "cache";
  return tvmjs.hasTensorInCache(modelUrl, "webllm/model", cacheType);
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
  const modelCache = getArtifactCache("webllm/model", appConfig);
  if (useCrossOrigin(appConfig)) {
    await deleteTensorCacheEntries(modelCache, modelUrl);
  } else {
    const backend = getCacheBackend(appConfig);
    const cacheType = backend === "indexeddb" ? "indexeddb" : "cache";
    await tvmjs.deleteTensorCache(modelUrl, "webllm/model", cacheType);
  }
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
  const configCache = getArtifactCache("webllm/config", appConfig);
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
  const wasmCache = getArtifactCache("webllm/wasm", appConfig);
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
  const modelCache = getArtifactCache("webllm/model", appConfig, logger);

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
