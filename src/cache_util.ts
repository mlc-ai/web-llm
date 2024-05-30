import * as tvmjs from "tvmjs";
import { AppConfig, ModelRecord, prebuiltAppConfig } from "./config";
import { cleanModelUrl } from "./support";

function findModelRecord(modelId: string, appConfig?: AppConfig): ModelRecord {
  const matchedItem = appConfig?.model_list.find(
    (item) => item.model_id == modelId,
  );
  if (matchedItem !== undefined) {
    return matchedItem;
  }
  throw Error("Cannot find model record in appConfig for " + modelId);
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
  const cacheType = appConfig.useIndexedDBCache ? "indexeddb" : "cache";
  return tvmjs.hasNDArrayInCache(modelUrl, "webllm/model", cacheType);
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
  let modelCache: tvmjs.ArtifactCacheTemplate;
  if (appConfig.useIndexedDBCache) {
    tvmjs.deleteNDArrayCache(modelUrl, "webllm/model", "indexeddb");
    modelCache = new tvmjs.ArtifactIndexedDBCache("webllm/model");
  } else {
    tvmjs.deleteNDArrayCache(modelUrl, "webllm/model", "cache");
    modelCache = new tvmjs.ArtifactCache("webllm/model");
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
  let configCache: tvmjs.ArtifactCacheTemplate;
  if (appConfig.useIndexedDBCache) {
    configCache = new tvmjs.ArtifactIndexedDBCache("webllm/config");
  } else {
    configCache = new tvmjs.ArtifactCache("webllm/config");
  }
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
  let wasmCache: tvmjs.ArtifactCacheTemplate;
  if (appConfig.useIndexedDBCache) {
    wasmCache = new tvmjs.ArtifactIndexedDBCache("webllm/wasm");
  } else {
    wasmCache = new tvmjs.ArtifactCache("webllm/wasm");
  }
  await wasmCache.deleteInCache(modelRecord.model_lib);
}
