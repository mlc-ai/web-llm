import { AppConfig, ChatOptions, ModelRecord } from "./config";

// Helper function to compare two arrays
export function areArraysEqual(arr1?: Array<any>, arr2?: Array<any>): boolean {
  if (!arr1 && !arr2) return true;
  if (!arr1 || !arr2) return false;
  if (arr1.length !== arr2.length) return false;
  for (let i = 0; i < arr1.length; i++) {
    if (arr1[i] !== arr2[i]) return false;
  }
  return true;
}

// Helper function to compare two objects deeply
function areObjectsEqual(obj1: any, obj2: any): boolean {
  if (obj1 === obj2) return true;
  if (typeof obj1 !== typeof obj2) return false;
  if (typeof obj1 !== "object" || obj1 === null || obj2 === null) return false;

  const keys1 = Object.keys(obj1);
  const keys2 = Object.keys(obj2);
  if (keys1.length !== keys2.length) return false;

  for (const key of keys1) {
    if (!keys2.includes(key) || !areObjectsEqual(obj1[key], obj2[key]))
      return false;
  }
  return true;
}

// Function to compare two ModelRecord instances
export function areModelRecordsEqual(
  record1: ModelRecord,
  record2: ModelRecord,
): boolean {
  // Compare primitive fields
  if (
    record1.model !== record2.model ||
    record1.model_id !== record2.model_id ||
    record1.model_lib !== record2.model_lib ||
    record1.vram_required_MB !== record2.vram_required_MB ||
    record1.low_resource_required !== record2.low_resource_required ||
    record1.buffer_size_required_bytes !== record2.buffer_size_required_bytes
  ) {
    return false;
  }

  // Compare required_features arrays
  if (
    (record1.required_features && !record2.required_features) ||
    (!record1.required_features && record2.required_features)
  ) {
    return false;
  }

  if (record1.required_features && record2.required_features) {
    if (record1.required_features.length !== record2.required_features.length) {
      return false;
    }

    for (let i = 0; i < record1.required_features.length; i++) {
      if (record1.required_features[i] !== record2.required_features[i]) {
        return false;
      }
    }
  }

  return true;
}

export function areAppConfigsEqual(
  config1?: AppConfig,
  config2?: AppConfig,
): boolean {
  if (config1 === undefined || config2 === undefined) {
    return config1 === config2;
  }

  // Check if both configurations have the same IndexedDB cache usage
  if (config1.useIndexedDBCache !== config2.useIndexedDBCache) {
    return false;
  }

  // Check if both configurations have the same number of model records
  if (config1.model_list.length !== config2.model_list.length) {
    return false;
  }

  // Compare each ModelRecord in the model_list
  for (let i = 0; i < config1.model_list.length; i++) {
    if (!areModelRecordsEqual(config1.model_list[i], config2.model_list[i])) {
      return false;
    }
  }

  // If all checks passed, the configurations are equal
  return true;
}

export function areChatOptionsEqual(
  options1?: ChatOptions,
  options2?: ChatOptions,
): boolean {
  if (options1 === undefined || options2 === undefined) {
    return options1 === options2;
  }
  // Compare each property of ChatOptions (which are Partial<ChatConfig>)
  if (!areArraysEqual(options1.tokenizer_files, options2.tokenizer_files))
    return false;
  if (!areObjectsEqual(options1.conv_config, options2.conv_config))
    return false;
  if (options1.conv_template !== options2.conv_template) return false;
  if (options1.repetition_penalty !== options2.repetition_penalty) return false;
  if (options1.frequency_penalty !== options2.frequency_penalty) return false;
  if (options1.presence_penalty !== options2.presence_penalty) return false;
  if (options1.top_p !== options2.top_p) return false;
  if (options1.temperature !== options2.temperature) return false;
  if (options1.bos_token_id !== options2.bos_token_id) return false;

  // If all checks passed, the options are equal
  return true;
}

export function areChatOptionsListEqual(
  options1?: ChatOptions[],
  options2?: ChatOptions[],
): boolean {
  if (options1 && options2) {
    // Both defined, need to compare
    if (options1.length !== options2.length) {
      return false;
    } else {
      for (let i = 0; i < options1.length; i++) {
        if (!areChatOptionsEqual(options1[i], options2[i])) {
          return false;
        }
      }
      return true;
    }
  } else if (!options1 && !options2) {
    // Both undefined, equal
    return true;
  } else {
    // One defined, other not
    return false;
  }
}
