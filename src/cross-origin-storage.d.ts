/**
 * Represents the dictionary for hash algorithms and values.
 */
interface CrossOriginStorageRequestFileHandleHash {
  value: string;
  algorithm: string;
}

/**
 * Represents the options for requesting file handles.
 */
interface CrossOriginStorageRequestFileHandleOptions {
  create?: boolean;
}

/**
 * The CrossOriginStorageManager interface.
 * [SecureContext]
 */
interface CrossOriginStorageManager {
  requestFileHandles(
    hashes: CrossOriginStorageRequestFileHandleHash[],
    options?: CrossOriginStorageRequestFileHandleOptions,
  ): Promise<FileSystemFileHandle[]>;
}

/**
 * Augment the standard Navigator interface.
 */
interface Navigator {
  readonly crossOriginStorage: CrossOriginStorageManager;
}
