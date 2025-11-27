import * as tvmjs from "@mlc-ai/web-runtime";
import CrossOriginStorage from "./cross_origin_storage";

type StoreType = string | undefined;

const DEFAULT_FETCH_OPTIONS: RequestInit = { method: "GET" };

export class CrossOriginStorageCache implements tvmjs.ArtifactCacheTemplate {
  private storage: CrossOriginStorage;

  constructor(
    _scope: string,
    storage: CrossOriginStorage = new CrossOriginStorage(),
  ) {
    this.storage = storage;
  }

  async fetchWithCache(
    url: string,
    storetype?: StoreType,
    signal?: AbortSignal,
  ): Promise<any> {
    const cachedResponse = await this.storage.match(url);
    if (cachedResponse !== undefined) {
      return this.responseToStoreType(cachedResponse, storetype);
    }

    await this.addToCache(url, storetype, signal);
    const hydrated = await this.storage.match(url);
    if (hydrated === undefined) {
      throw new Error(`CrossOriginStorageCache: failed to hydrate ${url}`);
    }
    return this.responseToStoreType(hydrated, storetype);
  }

  async addToCache(
    url: string,
    _storetype?: StoreType,
    signal?: AbortSignal,
  ): Promise<void> {
    const existing = await this.storage.match(url);
    if (existing !== undefined) {
      return;
    }
    const request = new Request(
      url,
      signal ? { ...DEFAULT_FETCH_OPTIONS, signal } : DEFAULT_FETCH_OPTIONS,
    );
    const response = await fetch(request);
    if (!response.ok) {
      throw new Error(
        `CrossOriginStorageCache: Unable to fetch ${url}, received status ${response.status}`,
      );
    }
    const cloned = response.clone();
    await this.storage.put(url, cloned);
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    const results = await Promise.all(
      keys.map(async (key) => {
        const cached = await this.storage.match(key);
        return cached !== undefined;
      }),
    );
    return results.every((item) => item);
  }

  async deleteInCache(_url: string): Promise<void> {
    await this.storage.delete(_url);
  }

  private async responseToStoreType(
    response: Response,
    storetype?: StoreType,
  ): Promise<any> {
    if (storetype === undefined) {
      return response;
    }
    const format = storetype.toLowerCase();
    if (format === "json") {
      return response.json();
    }
    if (format === "arraybuffer") {
      return response.arrayBuffer();
    }
    return response;
  }
}

export default CrossOriginStorageCache;
