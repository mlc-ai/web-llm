import * as tvmjs from "@mlc-ai/web-runtime";

declare global {
  interface CrossOriginStorageRequestFileHandleHash {
    value: string;
    algorithm: string;
  }

  interface CrossOriginStorageRequestFileHandleOptions {
    create?: boolean;
  }

  interface CrossOriginStorageManager {
    requestFileHandles(
      hashes: CrossOriginStorageRequestFileHandleHash[],
      options?: CrossOriginStorageRequestFileHandleOptions,
    ): Promise<FileSystemFileHandle[]>;
  }

  interface Navigator {
    readonly crossOriginStorage: CrossOriginStorageManager;
  }
}

/**
 * Name of the Cache API bucket used to persist the url→hash mapping.
 */
const HASH_CACHE_NAME = "webllm-cos-hash-cache";
const HASH_ALGORITHM = "SHA-256";

/**
 * Builds the hash descriptor object expected by the cross-origin storage API.
 */
const makeHashDescriptor = (
  value: string,
): { algorithm: string; value: string } => ({
  algorithm: HASH_ALGORITHM,
  value,
});

/**
 * ArtifactCrossOriginStorageCache implements the ArtifactCacheTemplate using the
 * experimental Cross-Origin Storage (COS) API.
 *
 * It acts as a progressive enhancement: if COS is available, it tries to use it.
 * If not, or if a file is not found in COS, it falls back to a primary cache (IndexedDB or Cache API).
 */
export class ArtifactCrossOriginStorageCache
  implements tvmjs.ArtifactCacheTemplate
{
  private primaryCache: tvmjs.ArtifactCacheTemplate;
  private hashCache: Promise<Cache> | null = null;
  private scope: string;

  constructor(scope: string, primaryCacheType: "indexeddb" | "cache") {
    this.scope = scope;
    if (primaryCacheType === "indexeddb") {
      this.primaryCache = new tvmjs.ArtifactIndexedDBCache(scope);
    } else {
      this.primaryCache = new tvmjs.ArtifactCache(scope);
    }
  }

  static isAvailable(): boolean {
    const available =
      typeof navigator !== "undefined" &&
      "crossOriginStorage" in (navigator as any);
    if (available) {
      console.log("[COS] Cross-Origin Storage API is available.");
    } else {
      console.log("[COS] Cross-Origin Storage API is NOT available.");
    }
    return available;
  }

  private _getHashCache(): Promise<Cache> {
    this.hashCache ??= caches.open(HASH_CACHE_NAME);
    return this.hashCache;
  }

  /**
   * Resolves the SHA-256 hash for a given URL.
   * Checks the local hash cache first, then tries to resolve via HF LFS if applicable.
   */
  private async _getFileHash(url: string): Promise<string | null> {
    try {
      const hashCache = await this._getHashCache();
      const cached = await hashCache.match(url);
      if (cached) {
        const hash = await cached.text();
        console.log(`[COS] Resolved hash for ${url} from local cache: ${hash}`);
        return hash;
      }

      // Try Hugging Face LFS resolution
      const hfHash = await this._getLfsFileHash(url);
      if (hfHash) {
        console.log(`[COS] Resolved hash for ${url} via HF LFS: ${hfHash}`);
        await hashCache.put(url, new Response(hfHash));
        return hfHash;
      }

      return null;
    } catch (err) {
      console.error(`[COS] Error resolving hash for ${url}:`, err);
      return null;
    }
  }

  private async _getLfsFileHash(url: string): Promise<string | null> {
    if (!url.includes("/resolve/")) {
      return null;
    }

    const rawUrl = url.replace("/resolve/", "/raw/");
    console.log(`[COS] Attempting HF LFS hash resolution via ${rawUrl}`);

    try {
      const response = await fetch(rawUrl);
      if (!response.ok) return null;
      const text = await response.text();
      const match = text.match(/^oid sha256:([0-9a-f]+)$/m);
      return match ? match[1] : null;
    } catch (err) {
      console.warn(`[COS] HF LFS fetch failed for ${rawUrl}:`, err);
      return null;
    }
  }

  private async _getBlobHash(blob: Blob): Promise<string> {
    const arrayBuffer = await blob.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }
  async fetchWithCache(
    url: string,
    storetype?: string,
    signal?: AbortSignal,
  ): Promise<any> {
    console.log(`[COS] fetchWithCache: ${url} (storetype: ${storetype})`);

    // 1. Try COS
    const hashValue = await this._getFileHash(url);
    if (hashValue) {
      try {
        const [handle] = await (
          navigator as any
        ).crossOriginStorage.requestFileHandles([
          makeHashDescriptor(hashValue),
        ]);
        if (handle) {
          console.log(`[COS] Match found in COS for ${url}`);
          const blob = await handle.getFile();
          if (storetype === "arraybuffer") {
            return await blob.arrayBuffer();
          } else if (storetype === "json") {
            return JSON.parse(await blob.text());
          }
          return blob;
        }
        console.log(
          `[COS] Hash ${hashValue} known but miss in COS for ${url}.`,
        );
      } catch (err) {
        console.warn(`[COS] COS match lookup failed for ${url}.`, err);
      }
    } else {
      console.log(`[COS] No pre-computed hash for ${url}.`);
    }

    if ((this.constructor as any).isAvailable()) {
      console.log(
        `[COS] Fetching ${url} directly from network to evade IDB...`,
      );
      const response = await fetch(url, { signal });
      if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
      let data: any;
      if (storetype === "arraybuffer") data = await response.arrayBuffer();
      else if (storetype === "json") data = await response.json();
      else data = await response.blob();

      this._backgroundStore(url, data, storetype, hashValue || undefined);

      if (
        url.endsWith("ndarray-cache.json") ||
        url.endsWith("tensor-cache.json")
      ) {
        try {
          const json =
            storetype === "json"
              ? data
              : JSON.parse(new TextDecoder().decode(data));
          this._parseNdarrayCache(url, json);
        } catch (err) {
          console.warn(
            `[COS] Failed to parse manifest ${url} for pre-population:`,
            err,
          );
        }
      }
      return data;
    }

    // 2. Fallback to primary cache (this also handles fetching and initial caching)
    const data = await this.primaryCache.fetchWithCache(url, storetype, signal);

    // 3. Background store in COS
    // We store if it was a miss in COS, using the hashValue we found (if any) or recomputing it.
    this._backgroundStore(url, data, storetype, hashValue || undefined);

    // 4. Special case: If we just fetched a manifest, parse it for shard hashes
    if (
      url.endsWith("ndarray-cache.json") ||
      url.endsWith("tensor-cache.json")
    ) {
      try {
        const json =
          storetype === "json"
            ? data
            : JSON.parse(new TextDecoder().decode(data));
        this._parseNdarrayCache(url, json);
      } catch (err) {
        console.warn(
          `[COS] Failed to parse manifest ${url} for pre-population:`,
          err,
        );
      }
    }

    return data;
  }

  private async _parseNdarrayCache(baseUrl: string, json: any) {
    console.log(`[COS] Parsing ndarray-cache.json for shard hashes...`);
    if (!json.records || !Array.isArray(json.records)) return;

    const hashCache = await this._getHashCache();
    for (const record of json.records) {
      // Record might have dataPath (e.g. params_shard_0.bin) and sha256sum
      if (record.dataPath && record.sha256sum) {
        const shardUrl = new URL(record.dataPath, baseUrl).href;
        console.log(
          `[COS] Pre-populating hash for shard: ${shardUrl} -> ${record.sha256sum}`,
        );
        await hashCache.put(shardUrl, new Response(record.sha256sum));
      }
    }
  }

  private async _backgroundStore(
    url: string,
    data: any,
    storetype?: string,
    precomputedHash?: string,
  ) {
    try {
      console.log(
        `[COS] Background storing ${url} in COS... (known hash: ${precomputedHash || "none"})`,
      );
      let blob: Blob;
      if (data instanceof Blob) {
        blob = data;
      } else if (data instanceof ArrayBuffer) {
        blob = new Blob([data]);
      } else if (typeof data === "object") {
        blob = new Blob([JSON.stringify(data)], { type: "application/json" });
      } else {
        blob = new Blob([data]);
      }

      let hashHex = precomputedHash;
      if (!hashHex) {
        console.log(`[COS] Computing hash for ${url} (size: ${blob.size})...`);
        hashHex = await this._getBlobHash(blob);
      }

      const [handle] = await (
        navigator as any
      ).crossOriginStorage.requestFileHandles([makeHashDescriptor(hashHex!)], {
        create: true,
      });

      const writableStream = await handle.createWritable();
      await writableStream.write(blob);
      await writableStream.close();
      console.log(
        `[COS] Successfully stored ${url} in COS with hash ${hashHex}`,
      );

      // Persist hash mapping if we computed it
      if (!precomputedHash) {
        const hashCache = await this._getHashCache();
        await hashCache.put(url, new Response(hashHex));
      }
    } catch (err) {
      console.warn(`[COS] Background store failed for ${url}:`, err);
    }
  }

  async addToCache(
    url: string,
    storetype?: string,
    signal?: AbortSignal,
  ): Promise<void> {
    console.log(`[COS] addToCache: ${url}`);

    // 1. Resolve hash
    const hashValue = await this._getFileHash(url);
    if (hashValue) {
      try {
        const [handle] = await (
          navigator as any
        ).crossOriginStorage.requestFileHandles([
          makeHashDescriptor(hashValue),
        ]);
        if (handle) {
          console.log(`[COS] Already exists in COS: ${url}`);
          return;
        }
      } catch {
        // Fall through
      }
    }

    if ((this.constructor as any).isAvailable()) {
      try {
        console.log(`[COS] Fetching ${url} directly to COS to evade IDB...`);
        const response = await fetch(url, { signal });
        if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
        let data: any;
        if (storetype === "arraybuffer") data = await response.arrayBuffer();
        else if (storetype === "json") data = await response.json();
        else data = await response.blob();

        await this._backgroundStore(
          url,
          data,
          storetype,
          hashValue || undefined,
        );
        return; // Success! Bypassed IDB entirely!
      } catch (err) {
        console.warn(`[COS] Direct COS store failed for ${url}:`, err);
        // Fall back to IDB
      }
    }

    // 2. Add to primary cache
    await this.primaryCache.addToCache(url, storetype, signal);

    // 3. Background store in COS
    const data = await this.primaryCache.fetchWithCache(url, storetype, signal);
    await this._backgroundStore(url, data, storetype, hashValue || undefined);
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    if ((this.constructor as any).isAvailable()) {
      let allInCos = true;
      for (const url of keys) {
        const hashValue = await this._getFileHash(url);
        if (!hashValue) {
          allInCos = false;
          break;
        }
        try {
          const [handle] = await (
            navigator as any
          ).crossOriginStorage.requestFileHandles([
            makeHashDescriptor(hashValue),
          ]);
          if (!handle) {
            allInCos = false;
            break;
          }
        } catch {
          allInCos = false;
          break;
        }
      }
      if (allInCos) {
        console.log(`[COS] hasAllKeys: true (all present in COS)`);
        return true;
      }
    }
    return await this.primaryCache.hasAllKeys(keys);
  }

  async deleteInCache(url: string): Promise<void> {
    console.log(`[COS] deleteInCache: ${url}`);
    // Remove from local hash mapping
    try {
      const hashCache = await this._getHashCache();
      await hashCache.delete(url);
    } catch {
      // Ignore
    }
    // Delete from primary cache
    await this.primaryCache.deleteInCache(url);
  }
}
