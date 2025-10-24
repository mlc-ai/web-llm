const HASH_ALGORITHM = "SHA-256";
const HASH_MATCH_REGEX = /[A-Fa-f0-9]{64}/;

export interface CrossOriginHashDescriptor {
  algorithm: string;
  value: string;
}

interface CrossOriginStorageHandle {
  getFile(): Promise<Blob>;
  createWritable(): Promise<FileSystemWritableFileStream>;
}

interface CrossOriginStorageAPI {
  requestFileHandles(
    descriptors: CrossOriginHashDescriptor[],
    options?: { create?: boolean },
  ): Promise<CrossOriginStorageHandle[]>;
  removeFileHandles?(descriptors: CrossOriginHashDescriptor[]): Promise<void>;
}

type RequestLike = string | URL | Request | { url?: string };

declare global {
  interface Navigator {
    crossOriginStorage?: CrossOriginStorageAPI;
  }
}

export default class CrossOriginStorage {
  private hashCache: Map<string, CrossOriginHashDescriptor>;

  constructor() {
    this.hashCache = new Map();
  }

  static isAvailable(): boolean {
    return (
      typeof navigator !== "undefined" &&
      "crossOriginStorage" in navigator &&
      navigator.crossOriginStorage !== undefined
    );
  }

  async match(request: RequestLike): Promise<Response | undefined> {
    const url = this.normalizeRequest(request);
    const hash = await this.resolveHashDescriptor(url);
    if (!hash) {
      return undefined;
    }
    try {
      const api = this.getApi();
      if (!api) {
        return undefined;
      }
      const handles = await api.requestFileHandles([hash]);
      const handle = handles[0];
      if (!handle) {
        return undefined;
      }
      const blob = await handle.getFile();
      return new Response(blob);
    } catch {
      return undefined;
    }
  }

  async put(request: RequestLike, response: Response): Promise<void> {
    const url = this.normalizeRequest(request);
    const blob = await response.blob();
    const hash = await this.getBlobHash(blob);
    const api = this.getApi();
    if (!api) {
      throw new Error("Cross-origin storage API unavailable.");
    }
    const handles = await api.requestFileHandles([hash], { create: true });
    const handle = handles[0];
    if (!handle) {
      throw new Error("Cross-origin storage API returned no handles.");
    }
    const writableStream = await handle.createWritable();
    await writableStream.write(blob);
    await writableStream.close();
    this.hashCache.set(url, hash);
  }

  async delete(request: RequestLike): Promise<void> {
    const url = this.normalizeRequest(request);
    const hash = await this.resolveHashDescriptor(url);
    if (!hash) {
      return;
    }
    const api = this.getApi();
    if (api && typeof api.removeFileHandles === "function") {
      await api.removeFileHandles([hash]);
    }
    this.hashCache.delete(url);
  }

  private getApi(): CrossOriginStorageAPI | undefined {
    if (!CrossOriginStorage.isAvailable()) {
      return undefined;
    }
    return navigator.crossOriginStorage;
  }

  private normalizeRequest(request: RequestLike): string {
    if (typeof request === "string") {
      return request;
    }
    if (request instanceof URL) {
      return request.href;
    }
    if (request instanceof Request) {
      return request.url;
    }
    if (request && typeof request.url === "string") {
      return request.url;
    }
    throw new Error("CrossOriginStorage: Unsupported request type.");
  }

  private async resolveHashDescriptor(
    url: string,
  ): Promise<CrossOriginHashDescriptor | null> {
    const cached = this.hashCache.get(url);
    if (cached) {
      return cached;
    }
    const hashValue = await this.getFileHash(url);
    if (!hashValue) {
      return null;
    }
    const descriptor: CrossOriginHashDescriptor = {
      algorithm: HASH_ALGORITHM,
      value: hashValue,
    };
    this.hashCache.set(url, descriptor);
    return descriptor;
  }

  // Gets the SHA-256 hash for large resources using request metadata.
  private async getFileHash(url: string): Promise<string | null> {
    const metadataHash = await this.extractHashFromHead(url);
    if (metadataHash) {
      return metadataHash;
    }
    if (/\/resolve\/main\//.test(url)) {
      const pointerHash = await this.extractHashFromPointer(url);
      if (pointerHash) {
        return pointerHash;
      }
    }
    return null;
  }

  private async extractHashFromHead(url: string): Promise<string | null> {
    try {
      const response = await fetch(url, { method: "HEAD" });
      if (!response.ok) {
        return null;
      }
      const headerNames = [
        "x-linked-etag",
        "x-linked-hash",
        "x-amz-meta-sha256",
        "x-oss-meta-sha256",
        "x-sha256",
        "etag",
      ];
      for (const name of headerNames) {
        const value = response.headers.get(name);
        const hash = this.extractSha256(value);
        if (hash) {
          return hash;
        }
      }
    } catch {
      // Swallow errors; fall back to other strategies.
    }
    return null;
  }

  private async extractHashFromPointer(url: string): Promise<string | null> {
    try {
      const rawUrl = url.replace(/\/resolve\//, "/raw/");
      const response = await fetch(rawUrl, {
        headers: { Range: "bytes=0-1023" },
      });
      if (!response.ok) {
        return null;
      }
      const text = await response.text();
      if (!text.includes("oid sha256:")) {
        return null;
      }
      const match = text.match(/oid sha256:([A-Fa-f0-9]+)/);
      return match ? match[1] : null;
    } catch {
      return null;
    }
  }

  private extractSha256(value: string | null): string | null {
    if (!value) {
      return null;
    }
    const match = value.match(HASH_MATCH_REGEX);
    return match ? match[0].toLowerCase() : null;
  }

  private async getBlobHash(blob: Blob): Promise<CrossOriginHashDescriptor> {
    const arrayBuffer = await blob.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");

    return {
      algorithm: HASH_ALGORITHM,
      value: hashHex,
    };
  }
}
