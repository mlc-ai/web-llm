# WebLLM - Cross-Origin Storage Integration Example

This example demonstrates how to use the experimental [Cross-Origin Storage (COS) API](https://github.com/WICG/cross-origin-storage) with WebLLM.

## Overview

The Cross-Origin Storage API allows decentralized storage of model shards across multiple domains. Instead of each website downloading and caching its own copy of a model (e.g., Llama-3 8B), they can share a common cache. This integration in WebLLM:

1.  **Resolves SHA-256 hashes** for model files (shards, tokenizers, configs).
2.  **Requests handles** from `navigator.crossOriginStorage`.
3.  **Falls back** to standard indexedDB or Cache API if COS is unavailable or a shard is missing.
4.  **Auto-populates** COS in the background after a successful fetch.

## Installing the COS Extension

For the best experience (and to see real cross-origin benefits), you should have the [Cross-Origin Storage Browser Extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih) installed. If the extension is not found, this demo will fall back to using the default cache.

## Getting Started

1.  Navigate to this directory:
    ```bash
    cd examples/cross-origin-storage
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm start
    ```
4.  Open `http://localhost:8888` and open the DevTools Console.
5.  Wait for the model resources to be loaded and stored in COS.
6.  Reload the app.
7.  With the [COS extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih) installed, the model resources should now be served from COS.
8.  Start the example on a different origin:
    ```bash
    npx parcel src/cross_origin_storage.html --port 1234
    ```
9.  Open `http://localhost:1234` and open the Console.
10. With the COS extension installed, the model resources should now be served from COS—cross-origin 🚀!

## Verifying Integration

Open the browser's developer console. Look for logs prefixed with `[COS]`:

- `[COS] Resolved hash for ... via HF LFS` - Searching for the content hash on Hugging Face.
- `[COS] Match found in COS for ...` - Successfully retrieving a file from the shared storage.
- `[COS] Background storing ... in COS` - Saving a newly downloaded model file to the shared storage.
