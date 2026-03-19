# WebLLM - Cross-Origin Storage Integration Example

This example demonstrates how to use the experimental [Cross-Origin Storage (COS) API](https://github.com/tomayac/cross-origin-storage) with WebLLM.

## Overview

The Cross-Origin Storage API allows decentralized storage of model shards across multiple domains. Instead of each website downloading and caching its own copy of a model (e.g., Llama-3 8B), they can share a common cache. This integration in WebLLM:
1.  **Resolves SHA-256 hashes** for model files (shards, tokenizers, configs).
2.  **Requests handles** from `navigator.crossOriginStorage`.
3.  **Falls back** to standard indexedDB or Cache API if COS is unavailable or a shard is missing.
4.  **Auto-populates** COS in the background after a successful fetch.

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
4.  Open the link in your browser.

## Using the COS Extension

For the best experience (and to see real cross-origin benefits), you should have the **Cross-Origin Storage Browser Extension** installed. If the extension is not found, this demo will use a mock to show the console logs of how the integration communicates with the API.

## Verifying Integration

Open the browser's developer console. Look for logs prefixed with `[COS]`:
- `[COS] Resolved hash for ... via HF LFS` - Searching for the content hash on Hugging Face.
- `[COS] Match found in COS for ...` - Successfully retrieving a file from the shared storage.
- `[COS] Background storing ... in COS` - Saving a newly downloaded model file to the shared storage.

## Smallest Model Recommendation

For quick testing, we recommend using **Qwen2-0.5B-Instruct-q4f16_1-MLC**. It is the smallest prebuilt model currently supported, allowing for rapid iteration and testing of the caching mechanism.
