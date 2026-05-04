# WebLLM Cache Usage

WebLLM supports multiple persistent cache backends. You can pick the classic Cache API, IndexedDB, [Origin Private File System (OPFS)](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system), or the experimental Chrome [Cross-Origin Storage](https://github.com/WICG/cross-origin-storage) extension by
setting `AppConfig.cacheBackend` to `"cache"`, `"indexeddb"`, `"opfs"`, or `"cross-origin"`.
This folder provides an example on how different caches are used in WebLLM. We also
demonstrate the utility cache functions such as deleting models, checking if models are in cache, etc.

> **Note:** The cross-origin backend requires installation of the [Cross-Origin Storage browser extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih) ([source code](https://github.com/web-ai-community/cross-origin-storage-extension)). This does not currently support programmatic tensor-cache deletion; deletion is extension-managed.

> **Note:** If `"opfs"` is selected in an environment without OPFS support, cache operations fail with an OPFS availability error.

For more information about Cache API and IndexedDB, see:
https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria#what_technologies_store_data_in_the_browser.

To inspect the downloaded artifacts in your browser, open up developer console, go to application,
and you will find the artifacts under `IndexedDB`, `Cache storage`, or OPFS (under the browser's origin-private file system). When `"cross-origin"` is selected,
the extension displays origins and resource hashes.

To run the example, you can do the following steps under this folder

```bash
npm install
npm start
```

Note if you would like to hack WebLLM core package.
You can change web-llm dependencies as `"file:../.."`, and follow the build from source
instruction in the project to build webllm locally. This option is only recommended
if you would like to hack WebLLM core package.
