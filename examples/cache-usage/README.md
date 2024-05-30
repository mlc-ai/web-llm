# WebLLM Cache Usage

WebLLM supports both the Cache API and IndexedDB, which you can specify via `AppConfig.useIndexedDBCache`.
This folder provides an example on how Cache and IndexedDB Cache are used in WebLLM. We also
demonstrate the utility cache functions such as deleting models, checking if models are in cache, etc.

For more information about the two caches, see: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria#what_technologies_store_data_in_the_browser.

To inspect the downloaded artifacts in your browser, open up developer console, go to application,
and you will find the artifacts under either `IndexedDB` or `Cache storage`.

To run the exapmle, you can do the following steps under this folder

```bash
npm install
npm start
```

Note if you would like to hack WebLLM core package.
You can change web-llm dependencies as `"file:../.."`, and follow the build from source
instruction in the project to build webllm locally. This option is only recommended
if you would like to hack WebLLM core package.
