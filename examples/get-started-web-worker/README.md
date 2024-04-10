# WebLLM Get Started with WebWorker

This folder provides a minimum demo to show WebLLM API using
[WebWorker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers).
The main benefit of web worker is that all ML workloads runs on a separate thread as a result
will less likely block the UI.

To try it out, you can do the following steps under this folder

```bash
npm install
npm start
```

Note if you would like to hack WebLLM core package.
You can change web-llm dependencies as `"file:../.."`, and follow the build from source
instruction in the project to build webllm locally. This option is only recommended
if you would like to hack WebLLM core package.
