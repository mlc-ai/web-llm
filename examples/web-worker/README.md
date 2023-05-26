# WebLLM Get Started with WebWorker

This folder provides a minimum demo to show WebLLM API using
[WebWorker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers).
The main benefit of web worker is that all ML workloads runs on a separate thread as a result
will less likely block the UI.

To try it out, you can do the following steps

- Modify [package.json](package.json) to make sure either
    - `@mlc-ai/web-llm` points to a valid npm version e.g.
      ```js
      "dependencies": {
        "@mlc-ai/web-llm": "^0.2.0"
      }
      ```
      Try this option if you would like to use WebLLM without building it yourself.
    - Or keep the dependencies as `"file:../.."`, and follow the build from source
      instruction in the project to build webllm locally. This option is more useful
      for developers who would like to hack WebLLM core package.
- Run the following command
  ```bash
  npm install
  npm start
  ```
