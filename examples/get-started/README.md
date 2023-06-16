# WebLLM Get Started App

This folder provides a minimum demo to show WebLLM API in a webapp setting.
To try it out, you can do the following steps

- Modify [package.json](package.json) to make sure either
    - `@mlc-ai/web-llm` points to a valid npm version e.g.
      ```js
      "dependencies": {
        "@mlc-ai/web-llm": "^0.2.0"
      }
      ```
  ```bash
  npm install
  npm start
  ```

Note if you would like to hack WebLLM core package.
You keep the dependencies as `"file:../.."`, and follow the build from source
instruction in the project to build webllm locally. This option is only recommended
if you would like to hack WebLLM core package.
