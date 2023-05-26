# SimpleChat

This folder provides a complete implementation of a simple
chat app based on WebLLM. To try it out, you can do the following steps

- Modify [package.json](package.json) to make sure either
    - Option 1: `@mlc-ai/web-llm` points to a valid npm version e.g.
      ```js
      "dependencies": {
          "@mlc-ai/web-llm": "^0.2.0"
       }
      ```
      Try this option if you would like to use WebLLM.
    - Option 2: Or keep the dependencies as `"file:../.."`, and follow the build from source
      instruction in the project to build webllm locally. This option is more useful
      for developers who would like to hack WebLLM core package.
- Run the following command
  ```bash
  npm install
  npm start
  ```
