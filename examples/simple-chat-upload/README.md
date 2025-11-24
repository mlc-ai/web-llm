# SimpleChat

This folder provides a complete implementation of a simple
chat app based on WebLLM. To try it out, you can do the following steps
under this folder

```bash
npm install
npm start
```

Note if you would like to hack WebLLM core package.
You can change web-llm dependencies as `"file:../.."`, and follow the build from source
instruction in the project to build webllm locally. This option is only recommended
if you would like to hack WebLLM core package.

Due to the differences in command-line tools between Unix/Linux and Windows systems, special adaptation is necessary for Windows. Unix/Linux systems natively support commands like `cp` for file operations, which are not directly available in Windows. To ensure cross-platform compatibility, we use a Node.js script for file copying in Windows.

### Steps for Windows Users

1. **Create a Node.js Script File**:
   - In the `examples\simple-chat` directory, create a file named `copy-config.js`.
   - Add the following code to handle file copying:
     ```javascript
     const fs = require("fs");
     // Copy file
     fs.copyFileSync("src/gh-config.js", "src/app-config.js");
     ```

2. **Modify `package.json`**:
   - In the `scripts` section of your `package.json`, replace Unix-style `cp` commands with our new Node.js script. For example:
     ```json
     "scripts": {
         "start": "node copy-config.js && parcel src/llm_chat.html --port 8888",
         "mlc-local": "node copy-config.js && parcel src/llm_chat.html --port 8888",
         "build": "node copy-config.js && parcel build src/llm_chat.html --dist-dir lib --no-content-hash"
     },
     ```

3. **Run the Application**:
   - Save your changes and run `npm start` in CMD or PowerShell to start the application.
