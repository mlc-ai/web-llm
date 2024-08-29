# WebLLM Chrome Extension

![Chrome Extension](https://github.com/mlc-ai/mlc-llm/assets/11940172/0d94cc73-eff1-4128-a6e4-70dc879f04e0)

To run the extension, do the following steps under this folder

```bash
npm install
npm run build
```

This will create a new directory at `chrome-extension/dist/`. To load the extension into Chrome, go to Extensions > Manage Extensions and select Load Unpacked. Add the `chrome-extension/dist/` directory. You can now pin the extension to your toolbar and use the drop-down menu to chat with your favorite model!
