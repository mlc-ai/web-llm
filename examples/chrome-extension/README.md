# WebLLM Chrome Extension

![Chrome Extension](https://github.com/mlc-ai/mlc-llm/assets/11940172/0d94cc73-eff1-4128-a6e4-70dc879f04e0)

To run the extension, first start a local server using the steps outlined [here](https://mlc.ai/mlc-llm/docs/deploy/rest.html). Next, you can do the following steps under this folder

```bash
npm install
npm run build
```

This will create a new directory at `chrome-extension/dist/`. To load the extension into Chrome, first launch Chrome with web security disabled (this is to disable CORS).

```bash
google-chrome --disable-web-security --user-data-dir=/some/dir/
```

Next, go to Extensions > Manage Extensions and select Load Unpacked. Add the `chrome-extension/dist/` directory. You can now pin the extension to your toolbar and use it to chat with your favorite model!