# WebLLM Chrome Extension using WebGPU Running on Service Worker

![Chrome Extension](https://github.com/mlc-ai/mlc-llm/assets/11940172/0d94cc73-eff1-4128-a6e4-70dc879f04e0)

> [!WARNING]  
> Service worker support in WebGPU is enabled by default in [Chrome 124](https://chromiumdash.appspot.com/commit/8d78510e4aca5ac3cd8ee4a33e96b404eaa43246).
> If you are using Chrome 123, go to `chrome://flags/#enable-experimental-web-platform-features`, enable the `#enable-experimental-web-platform-features` flag, and **relaunch the browser**.

This example shows how we can create a Chrome extension using WebGPU and service worker.

- The project structure is as follows:
  - `manifest.json`: A required file that lists important information about the structure and behavior of that extension. Here we are using manifest V3.
  - `popup.ts`: Script of the extension pop-up window.
  - `background.ts`: Script of the service worker. An extension service worker is loaded when it is needed, and unloaded when it goes dormant.
  - `content.js`: Content script that interacts with DOM.
- Run

  ```bash
  npm install
  npm run build
  ```

  This will create a new directory at `./dist/`. To load the extension into Chrome, go to Extensions > Manage Extensions and select Load Unpacked. Add the `./dist/` directory. You can now pin the extension to your toolbar and use it to chat with your favorite model!

**Note**: This example disables chatting using the contents of the active tab by default.
To enable it, set `useContext` in `popup.ts` to `true`. More info about this feature can be found
[here](https://github.com/mlc-ai/web-llm/pull/190).
However, if the web content is too large, it might run into issues. We recommend using `example.html` to
test this feature.
