# WebLLM Subgroups Usage App

This folder provides a minimum demo to show capability-based routing between
baseline and subgroup WebGPU WASM builds in a webapp setting.
To try it out, you can do the following steps under this folder

```bash
npm install
npm start
```

Edit `src/subgroups_usage.ts` if you would like to point the example at your own
model path and baseline `model_lib`. The example will suffix the WASM filename
with `-subgroups` before the `.wasm` extension when the adapter reports
subgroup support.

Note if you would like to hack WebLLM core package.
You can change the WebLLM dependency to `"file:../.."`, and follow the build
from source instruction in the project to build webllm locally. This option is only recommended
if you would like to hack WebLLM core package.
