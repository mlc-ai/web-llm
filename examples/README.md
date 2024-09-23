# Awesome WebLLM

This page contains a curated list of examples, tutorials, blogs about WebLLM usecases.
Please send a pull request if you find things that belongs to here.

## Example Projects

Note that all examples below run in-browser and use WebGPU as a backend.

#### Project List

- [get-started](get-started): minimum get started example with chat completion.

  [![Open in JSFiddle](https://img.shields.io/badge/open-JSFiddle-blue?logo=jsfiddle&logoColor=white)](https://jsfiddle.net/neetnestor/yac9gbwf/)
  [![Open in Codepen](https://img.shields.io/badge/open-codepen-gainsboro?logo=codepen)](https://codepen.io/neetnestor/pen/NWVdgey)

- [simple-chat-js](simple-chat-js): a mininum and complete chat bot app in vanilla JavaScript.

  [![Open in JSFiddle](https://img.shields.io/badge/open-JSFiddle-blue?logo=jsfiddle&logoColor=white)](https://jsfiddle.net/neetnestor/4nmgvsa2/)
  [![Open in Codepen](https://img.shields.io/badge/open-codepen-gainsboro?logo=codepen)](https://codepen.io/neetnestor/pen/vYwgZaG)

- [simple-chat-ts](simple-chat-ts): a mininum and complete chat bot app in TypeScript.
- [get-started-web-worker](get-started-web-worker): same as get-started, but using web worker.
- [next-simple-chat](next-simple-chat): a mininum and complete chat bot app with [Next.js](https://nextjs.org/).
- [multi-round-chat](multi-round-chat): while APIs are functional, we internally optimize so that multi round chat usage can reuse KV cache
- [text-completion](text-completion): demonstrates API `engine.completions.create()`, which is pure text completion with no conversation, as opposed to `engine.chat.completions.create()`
- [embeddings](embeddings): demonstrates API `engine.embeddings.create()`, integration with `EmbeddingsInterface` and `MemoryVectorStore` of [Langchain.js](js.langchain.com), and RAG with Langchain.js using WebLLM for both LLM and Embedding in a single engine
- [multi-models](multi-models): demonstrates loading multiple models in a single engine concurrently

#### Advanced OpenAI API Capabilities

These examples demonstrate various capabilities via WebLLM's OpenAI-like API.

- [streaming](streaming): return output as chunks in real-time in the form of an AsyncGenerator
- [json-mode](json-mode): efficiently ensure output is in json format, see [OpenAI Reference](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) for more.
- [json-schema](json-schema): besides guaranteeing output to be in JSON, ensure output to adhere to a specific JSON schema specified the user
- [seed-to-reproduce](seed-to-reproduce): use seeding to ensure reproducible output with fields `seed`.
- [function-calling](function-calling) (WIP): function calling with fields `tools` and `tool_choice` (with preliminary support).
- [vision-model](vision-model): process request with image input using Vision Language Model (e.g. Phi3.5-vision)

#### Chrome Extension

- [chrome-extension](chrome-extension): chrome extension that does not have a persistent background
- [chrome-extension-webgpu-service-worker](chrome-extension-webgpu-service-worker): chrome extension using service worker, hence having a persistent background

#### Others

- [logit-processor](logit-processor): while `logit_bias` is supported, we additionally support stateful logit processing where users can specify their own rules. We also expose low-level API `forwardTokensAndSample()`.
- [cache-usage](cache-usage): demonstrates how WebLLM supports both the [Cache API](https://developer.mozilla.org/en-US/docs/Web/API/Cache) and [IndexedDB cache](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API), and
  users can pick with `appConfig.useIndexedDBCache`. Also demonstrates various cache utils such as checking
  whether a model is cached, deleting a model's weights from cache, deleting a model library wasm from cache, etc.
- [simple-chat-upload](simple-chat-upload): demonstrates how to upload local models to WebLLM instead of downloading via a URL link

## Demo Spaces

- [web-llm-embed](https://huggingface.co/spaces/matthoffner/web-llm-embed): document chat prototype using react-llm with transformers.js embeddings
- [DeVinci](https://x6occ-biaaa-aaaai-acqzq-cai.icp0.io/): AI chat app based on WebLLM and hosted on decentralized cloud platform
