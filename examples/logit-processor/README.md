# WebLLM Logit Processor and Low-Level API Example

This folder explains the usage of `LogitProcessor`, demonstrating how it can be used to
manipulate the raw logits before sampling the token (e.g. setting certain tokens to `inf` or `-inf`).
We demonstrate how to use it with and without a web worker, which can be toggled with `USE_WEB_WORKER`
in `logit_processor.ts` (see `worker.ts` on how `LogitProcessor` plays a role there).

We also demonstrate the usage of a low-level API `forwardTokenAndSample()`, which, unlike `chat.completions.create()`
that assumes the usage is for autoregressive chatting, here we have more fine-grained control.

See `my_logit_processor.ts` on how to customize your own logit processor. Here we make the logit
of token 0 `100.0` manually, large enough that we should expect to always sample token 0, which
is indeed the case if we observe the console log. We also demonstarte that a LogitProcessor can be
stateful, and the state can also be cleaned with `LogitProcessor.resetState()`.

To try it out, you can do the following steps under this folder

```bash
npm install
npm start
```

Note if you would like to hack WebLLM core package, you can change web-llm dependencies as `"file:../.."`, and follow the build from source instruction in the project to build webllm locally. This option is only recommended if you would like to hack WebLLM core package.
