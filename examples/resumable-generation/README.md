# WebLLM Resumable Generation Example

This example is a browser harness for testing crash-resumable generation with
real WebGPU.

It intentionally targets only:

```text
Qwen3-0.6B-q4f16_1-MLC
```

The weights/tokenizer come from WebLLM's prebuilt model config. Supply your own
matching WebGPU model library; this example does not use a personal binary host.
For KV recovery, the WASM must contain the TVM checkpoint primitives. Without
those primitives, the core implementation falls back to token replay.

Create `.env.local` in this directory (the URL must be accessible to the browser):

```dotenv
VITE_WEBLLM_MODEL_LIB_URL=https://your-model-host/Qwen3-0.6B-q4f16_1-webgpu.wasm
# Optional: match your model's compile-time limits. These values describe the
# small artifact used during validation, not requirements for every build.
VITE_WEBLLM_CONTEXT_WINDOW_SIZE=512
VITE_WEBLLM_PREFILL_CHUNK_SIZE=128
```

Restart Vite after changing these settings. Keep the same model library,
weights, tokenizer, and configuration when resuming a saved session.

## Run

```bash
cd examples/resumable-generation
npm install
npm run dev
```

Open the printed localhost URL in Chrome or Edge with WebGPU enabled.

The example depends on the repo root through `file:../..`. Its `dev` and
`build` scripts rebuild the root package first, then Vite serves the generated
`lib/index.js`. This still exercises the current checkout, while avoiding raw
TypeScript runtime export issues in Vite.

## Manual Crash Test

1. Keep `Dedicated Worker` selected.
2. Click `Load`.
3. Click `Start`.
4. After several chunks appear, click `Reload Page` or close and reopen the tab.
5. Click `Load`.
6. Click `List Sessions`.
7. Click `Resume Continue`.

Expected results:

- If no committed KV checkpoint existed yet, recovery should use
  `token_replay`.
- If a committed checkpoint existed, recovery should use `kv`.
- In exact mode, exposed tokens are persisted first. In relaxed mode, a crash
  can lose an unflushed suffix. Decoding can also revise a partial Unicode/text
  suffix; use recovered text as authoritative rather than assuming text is
  strictly append-only.

## Automated Reload Test

Enable `Reload after chunks`, set `Chunks`, then click `Start`. The page reloads
itself after that many streamed chunks. After reload, click `Load`, then
`Resume Continue`.

## Validation Runner

Click `Run Validation` to exercise real WebGPU resumability checks in worker
mode:

- Worker-mode streaming resume after an interrupted generation.
- Same-session repeated resume by interrupting a resumed continuation and
  resuming it again.
- Session lock exclusion from a second worker client against the same OPFS
  session.
- Low-quota behavior when the browser reports less than 512 MiB free storage.

The low-quota case reports `skip` on normal profiles with enough free quota.
For a hard low-quota check, run the page in a constrained browser profile or
manually fill origin storage, then rerun `Run Validation`. The expected behavior
is that KV checkpointing is skipped while token journaling remains recoverable.

For an exact cross-tab lock check, start a resumable generation in one tab,
open this same page in a second tab, enter the same session ID, and click
`Resume Continue`. The second tab should fail with
`Resumable session is already active`.

## Metrics

`Dedicated Worker` mode matches the intended runtime path, but internal engine
metrics live inside the worker. Switch to `Main Thread` mode if you need to
inspect `lastResumableMetrics` directly from the page.
