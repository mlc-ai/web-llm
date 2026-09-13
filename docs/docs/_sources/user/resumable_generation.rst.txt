Resumable generation
====================

Resumability is opt-in, per request. It journals a text-only request's full
prompt, generated tokens, text patches, and RNG state in same-origin OPFS.
Optional KV checkpoints accelerate recovery; token replay does not require
checkpoint-capable model libraries.

.. code-block:: typescript

   const sessionId = crypto.randomUUID();
   const chunks = await engine.chat.completions.create({
     messages,
     stream: true,
     extra_body: {
       resumable: { enabled: true, sessionId, strictPersistence: true },
     },
   });
   for await (const chunk of chunks) { /* display chunk */ }

   // After a crash, load the SAME model/configuration before continuing.
   const saved = await engine.resumeChatCompletion(sessionId);
   // Replace any stale UI response with saved.recoveredText first.
   const resumed = await engine.resumeChatCompletion(sessionId, {
     continueGeneration: true,
   });

Request and recovery behavior
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Behavior
   * - Resumability disabled
     - Normal WebLLM behavior, including in-memory conversation prefix reuse.
       No session journal or checkpoint is created.
   * - New resumable request; any number of prior turns
     - Supply the full message history. The request resets and prefills that
       history, even if its prefix is in memory. The persisted prompt is
       self-contained, not a delta referring to another session.
   * - Another session on the same model
     - Uses a new ID and waits for the current request's model lock. Its prefill
       replaces the in-memory cache; the previous session's persisted data remains.
       As in ordinary WebLLM, ``interruptGenerate()`` can also abort a queued
       non-streaming request before prefill; that skipped request creates no session.
       Explicit iterator ``return()`` does not set this engine-wide interrupt flag.
   * - Reusing a session ID
     - A new request rejects an existing ID, even after completion. Use
       ``resumeChatCompletion`` for continuation, or explicitly delete the session
       before reusing its ID. Prefer a fresh ID per request.
   * - Read-only recovery
     - ``resumeChatCompletion(id)`` returns saved text without loading a model or
       generating tokens. It may repair a torn tail and clean checkpoints if it
       can obtain the session lock; an active session is read without mutation.
       Reads and session listing take a short journal I/O lock, not the lifetime
       generation lock. Without a supported locking backend, text inspection
       remains best-effort.
   * - Continued recovery with usable KV
     - Import the newest readable committed checkpoint, rebuild output and
       penalties, and forward only journaled tokens beyond that checkpoint.
       Restore the RNG before sampling new tokens.
   * - KV absent, corrupt, unsupported, or import rejected
     - Re-prefill the persisted prompt and forward the saved generated tokens.
       Saved tokens are not resampled. Restore the RNG and continue.
   * - No generated token committed yet
     - Sample the first token from checkpoint logits, or from replayed prompt
       logits. A supplied request seed is restored first.
   * - Checkpoint has no saved logits
     - KV recovery works if a journaled token follows the checkpoint. Otherwise
       fall back to prompt/token replay to obtain next-token logits.
   * - Missing model, RNG state, or replay context; already finished
     - No further generation; recover saved text where possible. A missing
       session or malformed resumable configuration raises an error.
   * - Streaming continuation
     - Available when both the original request and resume options use
       ``stream: true``. Emits only newly generated deltas, not the saved prefix.
       Other continuation paths return a ``ResumeResult`` containing full text.
   * - Unconsumed or cancelled stream
     - An unstarted direct-engine stream holds no generation/session locks.
       ``return()`` or breaking ``for await`` on a started stream records an abort
       and releases its locks. Merely abandoning a started iterator cannot be
       detected; close it explicitly.
   * - Interrupt, crash, or another crash during recovery
     - Continue the same unfinished session from its last valid journal prefix.
       An interrupted generation is resumable; a normally finished one is not.
   * - Concurrent tab/worker access
     - One writer per session, using Web Locks or a sync-access-handle lock.
       Continuing/deleting an active session rejects. Without cross-context
       locking, persistence is disabled or fails according to strictness.

Persistence and storage
-----------------------

``durabilityMode: "exact"`` (default) waits for each token's OPFS append before
exposing it. ``"relaxed"`` queues writes and waits every eight tokens or when
250 ms have elapsed at a token boundary; a crash can lose an unflushed suffix.
Neither mode protects against origin eviction, clearing browser data, or all
OS/power failures.

``strictPersistence: false`` (default) lets generation continue after journal
failure, without further persistence. If only a KV checkpoint write fails, that
checkpoint is skipped and healthy token journaling can continue.
``true`` makes persistence failures reject
generation, including an abort write/flush failure during explicit stream
cancellation. Unsupported KV export and a low reported storage estimate skip KV
capture but still permit token journaling; strictness does not require KV recovery.
Estimates are advisory: a browser can report headroom while enforcing a lower
quota. An actual checkpoint write failure follows the same strictness policy.
Cleanup failure after a durable finished record is logged and retried on later
session inspection, not treated as a failed generation.

Prompt checkpointing defaults to enabled. Decode checkpoints default to every
512 tokens, aligned to the cache page size. KV capture is skipped below the
storage headroom threshold (at least 512 MiB, or twice the checkpoint size).
Retention keeps the newest two payload-valid committed checkpoints; incomplete,
uncommitted, and invalid checkpoints are removed under the session lock.
Normal completion removes residual KV but retains the journal/text. Use
``listResumableSessions()`` and ``deleteResumableSession(id)`` to manage sessions.
There is no TTL or automatic journal eviction.

Persisted data includes the full conversation and is not encrypted by WebLLM.
Deletion does not clear the active model's in-memory prefix cache. After normal
or resumed completion, an ordinary non-resumable follow-up can reuse a matching
in-memory conversation prefix; another resumable request always prefills anew.

Supported scope and validation
------------------------------

Supported: text-only chat completions with one choice. The legacy
``completions`` endpoint rejects resumability. Resumable chat requests reject
images, grammar/JSON/structural-tag constraints, and custom LogitProcessors.
KV export additionally requires a compatible model library containing TVM's
checkpoint primitives and a supported pure KV-cache layout.
Unsupported cache layouts use token replay.

Model/tokenizer/weights/configuration fingerprints are not implemented. Keep
those artifacts unchanged when resuming; a matching model ID or KV layout hash
alone does not verify model identity. This is not a persistent cross-request
prefix cache.

Run ``npm ci``, ``npm test -- --runInBand``, and ``npm run test:browser`` for the
unit and model-free browser regressions. The latter exercises real OPFS, Web
Locks, and XGrammar WASM; its tensor-boundary fixtures are not GPU inference.

The opt-in real-model suite requires WebGPU with shader-f16 support, enough
storage for model weights/checkpoints, and a checkpoint-capable
``Qwen3-0.6B-q4f16_1-MLC`` model library:

.. code-block:: bash

   npm run build:browser-tests
   WEBLLM_TEST_MODEL_LIB=https://your-host/Qwen3-0.6B.wasm \
     npx playwright test --config tests/browser/playwright.config.mjs \
       --headed --grep 'real WebGPU'

It compares uninterrupted seeded output with repeated page-crash recovery,
with four prior messages and a prompt spanning multiple prefill chunks,
separately forcing token replay and KV recovery. It covers all combinations of
exact/relaxed durability and strict/best-effort persistence, completed-session
inspection, ordinary prefix reuse after session deletion, and full-history
prefill for a new resumable session. Opting in fails, rather than skips, if GPU
inference or KV import is unavailable.

Additional real-GPU cases cover zero-token recovery at checkpoint write/commit
boundaries, saved logits present or absent, decode-checkpoint retention and
corruption, torn journal repair, token-write failures, cancellation, competing
sessions/tabs, low browser quota, and deliberate browser-process crashes followed
by relaunching the same profile. Fault hooks pause writes or inject storage
errors; they do not replace inference. Unsupported requests are checked for
rejection without creating sessions.

For a locally compiled model library and downloaded model directory:

.. code-block:: bash

   WEBLLM_TEST_MODEL_LIB_PATH=/absolute/path/to/Qwen3-0.6B.wasm \
   WEBLLM_TEST_MODEL_PATH=/absolute/path/to/Qwen3-0.6B-q4f16_1-MLC \
   WEBLLM_TEST_BROWSER_EXECUTABLE=/absolute/path/to/chromium \
     npx playwright test --config tests/browser/playwright.config.mjs

The model directory must contain the matching configuration, tokenizer files,
tensor-cache manifest, and weight shards. Local files are served over loopback
HTTP. Do not set both the model-library URL and local-path options.

The browser bundle uses the package-locked runtime by default. To test a TVM
source build, run ``WEBLLM_TEST_RUNTIME_PATH=/absolute/path/to/tvm/web npm run
build:browser-tests`` after building that checkout's WASM runtime and JavaScript
package. Run without this variable to rebuild against the published runtime.
Record the TVM/MLC-LLM revisions and model-library hash alongside each result.

Each test uses a fresh persistent browser profile, retaining its OPFS contents
across page reloads. Set ``WEBLLM_TEST_PROFILE_ROOT`` to an existing directory
on a volume with ample free space if the default temporary volume is nearly
full; Chromium's blob-storage reserve can reject large cache writes even when
the model itself would fit. On macOS, also set ``MAC_CHROMIUM_TMPDIR`` to that
volume for Chromium's temporary files; ``TMPDIR`` alone does not redirect them.
Use Playwright's ``--output`` option to move test output if needed.
Profiles are removed after each test. These tests
exercise browser-process and page recovery, not OS/power-loss durability.
