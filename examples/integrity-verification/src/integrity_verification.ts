import * as webllm from "@mlc-ai/web-llm";

const statusEl = document.getElementById("status")!;

function setStatus(msg: string) {
  console.log(msg);
  statusEl.innerText = msg;
}

// ---------------------------------------------------------------------------
// Example: Basic integrity verification using the native `integrity` field.
//
// When `integrity` is set on a ModelRecord, WebLLM will verify the downloaded
// config, WASM, and tokenizer files against the provided SRI hashes before
// loading the model. If a hash does not match, an IntegrityError is thrown
// (or a warning is logged if `onFailure: "warn"` is set).
//
// To generate SRI hashes for model files, you can use either:
//   1. `openssl dgst -sha256 -binary <file> | openssl base64 -A`
//   2. `npx verifyfetch sign <file>` (from the @verifyfetch/cli package)
// ---------------------------------------------------------------------------

async function main() {
  setStatus("Initializing...");

  // Example model configuration with integrity hashes.
  // Replace the placeholder hashes below with real SRI hashes for your model.
  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        model:
          "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
        model_id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
        model_lib:
          webllm.modelLibURLPrefix +
          webllm.modelVersion +
          "/Llama-3.2-1B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
        low_resource_required: true,
        vram_required_MB: 879.04,
        overrides: {
          context_window_size: 4096,
        },
        // Uncomment and fill in real SRI hashes to enable verification:
        // integrity: {
        //   config: "sha256-<base64-hash-of-mlc-chat-config.json>",
        //   model_lib: "sha256-<base64-hash-of-wasm-file>",
        //   tokenizer: {
        //     "tokenizer.json": "sha256-<base64-hash-of-tokenizer.json>",
        //   },
        //   onFailure: "error",  // or "warn" to log and continue
        // },
      },
    ],
  };

  try {
    const engine = await webllm.CreateMLCEngine(
      "Llama-3.2-1B-Instruct-q4f16_1-MLC",
      {
        appConfig,
        initProgressCallback: (report) => {
          setStatus(report.text);
        },
      },
    );

    setStatus("Model loaded! Generating response...");

    const reply = await engine.chat.completions.create({
      messages: [{ role: "user", content: "Hello! What can you do?" }],
    });

    setStatus("Response: " + reply.choices[0].message.content);
  } catch (error) {
    if (error instanceof webllm.IntegrityError) {
      setStatus(
        `Integrity verification failed!\n` +
          `URL: ${error.url}\n` +
          `Expected: ${error.expected}\n` +
          `Got: ${error.actual}`,
      );
    } else {
      setStatus("Error: " + error);
    }
  }
}

main();

// ---------------------------------------------------------------------------
// Advanced usage: For full model weight verification, resumable downloads,
// chunked verification, and progress reporting, use @verifyfetch/webllm:
//
//   import { VerifiedMLCEngine } from "@verifyfetch/webllm";
//
//   const engine = new VerifiedMLCEngine({
//     verification: { manifestUrl: "/models/vf.manifest.json" },
//     initProgressCallback: (report) => console.log(report.text),
//   });
//   await engine.reload("Llama-3.2-1B-Instruct-q4f16_1-MLC");
//
// Learn more: https://github.com/hamzaydia/verifyfetch
// ---------------------------------------------------------------------------
