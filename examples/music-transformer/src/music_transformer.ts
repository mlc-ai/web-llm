import * as webllm from "@mlc-ai/web-llm";
import { MusicLogitProcessor } from "./music_logit_processor";

const DEBUG_MAX_CYCLES = 2;

/**
 * 
 * @param chat The ChatModule reloaded already.
 * @param musicLogitProcessor Music logit processor that resides in `chat` as well.
 * @yields First chunk has 1020 tokens, subsequent ones are 510 tokens.
 */
async function* chunkGenerator(chat: webllm.ChatInterface, musicLogitProcessor: MusicLogitProcessor) {
  // Generate first token
  let prompt: Array<number> = [55026];
  let nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, /*isPrefill=*/true);

  // 1. Generate first 1020 tokens
  let curTime = 0;  // For debugging
  while (musicLogitProcessor.tokenSequence.length < 1020) {
    nextToken = await chat.forwardTokensAndSample(
      [nextToken],
      musicLogitProcessor.tokenSequence.length + prompt.length,
      /*isPrefill=*/false
    );

    // For debugging
    if ((musicLogitProcessor.tokenSequence.length - 1) % 3 == 0) {
      if (nextToken < curTime) {
        throw Error("Generated past time. curTime=" + curTime + ", nextToken=" + nextToken);
      }
      curTime = nextToken;
    }
  }

  // 2. Take last 510 tokens, make it the new prompt, starting from time 0
  prompt = [...musicLogitProcessor.tokenSequence.slice(510)];
  let startTime = prompt[0];
  for (let i = 0; i < 510; i += 3) {
    prompt[i] -= startTime;
  }
  prompt.unshift(55026);  // Add the AUTOREGRESS prompt back in, making the prompt 511

  // 3. Clear KV cache and logitProcessor.tokenSequence
  yield musicLogitProcessor.tokenSequence;
  chat.resetChat(/*keepStats=*/true);

  // 4. Keep generating chunks of 510 tokens
  let cycles = 0; // Number of 510-token chunks generated after the first 1020 tokens
  while (cycles < DEBUG_MAX_CYCLES) {  // TODO: change to a user-triggered stop
    let curTime = prompt[-3];  // for debugging

    // 4.1. Prefill prompt and get first token
    musicLogitProcessor.curTime = prompt[-3];  // Update curTime so `future_logits()` still work
    nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, /*isPrefill=*/true);
    if (musicLogitProcessor.tokenSequence.length != 1) {
      throw Error("tokenSequence length should be 1 after prefill.");
    }

    // 4.2. Decode autoregressively
    while (musicLogitProcessor.tokenSequence.length < 510) {
      nextToken = await chat.forwardTokensAndSample(
        [nextToken],
        musicLogitProcessor.tokenSequence.length + prompt.length,
        /*isPrefill=*/false,
      );
      // For debugging
      if ((musicLogitProcessor.tokenSequence.length - 1) % 3 == 0) {
        if (nextToken < curTime) {
          throw Error("Generated past time. curTime=" + curTime + ", nextToken=" + nextToken);
        }
        curTime = nextToken;
      }
    }

    // 4.3. Take all the tokenSequences (510 of tokens)
    prompt = [...musicLogitProcessor.tokenSequence];  // there are 510 newly generated tokens
    let startTime = prompt[0];
    for (let i = 0; i < 510; i += 3) {
      prompt[i] -= startTime;
    }
    prompt.unshift(55026);  // Add the AUTOREGRESS prompt back in, making the prompt 511

    // 4.4. Clear KV cache and logitProcessor.tokenSequence
    yield musicLogitProcessor.tokenSequence;
    chat.resetChat(/*keepStats=*/true);
    cycles += 1;
  }
}

async function main() {
  const musicLogitProcessor = new MusicLogitProcessor();
  const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
  logitProcessorRegistry.set("music-medium-800k-q0f32", musicLogitProcessor);
  const chat = new webllm.ChatModule(logitProcessorRegistry);

  // Define modelRecord
  const myAppConfig: webllm.AppConfig = {
    model_list: [
      {
        "model_url": "https://huggingface.co/mlc-ai/mlc-chat-stanford-crfm-music-medium-800k-q0f32/resolve/main/",
        "local_id": "music-medium-800k-q0f32",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/music-medium-800k-q0f32.wasm",
      },
    ]
  }

  // Reload chat module with a logit processor
  await chat.reload("music-medium-800k-q0f32", undefined, myAppConfig);

  for await (const nextChunk of chunkGenerator(chat, musicLogitProcessor)) {
    console.log(nextChunk);
  };

  console.log(await chat.runtimeStatsText());
}

main();
