import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

/**
 * We demonstrate multiround chatting. Though users are required to maintain chat history, internally
 * we compare provided `messages` with the internal chat history. If it matches, we will reuse KVs
 * and hence save computation -- essentially an implicit internal optimization.
 */
async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );

  // Round 0
  const messages: webllm.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "You are a helpful, respectful and honest assistant. " +
        "Be as happy as you can when speaking please. ",
    },
    { role: "user", content: "Provide me three US states." },
  ];

  const request0: webllm.ChatCompletionRequest = {
    stream: false, // can be streaming, same behavior
    messages: messages,
  };

  const reply0 = await engine.chat.completions.create(request0);
  const replyMessage0 = await engine.getMessage();
  console.log(reply0);
  console.log(replyMessage0);
  console.log(reply0.usage);

  // Round 1
  // Append generated response to messages
  messages.push({ role: "assistant", content: replyMessage0 });
  // Append new user input
  messages.push({ role: "user", content: "Two more please!" });
  // Below line would cause an internal reset (clear KV cache, etc.) since the history no longer
  // matches the new request
  // messages[0].content = "Another system prompt";

  const request1: webllm.ChatCompletionRequest = {
    stream: false, // can be streaming, same behavior
    messages: messages,
  };

  const reply1 = await engine.chat.completions.create(request1);
  const replyMessage1 = await engine.getMessage();
  console.log(reply1);
  console.log(replyMessage1);
  console.log(reply1.usage);

  // If we used multiround chat, request1 should only prefill a small number of tokens
  const prefillTokens0 = reply0.usage?.prompt_tokens;
  const prefillTokens1 = reply1.usage?.prompt_tokens;
  console.log("Requset 0 prompt tokens: ", prefillTokens0);
  console.log("Requset 1 prompt tokens: ", prefillTokens1);
  if (
    prefillTokens0 === undefined ||
    prefillTokens1 === undefined ||
    prefillTokens1 > prefillTokens0
  ) {
    throw Error("Multi-round chat is not triggered as expected.");
  }
}

main();
