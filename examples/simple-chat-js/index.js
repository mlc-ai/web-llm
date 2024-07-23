import * as webllm from "https://esm.run/@mlc-ai/web-llm";

/*************** WebLLM logic ***************/
const messages = [
  {
    content: "You are a helpful AI agent helping users.",
    role: "system",
  },
];

const availableModels = webllm.prebuiltAppConfig.model_list.map(
  (m) => m.model_id,
);
let selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-1k";

// Callback function for initializing progress
function updateEngineInitProgressCallback(report) {
  console.log("initialize", report.progress);
  document.getElementById("download-status").textContent = report.text;
}

// Create engine instance
const engine = new webllm.MLCEngine();
engine.setInitProgressCallback(updateEngineInitProgressCallback);

async function initializeWebLLMEngine() {
  document.getElementById("download-status").classList.remove("hidden");
  selectedModel = document.getElementById("model-selection").value;
  const config = {
    temperature: 1.0,
    top_p: 1,
  };
  await engine.reload(selectedModel, config);
}

async function streamingGenerating(messages, onUpdate, onFinish, onError) {
  try {
    let curMessage = "";
    let usage;
    const completion = await engine.chat.completions.create({
      stream: true,
      messages,
      stream_options: { include_usage: true },
    });
    for await (const chunk of completion) {
      const curDelta = chunk.choices[0]?.delta.content;
      if (curDelta) {
        curMessage += curDelta;
      }
      if (chunk.usage) {
        usage = chunk.usage;
      }
      onUpdate(curMessage);
    }
    const finalMessage = await engine.getMessage();
    onFinish(finalMessage, usage);
  } catch (err) {
    onError(err);
  }
}

/*************** UI logic ***************/
function onMessageSend() {
  const input = document.getElementById("user-input").value.trim();
  const message = {
    content: input,
    role: "user",
  };
  if (input.length === 0) {
    return;
  }
  document.getElementById("send").disabled = true;

  messages.push(message);
  appendMessage(message);

  document.getElementById("user-input").value = "";
  document
    .getElementById("user-input")
    .setAttribute("placeholder", "Generating...");

  const aiMessage = {
    content: "typing...",
    role: "assistant",
  };
  appendMessage(aiMessage);

  const onFinishGenerating = (finalMessage, usage) => {
    updateLastMessage(finalMessage);
    document.getElementById("send").disabled = false;
    const usageText =
      `prompt_tokens: ${usage.prompt_tokens}, ` +
      `completion_tokens: ${usage.completion_tokens}, ` +
      `prefill: ${usage.extra.prefill_tokens_per_s.toFixed(4)} tokens/sec, ` +
      `decoding: ${usage.extra.decode_tokens_per_s.toFixed(4)} tokens/sec`;
    document.getElementById("chat-stats").classList.remove("hidden");
    document.getElementById("chat-stats").textContent = usageText;
  };

  streamingGenerating(
    messages,
    updateLastMessage,
    onFinishGenerating,
    console.error,
  );
}

function appendMessage(message) {
  const chatBox = document.getElementById("chat-box");
  const container = document.createElement("div");
  container.classList.add("message-container");
  const newMessage = document.createElement("div");
  newMessage.classList.add("message");
  newMessage.textContent = message.content;

  if (message.role === "user") {
    container.classList.add("user");
  } else {
    container.classList.add("assistant");
  }

  container.appendChild(newMessage);
  chatBox.appendChild(container);
  chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the latest message
}

function updateLastMessage(content) {
  const messageDoms = document
    .getElementById("chat-box")
    .querySelectorAll(".message");
  const lastMessageDom = messageDoms[messageDoms.length - 1];
  lastMessageDom.textContent = content;
}

/*************** UI binding ***************/
availableModels.forEach((modelId) => {
  const option = document.createElement("option");
  option.value = modelId;
  option.textContent = modelId;
  document.getElementById("model-selection").appendChild(option);
});
document.getElementById("model-selection").value = selectedModel;
document.getElementById("download").addEventListener("click", function () {
  initializeWebLLMEngine().then(() => {
    document.getElementById("send").disabled = false;
  });
});
document.getElementById("send").addEventListener("click", function () {
  onMessageSend();
});
