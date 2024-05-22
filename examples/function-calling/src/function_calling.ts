import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}


async function main() {

  const myAppConfig: webllm.AppConfig = {
    model_list: [
      {
        "model_url": "https://huggingface.co/mlc-ai/gorilla-openfunctions-v2-q4f16_1-MLC/resolve/main/",
        "model_id": "gorilla-openfunctions-v2-q4f16_1",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/gorilla-openfunctions-v2/gorilla-openfunctions-v2-q4f16_1.wasm",
      },
    ]
  }
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "gorilla-openfunctions-v2-q4f16_1"
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { appConfig: myAppConfig, initProgressCallback: initProgressCallback }
  );

  const tools: Array<webllm.ChatCompletionTool> = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
          },
          "required": ["location"],
        },
      },
    }
  ]

  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      { "role": "user", "content": "What is the current weather in celsius in Pittsburgh and Tokyo?" },
    ],
    tool_choice: 'auto',
    tools: tools,
  };

  const reply0 = await engine.chat.completions.create(request);
  console.log(reply0.choices[0].message.content);

  console.log(await engine.runtimeStatsText());
}

main();
