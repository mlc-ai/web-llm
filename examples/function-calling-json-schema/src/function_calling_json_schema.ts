import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

import { Type, type Static } from '@sinclair/typebox'
const T = Type.Object({
  tool_calls: Type.Array(
    Type.Object({
      arguments: Type.Any(),
      name: Type.String(),
    })
  )
});
type T = Static<typeof T>;
const schema = JSON.stringify(T);
console.log(schema);

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


async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Hermes-2-Pro-Mistral-7B-q4f16_1"
  const engine: webllm.EngineInterface = await webllm.CreateEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback
    }
  );

  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      { "role": "system", "content": `You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> ${JSON.stringify(tools)} </tools>. Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10.
      Calling multiple functions at once can overload the system and increase cost so call one function at a time please.
      If you plan to continue with analysis, always call another function.
      Return a valid json object (using double quotes) in the following schema: ${JSON.stringify(schema)}.`},
      { "role": "user", "content": "What is the current weather in celsius in Pittsburgh and Tokyo?" },
    ],
    response_format: { type: "json_object", schema: schema } as webllm.ResponseFormat,
  };

  const reply0 = await engine.chat.completions.create(request);
  console.log(reply0.choices[0].message.content);

  console.log(await engine.runtimeStatsText());
}

main();
