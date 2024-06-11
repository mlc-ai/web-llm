import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );

  const availableFunctions = {
    // mock get weather function
    getCurrentWeather: function (location, unit) {
      return {
        location,
        unit,
        weather: {
          wind: "10 mph",
          direction: "s",
          temperature: "50",
        },
      };
    },
  };

  // store messages history
  const messages: webllm.ChatCompletionMessage[] = [
    {
      role: "user",
      content:
        "What is the current weather in celsius in Pittsburgh and Tokyo?",
    },
  ];

  const tools: Array<webllm.ChatCompletionTool> = [
    {
      type: "function",
      function: {
        name: "getCurrentWeather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "location",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: ["location"],
        },
      },
    },
  ];

  const request: webllm.ChatCompletionRequest = {
    stream: false, // works with stream as well, where the last chunk returns tool_calls
    // stream_options: { include_usage: true },
    messages,
    tool_choice: "auto",
    tools: tools,
  };

  const reply0 = await engine.chat.completions.create(request);
  console.log(reply0.choices[0].message);
  // Errors here - system doesn't like that content is null, skipping for now...
  // messages.push(reply0.choices[0].message);
  console.log(reply0.usage);
  console.log("tool_calls:", reply0.choices[0].message.tool_calls);

  const toolCalls = reply0.choices[0].message.tool_calls;

  if (toolCalls) {
    toolCalls.forEach((tc) => {
      const args = JSON.parse(tc.function.arguments);
      const result = availableFunctions[tc.function.name].apply(
        this,
        Object.values(args),
      );
      messages.push({
        tool_call_id: tc.id,
        role: "tool",
        name: tc.function.name,
        content: JSON.stringify(result),
      });
    });
    const reply1 = await engine.chat.completions.create({
      messages,
    });

    console.log("second reply", reply1);
  }
  // const request2: webllm.ChatCompletionRequest = {
  //   stream: false, // works with stream as well, where the last chunk returns tool_calls
  //   // stream_options: { include_usage: true },
  //   messages: [
  //     ...messages,
  //     {
  //       role: "user",
  //       content: "What is the current weather in farenheit in Denver?",
  //     },
  //   ],
  //   tool_choice: "auto",
  //   tools: tools,
  // };

  // const reply1 = await engine.chat.completions.create(request2);
  // console.log(reply1.choices[0]);
  // console.log(reply1.usage);
  // console.log('')
  // // messages.push(reply1.choices[0].message);
}

main();
