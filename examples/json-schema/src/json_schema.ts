import * as webllm from "@mlc-ai/web-llm";
import { Type, Static } from "@sinclair/typebox";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function simpleStructuredTextExample() {
  // There are several options of providing such a schema
  // 1. You can directly define a schema in string
  const schema1 = `{
        "properties": {
            "size": {"title": "Size", "type": "integer"}, 
            "is_accepted": {"title": "Is Accepted", "type": "boolean"}, 
            "num": {"title": "Num", "type": "number"}
        },
        "required": ["size", "is_accepted", "num"], 
        "title": "Schema", "type": "object"
    }`;

  // 2. You can use 3rdparty libraries like typebox to create a schema
  const T = Type.Object({
    size: Type.Integer(),
    is_accepted: Type.Boolean(),
    num: Type.Number(),
  });
  type T = Static<typeof T>;
  const schema2 = JSON.stringify(T);
  console.log(schema2);
  // {"type":"object","properties":{"size":{"type":"integer"},"is_accepted":{"type":"boolean"},
  // "num":{"type":"number"}},"required":["size","is_accepted","num"]}

  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  // Pick any one of these models to start trying -- most models in WebLLM support grammar
  // const selectedModel = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
  const selectedModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback, logLevel: "INFO" },
  );

  // Note that you'd need to prompt the model to answer in JSON either in
  // user's message or the system prompt
  const request: webllm.ChatCompletionRequest = {
    stream: false, // works with streaming, logprobs, top_logprobs as well
    messages: [
      {
        role: "user",
        content:
          "Generate a json containing three fields: an integer field named size, a " +
          "boolean field named is_accepted, and a float field named num.",
      },
    ],
    max_tokens: 128,
    response_format: {
      type: "json_object",
      schema: schema2,
    } as webllm.ResponseFormat,
  };

  const reply0 = await engine.chatCompletion(request);
  console.log(reply0);
  console.log("Output:\n" + (await engine.getMessage()));
  console.log(reply0.usage);
}

// The json schema and prompt is taken from
// https://github.com/sgl-project/sglang/tree/main?tab=readme-ov-file#json-decoding
async function harryPotterExample() {
  const T = Type.Object({
    name: Type.String(),
    house: Type.Enum({
      Gryffindor: "Gryffindor",
      Hufflepuff: "Hufflepuff",
      Ravenclaw: "Ravenclaw",
      Slytherin: "Slytherin",
    }),
    blood_status: Type.Enum({
      "Pure-blood": "Pure-blood",
      "Half-blood": "Half-blood",
      "Muggle-born": "Muggle-born",
    }),
    occupation: Type.Enum({
      Student: "Student",
      Professor: "Professor",
      "Ministry of Magic": "Ministry of Magic",
      Other: "Other",
    }),
    wand: Type.Object({
      wood: Type.String(),
      core: Type.String(),
      length: Type.Number(),
    }),
    alive: Type.Boolean(),
    patronus: Type.String(),
  });

  type T = Static<typeof T>;
  const schema = JSON.stringify(T);
  console.log(schema);

  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  // Pick any one of these models to start trying -- most models in WebLLM support grammar
  const selectedModel = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";

  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback, logLevel: "INFO" },
  );

  // Note that you'd need to prompt the model to answer in JSON either in
  // user's message or the system prompt
  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      {
        role: "user",
        content:
          "Hermione Granger is a character in Harry Potter. Please fill in the following information about this character in JSON format." +
          "Name is a string of character name. House is one of Gryffindor, Hufflepuff, Ravenclaw, Slytherin. Blood status is one of Pure-blood, Half-blood, Muggle-born. Occupation is one of Student, Professor, Ministry of Magic, Other. Wand is an object with wood, core, and length. Alive is a boolean. Patronus is a string.",
      },
    ],
    max_tokens: 128,
    response_format: {
      type: "json_object",
      schema: schema,
    } as webllm.ResponseFormat,
  };

  const reply = await engine.chatCompletion(request);
  console.log(reply);
  console.log("Output:\n" + (await engine.getMessage()));
  console.log(reply.usage);
  console.log(reply.usage!.extra);
}

async function functionCallingExample() {
  const T = Type.Object({
    tool_calls: Type.Array(
      Type.Object({
        arguments: Type.Any(),
        name: Type.String(),
      }),
    ),
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
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: ["location"],
        },
      },
    },
  ];

  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  const selectedModel = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    {
      initProgressCallback: initProgressCallback,
    },
  );

  const request: webllm.ChatCompletionRequest = {
    stream: false,
    messages: [
      {
        role: "system",
        content: `You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> ${JSON.stringify(
          tools,
        )} </tools>. Do not stop calling functions until the task has been accomplished or you've reached max iteration of 10.
      Calling multiple functions at once can overload the system and increase cost so call one function at a time please.
      If you plan to continue with analysis, always call another function.
      Return a valid json object (using double quotes) in the following schema: ${JSON.stringify(
        schema,
      )}.`,
      },
      {
        role: "user",
        content:
          "What is the current weather in celsius in Pittsburgh and Tokyo?",
      },
    ],
    response_format: {
      type: "json_object",
      schema: schema,
    } as webllm.ResponseFormat,
  };

  const reply = await engine.chat.completions.create(request);
  console.log(reply.choices[0].message.content);

  console.log(reply.usage);
}

async function ebnfGrammarExample() {
  // You can directly define an EBNFGrammar string with ResponseFormat.grammar
  const jsonGrammarStr = String.raw`
root ::= basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
`;

  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };

  // Pick any one of these models to start trying -- most models in WebLLM support grammar
  const selectedModel = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
  // const selectedModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback, logLevel: "INFO" },
  );

  // Note that you'd need to prompt the model to answer in JSON either in
  // user's message or the system prompt
  const request: webllm.ChatCompletionRequest = {
    stream: false, // works with streaming, logprobs, top_logprobs as well
    messages: [
      {
        role: "user",
        content: "Introduce yourself in JSON",
      },
    ],
    max_tokens: 128,
    response_format: {
      type: "grammar",
      grammar: jsonGrammarStr,
    } as webllm.ResponseFormat,
  };

  const reply0 = await engine.chatCompletion(request);
  console.log(reply0);
  console.log("Output:\n" + (await engine.getMessage()));
  console.log(reply0.usage);
}

async function main() {
  // await simpleStructuredTextExample();
  await harryPotterExample();
  // await functionCallingExample();
  // await ebnfGrammarExample();
}

main();
