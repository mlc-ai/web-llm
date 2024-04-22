import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
    const label = document.getElementById(id);
    if (label == null) {
        throw Error("Cannot find label " + id);
    }
    label.innerText = text;
}

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
import { Type, type Static } from '@sinclair/typebox'
const T = Type.Object({
    size: Type.Integer(),
    is_accepted: Type.Boolean(),
    num: Type.Number(),
})
type T = Static<typeof T>;
const schema2 = JSON.stringify(T);
console.log(schema2);
// {"type":"object","properties":{"size":{"type":"integer"},"is_accepted":{"type":"boolean"},
// "num":{"type":"number"}},"required":["size","is_accepted","num"]}

async function main() {
    const initProgressCallback = (report: webllm.InitProgressReport) => {
        setLabel("init-label", report.text);
    };
    const engine: webllm.EngineInterface = await webllm.CreateEngine(
        "Llama-2-7b-chat-hf-q4f16_1", { initProgressCallback: initProgressCallback }
    );

    const request: webllm.ChatCompletionRequest = {
        stream: false,  // works with streaming, logprobs, top_logprobs as well
        messages: [
            {
                "role": "user",
                "content": "Generate a json containing three fields: an integer field named size, a " +
                    "boolean field named is_accepted, and a float field named num."
            }
        ],
        max_gen_len: 128,
        response_format: { type: "json_object", schema: schema2 } as webllm.ResponseFormat
    };

    const reply0 = await engine.chatCompletion(request);
    console.log(reply0);
    console.log("Output:\n" + await engine.getMessage());
    console.log(await engine.runtimeStatsText());
}

main();
