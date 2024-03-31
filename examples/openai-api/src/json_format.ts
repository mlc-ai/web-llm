import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
    const label = document.getElementById(id);
    if (label == null) {
        throw Error("Cannot find label " + id);
    }
    label.innerText = text;
}

async function demonstrateJSONFormat() {
    const chat: webllm.ChatInterface = new webllm.ChatModule();

    chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
        setLabel("init-label", report.text);
    });

    const myAppConfig: webllm.AppConfig = {
        model_list: [
            {
                "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC/resolve/main/",
                "local_id": "Llama-2-7b-chat-hf-q4f16_1",
                "model_lib_url": "https://raw.githubusercontent.com/CharlieFRuan/binary-mlc-llm-libs/temp-for-webllm/Llama-2-7b-chat-hf.wasm",
            },
        ]
    }
    await chat.reload("Llama-2-7b-chat-hf-q4f16_1", undefined, myAppConfig);

    const request: webllm.ChatCompletionRequest = {
        stream: false,  // works with streaming, logprobs, top_logprobs as well
        messages: [
            { "role": "user", "content": "Write a short JSON file introducign yourself." }
        ],
        n: 2,
        max_gen_len: 128,
        response_format: { type: "json_object" } as webllm.ResponseFormat
    };

    const reply0 = await chat.chatCompletion(request);
    console.log(reply0);
    console.log("First reply's last choice:\n" + await chat.getMessage());
    console.log(await chat.runtimeStatsText());
}

demonstrateJSONFormat();
