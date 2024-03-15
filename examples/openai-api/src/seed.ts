import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
    const label = document.getElementById(id);
    if (label == null) {
        throw Error("Cannot find label " + id);
    }
    label.innerText = text;
}

/**
 * We domnstrate the effect of seeding. The prompt is about writing a poem and we use a high
 * `temperature`, making the sampling distribution supposedly more random. However, we demonstrate
 * that with seeding, we should see the exact same result being generated across two trials.
 * With `n > 1`, all choices should also be exactly the same.
 */
async function demonstrateSeed() {
    const chat: webllm.ChatInterface = new webllm.ChatModule();

    chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
        setLabel("init-label", report.text);
    });

    await chat.reload("Llama-2-7b-chat-hf-q4f32_1");

    const request: webllm.ChatCompletionRequest = {
        stream: false,  // works with streaming as well
        messages: [
            { "role": "user", "content": "Write a creative Haiku about Pittsburgh" }
        ],
        n: 3,
        temperature: 1.2,  // high temperature gives much more random results
        seed: 42,
        max_gen_len: 128,  // To save time; enough to demonstrate the effect
    };

    const reply0 = await chat.chatCompletion(request);
    console.log(reply0);
    console.log("First reply's last choice:\n" + await chat.getMessage());

    const reply1 = await chat.chatCompletion(request);
    console.log(reply1);
    console.log("Second reply's last choice:\n" + await chat.getMessage());

    // Rigorously check the generation results of each choice for the two requests
    for (const choice0 of reply0.choices) {
        const id = choice0.index;
        const choice1 = reply1.choices[id];
        if (choice0.message.content !== choice1.message.content) {
            throw Error("Chocie " + id + " of the two generations are different despite seeding");
        }
    }

    console.log(await chat.runtimeStatsText());
}

// Run one of the functions
demonstrateSeed();
