import { EngineInterface, CreateEngine, InitProgressReport, ChatCompletionMessageParam } from "@mlc-ai/web-llm";

let model_loaded = false;

let engine: EngineInterface;
const chatHistory: ChatCompletionMessageParam[] = [];

let context = "";
chrome.runtime.onMessage.addListener(async function (request) {
    // check if the request contains a message that the user sent a new message
    if (request.input) {
        let inp = request.input;
        if (context.length > 0) {
            inp = "Use only the following context when answering the question at the end. Don't use any other knowledge.\n" + context + "\n\nQuestion: " + request.input + "\n\nHelpful Answer: ";
        }
        console.log("Input:", inp);
        chatHistory.push({ "role": "user", "content": inp });

        let curMessage = "";
        const completion = await engine.chat.completions.create({ stream: true, messages: chatHistory });
        for await (const chunk of completion) {
            const curDelta = chunk.choices[0].delta.content;
            if (curDelta) {
                curMessage += curDelta;
            }
            chrome.runtime.sendMessage({ answer: curMessage });
        }
        chatHistory.push({ "role": "assistant", "content": await engine.getMessage() });
    }
    if (request.context) {
        context = request.context;
        console.log("Got context:", context);
    }
    if (request.reload) {
        if (!model_loaded) {
            const appConfig = request.reload;
            console.log("Got appConfig: ", appConfig);
            const initProgressCallback = (report: InitProgressReport) => {
                console.log(report.text, report.progress);
                chrome.runtime.sendMessage({ initProgressReport: report.progress });
            }
            // const selectedModel = "TinyLlama-1.1B-Chat-v0.4-q4f16_1-1k";
            const selectedModel = "Mistral-7B-Instruct-v0.2-q4f16_1";
            engine = await CreateEngine(
                selectedModel,
                { appConfig: appConfig, initProgressCallback: initProgressCallback }
            );
            console.log("Model loaded");
            model_loaded = true;
        } else {
            chrome.runtime.sendMessage({ initProgressReport: 1.0 });
        }
    }
});
