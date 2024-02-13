import {ChatRestModule, ChatInterface, ChatModule, InitProgressReport} from "@mlc-ai/web-llm";

// TODO: Surface this as an option to the user 
const useWebGPU = true;
var model_loaded = false;

var cm: ChatInterface;
if (!useWebGPU) {
    cm = new ChatRestModule();
} else {
    cm = new ChatModule();
}

// Set reponse callback for chat module
const generateProgressCallback = (_step: number, message: string) => {
    // send the answer back to the content script
    chrome.runtime.sendMessage({ answer: message });
};

var context = "";
chrome.runtime.onMessage.addListener(async function (request) {
    // check if the request contains a message that the user sent a new message
    if (request.input) {
        var inp = request.input;
        if (context.length > 0) {
            inp = "Use only the following context when answering the question at the end. Don't use any other knowledge.\n"+ context + "\n\nQuestion: " + request.input + "\n\nHelpful Answer: ";
        }
        console.log("Input:", inp);
        const response = await cm.generate(inp, generateProgressCallback);
    }
    if (request.context) {
        context = request.context;
        console.log("Got context:", context);
    }
    if (request.reload) {
        if (!model_loaded) {
            var appConfig = request.reload;
            console.log("Got appConfig: ", appConfig);
            
            cm.setInitProgressCallback((report: InitProgressReport) => {
                console.log(report.text, report.progress);
                chrome.runtime.sendMessage({ initProgressReport: report.progress});
            });
        
            await cm.reload("Mistral-7B-Instruct-v0.2-q4f16_1", undefined, appConfig);
            console.log("Model loaded");
            model_loaded = true;
        } else {
            chrome.runtime.sendMessage({ initProgressReport: 1.0});
        }
    }
});
