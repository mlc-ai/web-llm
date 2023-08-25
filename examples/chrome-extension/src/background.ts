import {ChatRestModule} from "@mlc-ai/web-llm";


// Set reponse callback for chat module
const generateProgressCallback = (_step: number, message: string) => {
    // send the answer back to the content script
    chrome.runtime.sendMessage({ answer: message });
};

const cm = new ChatRestModule();

// listen for a request message from the content script
chrome.runtime.onMessage.addListener(async function (request) {
    // check if the request contains a message that the user sent a new message
    if (request.input) {
        const response = await cm.generate(request.input, generateProgressCallback);
    }
});