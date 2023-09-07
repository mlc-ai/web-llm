'use strict';

// This code is partially adapted from the openai-chatgpt-chrome-extension repo:
// https://github.com/jessedi0n/openai-chatgpt-chrome-extension

import './popup.css';

import {ChatModule, AppConfig, InitProgressReport} from "@mlc-ai/web-llm";

// TODO: Surface this as an option to the user
const useWebGPU = false;

var cm: ChatModule;
const generateProgressCallback = (_step: number, message: string) => {
    updateAnswer(message);
};
if (useWebGPU) {
    cm = new ChatModule();

    const appConfig : AppConfig = {
        model_list: [
            {
                "model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f32_1/resolve/main/",
                "local_id": "Llama-2-7b-chat-hf-q4f32_1"
            }
        ],
    model_lib_map: {
        "Llama-2-7b-chat-hf-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-webgpu.wasm"
    }
    }

    cm.setInitProgressCallback((report: InitProgressReport) => {
        console.log(report.text);
    });

    await cm.reload("Llama-2-7b-chat-hf-q4f32_1", undefined, appConfig);
}


const queryInput = document.getElementById("query-input")!;
const submitButton = document.getElementById("submit-button")!;

queryInput.focus();

// Disable submit button if input field is empty
queryInput.addEventListener("keyup", () => {
    if ((<HTMLInputElement>queryInput).value === "") {
        (<HTMLButtonElement>submitButton).disabled = true;
    } else {
        (<HTMLButtonElement>submitButton).disabled = false;
    }
});

// If user presses enter, click submit button
queryInput.addEventListener("keyup", (event) => {
    if (event.code === "Enter") {
        event.preventDefault();
        submitButton.click();
    }
});

// Listen for clicks on submit button
async function handleClick() {
    // Get the message from the input field
    const message = (<HTMLInputElement>queryInput).value;
    console.log("message", message);
    if (!useWebGPU) {
        // Send the query to the background script
        chrome.runtime.sendMessage({ input: message });
    }
    // Clear the answer
    document.getElementById("answer")!.innerHTML = "";
    // Hide the answer
    document.getElementById("answerWrapper")!.style.display = "none";
    // Show the loading indicator
    document.getElementById("loading-indicator")!.style.display = "block";

    if (useWebGPU) {
        // Generate response
        const response = await cm.generate(message, generateProgressCallback);
        console.log("response", response);
    }
}
submitButton.addEventListener("click", handleClick);

// Listen for messages from the background script
chrome.runtime.onMessage.addListener(({ answer, error }) => {
    if (answer) {
        updateAnswer(answer);
    }
});

function updateAnswer(answer: string) {
    // Show answer
    document.getElementById("answerWrapper")!.style.display = "block";
    const answerWithBreaks = answer.replace(/\n/g, '<br>');
    document.getElementById("answer")!.innerHTML = answerWithBreaks;
    // Add event listener to copy button
    document.getElementById("copyAnswer")!.addEventListener("click", () => {
      // Get the answer text
      const answerText = answer;
      // Copy the answer text to the clipboard
      navigator.clipboard.writeText(answerText)
        .then(() => console.log("Answer text copied to clipboard"))
        .catch((err) => console.error("Could not copy text: ", err));
    });
    const options: Intl.DateTimeFormatOptions = { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' };
    const time = new Date().toLocaleString('en-US', options);
    // Update timestamp
    document.getElementById("timestamp")!.innerText = time;
    // Hide loading indicator
    document.getElementById("loading-indicator")!.style.display = "none";
}


// Grab the page contents when the popup is opened
window.onload = function() {
    chrome.tabs.query({currentWindow: true,active: true}, function(tabs){
        var port = chrome.tabs.connect(tabs[0].id,{name: "channelName"});
        port.postMessage({});
        port.onMessage.addListener(function(msg) {
            console.log("Page contents:", msg.contents);
            chrome.runtime.sendMessage({ context: msg.contents });
        });
    });
}
