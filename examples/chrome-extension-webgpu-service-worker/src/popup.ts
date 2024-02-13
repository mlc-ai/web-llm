'use strict';

// This code is partially adapted from the openai-chatgpt-chrome-extension repo:
// https://github.com/jessedi0n/openai-chatgpt-chrome-extension

import './popup.css';

import { ChatModule, AppConfig, InitProgressReport } from "@mlc-ai/web-llm";
import { ProgressBar, Line } from "progressbar.js";

// Whether or not to use the content from the active tab as the context
const useContext = false;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const queryInput = document.getElementById("query-input")!;
const submitButton = document.getElementById("submit-button")!;


var isLoadingParams = false;

const generateProgressCallback = (_step: number, message: string) => {
    updateAnswer(message);
};


(<HTMLButtonElement>submitButton).disabled = true;

var progressBar: ProgressBar = new Line('#loadingContainer', {
    strokeWidth: 4,
    easing: 'easeInOut',
    duration: 1400,
    color: '#ffd166',
    trailColor: '#eee',
    trailWidth: 1,
    svgStyle: { width: '100%', height: '100%' }
});

const appConfig: AppConfig = {
    model_list: [
        {
            "model_url": "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/resolve/main/",
            "local_id": "Mistral-7B-Instruct-v0.2-q4f16_1",
            "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-webgpu.wasm",
            "required_features": ["shader-f16"],
        }
    ]
}

chrome.runtime.sendMessage({ reload: appConfig });
chrome.runtime.onMessage.addListener(function (request) {
    if (request.initProgressReport) {
        progressBar.animate(request.initProgressReport, {
            duration: 50
        });
        if (request.initProgressReport == 1.0) {
            enableInputs();
        }
    }
});

isLoadingParams = true;

function enableInputs() {
    if (isLoadingParams) {
        sleep(500);
        (<HTMLButtonElement>submitButton).disabled = false;
        const loadingBarContainer = document.getElementById("loadingContainer")!;
        loadingBarContainer.remove();
        queryInput.focus();
        isLoadingParams = false;
    }
}

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
    chrome.runtime.sendMessage({ input: message });

    // Clear the answer
    document.getElementById("answer")!.innerHTML = "";
    // Hide the answer
    document.getElementById("answerWrapper")!.style.display = "none";
    // Show the loading indicator
    document.getElementById("loading-indicator")!.style.display = "block";
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

function fetchPageContents() {
    chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
        var port = chrome.tabs.connect(tabs[0].id, { name: "channelName" });
        port.postMessage({});
        port.onMessage.addListener(function (msg) {
            console.log("Page contents:", msg.contents);
            chrome.runtime.sendMessage({ context: msg.contents });
        });
    });
}

// Grab the page contents when the popup is opened
window.onload = function () {
    if (useContext) {
        fetchPageContents();
    }
}
