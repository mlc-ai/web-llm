"use strict";

// This code is partially adapted from the openai-chatgpt-chrome-extension repo:
// https://github.com/jessedi0n/openai-chatgpt-chrome-extension

import "./popup.css";

import {
  ChatCompletionMessageParam,
  CreateExtensionServiceWorkerMLCEngine,
  MLCEngineInterface,
  InitProgressReport,
} from "@mlc-ai/web-llm";
import { ProgressBar, Line } from "progressbar.js";

/***************** UI elements *****************/
// Whether or not to use the content from the active tab as the context
const useContext = true;
console.log('useContext value:', useContext);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const queryInput = document.getElementById("query-input")!;
const submitButton = document.getElementById("submit-button")!;

let isLoadingParams = false;
let pageContext = ""; // Store the page context

(<HTMLButtonElement>submitButton).disabled = true;

const progressBar: ProgressBar = new Line("#loadingContainer", {
  strokeWidth: 4,
  easing: "easeInOut",
  duration: 1400,
  color: "#ffd166",
  trailColor: "#eee",
  trailWidth: 1,
  svgStyle: { width: "100%", height: "100%" },
});

/***************** Web-LLM MLCEngine Configuration *****************/
const initProgressCallback = (report: InitProgressReport) => {
  progressBar.animate(report.progress, {
    duration: 50,
  });
  if (report.progress == 1.0) {
    enableInputs();
  }
};

const engine: MLCEngineInterface = await CreateExtensionServiceWorkerMLCEngine(
  "Qwen2-0.5B-Instruct-q4f16_1-MLC",
  { initProgressCallback: initProgressCallback },
);
const chatHistory: ChatCompletionMessageParam[] = [];

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

/***************** Event Listeners *****************/

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
  chatHistory.push({ role: "user", content: message });
  console.log(chatHistory)
  // Clear the answer
  document.getElementById("answer")!.innerHTML = "";
  // Hide the answer
  document.getElementById("answerWrapper")!.style.display = "none";
  // Show the loading indicator
  document.getElementById("loading-indicator")!.style.display = "block";

  // Send the chat completion message to the engine
  let curMessage = "";
  const completion = await engine.chat.completions.create({
    stream: true,
    messages: chatHistory,
  });

  // Update the answer as the model generates more text
  for await (const chunk of completion) {
    const curDelta = chunk.choices[0].delta.content;
    if (curDelta) {
      curMessage += curDelta;
    }
    updateAnswer(curMessage);
  }
  chatHistory.push({ role: "assistant", content: await engine.getMessage() });
}

submitButton.addEventListener("click", handleClick);

function updateAnswer(answer: string) {
  // Show answer
  document.getElementById("answerWrapper")!.style.display = "block";
  const answerWithBreaks = answer.replace(/\n/g, "<br>");
  document.getElementById("answer")!.innerHTML = answerWithBreaks;
  // Add event listener to copy button
  document.getElementById("copyAnswer")!.addEventListener("click", () => {
    // Get the answer text
    const answerText = answer;
    // Copy the answer text to the clipboard
    navigator.clipboard
      .writeText(answerText)
      .then(() => console.log("Answer text copied to clipboard"))
      .catch((err) => console.error("Could not copy text: ", err));
  });
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  };
  const time = new Date().toLocaleString("en-US", options);
  // Update timestamp
  document.getElementById("timestamp")!.innerText = time;
  // Hide loading indicator
  document.getElementById("loading-indicator")!.style.display = "none";
}

function fetchPageContents() {
  // Query all tabs in the current window instead of just the active one
  chrome.tabs.query({ currentWindow: true }, function (tabs) {
    const allTabContents: { title: string; url: string; content: string }[] = [];
    let completedTabs = 0;
    
    if (tabs.length === 0) return;
    
    tabs.forEach((tab) => {
      if (tab.id) {
        try {
          const port = chrome.tabs.connect(tab.id, { name: "channelName" });
          port.postMessage({});
          port.onMessage.addListener(function (msg) {
            // Store each tab's content with metadata
            allTabContents.push({
              title: tab.title || "Untitled",
              url: tab.url || "Unknown URL",
              content: msg.contents
            });
            
            completedTabs++;
            
            // When all tabs have been processed
            if (completedTabs === tabs.length) {
              // Combine all tab contents into a single context
              pageContext = allTabContents
                .map((tabInfo, index) =>
                  `=== Tab ${index + 1}: ${tabInfo.title} ===\nURL: ${tabInfo.url}\n\n${tabInfo.content}\n\n`
                )
                .join("\n");
              
              // Add page context to chat history as a system message
              if (pageContext && chatHistory.length === 0) {
                chatHistory.push({
                  role: "system",
                  content: `You are a helpful assistant. Here is the content of all ${tabs.length} tabs currently open in the browser:\n\n${pageContext}\n\nPlease answer questions about these webpages based on the content provided above.`
                });
                console.log("content",pageContext);
              }
            }
          });
          
          // Handle connection errors (e.g., for chrome:// pages or pages without content script)
          port.onDisconnect.addListener(() => {
            completedTabs++;
            if (completedTabs === tabs.length && chatHistory.length === 0 && pageContext) {
              chatHistory.push({
                role: "system",
                content: `You are a helpful assistant. Here is the content of all ${allTabContents.length} accessible tabs currently open in the browser:\n\n${pageContext}\n\nPlease answer questions about these webpages based on the content provided above.`
              });
              // console.log("content",pageContext);
            }
          });
        } catch (error) {
          console.error(`Failed to connect to tab ${tab.id}:`, error);
          completedTabs++;
        }
      } else {
        completedTabs++;
      }
    });
  });
}

// Grab the page contents when the popup is opened
// Use DOMContentLoaded instead of window.onload to ensure it fires
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    if (useContext) {
      fetchPageContents();
    }
  });
} else {
  // Document already loaded
  if (useContext) {
    fetchPageContents();
  }
}
