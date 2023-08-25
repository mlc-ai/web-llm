'use strict';

// This code is partially adapted from the openai-chatgpt-chrome-extension repo:
// https://github.com/jessedi0n/openai-chatgpt-chrome-extension

import './popup.css';

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
    // Send the query to the background script
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