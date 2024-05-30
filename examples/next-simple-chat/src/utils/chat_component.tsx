import { useState } from "react";
import { MLCEngine } from "@mlc-ai/web-llm";
import ChatUI from "~/utils/chat_ui";

const ChatComponent = () => {
  const [messages, setMessages] = useState<{ kind: string; text: string }[]>(
    [],
  );
  const [prompt, setPrompt] = useState("");
  const [runtimeStats, setRuntimeStats] = useState("");
  const [chat_ui] = useState(new ChatUI(new MLCEngine()));
  const updateMessage = (kind: string, text: string, append: boolean) => {
    if (kind == "init") {
      text = "[System Initalize] " + text;
    }
    const msgCopy = [...messages];
    if (msgCopy.length == 0 || append) {
      setMessages([...msgCopy, { kind, text }]);
    } else {
      msgCopy[msgCopy.length - 1] = { kind, text };
      setMessages([...msgCopy]);
    }
  };
  return (
    <div className="flex flex-col items-center">
      <button
        className="chatui-btn"
        onClick={() => {
          chat_ui.asyncInitChat(updateMessage).catch((error) => {
            console.log(error);
          });
        }}
      >
        Download Model
      </button>

      <div className="chatui">
        <div className="chatui-chat" id="chatui-chat">
          {messages.map((value, index) => (
            <div key={index} className={`msg ${value.kind}-msg`}>
              <div className="msg-bubble">
                <div className="msg-text">${value.text}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="chatui-inputarea">
          <input
            id="chatui-input"
            type="text"
            className="chatui-input"
            placeholder="Enter your message..."
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                chat_ui
                  .onGenerate(prompt, updateMessage, setRuntimeStats)
                  .catch((error) => console.log(error));
              }
            }}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
          />
          <button
            className="chatui-btn"
            onClick={() => {
              chat_ui
                .onGenerate(prompt, updateMessage, setRuntimeStats)
                .catch((error) => console.log(error));
            }}
          >
            Send
          </button>
        </div>
      </div>

      <div className="chatui-extra-control">
        <button
          className="chatui-btn"
          onClick={() => {
            chat_ui
              .onReset(() => {
                setMessages([]);
              })
              .catch((error) => console.log(error));
          }}
        >
          Reset Chat
        </button>
        <label id="chatui-info-label">{runtimeStats}</label>
      </div>
    </div>
  );
};

export default ChatComponent;
