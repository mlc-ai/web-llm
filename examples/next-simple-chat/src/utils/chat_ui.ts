import { type ChatInterface } from "@mlc-ai/web-llm";

export default class ChatUI {
    private chat: ChatInterface;
    private chatLoaded = false;
    private requestInProgress = false;
    // We use a request chain to ensure that
    // all requests send to chat are sequentialized
    private chatRequestChain: Promise<void> = Promise.resolve();

    constructor(chat: ChatInterface) {
        this.chat = chat;
    }
    /**
     * Push a task to the execution queue.
     *
     * @param task The task to be executed;
     */
    private pushTask(task: () => Promise<void>) {
        const lastEvent = this.chatRequestChain;
        this.chatRequestChain = lastEvent.then(task);
    }
    // Event handlers
    // all event handler pushes the tasks to a queue
    // that get executed sequentially
    // the tasks previous tasks, which causes them to early stop
    // can be interrupted by chat.interruptGenerate
    async onGenerate(prompt: string, messageUpdate: (kind: string, text: string, append: boolean) => void, setRuntimeStats: (runtimeStats: string) => void) {
        if (this.requestInProgress) {
            return;
        }
        this.pushTask(async () => {
            await this.asyncGenerate(prompt, messageUpdate, setRuntimeStats);
        });
        return this.chatRequestChain
    }

    async onReset(clearMessages: () => void) {
        if (this.requestInProgress) {
            // interrupt previous generation if any
            this.chat.interruptGenerate();
        }
        // try reset after previous requests finishes
        this.pushTask(async () => {
            await this.chat.resetChat();
            clearMessages();
        });
        return this.chatRequestChain
    }

    async asyncInitChat(messageUpdate: (kind: string, text: string, append: boolean) => void) {
        if (this.chatLoaded) return;
        this.requestInProgress = true;
        messageUpdate("init", "", true);
        const initProgressCallback = (report: { text: string }) => {
            messageUpdate("init", report.text, false);
        }
        this.chat.setInitProgressCallback(initProgressCallback);

        try {
            await this.chat.reload("Llama-2-7b-chat-hf-q4f32_1", undefined, {
                "model_list": [
                    {
                        "model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f32_1-MLC/resolve/main/",
                        "local_id": "Llama-2-7b-chat-hf-q4f32_1",
                        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-webgpu.wasm",
                    },
                ]
            });
        } catch (err: unknown) {
            messageUpdate("error", "Init error, " + (err?.toString() ?? ""), true);
            console.log(err);
            await this.unloadChat();
            this.requestInProgress = false;
            return;
        }
        this.requestInProgress = false;
        this.chatLoaded = true;
    }

    private async unloadChat() {
        await this.chat.unload();
        this.chatLoaded = false;
    }

    /**
     * Run generate
     */
    private async asyncGenerate(prompt: string, messageUpdate: (kind: string, text: string, append: boolean) => void, setRuntimeStats: (runtimeStats: string) => void) {
        await this.asyncInitChat(messageUpdate);
        this.requestInProgress = true;
        // const prompt = this.uiChatInput.value;
        if (prompt == "") {
            this.requestInProgress = false;
            return;
        }

        messageUpdate("right", prompt, true);
        // this.uiChatInput.value = "";
        // this.uiChatInput.setAttribute("placeholder", "Generating...");

        messageUpdate("left", "", true);
        const callbackUpdateResponse = (step: number, msg: string) => {
            messageUpdate("left", msg, false);
        };

        try {
            const output = await this.chat.generate(prompt, callbackUpdateResponse);
            messageUpdate("left", output, false);
            this.chat.runtimeStatsText().then(stats => setRuntimeStats(stats)).catch(error => console.log(error));
        } catch (err: unknown) {
            messageUpdate("error", "Generate error, " + (err?.toString() ?? ""), true);
            console.log(err);
            await this.unloadChat();
        }
        this.requestInProgress = false;
    }
}