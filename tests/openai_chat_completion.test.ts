import { postInitAndCheckFields, ChatCompletionRequest } from "../src/openai_api_protocols/chat_completion"
import { describe, expect, test } from '@jest/globals';

describe('Check chat completion unsupported requests', () => {
    test('High-level unsupported fields', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                model: "Phi2-q4f32_1",  // this raises error
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "user", content: "Hello! " },
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("The following fields in ChatCompletionRequest are not yet supported");
    });

    test('Last message should be from user', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "user", content: "Hello! " },
                    { role: "assistant", content: "Hello! How may I help you today?" },
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("Last message should be from `user`.");
    });

    test('System prompt should always be the first one in `messages`', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                messages: [
                    { role: "user", content: "Hello! " },
                    { role: "assistant", content: "Hello! How may I help you today?" },
                    { role: "user", content: "Tell me about Pittsburgh" },
                    { role: "system", content: "You are a helpful assistant." },
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("System prompt should always be the first one in `messages`.");
    });

    test('When streaming `n` needs to be 1', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                stream: true,
                n: 2,
                messages: [
                    { role: "user", content: "Hello! " },
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("When streaming, `n` cannot be > 1.");
    });

    test('When stateful `n` needs to be 1', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                stateful: true,
                n: 2,
                messages: [
                    { role: "user", content: "Hello! " },
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("If the request is stateful, `n` cannot be > 1.");
    });

    test('Non-integer seed', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                messages: [
                    { role: "user", content: "Hello! " },
                ],
                max_gen_len: 10,
                seed: 42.2,  // Note that Number.isInteger(42.0) is true 
            };
            postInitAndCheckFields(request)
        }).toThrow("`seed` should be an integer, but got");
    });

    // Remove when we support image input (e.g. LlaVA model)
    test('Image input is unsupported', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: "What is in this image?" },
                            {
                                type: "image_url",
                                image_url: { url: "https://url_here.jpg" },
                            },
                        ],
                    },
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("User message only supports string `content` for now");
    });

    // Remove two tests below after we support function calling
    test('tool_calls is unsupported', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "assistant", tool_calls: [] }, // This raises error
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("`tool_calls` is not supported yet.");
    });

    test('tool is unsupported', () => {
        expect(() => {
            const request: ChatCompletionRequest = {
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "tool", content: "dummy content", tool_call_id: "dummy id" }, // This raises error
                ],
            };
            postInitAndCheckFields(request)
        }).toThrow("`tool` and `function` are not supported yet.");
    });
});

describe('Supported requests', () => {
    test('Supproted chat completion request', () => {
        const request: ChatCompletionRequest = {
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: "Hello! " },
                { role: "assistant", content: "How can I help you? " },
                { role: "user", content: "Give me 5 US states. " },
            ],
            n: 3,
            temperature: 1.5,
            max_gen_len: 25,
            frequency_penalty: 0.2,
            seed: 42,
            logprobs: true,
            top_logprobs: 2,
            logit_bias: {
                "13813": -100,
                "10319": 5,
                "7660": 5,
            },
        };
        postInitAndCheckFields(request)
    });
})
