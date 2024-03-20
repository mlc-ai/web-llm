import { Role } from '../src/config'
import { getConversation } from '../src/conversation'
import { ChatModule } from '../src/chat_module'
import { ChatCompletionRequest } from "../src/openai_api_protocols/chat_completion"


import { describe, expect, test } from '@jest/globals';

describe('Test conversation template', () => {
    test('Test getPromptArrayInternal', () => {
        const conv = getConversation("gorilla");
        conv.appendMessage(Role.user, "Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes", "Tom");
        const prompt_array = conv.getPromptArray();
        
        expect(prompt_array).toEqual([
            "A chat between a curious user and an artificial intelligence assistant. " +
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n",
            "Tom: <<question>> Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes <<function>> \n"]);
    });

    test('Test getPromptArrayInternal function call', () => {
        const conv = getConversation("gorilla");
        conv.appendMessage(Role.user, "Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes");
        conv.use_function_calling = true;
        conv.function_string = JSON.stringify([{
            "name": "Uber Carpool",
            "api_name": "uber.ride",
            "description": "Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters",
            "parameters":  [
                {"name": "loc", "description": "Location of the starting place of the Uber ride"},
                {"name": "type", "enum": ["plus", "comfort", "black"], "description": "Types of Uber ride user is ordering"},
                {"name": "time", "description": "The amount of time in minutes the customer is willing to wait"}
            ]
        }]);
        const prompt_array = conv.getPromptArray();
        
        expect(prompt_array).toEqual([
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n",
            "USER: <<question>> Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes <<function>> [{\"name\":\"Uber Carpool\",\"api_name\":\"uber.ride\",\"description\":\"Find suitable ride for customers given the location, type of ride, and the amount of time the customer is willing to wait as parameters\",\"parameters\":[{\"name\":\"loc\",\"description\":\"Location of the starting place of the Uber ride\"},{\"name\":\"type\",\"enum\":[\"plus\",\"comfort\",\"black\"],\"description\":\"Types of Uber ride user is ordering\"},{\"name\":\"time\",\"description\":\"The amount of time in minutes the customer is willing to wait\"}]}]\n"
        ]);
    })
});

describe('Test ChatModule', () => {
    test('Test checkFunctionCallUsage none', () => {
        const chat_module = new ChatModule();

        const request: ChatCompletionRequest = {
            model: "gorilla-openfunctions-v1-q4f16_1_MLC",
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: "Hello!" },
            ],
            tool_choice: 'none',
            tools: [
              { type: 'function', function: { description: 'A', name: 'fn_A', parameters: { foo: 'bar' } } },
              { type: 'function', function: { description: 'B', name: 'fn_B', parameters: { foo: 'bar' } } },
              { type: 'function', function: { description: 'C', name: 'fn_C', parameters: { foo: 'bar' } } },
            ],
        };

        jest.spyOn(chat_module as any, "getPipeline")
            .mockImplementation(() => {
                return {
                    overrideFunctionCalling: jest.fn((use_fn_calling, fn_str) => {
                        expect(use_fn_calling).toBe(false);
                        expect(fn_str).toEqual("");
                    })
                };
            });
        (chat_module as any).checkFunctionCallUsage(request);
    });

    test('Test checkFunctionCallUsage auto', () => {
        const chat_module = new ChatModule();

        const request: ChatCompletionRequest = {
            model: "gorilla-openfunctions-v1-q4f16_1_MLC",
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: "Hello!" },
            ],
            tool_choice: 'auto',
            tools: [
                { type: 'function', function: { description: 'A', name: 'fn_A', parameters: { foo: 'bar' } } },
                { type: 'function', function: { description: 'B', name: 'fn_B', parameters: { foo: 'bar' } } },
                { type: 'function', function: { description: 'C', name: 'fn_C', parameters: { foo: 'bar' } } },
            ],
        };

        jest.spyOn(chat_module as any, "getPipeline")
            .mockImplementation(() => {
                return {
                    overrideFunctionCalling: jest.fn((use_fn_calling, fn_str) => {
                        expect(use_fn_calling).toBe(true);
                        expect(fn_str).toEqual("[{\"description\":\"A\",\"name\":\"fn_A\",\"parameters\":{\"foo\":\"bar\"}},{\"description\":\"B\",\"name\":\"fn_B\",\"parameters\":{\"foo\":\"bar\"}},{\"description\":\"C\",\"name\":\"fn_C\",\"parameters\":{\"foo\":\"bar\"}}]");
                    })
                };
            });
        (chat_module as any).checkFunctionCallUsage(request);
    });

    test('Test checkFunctionCallUsage function', () => {
        const chat_module = new ChatModule();

        const request: ChatCompletionRequest = {
            model: "gorilla-openfunctions-v1-q4f16_1_MLC",
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: "Hello!" },
            ],
            tool_choice: {
                type: 'function',
                function: {
                    name: 'fn_B',
                },
            },
            tools: [
                { type: 'function', function: { description: 'A', name: 'fn_A', parameters: { foo: 'bar' } } },
                { type: 'function', function: { description: 'B', name: 'fn_B', parameters: { foo: 'bar' } } },
                { type: 'function', function: { description: 'C', name: 'fn_C', parameters: { foo: 'bar' } } },
            ],
        };

        jest.spyOn(chat_module as any, "getPipeline")
            .mockImplementation(() => {
                return {
                    overrideFunctionCalling: jest.fn((use_fn_calling, fn_str) => {
                        expect(use_fn_calling).toBe(true);
                        expect(fn_str).toEqual("[{\"description\":\"B\",\"name\":\"fn_B\",\"parameters\":{\"foo\":\"bar\"}}]");
                    })
                };
            });
        (chat_module as any).checkFunctionCallUsage(request);
    });
});