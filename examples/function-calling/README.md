### OpenAI API Demos - Function calling

This folder contains two main ways of using function calling with WebLLM.

`function-calling-manual` demonstrates how you can use function calling with Llama3.1 and Hermes2
without using the `tools`, `tool_choice`, and `tool_call` fields. This is the most flexible way and you can follow
the instruction given by the model releaser and iterate yourself on top of that. However, you need to do parsing on your own, which differs for each model. For instance, Hermes2 models use `<tool_call>` and `</tool_call>` to wrap around a tool call, which may be very different from other models' format.

`function-calling-openai` conforms to the OpenAI function calling usage, leveraging `tools`, `tool_choice`, and `tool_call`
fields. This is more usable, but sacrifices the flexibility since we have pre-defined system prompt
for this.
