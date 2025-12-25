# Structural tag MCP-style tool calls

Run `npm install`, then `npm start` to launch a minimal page that prints progress and logs to the browser console.

This example demonstrates how to:
- Define a structural tag that forces an MCP-style `<tool_call>...</tool_call>` block with `{"name": ..., "arguments": ...}` payloads.
- Ask WebLLM for a tool call with `response_format.type = "structural_tag"`, parse the call, and dispatch to a stubbed tool implementation.
- Send the tool result back via a `tool` message and request a final natural-language answer.

Open the console to see the enforced tool call, the stubbed tool response, and the final assistant reply.
