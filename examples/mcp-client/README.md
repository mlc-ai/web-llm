# MCP client example

Run `npm install`, then `npm start` to launch a page at `localhost:8890`.

This example demonstrates how to connect WebLLM to a **real** Model Context Protocol server —
as opposed to `structural-tag-tool-use`, which demonstrates the MCP-style tool-call _format_
against stubbed, hardcoded tools.

It shows how to:

- Implement the MCP **Streamable HTTP** transport (spec `2025-03-26`) directly with `fetch`,
  including the JSON-RPC handshake (`initialize` → `notifications/initialized`) and both
  plain-JSON and `text/event-stream` response handling.
- Discover a server's tools at runtime via `tools/list` and convert them into WebLLM's standard
  `ChatCompletionTool` format.
- Let the model decide when to call a tool via `tools` / `tool_choice: "auto"`, execute the call
  against the live server via `tools/call`, and feed the real result back as a `role: "tool"`
  message before requesting the model's final answer.

## Prerequisites: something to connect to

This example only speaks MCP's **Streamable HTTP** transport, and only to a server with CORS
enabled for this origin. You can test it with https://met-museum.caseyjhand.com/mcp for example.
Most local MCP servers (`npx @some/mcp-server`) speak **stdio**, not
HTTP — bridge one first with a tool like
[`supergateway`](https://github.com/supercorp-ai/supergateway):

```bash
npx -y supergateway --stdio "npx -y @modelcontextprotocol/server-everything" --port 8811 --cors
```

Then paste `http://localhost:8811/mcp` (or whatever URL that prints) into the input on the page.

## Model choice

Tool calling quality varies a lot by model — this example defaults to
`Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC`, which reliably produces well-formed tool calls. Smaller
models are more likely to hallucinate malformed `tool_calls` arguments.

## Note: expected 400 in console

If the target server doesn't implement 2026-07-28's `server/discover` (most don't yet), you'll
see one `400 Bad Request` in the console before the client falls back to the classic handshake.
This is the version-detection probe working as intended, not an error — the fallback in
`initialize()` handles it and the rest of the flow proceeds normally.
