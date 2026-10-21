import { test, expect, loadModel, modelId } from "./webgpu.mjs";

test("separated tool-loop reasoning matches model-formatted input on real WebGPU", async ({
  page,
}) => {
  await loadModel(page);
  const result = await page.evaluate(async (modelId) => {
    const engine = globalThis.gpuEngine;
    await engine.reload(modelId, {
      context_window_size: 512,
      prefill_chunk_size: 128,
      conv_config: {
        reasoning: "qwen3",
        roles: {
          user: "<|im_start|>user",
          assistant: "<|im_start|>assistant",
          tool: "<|im_start|>user",
        },
        role_templates: {
          user: "{user_message}",
          assistant: "{assistant_message}",
          tool: "<tool_response>\n{tool_message}\n</tool_response>",
        },
      },
    });
    const content =
      '<tool_call>\n{"name":"lookup","arguments":{"code":"A7"}}\n</tool_call>';
    const first = {
      role: "user",
      content: "Look up A7 and report the returned value.",
    };
    const last = { role: "tool", tool_call_id: "call", content: "42" };
    const request = {
      seed: 17,
      temperature: 0.6,
      max_tokens: 64,
      extra_body: { enable_thinking: true },
    };
    const records = [];
    for (const streaming of [false, true]) {
      const read = async (assistant) => {
        await engine.resetChat();
        const response = await engine.chat.completions.create({
          ...request,
          messages: [first, assistant, last],
          stream: streaming,
          ...(streaming && { stream_options: { include_usage: true } }),
        });
        if (!streaming)
          return {
            text: response.choices[0].message.content,
            promptTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens,
          };
        let text = "";
        let usage;
        for await (const chunk of response) {
          text += chunk.choices[0]?.delta?.content ?? "";
          if (chunk.usage) usage = chunk.usage;
        }
        return {
          text,
          promptTokens: usage.prompt_tokens,
          outputTokens: usage.completion_tokens,
        };
      };
      const separate = await read({
        role: "assistant",
        content,
        reasoning_content: "\nRead the result.\n",
      });
      const manual = await read({
        role: "assistant",
        content: "<think>\nRead the result.\n</think>\n\n" + content,
      });
      records.push({ separate, manual });
    }
    let unsupported;
    try {
      await engine.chat.completions.create({
        ...request,
        messages: [first],
        extra_body: { preserve_thinking: true },
      });
    } catch (error) {
      unsupported = String(error);
    }
    const recovery = await engine.chat.completions.create({
      ...request,
      messages: [{ role: "user", content: "Say hello." }],
    });
    return {
      records,
      unsupported,
      recovered: recovery.usage.completion_tokens > 0,
    };
  }, modelId);
  for (const { separate, manual } of result.records) {
    expect(separate).toEqual(manual);
    expect(separate.outputTokens).toBeGreaterThan(0);
  }
  expect(result.unsupported).toContain("does not support preserve_thinking");
  expect(result.recovered).toBe(true);
});
