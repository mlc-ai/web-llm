import { renderOmniPrompt } from "../src/omni_prompt";
test("matches the pinned HF completed-answer text prompt", () => {
  expect(renderOmniPrompt("Say hello in one short sentence.")).toBe(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n",
  );
});
test("keeps literal plain-text whitespace and substitutes the exact audio token count", () => {
  const text = "  Describe this sound.\t";
  const result = renderOmniPrompt(text, 32000);
  expect(result.match(/<\|AUDIO\|>/g)).toHaveLength(50);
  expect(result).toContain("<|audio_eos|>" + text + "<|im_end|>");
});
test.each([0, -1, 160, 320, NaN, 1.5])(
  "rejects invalid or too-short audio %s",
  (samples) => expect(() => renderOmniPrompt("Hi", samples)).toThrow(),
);
test.each(["", "  ", "<|AUDIO|>", "hello <|im_end|>"])(
  "rejects reserved or empty text %s",
  (text) => expect(() => renderOmniPrompt(text)).toThrow(),
);
