/** Pinned Qwen2.5-Omni single-turn plain-text or audio-then-text prompt. */
export function renderOmniPrompt(text: string, audioSamples?: number): string {
  if (typeof text !== "string" || !text.trim() || text.includes("<|"))
    throw new Error(
      "Enter plain text without reserved control or media tokens",
    );
  let media = "";
  if (audioSamples !== undefined) {
    if (!Number.isSafeInteger(audioSamples) || audioSamples < 1)
      throw new Error("Audio is empty");
    const frames = Math.ceil(audioSamples / 160),
      tokens = Math.floor(Math.ceil(frames / 2) / 2);
    if (tokens < 1) throw new Error("Audio is too short for the encoder");
    media = "<|audio_bos|>" + "<|AUDIO|>".repeat(tokens) + "<|audio_eos|>";
  }
  return (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
    "<|im_start|>user\n" +
    media +
    text +
    "<|im_end|>\n<|im_start|>assistant\n"
  );
}
