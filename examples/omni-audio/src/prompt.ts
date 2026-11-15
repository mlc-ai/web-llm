// Qwen/Qwen2.5-Omni-3B chat_template.json: one user audio item followed by text,
// add_generation_prompt=true, add_audio_id=false, add_vision_id=false.
export function renderAudioPrompt(text: string, samples: number): string {
  if (!text.trim() || text.includes("<|")) {
    throw new Error(
      "Enter plain text without reserved control or media tokens.",
    );
  }
  if (!Number.isSafeInteger(samples) || samples < 1) {
    throw new Error("Audio is empty.");
  }
  const frames = Math.ceil(samples / 160);
  const audioTokens = Math.floor(Math.ceil(frames / 2) / 2);
  if (audioTokens < 1) throw new Error("Audio is too short for the encoder.");
  return (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
    "<|im_start|>user\n<|audio_bos|>" +
    "<|AUDIO|>".repeat(audioTokens) +
    "<|audio_eos|>" +
    text +
    "<|im_end|>\n<|im_start|>assistant\n"
  );
}
