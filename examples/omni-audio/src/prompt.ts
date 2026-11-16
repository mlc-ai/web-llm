import { renderOmniPrompt } from "@mlc-ai/web-llm";

export function renderAudioPrompt(text: string, samples: number): string {
  return renderOmniPrompt(text, samples);
}
