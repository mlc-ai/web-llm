// Serve locally built artifacts under public/artifacts (see README).
export const config = {
  components: {
    features: "/artifacts/omni-features/",
    audio: "/artifacts/omni-audio/",
    thinker: "/artifacts/omni-thinker-f32accum-q0f16/",
  },
  tokenizer: "/artifacts/omni-reference/tokenizer.json",
  maxSeconds: 30,
  maxNewTokens: 128,
  contextWindowSize: 4096,
};
