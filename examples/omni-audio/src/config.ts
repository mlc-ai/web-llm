// Serve locally built artifacts under public/artifacts (see README).
export const config = {
  components: {
    features: "/artifacts/omni-features/",
    audio: "/artifacts/omni-audio/",
    thinker: "/artifacts/omni-public-thinker-speech/",
    talker: "/artifacts/omni-talker-graph-webgpu/",
    reference: "/artifacts/omni-ecapa-webgpu/",
    flow: "/artifacts/omni-dit-webgpu-512/",
    vocoder: "/artifacts/omni-vocoder-webgpu-512/",
  },
  speakers: "/artifacts/omni-public-speakers/speakers.json",
  tokenizer: "/artifacts/omni-reference/tokenizer.json",
  maxSeconds: 30,
  maxNewTokens: 64,
  maxCodes: 256,
  maxFrames: 512,
  contextWindowSize: 4096,
};
