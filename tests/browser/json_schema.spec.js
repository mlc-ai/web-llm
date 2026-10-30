import { test, expect } from "./fixtures.mjs";

test("JSON schema bounds formatting whitespace and preserves string contents", async ({
  page,
}) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { LLMChatPipeline, xgrammar } = await import("/harness.js");
    const vocab = [" ", '"', "<eos>"];
    const pipeline = Object.create(LLMChatPipeline.prototype);
    pipeline.xgTokenizerInfo = await xgrammar.TokenizerInfo.createTokenizerInfo(
      vocab,
      "raw",
      false,
      vocab.length,
      [2],
    );
    pipeline.grammarCompiler =
      await xgrammar.GrammarCompiler.createGrammarCompiler(
        pipeline.xgTokenizerInfo,
      );
    try {
      await pipeline.prepareGrammarMatcherForSampling({
        response_format: {
          type: "json_object",
          schema: JSON.stringify({
            type: "object",
            properties: { order_id: { type: "string" } },
            required: ["order_id"],
            additionalProperties: false,
          }),
        },
      });
      const matcher = pipeline.grammarMatcher;
      const prefix = matcher._acceptString('{"order_id":' + " ".repeat(16));
      const whitespaceMask = await matcher.getNextTokenBitmask();
      const opensString = matcher.acceptToken(1);
      const stringMask = await matcher.getNextTokenBitmask();
      const stringContents = matcher._acceptString(" ".repeat(64) + '1029"}');
      const stopsAfterObject = matcher.acceptToken(2);
      matcher.reset();
      const rejectsInvalidJSON = !matcher._acceptString('{"order_id": 1029}');
      return {
        prefix,
        blocksMoreWhitespace: (whitespaceMask[0] & 1) === 0,
        opensString,
        blocksEOSInsideString: (stringMask[0] & (1 << 2)) === 0,
        stringContents,
        stopsAfterObject,
        rejectsInvalidJSON,
      };
    } finally {
      pipeline.grammarMatcher?.dispose();
      pipeline.grammarCompiler.dispose();
      pipeline.xgTokenizerInfo.dispose();
    }
  });
  expect(Object.values(result)).toEqual(Array(7).fill(true));
});
