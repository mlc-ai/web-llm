import { MLCEngine } from "../../src/engine";
import { ModelType } from "../../src/config";
import { LLMChatPipeline } from "../../src/llm_chat";
import { EmbeddingPipeline } from "../../src/embedding";
import { CustomLock } from "../../src/support";
import { jest } from "@jest/globals";
type ChatConfig = import("../../src/config").ChatConfig;
type Conversation = import("../../src/conversation").Conversation;
type TVMInstance = import("@mlc-ai/web-runtime").Instance;
type Tokenizer = import("@mlc-ai/web-tokenizers").Tokenizer;

jest.mock("../../src/llm_chat", () => {
  const { getConversation } = jest.requireActual(
    "../../src/conversation",
  ) as typeof import("../../src/conversation");

  class MockLLMChatPipeline {
    public decodeLimit = 2;
    public prefillCallCount = 0;
    public curRoundCompletionTotalTokens = 0;
    public decodeCallCount = 0;
    public resetCount = 0;
    public enablePromptCheckpoint = false;
    public enableDecodeCheckpoint = false;
    public promptCheckpointRestoreCount = 0;
    public restoredCheckpointSeqLen = 0;
    public checkpointPageSize = 1;
    private conversation: Conversation = getConversation(
      {
        system_template: "{system_message}",
        system_message: "",
        roles: { user: "user", assistant: "assistant" },
        seps: ["\n"],
        stop_token_ids: [0],
        stop_str: [],
      } as any,
      undefined,
    );
    private stopFlag = true;
    private message = "";
    private finishReason: string | undefined = undefined;
    private curRoundPrefillTotalTokens = 0;
    private curRoundDecodingTotalTokens = 0;
    private curRoundPrefillTotalTime = 0.001;
    private curRoundDecodingTotalTime = 0.001;
    private curRoundGrammarPerTokenTotalTime = 0;
    private pendingPrefillMessage = "";
    private pendingDecodeMessage = "";
    private pendingDecodeStop = false;
    private rngState: unknown;

    constructor(_tvm: TVMInstance, _tokenizer: Tokenizer, config: ChatConfig) {
      this.conversation = getConversation(
        config.conv_template,
        config.conv_config,
      );
    }

    async asyncLoadWebGPUPipelines() {}
    dispose() {}
    async sync() {}
    setSeed(seed: number) {
      this.rngState = seed;
    }

    getConversationObject() {
      return this.conversation;
    }

    setConversation(newConv: Conversation) {
      this.conversation = newConv;
    }

    resetChat() {
      this.resetCount++;
      this.stopFlag = true;
      this.decodeCallCount = 0;
      this.message = "";
      this.finishReason = undefined;
      this.curRoundPrefillTotalTokens = 0;
      this.curRoundDecodingTotalTokens = 0;
    }

    async prefillStep(
      inp: string,
      msgRole: string,
      roleName?: string,
    ): Promise<void> {
      const step = await this.samplePrefillStep(inp, msgRole, roleName);
      this.commitSampledStep(step);
    }

    async samplePrefillStep(
      inp: string,
      msgRole: string,
      roleName?: string,
      _genConfig?: unknown,
      opts?: {
        capturePromptCheckpoint?: boolean;
        storeCheckpointLogits?: boolean;
      },
    ): Promise<any> {
      this.prefillCallCount++;
      const roleSuffix = roleName ? `(${roleName})` : "";
      this.pendingPrefillMessage = `${msgRole}${roleSuffix}:${inp}`;
      this.stopFlag = false;
      this.decodeCallCount = 0;
      this.curRoundPrefillTotalTokens = Math.max(1, inp.length);
      this.curRoundPrefillTotalTime = 0.01 * this.curRoundPrefillTotalTokens;
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundCompletionTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      this.curRoundGrammarPerTokenTotalTime = 0;
      this.finishReason = "length";
      const promptCheckpoint =
        this.enablePromptCheckpoint && opts?.capturePromptCheckpoint === true
          ? {
              processedSeqLen: this.curRoundPrefillTotalTokens,
              layoutHash: "mock-layout",
              metadata: {
                seqLength: this.curRoundPrefillTotalTokens,
                layoutHash: "mock-layout",
                pageSize: this.checkpointPageSize,
                groups: [
                  {
                    groupIndex: 0,
                    layerBegin: 0,
                    layerEnd: 1,
                    shape: [1, 1],
                    dtype: "uint8",
                  },
                ],
              },
              pageGroups: [
                {
                  groupId: 0,
                  layerStart: 0,
                  layerEnd: 1,
                  data: new Uint8Array([7]),
                },
              ],
              nextLogits:
                opts?.storeCheckpointLogits === false
                  ? undefined
                  : {
                      shape: [1],
                      dtype: "uint8",
                      data: new Uint8Array([8]),
                    },
            }
          : undefined;
      return {
        source: "prefill",
        tokenId: 100,
        globalTokenPos: this.curRoundPrefillTotalTokens,
        promptLen: this.curRoundPrefillTotalTokens,
        promptTokenIds: Array.from(inp).map((char) => char.charCodeAt(0)),
        assistantPrefixTokenIds: [],
        promptCheckpoint,
      };
    }

    async decodeStep(genConfig?: { max_tokens?: number | null }) {
      const step = await this.sampleDecodeStep(genConfig);
      this.commitSampledStep(step);
    }

    async sampleDecodeStep(
      genConfig?: { max_tokens?: number | null },
      opts?: {
        captureCheckpoint?: boolean;
        storeCheckpointLogits?: boolean;
      },
    ) {
      if (this.stopFlag) return;
      this.decodeCallCount++;
      const globalTokenPos =
        this.curRoundPrefillTotalTokens + this.decodeCallCount;
      this.pendingDecodeMessage = `|token${this.decodeCallCount}|`;
      this.curRoundDecodingTotalTokens = this.decodeCallCount;
      this.curRoundDecodingTotalTime = this.curRoundDecodingTotalTokens * 0.02;
      this.curRoundGrammarPerTokenTotalTime =
        this.curRoundDecodingTotalTokens * 0.001;
      this.pendingDecodeStop =
        this.decodeCallCount >= this.decodeLimit ||
        (genConfig?.max_tokens !== null &&
          genConfig?.max_tokens !== undefined &&
          this.decodeCallCount >= genConfig.max_tokens);
      return {
        source: "decode",
        tokenId: 100 + this.decodeCallCount,
        globalTokenPos,
        decodeCheckpoint:
          this.enableDecodeCheckpoint && opts?.captureCheckpoint === true
            ? {
                processedSeqLen: globalTokenPos,
                layoutHash: "mock-layout",
                metadata: {
                  seqLength: globalTokenPos,
                  layoutHash: "mock-layout",
                  pageSize: this.checkpointPageSize,
                  groups: [
                    {
                      groupIndex: 0,
                      layerBegin: 0,
                      layerEnd: 1,
                      shape: [1, 1],
                      dtype: "uint8",
                    },
                  ],
                },
                pageGroups: [
                  {
                    groupId: 0,
                    layerStart: 0,
                    layerEnd: 1,
                    data: new Uint8Array([7 + this.decodeCallCount]),
                  },
                ],
                nextLogits:
                  opts?.storeCheckpointLogits === false
                    ? undefined
                    : {
                        shape: [1],
                        dtype: "uint8",
                        data: new Uint8Array([8 + this.decodeCallCount]),
                      },
              }
            : undefined,
      };
    }

    commitSampledStep(step: any) {
      const prevMessage = this.message;
      this.curRoundCompletionTotalTokens += 1;
      if (step.source === "prefill") {
        this.message = this.pendingPrefillMessage;
      } else {
        this.message += this.pendingDecodeMessage;
        if (this.pendingDecodeStop) {
          this.stopFlag = true;
          this.finishReason = "stop";
        }
      }
      return {
        source: step.source,
        tokenId: step.tokenId,
        globalTokenPos: step.globalTokenPos,
        textDelta: this.message.slice(prevMessage.length),
        textPrefixLength: prevMessage.length,
        outputMessage: this.message,
        stopped: this.stopFlag,
        finishReason: this.finishReason,
      };
    }

    getRNGState() {
      return this.prefillCallCount * 1000 + this.decodeCallCount;
    }

    setRNGState(state: unknown) {
      this.rngState = state;
      return state !== undefined;
    }

    async replayGenerationTokens(
      promptTokenIds: number[],
      _assistantPrefixTokenIds: number[],
      generatedTokens: Array<{
        tokenId: number;
        textDelta: string;
        textPrefixLength?: number;
      }>,
    ) {
      this.resetChat();
      this.stopFlag = false;
      this.finishReason = "length";
      this.curRoundPrefillTotalTokens = promptTokenIds.length;
      this.curRoundPrefillTotalTime = Math.max(
        0.001,
        promptTokenIds.length * 0.01,
      );
      this.decodeCallCount = Math.max(0, generatedTokens.length - 1);
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundCompletionTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      if (generatedTokens.length === 0) {
        this.message = "first";
        return {
          replayedTokens: 0,
          sampledFromCheckpointLogits: false,
          sampledToken: {
            source: "prefill",
            tokenId: 100,
            globalTokenPos: promptTokenIds.length,
          },
          committedToken: {
            source: "prefill",
            tokenId: 100,
            globalTokenPos: promptTokenIds.length,
            textDelta: "first",
            textPrefixLength: 0,
            outputMessage: "first",
            stopped: false,
          },
        };
      }
      this.message = generatedTokens.reduce(
        (message, token) =>
          message.slice(0, token.textPrefixLength ?? message.length) +
          token.textDelta,
        "",
      );
      return {
        replayedTokens: generatedTokens.length,
        sampledFromCheckpointLogits: false,
      };
    }

    async replayFromPromptCheckpoint(
      checkpoint: { processedSeqLen?: number },
      _assistantPrefixTokenIds: number[],
      coveredTokens: Array<{
        tokenId: number;
        textDelta: string;
        textPrefixLength?: number;
      }>,
      tailTokens: Array<{
        tokenId: number;
        textDelta: string;
        textPrefixLength?: number;
      }>,
    ) {
      this.promptCheckpointRestoreCount++;
      this.restoredCheckpointSeqLen = checkpoint.processedSeqLen ?? 0;
      this.resetChat();
      this.stopFlag = false;
      this.finishReason = "length";
      const generatedTokens = [...coveredTokens, ...tailTokens];
      this.decodeCallCount = Math.max(0, generatedTokens.length - 1);
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundCompletionTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      if (generatedTokens.length === 0) {
        this.message = "first";
        return {
          replayedTokens: 0,
          sampledFromCheckpointLogits: true,
          sampledToken: {
            source: "prefill",
            tokenId: 100,
            globalTokenPos: checkpoint.processedSeqLen ?? 0,
          },
          committedToken: {
            source: "prefill",
            tokenId: 100,
            globalTokenPos: checkpoint.processedSeqLen ?? 0,
            textDelta: "first",
            textPrefixLength: 0,
            outputMessage: "first",
            stopped: false,
          },
        };
      }
      this.message = generatedTokens.reduce(
        (message, token) =>
          message.slice(0, token.textPrefixLength ?? message.length) +
          token.textDelta,
        "",
      );
      return {
        replayedTokens: tailTokens.length,
        sampledFromCheckpointLogits: false,
      };
    }

    stopped() {
      return this.stopFlag;
    }

    triggerStop() {
      this.stopFlag = true;
      this.finishReason = "abort";
    }

    getMessage() {
      return this.message;
    }

    getFinishReason() {
      return this.finishReason ?? "stop";
    }

    getCurRoundDecodingTotalTokens() {
      return this.curRoundDecodingTotalTokens;
    }

    getCurRoundCompletionTotalTokens() {
      return this.curRoundCompletionTotalTokens;
    }

    getCurRoundPrefillTotalTokens() {
      return this.curRoundPrefillTotalTokens;
    }

    getCurRoundPrefillTokensPerSec() {
      return this.curRoundPrefillTotalTokens / this.curRoundPrefillTotalTime;
    }

    getCurRoundDecodingTokensPerSec() {
      return this.curRoundDecodingTotalTokens / this.curRoundDecodingTotalTime;
    }

    getCurRoundGrammarInitTotalTime() {
      return 0.001;
    }

    getCurRoundPrefillTotalTime() {
      return this.curRoundPrefillTotalTime;
    }

    getCurRoundDecodingTotalTime() {
      return this.curRoundDecodingTotalTime;
    }

    getCurRoundGrammarPerTokenTotalTime() {
      return this.curRoundGrammarPerTokenTotalTime;
    }

    getCurRoundLatencyBreakdown() {
      return {
        logitProcessorTime: [0.001],
        logitBiasTime: [0.001],
        penaltyTime: [0.001],
        sampleTime: [0.001],
        totalTime: [0.001],
        grammarBitmaskTime: [0.001],
      };
    }

    getTokenLogprobArray() {
      return [];
    }

    async forwardTokensAndSample(inputIds: Array<number>): Promise<number> {
      return inputIds[0] ?? 0;
    }

    async runtimeStatsText() {
      return `prefills=${this.prefillCallCount}`;
    }
  }

  return { LLMChatPipeline: MockLLMChatPipeline };
});

jest.mock("../../src/embedding", () => {
  class MockEmbeddingPipeline {
    public inputs: any;
    public embedResult: Array<Array<number>> = [[0.1, 0.2, 0.3]];
    dispose() {}
    async sync() {}
    async embedStep(
      input: string | Array<string> | Array<number> | Array<Array<number>>,
    ): Promise<Array<Array<number>>> {
      this.inputs = input;
      return this.embedResult;
    }
    getCurRoundEmbedTotalTokens(): number {
      if (typeof this.inputs === "string") {
        return this.inputs.length;
      } else if (Array.isArray(this.inputs)) {
        return this.inputs.length;
      }
      return 0;
    }
    getCurRoundEmbedTokensPerSec(): number {
      const tokens = this.getCurRoundEmbedTotalTokens();
      return tokens === 0 ? 0 : tokens / 0.01;
    }
  }
  return { EmbeddingPipeline: MockEmbeddingPipeline };
});

export const MODEL_ID = "mock-model";
export const SECOND_MODEL_ID = "mock-model-2";
export const EMBED_MODEL_ID = "mock-embed";
export const FIXED_CREATED_DATE = new Date("2024-04-05T06:34:56.789Z");
export const FIXED_CREATED_SECONDS = 1712298896;

export const mockChatConfig: ChatConfig = {
  tokenizer_files: ["tokenizer.json"],
  vocab_size: 10,
  conv_template: {
    system_template: "{system_message}",
    system_message: "You are a helpful assistant.",
    system_prefix_token_ids: [1],
    add_role_after_system_message: false,
    roles: {
      user: "User",
      assistant: "Assistant",
      tool: "Tool",
    },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
      tool: "{tool_message}",
    },
    seps: ["\n"],
    role_content_sep: ": ",
    role_empty_sep: ": ",
    stop_str: [],
    stop_token_ids: [0],
  },
  conv_config: undefined,
  context_window_size: 8,
  sliding_window_size: -1,
  attention_sink_size: -1,
  temperature: 0.8,
  presence_penalty: 0,
  frequency_penalty: 0,
  repetition_penalty: 1,
  top_p: 1,
};

export function createEngineWithPipeline(decodeLimit = 2, modelId = MODEL_ID) {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/model",
          model_id: modelId,
          model_lib: "https://example.com/model.wasm",
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  pipeline.decodeLimit = decodeLimit;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(modelId, pipeline);
  internal.loadedModelIdToChatConfig.set(modelId, mockChatConfig);
  internal.loadedModelIdToModelType.set(modelId, ModelType.LLM);
  internal.loadedModelIdToLock.set(modelId, new CustomLock());
  return { engine, pipeline };
}

export function createEngineWithMultiplePipelines() {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/model",
          model_id: MODEL_ID,
          model_lib: "https://example.com/model.wasm",
        },
        {
          model: "https://example.com/model2",
          model_id: SECOND_MODEL_ID,
          model_lib: "https://example.com/model2.wasm",
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline1 = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  const pipeline2 = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(MODEL_ID, pipeline1);
  internal.loadedModelIdToPipeline.set(SECOND_MODEL_ID, pipeline2);
  internal.loadedModelIdToChatConfig.set(MODEL_ID, mockChatConfig);
  internal.loadedModelIdToChatConfig.set(SECOND_MODEL_ID, mockChatConfig);
  internal.loadedModelIdToModelType.set(MODEL_ID, ModelType.LLM);
  internal.loadedModelIdToModelType.set(SECOND_MODEL_ID, ModelType.LLM);
  internal.loadedModelIdToLock.set(MODEL_ID, new CustomLock());
  internal.loadedModelIdToLock.set(SECOND_MODEL_ID, new CustomLock());
  return engine;
}

const mockEmbeddingConfig: ChatConfig = {
  ...mockChatConfig,
};

export function createEngineWithEmbeddingPipeline() {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/embed",
          model_id: EMBED_MODEL_ID,
          model_lib: "https://example.com/embed.wasm",
          model_type: ModelType.embedding,
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline = new EmbeddingPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockEmbeddingConfig,
  ) as any;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(EMBED_MODEL_ID, pipeline);
  internal.loadedModelIdToChatConfig.set(EMBED_MODEL_ID, mockEmbeddingConfig);
  internal.loadedModelIdToModelType.set(EMBED_MODEL_ID, ModelType.embedding);
  internal.loadedModelIdToLock.set(EMBED_MODEL_ID, new CustomLock());
  return { engine, pipeline };
}
