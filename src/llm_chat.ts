/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable no-prototype-builtins */
import * as tvmjs from "@mlc-ai/web-runtime";
import * as xgrammar from "@mlc-ai/web-xgrammar";
import log from "loglevel";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig, GenerationConfig, Role } from "./config";
import { getConversation, Conversation } from "./conversation";
import { LogitProcessor } from "./types";
import {
  getChunkedPrefillInputData,
  getImageDataFromURL,
  getRGBArrayFromImageData,
  getTokenTableFromTokenizer,
  getTopProbs,
  IMAGE_EMBED_SIZE,
} from "./support";
import {
  ChatCompletionFinishReason,
  ChatCompletionTokenLogprob,
  TopLogprob,
  ResponseFormat,
  ChatCompletionContentPartImage,
} from "./openai_api_protocols/index";
import {
  AttentionSinkSizeError,
  ContextWindowSizeExceededError,
  MinValueError,
  RangeError,
  WindowSizeConfigurationError,
  WindowSizeSpecificationError,
  MessageOrderError,
  TextCompletionExpectsKVEmptyError,
  PrefillChunkSizeSmallerThanImageError,
  CannotFindImageEmbedError,
} from "./error";

type ImageURL = ChatCompletionContentPartImage.ImageURL;

export class LLMChatPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private decoding: tvmjs.PackedFunc;
  private image_embed: tvmjs.PackedFunc | undefined;
  private embed: tvmjs.PackedFunc;
  private fapplyBitmask: tvmjs.PackedFunc;
  // Functions related to PagedKVCache
  private fclearKVCaches: tvmjs.PackedFunc;
  private fKVCacheAddSequence: tvmjs.PackedFunc;
  private fKVCacheRemoveSequence: tvmjs.PackedFunc;
  private fKVCacheBeginForward: tvmjs.PackedFunc;
  private fKVCacheEndForward: tvmjs.PackedFunc;
  private fKVCacheEnableSlidingWindowForSeq: tvmjs.PackedFunc;

  // parameter states
  private params: tvmjs.TVMObject;
  private kvCache: tvmjs.TVMObject;
  private logitsOnCPU?: tvmjs.NDArray = undefined;
  private filledKVCacheLength = 0;

  // meta data
  private bosTokenId = 1;
  private contextWindowSize = -1;
  private slidingWindowSize = -1;
  private attentionSinkSize = -1;
  private prefillChunkSize = -1;
  private resetStatsPerPrefill = true;
  private stopStr: string[];
  private stopTokens: Array<number>;

  // states
  private outputMessage = "";
  private outputIds: Array<number> = [];
  private stopTriggered = false;
  private finishReason: ChatCompletionFinishReason | undefined = undefined;
  // frequency of appeared token ids till now (refresh after PrefillStep); token_id mapped to freq
  private appearedTokensFreq = new Map<number, number>();
  private conversation: Conversation;
  // The logprob information of all tokens for this current round (cleared upon each prefillStep)
  // Cleared & updated at the exact same spots as `outputMessage`. Only updated when
  // `genConfig.logprobs` is true. Each entry corresponds to a single autoregressive step.
  private tokenLogprobArray: Array<ChatCompletionTokenLogprob> = [];

  // stats, reset at every `resetChat(keepstats=false)`
  private decodingTotalTime = 0;
  private decodingTotalTokens = 0;
  private prefillTotalTime = 0;
  private prefillTotalTokens = 0;
  // same stats as above, but reset at every `prefillStep()`
  private curRoundDecodingTotalTokens = 0;
  private curRoundPrefillTotalTokens = 0;
  private curRoundDecodingTotalTime = 0;
  private curRoundPrefillTotalTime = 0;

  // LogitProcessor
  private logitProcessor?: LogitProcessor = undefined;

  // Grammar-related
  // A grammar state matcher for this current round if response_format is set. Reinitialized upon
  // each step regardless of whether the chat is multi-round or not.
  private grammarStateMatcher?: xgrammar.GrammarStateMatcher = undefined;
  // The current schema used for grammarStateMatcher; if undefined, grammarStateMatcher is simply
  // using JSON mode. We use this field to determine whether we re-initiate a GrammarStateMatcher
  // or simply reset the state during each round (i.e. during prefillStep).
  private schema?: string = undefined;
  // A string list of tokens ordered by their token id, post-processed. Once initialized, will not
  // be reinitialized since `this.tokenizer` does not change throughout the lifetime of LLMChatPipeline.
  private tokenTable?: xgrammar.XGTokenTable = undefined;
  private bitmaskSize: number;
  // `vocab_size` read from `config.json`. Can be different from the size of the tokenTable for some
  // models due to dummy padded tokens.
  private fullVocabSize: number;
  // Method to post process the token for grammar; either "byte_level" or default "byte_fallback".
  private token_postproc_method: string;

  constructor(
    tvm: tvmjs.Instance,
    tokenizer: Tokenizer,
    config: ChatConfig,
    logitProcessor?: LogitProcessor,
  ) {
    // 0. Setting attributes
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;
    this.logitProcessor = logitProcessor;
    this.fullVocabSize = this.config.vocab_size;
    this.bitmaskSize = Math.ceil(this.fullVocabSize / 32);

    this.conversation = getConversation(
      config.conv_template,
      config.conv_config,
    );
    this.stopStr = this.conversation.getStopStr();
    this.stopTokens = this.conversation.getStopTokens();
    if (config.bos_token_id !== undefined) {
      this.bosTokenId = config.bos_token_id;
    }
    // Set token_post_proc_method, currently mlc-chat-config.json are unstable, hence various
    // fallback mechanisms
    if (config.tokenizer_info !== undefined) {
      this.token_postproc_method = config.tokenizer_info.token_postproc_method;
    } else if (config.token_table_postproc_method !== undefined) {
      this.token_postproc_method = config.token_table_postproc_method;
    } else {
      log.warn(
        "Cannot find `tokenizer_info` or `token_table_postproc_method` in `mlc-chat-config.json`, " +
          "using default token_postproc_method `byte_fallback`.\n" +
          "Models that should not use `byte_fallback` include: Llama3, Qwen1.5-1.8B, StableLM-zerphyr-1.6B.\n" +
          "This field is only used for json mode.",
      );
      this.token_postproc_method = "byte_fallback";
    }
    log.info("token_postproc_method: ", this.token_postproc_method);

    this.device = this.tvm.webgpu();

    // 1. Create VM and get the core functions
    tvm.beginScope();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device),
    );
    this.prefill = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("prefill"),
    );
    this.embed = this.tvm.detachFromCurrentScope(this.vm.getFunction("embed"));
    this.decoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("decode"),
    );
    this.fapplyBitmask = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("apply_bitmask_inplace"),
    );
    try {
      this.image_embed = this.tvm.detachFromCurrentScope(
        this.vm.getFunction("image_embed"),
      );
    } catch {
      log.info("Cannot find function image_embed.");
    }

    // 2. Get json stored in the vm's metadata function
    const fgetMetadata = this.vm.getFunction("_metadata");
    const ret_value = fgetMetadata();
    const metadataStr = this.tvm.detachFromCurrentScope(ret_value).toString();
    const metadata = JSON.parse(metadataStr);

    // 3. Load parameters by name
    const paramNames: string[] = [];
    metadata.params.forEach((param: any) => {
      paramNames.push(param.name);
    });
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCacheByName(paramNames),
    );

    // 4. Read in compilation configurations from metadata
    this.prefillChunkSize = metadata.prefill_chunk_size;
    log.info("Using prefillChunkSize: ", this.prefillChunkSize);
    if (this.prefillChunkSize <= 0) {
      throw new MinValueError("prefill_chunk_size", 0);
    }

    // 5. Consolidate KVCache settings: context window, sliding window, attention sink
    this.slidingWindowSize = config.sliding_window_size;
    this.contextWindowSize = config.context_window_size;
    this.attentionSinkSize = config.attention_sink_size;
    if (this.contextWindowSize !== -1 && this.slidingWindowSize !== -1) {
      throw new WindowSizeConfigurationError(
        this.contextWindowSize,
        this.slidingWindowSize,
      );
    } else if (this.slidingWindowSize != -1) {
      // Use sliding window and attention sink
      log.info("Using slidingWindowSize: ", this.slidingWindowSize);
      if (this.attentionSinkSize >= 0) {
        log.info("Using attentionSinkSize: ", this.attentionSinkSize);
      } else {
        throw new AttentionSinkSizeError();
      }
    } else if (this.contextWindowSize != -1) {
      // Use default kv cache without sliding window
      log.info("Using contextWindowSize: ", this.contextWindowSize);
    } else {
      throw new WindowSizeSpecificationError();
    }

    // 5. Create cache
    // Load cache functions and instantiate KVCache
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_clear"),
    );
    this.fKVCacheAddSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_add_sequence"),
    );
    this.fKVCacheRemoveSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_remove_sequence"),
    );
    this.fKVCacheBeginForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_begin_forward"),
    );
    this.fKVCacheEndForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_end_forward"),
    );
    this.fKVCacheEnableSlidingWindowForSeq = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc(
        "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq",
      ),
    );

    // Create PagedKVCache; we do not expose KVCache config for now
    const fcreateCache = this.vm.getFunction("create_tir_paged_kv_cache");
    const defaultPageSize = 16;
    const defaultMaxNumSequence = 1;
    const maxTotalSeqLen =
      this.slidingWindowSize != -1
        ? this.slidingWindowSize
        : this.contextWindowSize;
    this.kvCache = this.tvm.detachFromCurrentScope(
      fcreateCache(
        this.tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
        this.tvm.makeShapeTuple([maxTotalSeqLen]), // max_total_sequence_length
        this.tvm.makeShapeTuple([this.prefillChunkSize]), // prefill_chunk_size
        this.tvm.makeShapeTuple([defaultPageSize]), // page_size, hard coded for now
        this.tvm.makeShapeTuple([this.slidingWindowSize != -1 ? 1 : 0]),
      ),
    );

    this.filledKVCacheLength = 0;
    this.resetChat(); // especially needed for PagedKVCache as we need to call fKVCacheAddSequence
    tvm.endScope();
  }

  dispose() {
    // TODO: Do we need to dispose all PackedFuncs here?
    this.grammarStateMatcher?.dispose();
    this.params.dispose();
    this.decoding.dispose();
    this.prefill.dispose();
    this.embed.dispose();
    this.image_embed?.dispose();
    this.vm.dispose();
    this.kvCache.dispose();
    this.fclearKVCaches.dispose();
    this.logitsOnCPU?.dispose();
    this.tvm.dispose();
    this.tokenizer.dispose();
    this.tokenTable?.dispose();
  }

  /**
   * Get the current message.
   */
  getMessage() {
    return this.outputMessage;
  }

  /**
   * Reset the runtime statistics
   */
  resetRuntimeStats() {
    this.prefillTotalTime = 0;
    this.prefillTotalTokens = 0;
    this.decodingTotalTime = 0;
    this.decodingTotalTokens = 0;
  }

  /**
   * Reset the chat history
   */
  resetChat(keepStats = false) {
    this.tvm.beginScope();
    this.conversation.reset();
    if (!keepStats) {
      this.resetRuntimeStats();
    }
    this.resetKVCache();
    this.filledKVCacheLength = 0;
    this.logitProcessor?.resetState();
    this.tvm.endScope();
  }

  /**
   * Reset KV Cache
   */
  resetKVCache() {
    this.fclearKVCaches(this.kvCache);
    this.fKVCacheAddSequence!(this.kvCache, new tvmjs.Scalar(0, "int64"));
    if (this.slidingWindowSize != -1) {
      this.fKVCacheEnableSlidingWindowForSeq(
        this.kvCache,
        new tvmjs.Scalar(0, "int64"),
        new tvmjs.Scalar(this.slidingWindowSize, "int32"),
        new tvmjs.Scalar(this.attentionSinkSize, "int32"),
      );
    }
  }

  /**
   * @returns Whether stop is triggered.
   */
  stopped(): boolean {
    return this.stopTriggered;
  }

  /**
   * @returns Finish reason; undefined if generation not started/stopped yet.
   */
  getFinishReason(): ChatCompletionFinishReason | undefined {
    return this.finishReason;
  }

  /**
   * @returns tokenLogprobArray for this current round of autoregressive generation.
   * Updated upon each sampled token, cleared upon each prefillStep().
   */
  getTokenLogprobArray(): Array<ChatCompletionTokenLogprob> {
    return this.tokenLogprobArray;
  }

  /**
   * @returns the number of tokens decoded for a single request or a single choice in the request.
   */
  getCurRoundDecodingTotalTokens(): number {
    return this.curRoundDecodingTotalTokens;
  }

  /**
   * @returns the number of tokens decoded for a single request or a single choice in the request.
   */
  getCurRoundPrefillTotalTokens(): number {
    return this.curRoundPrefillTotalTokens;
  }

  /**
   * @returns the time spent on decode for a single request or a single choice in the request.
   */
  getCurRoundDecodingTotalTime(): number {
    return this.curRoundDecodingTotalTime;
  }

  /**
   * @returns the time spent on  for a single request or a single choice in the request.
   */
  getCurRoundPrefillTotalTime(): number {
    return this.curRoundPrefillTotalTime;
  }

  /**
   * @returns Runtime stats information.
   */
  runtimeStatsText(): string {
    return (
      `prefill: ${(this.prefillTotalTokens / this.prefillTotalTime).toFixed(4)} tokens/sec, ` +
      `decoding: ${(this.decodingTotalTokens / this.decodingTotalTime).toFixed(4)} tokens/sec`
    );
  }

  /**
   * @returns Runtime stats information, starting from the last prefill performed.
   */
  curRoundRuntimeStatsText(): string {
    return (
      `prefill: ${this.getCurRoundPrefillTokensPerSec().toFixed(4)} tokens/sec, ` +
      `decoding: ${this.getCurRoundDecodingTokensPerSec().toFixed(4)} tokens/sec`
    );
  }

  /**
   * @returns Prefill tokens per second, starting from the last prefill performed.
   */
  getCurRoundPrefillTokensPerSec(): number {
    return this.curRoundPrefillTotalTokens / this.curRoundPrefillTotalTime;
  }

  /**
   * @returns Prefill tokens per second, starting from the last prefill performed.
   */
  getCurRoundDecodingTokensPerSec(): number {
    return this.curRoundDecodingTotalTokens / this.curRoundDecodingTotalTime;
  }

  /**
   * Set the seed for the RNG `this.tvm.rng`.
   */
  setSeed(seed: number): void {
    this.tvm.setSeed(seed);
  }

  // Getters and setters for this.conversation.
  /**
   * @returns The conversation object (not a deep copy).
   */
  getConversationObject(): Conversation {
    return this.conversation;
  }

  /**
   * Set this.conversation to a new conversation object.
   */
  setConversation(newConv: Conversation) {
    this.conversation = newConv;
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  /**
   * Generate the first token given input prompt
   */
  async prefillStep(
    inp: string,
    msgRole: Role, // either user or tool
    inp_role_str?: string,
    genConfig?: GenerationConfig,
  ): Promise<void> {
    if (msgRole !== Role.user && msgRole !== Role.tool) {
      throw new MessageOrderError(
        "The last message should be from `user` or `tool`.",
      );
    }
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }

    const tstart = performance.now();

    // cleanup the per convo states
    this.outputIds = [];
    this.appearedTokensFreq.clear();
    this.outputMessage = "";
    this.tokenLogprobArray = [];
    this.curRoundDecodingTotalTokens = 0;
    this.curRoundPrefillTotalTokens = 0;
    this.curRoundPrefillTotalTime = 0;
    this.curRoundDecodingTotalTime = 0;
    this.stopTriggered = false;
    const conversation = this.conversation;

    // 0. Get inputData from conversation
    if (conversation.isTextCompletion) {
      conversation.prompt = inp;
    } else {
      conversation.appendMessage(msgRole, inp, inp_role_str);
      conversation.appendReplyHeader(Role.assistant);
    }
    const retGetInputData = this.getInputData();
    const inputData: Array<Array<number> | ImageURL> = retGetInputData[0];
    const promptLen: number = retGetInputData[1];

    // Check if LLMChatPipeline fits for forwarding image input
    let hasImageInput = false;
    inputData.forEach((data) => {
      if (!Array.isArray(data)) {
        hasImageInput = true;
      }
    });
    if (hasImageInput && this.prefillChunkSize < IMAGE_EMBED_SIZE) {
      throw new PrefillChunkSizeSmallerThanImageError(
        this.prefillChunkSize,
        IMAGE_EMBED_SIZE,
      );
    }
    if (hasImageInput && this.image_embed === undefined) {
      throw new CannotFindImageEmbedError();
    }

    // 1. Chunk inputData to embed and forward in one shot for each, minimize intermediate data
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize,
    );
    const chunks: Array<Array<number> | ImageURL>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    // 2. Prefill each chunk
    this.tvm.beginScope();
    let logits: tvmjs.NDArray;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkLen = chunkLens[i];
      const prevFilledLen = this.filledKVCacheLength;
      logits = this.tvm.detachFromCurrentScope(
        await this.embedAndForward(chunk, chunkLen),
      );
      if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
        throw new Error(
          "Internal Error: filledKVCacheLength does not match expected value.",
        );
      }
    }

    // 3. Instantiate grammar state matcher according to generation config
    if (genConfig?.response_format?.type === "json_object") {
      const curSchema = genConfig.response_format.schema;
      if (curSchema === this.schema && this.grammarStateMatcher) {
        // If we did not change the schema and have instantiated a GrammarStateMatcher, we reuse it.
        this.grammarStateMatcher.reset();
      } else {
        // Else dispose current grammarStateMatcher, reinitialize, and update this.schema.
        if (this.grammarStateMatcher) {
          this.grammarStateMatcher.dispose();
        }
        if (this.tokenTable === undefined) {
          // Post process entire table
          const rawTokenTable = getTokenTableFromTokenizer(this.tokenizer);
          this.tokenTable = await xgrammar.XGTokenTable.createXGTokenTable(
            rawTokenTable,
            this.token_postproc_method,
          );
        }
        const grammar: xgrammar.BNFGrammar =
          curSchema === undefined
            ? await xgrammar.BuiltinGrammar.json()
            : await xgrammar.BuiltinGrammar.jsonSchema(curSchema);
        this.grammarStateMatcher =
          await xgrammar.GrammarStateMatcher.createGrammarStateMatcher(
            grammar,
            this.tokenTable,
          );
        grammar.dispose();
        this.schema = curSchema;
      }
    }

    this.tvm.endScope();

    // 4. Sample, stats, post process token sampled.
    const nextToken = await this.sampleTokenFromLogits(logits!, genConfig);
    logits!.dispose();
    const tend = performance.now();

    this.prefillTotalTime += (tend - tstart) / 1e3;
    this.prefillTotalTokens += promptLen;
    this.curRoundPrefillTotalTokens += promptLen;
    this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;

    this.processNextToken(nextToken, genConfig);
  }

  async decodeStep(genConfig?: GenerationConfig): Promise<void> {
    if (this.stopTriggered) {
      throw Error("Cannot run decode when stopped");
    }

    const tstart = performance.now();

    this.tvm.beginScope();
    const chunk: Array<Array<number>> = [
      this.outputIds.slice(this.outputIds.length - 1),
    ];
    const chunkLen = chunk.length;
    const prevFilledLen = this.filledKVCacheLength;
    const logits = this.tvm.detachFromCurrentScope(
      await this.embedAndForward(chunk, chunkLen),
    );
    if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
      throw new Error(
        "Internal Error: filledKVCacheLength does not match expected value.",
      );
    }
    this.tvm.endScope();

    // sample from logits
    const nextToken = await this.sampleTokenFromLogits(logits, genConfig);
    logits.dispose();
    const tend = performance.now();

    this.decodingTotalTime += (tend - tstart) / 1e3;
    this.decodingTotalTokens += 1;
    this.curRoundDecodingTotalTokens += 1;
    this.curRoundDecodingTotalTime += (tend - tstart) / 1e3;

    this.processNextToken(nextToken, genConfig);
  }

  /**
   * Manually trigger stop if it is not stopped.
   */
  triggerStop() {
    if (this.stopTriggered) {
      return;
    }
    this.stopTriggered = true;
    this.finishReason = "abort";
    if (!this.conversation.isTextCompletion) {
      this.conversation.finishReply(this.outputMessage);
    }
  }

  /**
   * Add a generated token and check for stop.
   *
   * @param nextToken The next token.
   * @param genConfig Configs that override `this.config` for this round of generation.
   */
  private processNextToken(
    nextToken: number,
    genConfig?: GenerationConfig,
  ): void {
    if (this.stopTriggered) {
      throw Error("Cannot call process when it is stoppped");
    }

    // Get max_tokens from generationConfig (specified by user in completion request)
    // If not specified, do not set a limit
    let max_tokens = Infinity;
    if (genConfig !== undefined && genConfig.max_tokens) {
      max_tokens = genConfig.max_tokens;
    }
    if (max_tokens <= 0) {
      throw new MinValueError("max_tokens", 0);
    }
    // Get stopStrs, possibly overridden by genConfig for this round
    let stopStrs = this.stopStr;
    if (genConfig !== undefined && genConfig.stop) {
      stopStrs = stopStrs.concat(genConfig.stop);
    }

    // Stop condition 1: stop token; otherwise, append to `this.outputIds`
    if (this.stopTokens.includes(nextToken)) {
      this.stopTriggered = true;
      this.finishReason = "stop";
    }
    if (!this.stopTriggered) {
      this.outputIds.push(nextToken);
      // Update token appearance frequency
      const curFreq = this.appearedTokensFreq.get(nextToken);
      if (curFreq !== undefined) {
        this.appearedTokensFreq.set(nextToken, curFreq + 1);
      } else {
        this.appearedTokensFreq.set(nextToken, 1);
      }
    }

    // Stop condition 2: stop string; update `this.outputMessage` subsequently
    let outputMessage = this.tokenizer.decode(new Int32Array(this.outputIds));
    let stopPos = -1;
    for (const stopStr of stopStrs) {
      // Stop at the first stopStr we find
      stopPos = outputMessage.lastIndexOf(stopStr);
      if (stopPos != -1) {
        outputMessage = outputMessage.substring(0, stopPos);
        this.stopTriggered = true;
        this.finishReason = "stop";
        break;
      }
    }
    this.outputMessage = outputMessage;

    // Stop condition 3: exceed max_tokens
    if (this.outputIds.length >= max_tokens) {
      this.stopTriggered = true;
      this.finishReason = "length";
      log.info("Generation stopped due to exceeding max_tokens.");
    }

    // Stop condition 4: exceed KVCache's context window size
    if (
      this.slidingWindowSize == -1 &&
      this.filledKVCacheLength == this.contextWindowSize
    ) {
      this.stopTriggered = true;
      this.finishReason = "length";
      log.info("Generation stopped due to exceeding context_window_size.");
    }

    // Finally, modify conversation history if stopped
    if (this.stopTriggered) {
      if (!this.conversation.isTextCompletion) {
        this.conversation.finishReply(this.outputMessage);
      }
    }
  }

  /**
   * Given input tokens, return embeddings of them by calling embed kernel.
   *
   * @note precondition: inputTokens.length <= prefillChunkSize, since we take care of
   * chunking in `getChunkedPrefillInputData()`.
   */
  private getTokensEmbeddings(inputTokens: number[]): tvmjs.NDArray {
    this.tvm.beginScope();
    if (inputTokens.length > this.prefillChunkSize) {
      throw new Error(
        "Internal Error: getTokensEmbeddings input should be <= prefillChunkSize.",
      );
    }
    const inputData = this.tvm.empty(
      [inputTokens.length],
      "int32",
      this.device,
    );
    inputData.copyFrom(inputTokens);
    const embed: tvmjs.NDArray = this.tvm.detachFromCurrentScope(
      this.embed!(inputData, this.params),
    );
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embed); // tracked by scope of embedAndForward
    return embed;
  }

  /**
   * Embed an image input.
   */
  private async getImageEmbeddings(
    inputImage: ImageURL,
  ): Promise<tvmjs.NDArray> {
    this.tvm.beginScope();
    // 1. Transform ImageURL into image input in NDArray
    const url = inputImage.url;
    // url starting with `data:image` and `http` share the same loading method
    const imgData: ImageData = await getImageDataFromURL(url);
    const pixelValues: Uint8ClampedArray = getRGBArrayFromImageData(imgData);
    const pixelArray = this.tvm
      // .empty([imgData.height, imgData.width, 3], "uint8", this.device)
      .empty([imgData.height, imgData.width, 3], "uint32", this.device)
      .copyFrom(pixelValues)
      .view([1, imgData.height, imgData.width, 3]); // NHWC

    // 2. Call image embed kernel
    const embed: tvmjs.NDArray = this.tvm.detachFromCurrentScope(
      this.image_embed!(pixelArray, this.params),
    );
    if (embed.shape[0] !== IMAGE_EMBED_SIZE) {
      throw new Error(
        `InternalError: expect embed.shape[0] to be ${IMAGE_EMBED_SIZE}, ` +
          `but got ${embed.shape[0]}`,
      );
    }
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embed); // tracked by scope of embedAndForward
    return embed;
  }

  /**
   * Embed and forward input data, that can be either array of tokens, or an image.
   * This will increment `this.filledKVCacheLength`.
   *
   * @param inputData data to embed and forward
   * @param inputDataLen length of this inputData, should smaller than prefill chunk size.
   * @returns The logits returned by this forward as tvmjs.NDArray on GPU.
   *
   * @note Precondition: inputData's data length is smaller than prefill chunk size
   */
  private async embedAndForward(
    inputData: Array<Array<number> | ImageURL>,
    inputDataLen: number,
  ): Promise<tvmjs.NDArray> {
    if (inputDataLen > this.prefillChunkSize) {
      throw new Error(
        "InternalError: expect inputDataLen <= this.prefillChunkSize.",
      );
    }
    // TODO: we should combine string data to embed once, then rearrange the embeddings; currently
    // ["hi", imageUrl, "hi"] would call embed kernels 3 times, while 2 would suffice.

    // 1. Embed all inputData
    this.tvm.beginScope();
    const embeddings: tvmjs.NDArray[] = [];
    for (let i = 0; i < inputData.length; i++) {
      const data = inputData[i];
      if (Array.isArray(data)) {
        embeddings.push(this.getTokensEmbeddings(data));
      } else {
        embeddings.push(await this.getImageEmbeddings(data));
      }
    }

    // 2. Concatenate embeddings
    let allEmbeddings: tvmjs.NDArray;
    if (embeddings.length === 1) {
      allEmbeddings = embeddings[0];
    } else {
      allEmbeddings = this.tvm.concatEmbeddings(embeddings);
    }
    if (inputDataLen !== allEmbeddings.shape[0]) {
      throw new Error("InternalError: expect seqLen == allEmbeddings.shape[0]");
    }
    allEmbeddings = allEmbeddings.view([1].concat(allEmbeddings.shape));
    // TODO: Should we end this scope here and begin another scope? Will this dispose embeddings to
    // save RAM? We will detach allEmbeddings from this scope and attach to the next scope.

    // 3. Forward the concatenated embeddings
    const inputLenShape = this.tvm.makeShapeTuple([inputDataLen]);
    const seqIdsTuple = this.tvm.makeShapeTuple([0]);
    this.fKVCacheBeginForward!(this.kvCache, seqIdsTuple, inputLenShape);
    let retValue;
    if (inputDataLen > 1) {
      retValue = this.prefill(allEmbeddings, this.kvCache, this.params);
    } else {
      retValue = this.decoding(allEmbeddings, this.kvCache, this.params);
    }

    // Epilogue
    this.fKVCacheEndForward!(this.kvCache);
    this.filledKVCacheLength += inputDataLen;
    const logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  // NOTE: caller must call device.sync()
  private updateLogitsOnCPU(logits: tvmjs.NDArray): tvmjs.NDArray {
    if (this.logitsOnCPU == undefined) {
      this.logitsOnCPU = this.tvm.detachFromCurrentScope(
        this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu()),
      );
    } else {
      if (logits.shape[0] != this.logitsOnCPU.shape[0]) {
        throw Error("We expect the size of logits to remain unchanged");
      }
    }
    this.logitsOnCPU.copyFrom(logits);
    return this.logitsOnCPU;
  }

  private async sampleTokenFromLogits(
    logitsOnGPU: tvmjs.NDArray,
    genConfig?: GenerationConfig,
  ) {
    // 0. Get value of temperature, top_p, and various penalties, possibly overridden by genConfig
    // Also load other genConfig items like logit_bias. Consume all fields of `genConfig` here.
    function _hasValue(value: any): boolean {
      // if we use `if value` directly, `value` being 0 evaluates to false, violating semantics
      return value !== undefined && value !== null;
    }
    let temperature: number = this.config.temperature;
    let top_p: number = this.config.top_p;
    let repetition_penalty: number = this.config.repetition_penalty;
    let frequency_penalty: number = this.config.frequency_penalty;
    let presence_penalty: number = this.config.presence_penalty;
    let logit_bias: Record<string, number> | undefined = undefined;
    let logprobs: boolean | undefined = undefined;
    let top_logprobs: number | undefined = undefined;
    let response_format: ResponseFormat | undefined = undefined;

    if (genConfig !== undefined) {
      if (_hasValue(genConfig.temperature)) {
        temperature = genConfig.temperature!;
      }
      if (_hasValue(genConfig.top_p)) {
        top_p = genConfig.top_p!;
      }
      if (_hasValue(genConfig.repetition_penalty)) {
        repetition_penalty = genConfig.repetition_penalty!;
      }
      if (_hasValue(genConfig.frequency_penalty)) {
        frequency_penalty = genConfig.frequency_penalty!;
      }
      if (_hasValue(genConfig.presence_penalty)) {
        presence_penalty = genConfig.presence_penalty!;
      }
      // If only one of frequency or presence penatly is set, make the other one 0.0
      if (_hasValue(frequency_penalty) && !_hasValue(presence_penalty)) {
        presence_penalty = 0.0;
      }
      if (_hasValue(presence_penalty) && !_hasValue(frequency_penalty)) {
        frequency_penalty = 0.0;
      }
      if (_hasValue(genConfig.logit_bias)) {
        logit_bias = genConfig.logit_bias!;
      }
      if (_hasValue(genConfig.logprobs)) {
        logprobs = genConfig.logprobs!;
      }
      if (_hasValue(genConfig.top_logprobs)) {
        top_logprobs = genConfig.top_logprobs!;
      }
      if (_hasValue(genConfig.response_format)) {
        response_format = genConfig.response_format!;
      }
    }
    // Check range validity
    if (top_p <= 0 || top_p > 1) {
      throw new RangeError("top_p", 0, 1);
    }
    if (temperature < 0) {
      throw new MinValueError("temperature", 0);
    }
    if (repetition_penalty <= 0) {
      throw new MinValueError("repetition_penalty", 0);
    }
    if (
      frequency_penalty &&
      (frequency_penalty < -2.0 || frequency_penalty > 2.0)
    ) {
      throw new RangeError("frequency_penalty", -2.0, 2.0);
    }
    if (
      presence_penalty &&
      (presence_penalty < -2.0 || presence_penalty > 2.0)
    ) {
      throw new RangeError("presence_penalty", -2.0, 2.0);
    }

    // 0. Update logitsOnGPU with on-GPU grammar bitmasking
    if (response_format?.type === "json_object") {
      this.tvm.beginScope();
      if (this.grammarStateMatcher === undefined) {
        throw Error("Expect grammar state matcher to be initialized.");
      }
      const bitMaskOnCPU: Int32Array =
        await this.grammarStateMatcher.findNextTokenBitmask();
      if (bitMaskOnCPU.length !== this.bitmaskSize) {
        throw new Error(
          `InternalError: Expect grammar bitmask to be ` +
            `size ${this.bitmaskSize}, but got ${bitMaskOnCPU.length}.`,
        );
      }
      const bitMaskOnGPU = this.tvm
        .empty([1, this.bitmaskSize], "int32", this.device)
        .copyFrom(bitMaskOnCPU);
      const seqIdsArray = this.tvm
        .empty([1], "int32", this.device)
        .copyFrom([0]);
      this.fapplyBitmask(
        logitsOnGPU.view([1, this.fullVocabSize]),
        seqIdsArray,
        bitMaskOnGPU,
      );
      this.tvm.endScope();
    }

    // 1. Move logits to CPU
    this.tvm.beginScope();
    this.updateLogitsOnCPU(logitsOnGPU);
    this.tvm.endScope();
    await this.device.sync();

    if (this.logitsOnCPU == undefined) {
      throw Error("logits should be assigned");
    }

    // 2. Post process logits via logitProcessor and/or logit_bias
    if (this.logitProcessor !== undefined || _hasValue(logit_bias)) {
      let logitsOnCPUArray: Float32Array = <Float32Array>(
        this.logitsOnCPU.toArray()
      );
      const vocab_size = logitsOnCPUArray.length;
      if (this.logitProcessor !== undefined) {
        logitsOnCPUArray = this.logitProcessor.processLogits(logitsOnCPUArray);
      }
      if (_hasValue(logit_bias)) {
        for (const tokenID in logit_bias) {
          const curBias = logit_bias[tokenID];
          const curTokenID = parseInt(tokenID);
          if (curTokenID > vocab_size) {
            throw Error(
              "Token " +
                curTokenID +
                " in logit_bias exceeds vocab_size " +
                vocab_size,
            );
          }
          logitsOnCPUArray[curTokenID] += curBias;
        }
      }
      this.logitsOnCPU.copyFrom(logitsOnCPUArray);
    }

    // 3. Apply penalties to logits
    if (_hasValue(frequency_penalty) && _hasValue(presence_penalty)) {
      // 3.1. Use frequency and presence penalty
      this.tvm.beginScope();
      // Both `keys()` and `values()` are in insertion order.
      const appearedTokens = [...this.appearedTokensFreq.keys()];
      const appearedTokensFreqs = [...this.appearedTokensFreq.values()];
      const appeared_tokens_ndarray = this.tvm.empty(
        [1, appearedTokens.length],
        "int32",
        this.tvm.cpu(),
      );
      const appeared_tokens_freqs_ndarray = this.tvm.empty(
        [1, appearedTokensFreqs.length],
        "int32",
        this.tvm.cpu(),
      );
      appeared_tokens_ndarray.copyFrom(appearedTokens);
      appeared_tokens_freqs_ndarray.copyFrom(appearedTokensFreqs);
      this.tvm.applyPresenceAndFrequencyPenalty(
        this.logitsOnCPU,
        appeared_tokens_ndarray,
        appeared_tokens_freqs_ndarray,
        presence_penalty!,
        frequency_penalty!,
      );
      this.tvm.endScope();
    } else if (repetition_penalty != 1.0) {
      // 3.2. Use repetition penalty
      this.tvm.beginScope();
      const appearedTokens = [...this.appearedTokensFreq.keys()];
      const appeared_tokens_ndarray = this.tvm.empty(
        [1, appearedTokens.length],
        "int32",
        this.tvm.cpu(),
      );
      appeared_tokens_ndarray.copyFrom(appearedTokens);
      this.tvm.applyRepetitionPenalty(
        this.logitsOnCPU,
        appeared_tokens_ndarray,
        repetition_penalty,
      );
      this.tvm.endScope();
    }

    // 4. Sample token from logits
    // If logprobs, need the actual distribution via softmax, otherwise directly sample from logits
    let sampledToken: number;
    if (logprobs) {
      // Inplace transform logitsOnCPU to a distribution
      temperature = Math.max(1e-6, temperature); // to prevent division by zero
      this.tvm.applySoftmaxWithTemperature(this.logitsOnCPU, temperature);
      sampledToken = this.tvm.sampleTopPFromProb(this.logitsOnCPU, top_p);
      this.tokenLogprobArray.push(
        this.getTokenLogprob(sampledToken, top_logprobs!),
      );
    } else {
      // temperature being 0 is allowed here, equivalent to argmax
      sampledToken = this.tvm.sampleTopPFromLogits(
        this.logitsOnCPU,
        temperature,
        top_p,
      );
    }

    // 5. Update logit processor
    this.logitProcessor?.processSampledToken(sampledToken);

    // 6. Update grammar state matcher with new token
    if (response_format?.type === "json_object") {
      this.tvm.beginScope();
      if (this.grammarStateMatcher === undefined) {
        throw Error("Expect grammar state matcher to be initialized.");
      }
      const accepted = this.grammarStateMatcher.acceptToken(sampledToken);
      if (!accepted) {
        throw Error("Grammar state matcher rejected the newly sampled token.");
      }
      this.tvm.endScope();
    }

    return sampledToken;
  }

  /**
   * Return the an array of a mixture of token arrays and imageURLs (which cannot be represented
   * as tokens). Also return the number of tokens this represents.
   *
   * We first convert the Conversation into a prompt array to be prefilled. Then we encode the
   * text parts, leaving the imageURLs as it is.
   * Example prompts:
   * [
   *   "<|system|>\nSome system prompt\n",
   *   [
   *     "<|user|>\n",
   *     imageURL1,
   *     "\n",
   *     imageURL2,
   *     "\n",
   *     "Some user input<|end|>\n"
   *   ],
   * ]
   *
   * Expected output:
   * [
   *   token array for "<|system|>\nSome system prompt\n<|user|>\n",
   *   imageUrl1,
   *   token array for "\n",
   *   imageUrl2,
   *   token array for "\nSome user input<|end|>\n"
   */
  private getInputData(): [Array<Array<number> | ImageURL>, number] {
    const ret: Array<Array<number> | ImageURL> = [];
    let curTokens: Array<number> = [];
    let prompts: Array<string | Array<string | ImageURL>>;

    // 1. Get prompts
    if (this.conversation.isTextCompletion) {
      // 1.1. Non-conversation style
      if (this.filledKVCacheLength !== 0) {
        throw new TextCompletionExpectsKVEmptyError();
      }
      prompts = this.conversation.getPromptArrayTextCompletion();
    } else {
      // 1.2. Conversation style
      if (this.filledKVCacheLength === 0) {
        if (
          this.conversation.config.system_prefix_token_ids !== undefined &&
          this.conversation.config.system_prefix_token_ids !== null
        ) {
          curTokens = [...this.conversation.config.system_prefix_token_ids];
        }
        prompts = this.conversation.getPromptArray();
      } else {
        prompts = this.conversation.getPromptArrayLastRound();
      }
    }

    // 2. Encode all prompts. Iterate through each message in the prompt array, where each
    // prompt can either be a string, or an array of a mixture of string and ImageURLs.
    let numPromptTokens = 0;
    for (let i = 0; i < prompts.length; i++) {
      const curPrompt = prompts[i];
      if (typeof curPrompt === "string") {
        const encoded = this.tokenizer.encode(curPrompt);
        numPromptTokens += encoded.length;
        curTokens.push(...encoded);
      } else {
        for (let j = 0; j < curPrompt.length; j++) {
          const curPromptContent: string | ImageURL = curPrompt[j];
          if (typeof curPromptContent === "string") {
            const encoded = this.tokenizer.encode(curPromptContent);
            numPromptTokens += encoded.length;
            curTokens.push(...encoded);
          } else {
            // push curTokens to ret, push imageUrl, create a new curTokens
            ret.push([...curTokens]);
            ret.push(curPromptContent);
            numPromptTokens += IMAGE_EMBED_SIZE;
            curTokens = [];
          }
        }
      }
    }
    // Deal with last curTokens
    if (curTokens.length !== 0) {
      ret.push([...curTokens]);
    }

    // Check if input tokens exceed context window size
    if (
      this.slidingWindowSize == -1 && // There is no limit on contextWindowSize for sliding window
      numPromptTokens + this.filledKVCacheLength > this.contextWindowSize
    ) {
      throw new ContextWindowSizeExceededError(
        numPromptTokens,
        this.contextWindowSize,
      );
    }
    return [ret, numPromptTokens];
  }

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
  ): Promise<number> {
    const tstart = performance.now();
    this.tvm.beginScope();
    // 1. Chunk inputData if needed
    const inputData: Array<Array<number>> = [inputIds];
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize,
    );
    const chunks: Array<Array<number> | ImageURL>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    // 2. Prefill each chunk
    let logitsOnGPU: tvmjs.NDArray;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkLen = chunkLens[i];
      const prevFilledLen = this.filledKVCacheLength;
      logitsOnGPU = await this.embedAndForward(chunk, chunkLen);
      if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
        throw new Error(
          "Internal Error: filledKVCacheLength does not match expected value.",
        );
      }
    }

    // 3. Sample next token
    const nextToken = await this.sampleTokenFromLogits(logitsOnGPU!);
    this.tvm.endScope();

    // 4. Stats
    const tend = performance.now();
    if (isPrefill) {
      // We assume that if the input has more than 1 token
      this.prefillTotalTime += (tend - tstart) / 1e3;
      this.prefillTotalTokens += inputIds.length;
      this.curRoundPrefillTotalTokens += inputIds.length;
      this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;
    } else {
      this.decodingTotalTime += (tend - tstart) / 1e3;
      this.decodingTotalTokens += 1;
      this.curRoundDecodingTotalTokens += 1;
      this.curRoundDecodingTotalTime += (tend - tstart) / 1e3;
    }
    return nextToken;
  }

  /**
   * Based on `sampledToken` and `this.logitsOnCPU`, which becomes a distribution after
   * calling `this.tvm.applySoftmaxWithTemperature()`, generate `ChatCompletionTokenLogprob` and
   * update `this.tokenLogprobArray`.
   *
   * @param sampledToken The token ID sampled.
   * @param top_logprobs Number of top tokens to include; `top_logprobs` in `ChatCompletionRequest`.
   *
   * @return The `ChatCompletionTokenLogprob` for this single autoregressive step.
   */
  private getTokenLogprob(
    sampledToken: number,
    top_logprobs: number,
  ): ChatCompletionTokenLogprob {
    if (this.logitsOnCPU == undefined) {
      throw Error("logits should be assigned");
    }
    // Array of [token, prob] pairs, sorted with highest prob first.
    const logitsOnCPUArray = <Float32Array>this.logitsOnCPU.toArray();
    const topLogprobs = getTopProbs(top_logprobs!, logitsOnCPUArray);

    // Get entry for sampled token first
    const textEncoder = new TextEncoder();
    const tokenStr = this.tokenizer.decode(new Int32Array([sampledToken]));
    const bytes: Array<number> = Array.from(textEncoder.encode(tokenStr));
    const logprob = Math.log(logitsOnCPUArray[sampledToken]);

    // Populate `top_logprobs`
    const topLogprobArray: Array<TopLogprob> = [];
    for (let i = 0; i < top_logprobs; i++) {
      const tokenID_i = topLogprobs[i][0];
      const prob_i = topLogprobs[i][1];
      const tokenStr_i = this.tokenizer.decode(new Int32Array([tokenID_i]));
      topLogprobArray.push({
        token: tokenStr_i,
        bytes: Array.from(textEncoder.encode(tokenStr_i)) as Array<number>,
        logprob: Math.log(prob_i),
      } as TopLogprob);
    }

    return {
      token: tokenStr,
      bytes: bytes,
      logprob: logprob,
      top_logprobs: topLogprobArray,
    } as ChatCompletionTokenLogprob;
  }

  /**
   * Synchronize the device.
   */
  async sync(): Promise<void> {
    // Is it equivalent to this.tvm.sync()?
    await this.device.sync();
  }

  async evaluate() {
    // run a canonical evaluation of the flow
    this.resetKVCache();
    this.filledKVCacheLength = 0;

    const testPrompt = "The capital of Canada is";
    const ids = await this.tokenizer.encode(testPrompt);
    const tokens = Array.from(ids);
    tokens.unshift(this.bosTokenId);
    if (tokens.length == 0) {
      throw Error("empty token");
    }

    this.tvm.beginScope();
    const prefillChunk: Array<Array<number>> = [tokens];
    const prefillChunkLen = tokens.length;
    const prefillStart = performance.now();
    await this.embedAndForward(prefillChunk, prefillChunkLen);
    this.tvm.endScope();
    await this.device.sync();

    const decodingStart = performance.now();

    this.tvm.beginScope();
    const decodeChunk: Array<Array<number>> = [[6234]];
    const decodeChunkLen = 1;
    const logitsOnCPU = this.updateLogitsOnCPU(
      await this.embedAndForward(decodeChunk, decodeChunkLen),
    );
    await this.device.sync();
    this.tvm.endScope();

    const decodingEnd = performance.now();
    const msg =
      `prefill-time=${((decodingStart - prefillStart) / 1000).toFixed(4)} sec` +
      `decoding-time=${((decodingEnd - decodingStart) / 1000).toFixed(4)} sec`;

    // simply log tokens for eyeballing.
    log.info("Logits:");
    log.info(logitsOnCPU.toArray());
    log.info(msg);
  }
}
