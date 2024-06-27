/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable no-prototype-builtins */
import * as tvmjs from "tvmjs";
import log from "loglevel";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig, GenerationConfig, Role } from "./config";
import { getConversation, Conversation } from "./conversation";
import { LogitProcessor } from "./types";
import { getTokenTableFromTokenizer, getTopProbs } from "./support";
import {
  ChatCompletionFinishReason,
  ChatCompletionTokenLogprob,
  TopLogprob,
  ResponseFormat,
} from "./openai_api_protocols/index";
import { BNFGrammar, GrammarFactory, GrammarStateMatcher } from "./grammar";
import {
  AttentionSinkSizeError,
  ContextWindowSizeExceededError,
  MinValueError,
  RangeError,
  WindowSizeConfigurationError,
  WindowSizeSpecificationError,
} from "./error";

export class LLMChatPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private decoding: tvmjs.PackedFunc;
  private embed: tvmjs.PackedFunc;
  private fapplyBitmask: tvmjs.PackedFunc;
  private fpostProcessTokenTable: tvmjs.PackedFunc;
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
  // A factory to instantiate and maintain the BNF grammars and grammar state matchers.
  private grammarFactory: GrammarFactory;
  // A grammar state matcher for this current round if response_format is set. Reinitialized upon
  // each step regardless of whether the chat is multi-round or not.
  private grammarStateMatcher?: GrammarStateMatcher = undefined;
  // The current schema used for grammarStateMatcher; if undefined, grammarStateMatcher is simply
  // using JSON mode. We use this field to determine whether we re-initiate a GrammarStateMatcher
  // or simply reset the state during each round (i.e. during prefillStep).
  private schema?: string = undefined;
  // A string list of tokens ordered by their token id, post-processed. Once initialized, will not
  // be reinitialized since `this.tokenizer` does not change throughout the lifetime of LLMChatPipeline.
  private tokenTable?: tvmjs.TVMObject = undefined;
  private bitmaskSize: number;
  private vocabSize: number;
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
    this.grammarFactory = new GrammarFactory(tvm);
    this.vocabSize = this.tokenizer.getVocabSize();
    this.bitmaskSize = Math.ceil(this.vocabSize / 32);

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
    this.fpostProcessTokenTable = this.tvm.detachFromCurrentScope(
      tvm.getGlobalFunc("mlc.tokenizers.PostProcessTokenTable"),
    );

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
    this.grammarFactory.dispose();
    this.grammarStateMatcher?.dispose();
    this.params.dispose();
    this.decoding.dispose();
    this.prefill.dispose();
    this.vm.dispose();
    this.kvCache.dispose();
    this.fclearKVCaches.dispose();
    this.logitsOnCPU?.dispose();
    this.tvm.dispose();
    this.tokenizer.dispose();
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
    inp_role_str?: string,
    genConfig?: GenerationConfig,
  ): Promise<void> {
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }

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

    // initialize
    conversation.appendMessage(Role.user, inp, inp_role_str);
    conversation.appendReplyHeader(Role.assistant);
    const promptTokens = this.getInputTokens();

    const tstart = performance.now();
    this.tvm.beginScope();

    let newSeqLen = this.filledKVCacheLength;
    const tokenLen = promptTokens.length;
    let logits = this.tvm.empty([1, 1], "int32", this.device); // Dummy value to avoid type error
    // Use prefill chunking regardless whether we use SWA (see Mistral paper figure 3)
    for (let begin = 0; begin < tokenLen; begin += this.prefillChunkSize) {
      const end = Math.min(tokenLen, begin + this.prefillChunkSize);
      const chunk = promptTokens.slice(begin, end);
      const inputData = this.tvm.empty([chunk.length], "int32", this.device);
      inputData.copyFrom(chunk);
      newSeqLen += chunk.length;
      logits = this.tvm.detachFromCurrentScope(this.forward(inputData));
    }
    if (newSeqLen != this.filledKVCacheLength + tokenLen) {
      throw Error("Expect chunking process all tokens.");
    }
    this.filledKVCacheLength = newSeqLen;

    // Instantiate grammar state matcher according to generation config
    if (genConfig?.response_format?.type === "json_object") {
      const curSchema = genConfig.response_format.schema;
      if (curSchema === this.schema && this.grammarStateMatcher) {
        // If we did not change the schema and have instantiated a GrammarStateMatcher, we reuse it.
        this.grammarFactory.resetState(this.grammarStateMatcher);
      } else {
        // Else dispose current grammarStateMatcher, reinitialize, and update this.schema.
        if (this.grammarStateMatcher) {
          this.grammarStateMatcher.dispose();
        }
        if (this.tokenTable === undefined) {
          const rawTokenTable = getTokenTableFromTokenizer(this.tokenizer);
          // Post process entire table
          this.tokenTable = this.fpostProcessTokenTable(
            rawTokenTable,
            this.token_postproc_method,
          );
        }
        const grammar: BNFGrammar =
          curSchema === undefined
            ? this.grammarFactory.getBNFGrammarOfJSON()
            : this.grammarFactory.getBNFGrammarFromSchema(curSchema);
        this.grammarStateMatcher = this.tvm.detachFromCurrentScope(
          this.grammarFactory.getGrammarStateMatcherFromTokenTable(
            grammar,
            this.tokenTable!,
          ),
        );
        this.schema = curSchema;
      }
    }

    this.tvm.endScope();

    const nextToken = await this.sampleTokenFromLogits(logits, genConfig);
    logits.dispose();
    const tend = performance.now();

    this.prefillTotalTime += (tend - tstart) / 1e3;
    this.prefillTotalTokens += promptTokens.length;
    this.curRoundPrefillTotalTokens += promptTokens.length;
    this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;

    this.processNextToken(nextToken, genConfig);
  }

  async decodeStep(genConfig?: GenerationConfig): Promise<void> {
    if (this.stopTriggered) {
      throw Error("Cannot run decode when stopped");
    }

    const tstart = performance.now();

    this.tvm.beginScope();
    const inputData = this.tvm.empty([1], "int32", this.device);
    inputData.copyFrom(this.outputIds.slice(this.outputIds.length - 1));

    const logits = this.tvm.detachFromCurrentScope(this.forward(inputData));
    this.filledKVCacheLength += 1;
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
    this.conversation.finishReply(this.outputMessage);
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
      this.conversation.finishReply(this.outputMessage);
    }
  }

  private forward(inputs: tvmjs.NDArray): tvmjs.NDArray {
    this.tvm.beginScope();
    let retValue;
    const seqLen = inputs.shape[0]; // Num input tokens
    const seqIdsTuple = this.tvm.makeShapeTuple([0]);
    const inputLenShape = this.tvm.makeShapeTuple([seqLen]);
    this.fKVCacheBeginForward!(this.kvCache, seqIdsTuple, inputLenShape);
    let embed = this.embed!(inputs, this.params);
    embed = embed.view([1].concat(embed.shape)); // Reshape to [1, seqLen, hiddenSize]
    if (seqLen > 1) {
      retValue = this.prefill(embed, this.kvCache, this.params);
    } else {
      retValue = this.decoding(embed, this.kvCache, this.params);
    }
    this.fKVCacheEndForward!(this.kvCache);
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
    let frequency_penalty: number | undefined = undefined;
    let presence_penalty: number | undefined = undefined;
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
      // TODO(Charlie): Do we detach from current scope here for bitmask?
      const bitMaskOnCPU = this.grammarFactory.findNextTokenBitmask(
        this.grammarStateMatcher,
      ) as unknown as tvmjs.NDArray;
      const bitMaskOnGPU = this.tvm
        .empty([1, this.bitmaskSize], "int32", this.device)
        .copyFrom(bitMaskOnCPU);
      const seqIdsArray = this.tvm
        .empty([1], "int32", this.device)
        .copyFrom([0]);
      this.fapplyBitmask(
        logitsOnGPU.view([1, this.vocabSize]),
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
      const accepted = this.grammarFactory.acceptToken(
        this.grammarStateMatcher,
        sampledToken,
      );
      if (!accepted) {
        throw Error("Grammar state matcher rejected the newly sampled token.");
      }
      this.tvm.endScope();
    }

    return sampledToken;
  }

  private getInputTokens(): Array<number> {
    let tokens: Array<number> = [];
    let prompts: string[];
    // beginning of the conversation
    if (this.filledKVCacheLength === 0) {
      if (
        this.conversation.config.system_prefix_token_ids !== undefined &&
        this.conversation.config.system_prefix_token_ids !== null
      ) {
        tokens = [...this.conversation.config.system_prefix_token_ids];
      }
      prompts = this.conversation.getPromptArray();
    } else {
      prompts = this.conversation.getPrompArrayLastRound();
    }

    // Encode all prompts
    let numPromptTokens = 0;
    for (let i = 0; i < prompts.length; i++) {
      const encoded = this.tokenizer.encode(prompts[i]);
      numPromptTokens += encoded.length;
      tokens.push(...encoded);
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
    return tokens;
  }

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
  ): Promise<number> {
    // 1. Convert input to NDArray
    const tstart = performance.now();
    this.tvm.beginScope();
    const inputData = this.tvm.empty([inputIds.length], "int32", this.device);
    inputData.copyFrom(inputIds);

    // 2. Forward tokens and get logits
    const logitsOnGPU: tvmjs.NDArray = this.forward(inputData);
    const nextToken = await this.sampleTokenFromLogits(logitsOnGPU);
    this.tvm.endScope();

    // 3. Stats
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
    const inputData = this.tvm.empty([tokens.length], "int32", this.device);
    inputData.copyFrom(tokens);
    const prefillStart = performance.now();
    this.forward(inputData);
    this.tvm.endScope();
    await this.device.sync();

    const decodingStart = performance.now();

    this.tvm.beginScope();
    const firstSampleToken = this.tvm
      .empty([1], "int32", this.device)
      .copyFrom([6234]);
    const logitsOnCPU = this.updateLogitsOnCPU(this.forward(firstSampleToken));
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
