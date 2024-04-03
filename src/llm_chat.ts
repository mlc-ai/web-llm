/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable no-prototype-builtins */
import * as tvmjs from "tvmjs";
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
} from "./openai_api_protocols/index"
import { GrammarFactory, GrammarStateMatcher } from "./grammar";

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
  private maxWindowLength = -1;
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
  // same as `prefillTotalTokens` and `decodingTotalTokens`, but reset at every `prefillStep()`
  private curRoundDecodingTotalTokens = 0;
  private curRoundPrefillTotalTokens = 0;

  // logger
  private logger = console.log;

  // LogitProcessor
  private logitProcessor?: LogitProcessor = undefined;

  // Grammar-related
  // A factory to instantiate and maintain the BNF grammars and grammar state matchers.
  private grammarFactory: GrammarFactory;
  // A grammar state matcher for this current round (reinitialized upon each prefillStep) if
  // response_format is set.
  private grammarStateMatcher?: GrammarStateMatcher = undefined;
  // A string list of tokens ordered by their token id. Once initialized, will not be reinitialized
  // since `this.tokenizer` does not change throughout the lifetime of LLMChatPipeline.
  private tokenTable?: string[] = undefined;
  private bitmaskSize: number;
  private vocabSize: number;

  constructor(tvm: tvmjs.Instance, tokenizer: Tokenizer, config: ChatConfig, logitProcessor?: LogitProcessor) {
    // 0. Setting attributes
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;
    this.logitProcessor = logitProcessor;
    this.grammarFactory = new GrammarFactory(tvm);
    this.vocabSize = this.tokenizer.getVocabSize();
    this.bitmaskSize = Math.ceil(this.vocabSize / 32);

    this.conversation = getConversation(config.conv_template, config.conv_config);
    this.stopStr = this.conversation.getStopStr();
    this.stopTokens = this.conversation.getStopTokens();
    if (config.bos_token_id !== undefined) {
      this.bosTokenId = config.bos_token_id;
    }

    this.device = this.tvm.webgpu();

    // 1. Create VM and get the core functions
    tvm.beginScope();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );
    this.prefill = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("prefill")
    );
    this.embed = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("embed")
    );
    this.decoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("decode")
    );
    this.fapplyBitmask = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("apply_bitmask_inplace")
    )

    // 2. Get json stored in the vm's metadata function
    const fgetMetadata = this.vm.getFunction("_metadata");
    const ret_value = fgetMetadata();
    const metadataStr = this.tvm.detachFromCurrentScope(ret_value).toString();
    const metadata = JSON.parse(metadataStr);

    // 3. Load parameters by name
    const paramNames: string[] = [];
    metadata.params.forEach((param: any) => { paramNames.push(param.name) });
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCacheByName(paramNames)
    );

    // 4. Read in compilation configurations from metadata
    this.prefillChunkSize = metadata.prefill_chunk_size;
    this.logger("Using prefillChunkSize: ", this.prefillChunkSize);
    if (this.prefillChunkSize <= 0) {
      throw Error("Prefill chunk size needs to be positive.");
    }
    // Only use one of slidingWindowSize and maxWindowLength
    if (metadata.hasOwnProperty("sliding_window_size") && metadata.sliding_window_size != -1) {
      this.slidingWindowSize = metadata.sliding_window_size;
      this.logger("Using slidingWindowSize: ", this.slidingWindowSize);
      // Parse attention sink size
      if (metadata.hasOwnProperty("attention_sink_size") && metadata.attention_sink_size >= 0) {
        this.attentionSinkSize = metadata.attention_sink_size;
        this.logger("Using attentionSinkSize: ", this.attentionSinkSize);
      } else {
        throw Error(
          "Need to specify non-negative attention_sink_size if using sliding window. " +
          "Consider re-compiling the model with the most recent mlc-llm. " +
          "Use `attention_sink_size=0` for default sliding window."
        );
      }
    } else if (metadata.hasOwnProperty("context_window_size") && metadata.context_window_size != -1) {
      this.maxWindowLength = metadata.context_window_size;
      this.logger("Using maxWindowLength: ", this.maxWindowLength);
    } else {
      throw Error("Need to specify either sliding window size or max window size.");
    }

    // 5. Create cache
    // Load cache functions and instantiate KVCache
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_clear")
    );
    this.fKVCacheAddSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_add_sequence")
    );
    this.fKVCacheRemoveSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_remove_sequence")
    );
    this.fKVCacheBeginForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_begin_forward")
    );
    this.fKVCacheEndForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_end_forward")
    );
    this.fKVCacheEnableSlidingWindowForSeq = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.attention_kv_cache_enable_sliding_window_for_seq")
    );

    // Create PagedKVCache; we do not expose KVCache config for now
    const fcreateCache = this.vm.getFunction("create_tir_paged_kv_cache");
    const defaultPageSize = 16;
    const defaultMaxNumSequence = 1;
    const maxTotalSeqLen =
      this.slidingWindowSize != -1 ? this.slidingWindowSize : this.maxWindowLength;
    this.kvCache = this.tvm.detachFromCurrentScope(fcreateCache(
      this.tvm.makeShapeTuple([defaultMaxNumSequence]),  // max_num_sequence
      this.tvm.makeShapeTuple([maxTotalSeqLen]),  // max_total_sequence_length
      this.tvm.makeShapeTuple([this.prefillChunkSize]),  // prefill_chunk_size
      this.tvm.makeShapeTuple([defaultPageSize]),  // page_size, hard coded for now
      this.tvm.makeShapeTuple([this.slidingWindowSize != -1 ? 1 : 0]),
    ));

    this.filledKVCacheLength = 0;
    this.resetChat();  // especially needed for PagedKVCache as we need to call fKVCacheAddSequence
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
        new tvmjs.Scalar(this.attentionSinkSize, "int32")
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
   * @returns Runtime stats information.
   */
  runtimeStatsText(): string {
    return (
      `prefill: ${(this.prefillTotalTokens / this.prefillTotalTime).toFixed(4)} tokens/sec, ` +
      `decoding: ${(this.decodingTotalTokens / this.decodingTotalTime).toFixed(4)} tokens/sec`
    )
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
  async prefillStep(inp: string, inp_role_str?: string, genConfig?: GenerationConfig): Promise<void> {
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
    this.stopTriggered = false;
    const conversation = this.conversation;

    // initialize
    conversation.appendMessage(Role.user, inp, inp_role_str);
    conversation.appendReplyHeader(Role.assistant);
    const promptTokens = this.getInputTokens(genConfig);

    const tstart = performance.now();
    this.tvm.beginScope();

    let newSeqLen = this.filledKVCacheLength;
    const tokenLen = promptTokens.length;
    let logits = this.tvm.empty([1, 1], "int32", this.device);  // Dummy value to avoid type error
    // Use prefill chunking regardless whether we use SWA (see Mistral paper figure 3)
    for (let begin = 0; begin < tokenLen; begin += this.prefillChunkSize) {
      const end = Math.min(tokenLen, begin + this.prefillChunkSize);
      const chunk = promptTokens.slice(begin, end);
      const inputData = this.tvm.empty([chunk.length], "int32", this.device);
      inputData.copyFrom(chunk);
      newSeqLen += chunk.length;
      logits = this.tvm.detachFromCurrentScope(
        this.forward(inputData)
      );
    }
    if (newSeqLen != this.filledKVCacheLength + tokenLen) {
      throw Error("Expect chunking process all tokens.")
    }
    this.filledKVCacheLength = newSeqLen;

    // Instantiate grammar state matcher according to generation config
    if (this.grammarStateMatcher) {
      this.grammarStateMatcher.dispose();
    }
    if (genConfig?.response_format?.type === "json_object") {
      // Currently only support JSON grammar
      const JSONgrammar = this.grammarFactory.getBNFGrammarOfJSON();
      this.tokenTable = getTokenTableFromTokenizer(this.tokenizer);
      this.grammarStateMatcher = this.tvm.detachFromCurrentScope(
        this.grammarFactory.getGrammarStateMatcherFromTokenTable(JSONgrammar, this.tokenTable)
      );
    }

    this.tvm.endScope();

    const nextToken = await this.sampleTokenFromLogits(logits, genConfig);
    logits.dispose();
    const tend = performance.now();

    this.prefillTotalTime += (tend - tstart) / 1e3;
    this.prefillTotalTokens += promptTokens.length;
    this.curRoundPrefillTotalTokens += promptTokens.length;

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

    const logits = this.tvm.detachFromCurrentScope(
      this.forward(inputData)
    );
    this.filledKVCacheLength += 1;
    this.tvm.endScope();

    // sample from logits
    const nextToken = await this.sampleTokenFromLogits(logits, genConfig);
    logits.dispose();
    const tend = performance.now();

    this.decodingTotalTime += (tend - tstart) / 1e3;
    this.decodingTotalTokens += 1;
    this.curRoundDecodingTotalTokens += 1;

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
  private processNextToken(nextToken: number, genConfig?: GenerationConfig): void {
    if (this.stopTriggered) {
      throw Error("Cannot call process when it is stoppped");
    }

    // Get max_gen_len and stopStrs, possibly overridden by genConfig for this round
    let max_gen_len = this.config.max_gen_len;
    if (genConfig !== undefined && genConfig.max_gen_len) {
      max_gen_len = genConfig.max_gen_len;
    }
    if (max_gen_len <= 0) {
      throw new Error("`max_gen_len` should be greater than 0.")
    }
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

    // Stop condition 3: exceed max_gen_len
    if (this.outputIds.length >= max_gen_len) {
      this.stopTriggered = true;
      this.finishReason = "length";
    }

    // Finally, modify conversation history if stopped
    if (this.stopTriggered) {
      this.conversation.finishReply(this.outputMessage);
    }
  }

  private forward(inputs: tvmjs.NDArray): tvmjs.NDArray {
    this.tvm.beginScope();
    let retValue;
    const seqLen = inputs.shape[0];  // Num input tokens
    const seqIdsTuple = this.tvm.makeShapeTuple([0]);
    const inputLenShape = this.tvm.makeShapeTuple([seqLen]);
    this.fKVCacheBeginForward!(this.kvCache, seqIdsTuple, inputLenShape);
    let embed = this.embed!(inputs, this.params);
    embed = embed.view([1].concat(embed.shape));  // Reshape to [1, seqLen, hiddenSize]
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
        this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu())
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
      if (_hasValue(genConfig.temperature)) { temperature = genConfig.temperature!; }
      if (_hasValue(genConfig.top_p)) { top_p = genConfig.top_p!; }
      if (_hasValue(genConfig.repetition_penalty)) { repetition_penalty = genConfig.repetition_penalty!; }
      if (_hasValue(genConfig.frequency_penalty)) { frequency_penalty = genConfig.frequency_penalty!; }
      if (_hasValue(genConfig.presence_penalty)) { presence_penalty = genConfig.presence_penalty!; }
      // If only one of frequency or presence penatly is set, make the other one 0.0
      if (_hasValue(frequency_penalty) && !_hasValue(presence_penalty)) { presence_penalty = 0.0; }
      if (_hasValue(presence_penalty) && !_hasValue(frequency_penalty)) { frequency_penalty = 0.0; }
      if (_hasValue(genConfig.logit_bias)) { logit_bias = genConfig.logit_bias!; }
      if (_hasValue(genConfig.logprobs)) { logprobs = genConfig.logprobs!; }
      if (_hasValue(genConfig.top_logprobs)) { top_logprobs = genConfig.top_logprobs!; }
      if (_hasValue(genConfig.response_format)) { response_format = genConfig.response_format!; }
    }
    // Check range validity
    if (top_p <= 0 || top_p > 1) { throw new Error("Make sure 0 < `top_p` <= 1."); }
    if (temperature < 0) { throw new Error("Make sure `temperature` >= 0."); }
    if (repetition_penalty <= 0) { throw new Error("Make sure `repetition_penalty` > 0."); }
    if (frequency_penalty && (frequency_penalty < -2.0 || frequency_penalty > 2.0)) {
      throw new Error("`frequency_penalty` should be between -2.0 and 2.0.");
    }
    if (presence_penalty && (presence_penalty < -2.0 || presence_penalty > 2.0)) {
      throw new Error("`presence_penalty` should be between -2.0 and 2.0.");
    }

    // 0. Update logitsOnGPU with on-GPU grammar bitmasking
    if (response_format?.type === "json_object") {
      this.tvm.beginScope();
      if (this.grammarStateMatcher === undefined) {
        throw Error("Expect grammar state matcher to be initialized.");
      }
      // TODO(Charlie): Do we detach from current scope here for bitmask?
      const bitMaskOnCPU = this.grammarFactory.findNextTokenBitmask(
        this.grammarStateMatcher) as unknown as tvmjs.NDArray;
      const bitMaskOnGPU = this.tvm.empty([1, this.bitmaskSize], "int32",
        this.device).copyFrom(bitMaskOnCPU);
      const seqIdsArray = this.tvm.empty([1], "int32", this.device).copyFrom([0]);
      this.fapplyBitmask(logitsOnGPU.view([1, this.vocabSize]), seqIdsArray, bitMaskOnGPU);
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
      let logitsOnCPUArray: Float32Array = <Float32Array>(this.logitsOnCPU.toArray());
      const vocab_size = logitsOnCPUArray.length;
      if (this.logitProcessor !== undefined) {
        logitsOnCPUArray = this.logitProcessor.processLogits(logitsOnCPUArray);
      }
      if (_hasValue(logit_bias)) {
        for (const tokenID in logit_bias) {
          const curBias = logit_bias[tokenID];
          const curTokenID = parseInt(tokenID);
          if (curTokenID > vocab_size) {
            throw Error("Token " + curTokenID + " in logit_bias exceeds vocab_size " + vocab_size);
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
        [1, appearedTokens.length], "int32", this.tvm.cpu());
      const appeared_tokens_freqs_ndarray = this.tvm.empty(
        [1, appearedTokensFreqs.length], "int32", this.tvm.cpu());
      appeared_tokens_ndarray.copyFrom(appearedTokens);
      appeared_tokens_freqs_ndarray.copyFrom(appearedTokensFreqs);
      this.tvm.applyPresenceAndFrequencyPenalty(
        this.logitsOnCPU,
        appeared_tokens_ndarray,
        appeared_tokens_freqs_ndarray,
        presence_penalty!,
        frequency_penalty!
      );
      this.tvm.endScope();
    } else if (repetition_penalty != 1.0) {
      // 3.2. Use repetition penalty
      this.tvm.beginScope();
      const appearedTokens = [...this.appearedTokensFreq.keys()];
      const appeared_tokens_ndarray = this.tvm.empty(
        [1, appearedTokens.length], "int32", this.tvm.cpu());
      appeared_tokens_ndarray.copyFrom(appearedTokens);
      this.tvm.applyRepetitionPenalty(
        this.logitsOnCPU, appeared_tokens_ndarray, repetition_penalty);
      this.tvm.endScope();
    }

    // 4. Sample token from logits
    // If logprobs, need the actual distribution via softmax, otherwise directly sample from logits
    let sampledToken: number;
    if (logprobs) {
      // Inplace transform logitsOnCPU to a distribution
      temperature = Math.max(1e-6, temperature);  // to prevent division by zero
      this.tvm.applySoftmaxWithTemperature(this.logitsOnCPU, temperature);
      sampledToken = this.tvm.sampleTopPFromProb(this.logitsOnCPU, top_p);
      this.tokenLogprobArray.push(this.getTokenLogprob(sampledToken, top_logprobs!));
    } else {
      // temperature being 0 is allowed here, equivalent to argmax
      sampledToken = this.tvm.sampleTopPFromLogits(this.logitsOnCPU, temperature, top_p);
    }

    // 5. Update logit processor
    this.logitProcessor?.processSampledToken(sampledToken);

    // 6. Update grammar state matcher with new token
    if (response_format?.type === "json_object") {
      this.tvm.beginScope();
      if (this.grammarStateMatcher === undefined) {
        throw Error("Expect grammar state matcher to be initialized.");
      }
      const accepted = this.grammarFactory.acceptToken(this.grammarStateMatcher, sampledToken);
      if (!accepted) {
        throw Error("Grammar state matcher rejected the newly sampled token.");
      }
      this.tvm.endScope();
    }

    return sampledToken;
  }

  private getInputTokens(genConfig?: GenerationConfig): Array<number> {
    // Get mean_gen_len and max_gen_len, possibly overriden by genConfig
    let mean_gen_len = this.config.mean_gen_len;
    let shift_fill_factor = this.config.shift_fill_factor;
    if (genConfig !== undefined) {
      if (genConfig.mean_gen_len !== undefined && genConfig.mean_gen_len !== null) {
        mean_gen_len = genConfig.mean_gen_len;
      }
      if (genConfig.shift_fill_factor !== undefined && genConfig.shift_fill_factor !== null) {
        shift_fill_factor = genConfig.shift_fill_factor;
      }
    }
    // Check range validity
    if (shift_fill_factor <= 0 || shift_fill_factor > 1) {
      throw new Error("Make sure 0 < `shift_fill_factor` <= 1.");
    }
    if (mean_gen_len <= 0) {
      throw new Error("`mean_gen_len` should be greater than zero.");
    }

    let tokens: Array<number> = [];
    let prompts;
    // beginning of the conversation
    if (this.filledKVCacheLength === 0) {
      if (this.conversation.config.system_prefix_token_ids !== undefined) {
        tokens = [...this.conversation.config.system_prefix_token_ids];
      }
      prompts = this.conversation.getPromptArray();
    } else {
      prompts = this.conversation.getPrompArrayLastRound();
    }
    // keep system prompt when possible
    tokens.push(...this.tokenizer.encode(prompts[0]));

    let ctxLength = tokens.length;
    let context = [];

    // detect if we go out of the range
    let needShiftWindow = false;
    for (let i = prompts.length - 1; i > 0; --i) {
      const encoded = this.tokenizer.encode(prompts[i]);
      ctxLength += encoded.length;
      if (this.slidingWindowSize == -1 &&  // There is no maxWindowLength if we use sliding window
        this.filledKVCacheLength + ctxLength + mean_gen_len >= this.maxWindowLength) {
        needShiftWindow = true;
        break;
      }
      context.unshift(encoded);
    }
    if (!needShiftWindow) {
      for (const ctx of context) {
        tokens.push(...ctx);
      }
      return tokens;
    }

    // Code starting below should not be reached when using sliding window.
    if (this.slidingWindowSize != -1) {
      throw Error("Should not shift window when using sliding window attention.");
    }

    // need shift window and re-encode
    this.logger("need shift window")
    this.filledKVCacheLength = 0;
    this.resetKVCache();

    // abandon all tokens we collected
    if (this.conversation.config.system_prefix_token_ids !== undefined) {
      tokens = [...this.conversation.config.system_prefix_token_ids];
    } else {
      tokens = [];
    }

    const all_prompts = this.conversation.getPromptArray();
    tokens.push(...this.tokenizer.encode(all_prompts[0]));
    context = [];
    ctxLength = tokens.length;
    // only keep shift_fill_factor of the window context
    for (let i = all_prompts.length - 1; i > 0; --i) {
      const encoded = this.tokenizer.encode(all_prompts[i]);
      ctxLength += encoded.length;
      if (ctxLength >= shift_fill_factor * this.maxWindowLength && i + 2 < all_prompts.length) {
        break;
      }
      context.unshift(encoded);
    }
    for (const ctx of context) {
      tokens.push(...ctx);
    }
    if (tokens.length + mean_gen_len >= this.maxWindowLength) {
      throw Error("Exceed max window length curr=" + tokens.length);
    }
    return tokens;
  }

  async forwardTokensAndSample(inputIds: Array<number>, isPrefill: boolean): Promise<number> {
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
    } else {
      this.decodingTotalTime += (tend - tstart) / 1e3;
      this.decodingTotalTokens += 1;
      this.curRoundDecodingTotalTokens += 1;
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
  private getTokenLogprob(sampledToken: number, top_logprobs: number): ChatCompletionTokenLogprob {
    if (this.logitsOnCPU == undefined) {
      throw Error("logits should be assigned");
    }
    // Array of [token, prob] pairs, sorted with highest prob first.
    const logitsOnCPUArray = <Float32Array>(this.logitsOnCPU.toArray())
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
      top_logprobs: topLogprobArray
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
    const firstSampleToken = this.tvm.empty([1], "int32", this.device).copyFrom([6234]);
    const logitsOnCPU = this.updateLogitsOnCPU(this.forward(firstSampleToken));
    await this.device.sync();
    this.tvm.endScope();

    const decodingEnd = performance.now();
    const msg = (
      `prefill-time=${((decodingStart - prefillStart) / 1000).toFixed(4)} sec` +
      `decoding-time=${((decodingEnd - decodingStart) / 1000).toFixed(4)} sec`
    );

    // simply log tokens for eyeballing.
    console.log("Logits:");
    console.log(logitsOnCPU.toArray());
    console.log(msg);
  }
}
