/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable no-prototype-builtins */
import * as tvmjs from "tvmjs";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig, GenerationConfig } from "./config";
import { getConversation, Conversation } from "./conversation";
import { LogitProcessor } from "./types";
import { getTopProbs } from "./support";
import {
  ChatCompletionFinishReason,
  ChatCompletionTokenLogprob,
  TopLogprob,
} from "./openai_api_protocols/index"

export class LLMChatPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private decoding: tvmjs.PackedFunc;
  private fclearKVCaches: tvmjs.PackedFunc;
  // Functions for PagedKVCache only
  private embed?: tvmjs.PackedFunc = undefined;
  private fKVCacheAddSequence?: tvmjs.PackedFunc = undefined;
  private fKVCacheRemoveSequence?: tvmjs.PackedFunc = undefined;
  private fKVCacheBeginForward?: tvmjs.PackedFunc = undefined;
  private fKVCacheEndForward?: tvmjs.PackedFunc = undefined;

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
  private stopStr: string;
  private stopTokens: Array<number>;

  // states
  private outputMessage = "";
  private outputIds: Array<number> = [];
  private stopTriggered = false;
  private finishReason: ChatCompletionFinishReason | undefined = undefined;
  // frequency of appeared token ids till now (refresh after PrefillStep); token_id mapped to freq
  private appearedTokensFreq = new Map<number, number>();
  private conversation: Conversation;
  // Whether sink is in action
  private sinkTriggered = false;
  // sliding window cache offset (Next position to be overridden on the rolling kv cache.)
  private slidingWindowCacheOffset = 0;
  // Whether we are using PagedKVCache (eventually this will become default)
  private usePagedKVCache = false;
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

  constructor(tvm: tvmjs.Instance, tokenizer: Tokenizer, config: ChatConfig, logitProcessor?: LogitProcessor) {
    // 0. Setting attributes
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;
    this.logitProcessor = logitProcessor;

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
    try {
      // We expect to find `embed` if we usePagedKVCache
      this.embed = this.tvm.detachFromCurrentScope(
        this.vm.getFunction("embed")
      );
    } catch {
      // Do nothing
    }
    this.decoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("decode")
    );

    // 2. Get json stored in the vm's metadata function
    let fgetMetadata;
    let useSLIM = false;  // SLIM is the new workflow
    try {
      fgetMetadata = this.vm.getFunction("get_metadata");
    } catch (err) {
      fgetMetadata = this.vm.getFunction("_metadata");
      useSLIM = true;
    }
    const ret_value = fgetMetadata();
    const metadataStr = this.tvm.detachFromCurrentScope(ret_value).toString();
    const metadata = JSON.parse(metadataStr);

    // 3. Load parameters
    if (useSLIM) {
      // Under SLIM workflow, we load parameters by name
      const paramNames: string[] = [];
      metadata.params.forEach((param: any) => { paramNames.push(param.name) });
      this.params = this.tvm.detachFromCurrentScope(
        this.tvm.getParamsFromCacheByName(paramNames)
      );
    } else {
      // Backward compatibility -- load parameters by ids
      this.params = this.tvm.detachFromCurrentScope(
        this.tvm.getParamsFromCache("param", -1)
      );
    }

    // 4. Read in compilation configurations from metadata
    if (metadata.hasOwnProperty("prefill_chunk_size")) {
      this.prefillChunkSize = metadata.prefill_chunk_size;
      this.logger("Using prefillChunkSize: ", this.prefillChunkSize);
      if (this.prefillChunkSize <= 0) {
        throw Error("Prefill chunk size needs to be positive.");
      }
    } else {
      throw Error("Cannot find `prefill_chunk_size` in metadta; make sure the wasm is up to date.");
    }
    // Only use one of slidingWindowSize and maxWindowLength
    if (metadata.hasOwnProperty("sliding_window_size") && metadata.sliding_window_size != -1) {
      this.slidingWindowSize = metadata.sliding_window_size;
      if (this.prefillChunkSize <= 0) {
        throw Error("Need to specify prefill chunk size if using sliding window attention.");
      }
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
    } else {
      // Depending on when the model is compiled, it can be either called
      // `context_window_size` or `max_window_size`
      if (metadata.hasOwnProperty("context_window_size") && metadata.context_window_size != -1) {
        this.maxWindowLength = metadata.context_window_size;
        this.logger("Using maxWindowLength: ", this.maxWindowLength);
      } else if (metadata.hasOwnProperty("max_window_size") && metadata.max_window_size != -1) {
        this.maxWindowLength = metadata.max_window_size;
        this.logger("Using maxWindowLength: ", this.maxWindowLength);
      } else {
        throw Error("Need to specify either sliding window size or max window size.");
      }
    }

    // 5. Create cache
    // Use `fcreateCache` to determine whether we are using the new KVCache implementation
    let fcreateCache;
    try {
      if (useSLIM) {
        fcreateCache = this.vm.getFunction("_initialize_effect");
      } else {
        fcreateCache = this.vm.getFunction("create_kv_cache");
      }
    } catch (err) {
      // If we cannot find function above, it means that we are using the new PagedKVCache
      this.usePagedKVCache = true;
      fcreateCache = this.vm.getFunction("create_tir_paged_kv_cache");
      console.log("Using Paged KVCache");
      if (this.embed === undefined) {
        throw Error("If using paged KVCache, method `embed()` needs to be defined.");
      }
    }

    // Load cache functions and instantiate KVCache
    if (this.usePagedKVCache) {
      this.fclearKVCaches = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_clear")
      );
      this.fKVCacheAddSequence = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_add_sequence")
      );
      this.fKVCacheRemoveSequence = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_remove_sequence")
      );
      this.fKVCacheBeginForward = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_begin_forward")
      );
      this.fKVCacheEndForward = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_end_forward")
      );

      // Create PagedKVCache; we do not expose KVCache config for now
      const defaultPageSize = 16;
      const defaultMaxNumSequence = 1;
      this.kvCache = this.tvm.detachFromCurrentScope(fcreateCache(
        this.tvm.makeShapeTuple([defaultMaxNumSequence]),  // max_num_sequence
        this.tvm.makeShapeTuple([this.maxWindowLength]),  // max_total_sequence_length
        this.tvm.makeShapeTuple([this.prefillChunkSize]),  // prefill_chunk_size
        this.tvm.makeShapeTuple([defaultPageSize]),  // page_size, hard coded for now
      ));
    } else {
      this.fclearKVCaches = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc("vm.builtin.attention_kv_cache_array_clear")
      );
      this.kvCache = this.tvm.detachFromCurrentScope(fcreateCache());
    }
    this.filledKVCacheLength = 0;
    this.resetChat();  // especially needed for PagedKVCache as we need to call fKVCacheAddSequence
    tvm.endScope();
  }

  dispose() {
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
    this.conversation.reset();
    if (!keepStats) {
      this.resetRuntimeStats();
    }
    this.resetKVCache();
    this.filledKVCacheLength = 0;
    this.sinkTriggered = false;
    this.slidingWindowCacheOffset = 0;
    this.logitProcessor?.resetState();
  }

  /**
   * Reset KV Cache
   */
  resetKVCache() {
    this.fclearKVCaches(this.kvCache);
    if (this.usePagedKVCache) {
      this.fKVCacheAddSequence!(this.kvCache, new tvmjs.Scalar(0, "int64"));
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
   * Overrides the system prompt.
   */
  overrideSystemPrompt(system: string): void {
    this.conversation.config.system = system;
  }

  /**
   * Append a new message to `this.conversation`.
   */
  appendConversationMessage(role: string, input: string): void {
    this.conversation.appendMessage(role, input);
  }

  /**
   * Get `this.conversation.messages`.
   */
  getConversationMessages(): Array<[string, string | undefined]> {
    // TODO(Charlie): Do we need to make a deep copy here?
    return this.conversation.messages;
  }

  /**
   * @returns the roles of this.conversation's conversation template of lengths of two.
   */
  getRoles(): Array<string> {
    const res = this.conversation.config.roles;
    if (res.length !== 2) {
      throw new Error("Expect the conversation template to have two roles.");
    }
    return res;
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  /**
   * Generate the first token given input prompt
   */
  async prefillStep(inp: string, genConfig?: GenerationConfig): Promise<void> {
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
    conversation.appendMessage(conversation.config.roles[0], inp);
    conversation.appendReplyHeader(conversation.config.roles[1]);
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
      const inputData = this.tvm.empty([1, chunk.length], "int32", this.device);
      inputData.copyFrom(chunk);
      newSeqLen += chunk.length;
      logits = this.tvm.detachFromCurrentScope(
        this.forward(inputData, newSeqLen)
      );

      // Update window cache offset (prefill)
      if (this.slidingWindowSize != -1) {
        if (this.sinkTriggered) {
          this.slidingWindowCacheOffset = Math.max(
            (this.slidingWindowCacheOffset + chunk.length) % this.slidingWindowSize,
            this.attentionSinkSize
          )
        } else {
          this.slidingWindowCacheOffset += chunk.length;
          this.sinkTriggered = this.slidingWindowCacheOffset >= this.attentionSinkSize;
        }
      }
    }
    if (newSeqLen != this.filledKVCacheLength + tokenLen) {
      throw Error("Expect chunking process all tokens.")
    }
    this.filledKVCacheLength = newSeqLen;
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
    const inputData = this.tvm.empty([1, 1], "int32", this.device);
    inputData.copyFrom(this.outputIds.slice(this.outputIds.length - 1));

    const logits = this.tvm.detachFromCurrentScope(
      this.forward(inputData, this.filledKVCacheLength + 1)
    );
    this.filledKVCacheLength += 1;

    // Update window cache offset (decoding)
    if (this.slidingWindowSize != -1) {
      if (this.sinkTriggered) {
        this.slidingWindowCacheOffset = Math.max(
          (this.slidingWindowCacheOffset + 1) % this.slidingWindowSize,
          this.attentionSinkSize
        )
      } else {
        this.slidingWindowCacheOffset += 1;
        this.sinkTriggered = this.slidingWindowCacheOffset >= this.attentionSinkSize;
      }
    }
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
    let stopStrs = [this.stopStr];
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

  private forward(inputs: tvmjs.NDArray, curPos: number): tvmjs.NDArray {
    this.tvm.beginScope();
    let retValue;
    const seqLen = inputs.shape[1];  // Num input tokens
    const seqLenShape = this.tvm.makeShapeTuple([curPos]);
    if (seqLen > 1) {
      // Prefill
      if (this.slidingWindowSize == -1) {
        if (this.usePagedKVCache) {
          const seqIdsTuple = this.tvm.makeShapeTuple([0]);
          const inputLenShape = this.tvm.makeShapeTuple([seqLen]);
          this.fKVCacheBeginForward!(this.kvCache, seqIdsTuple, inputLenShape);
          const embed = this.embed!(inputs, this.params);
          retValue = this.prefill(embed, this.kvCache, this.params);
          this.fKVCacheEndForward!(this.kvCache);
        } else {
          retValue = this.prefill(inputs, seqLenShape, this.kvCache, this.params);
        }
      } else {
        // Sliding window attention needs extra shape parameters
        const cacheLen = Math.min(this.slidingWindowSize, curPos - seqLen);  // Num elements in the cache
        const cacheLenShape = this.tvm.makeShapeTuple([cacheLen]);
        const kvSeqLenShape = this.tvm.makeShapeTuple([cacheLen + seqLen]);
        // Next position to be overridden on the rolling kv cache.
        const slidingWindowCacheOffsetShape = this.tvm.makeShapeTuple([this.slidingWindowCacheOffset]);
        retValue = this.prefill(
          inputs, cacheLenShape, kvSeqLenShape, slidingWindowCacheOffsetShape, this.kvCache, this.params
        );
      }
    } else {
      // Decode
      if (this.slidingWindowSize == -1) {
        if (this.usePagedKVCache) {
          const seqIdsTuple = this.tvm.makeShapeTuple([0]);
          const appendLength = this.tvm.makeShapeTuple([1]);
          this.fKVCacheBeginForward!(this.kvCache, seqIdsTuple, appendLength);
          const embed = this.embed!(inputs, this.params);
          retValue = this.decoding(embed, this.kvCache, this.params);
          this.fKVCacheEndForward!(this.kvCache);
        } else {
          retValue = this.decoding(inputs, seqLenShape, this.kvCache, this.params);
        }
      } else {
        // Same logic as above; keeping this if-else structure to match mlc-llm's llm_chat.cc
        const seqLen = inputs.shape[1];  // Num input tokens
        const cacheLen = Math.min(this.slidingWindowSize, curPos - seqLen);  // Num elements in the cache
        const cacheLenShape = this.tvm.makeShapeTuple([cacheLen]);
        const kvSeqLenShape = this.tvm.makeShapeTuple([cacheLen + seqLen]);
        // Next position to be overridden on the rolling kv cache.
        const slidingWindowCacheOffsetShape = this.tvm.makeShapeTuple([this.slidingWindowCacheOffset]);
        retValue = this.decoding(
          inputs, cacheLenShape, kvSeqLenShape, slidingWindowCacheOffsetShape, this.kvCache, this.params
        );
      }
    }
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
    }
    // Check range validity
    if (top_p <= 0 || top_p >= 1) { throw new Error("Make sure 0 < `top_p` < 1."); }
    if (temperature < 0) { throw new Error("Make sure `temperature` >= 0."); }
    if (repetition_penalty <= 0) { throw new Error("Make sure `repetition_penalty` > 0."); }
    if (frequency_penalty && (frequency_penalty < -2.0 || frequency_penalty > 2.0)) {
      throw new Error("`frequency_penalty` should be between -2.0 and 2.0.");
    }
    if (presence_penalty && (presence_penalty < -2.0 || presence_penalty > 2.0)) {
      throw new Error("`presence_penalty` should be between -2.0 and 2.0.");
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
      if (this.conversation.config.add_bos) {
        tokens = [this.bosTokenId];
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
    if (this.conversation.config.add_bos) {
      tokens = [this.bosTokenId];
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

  async forwardTokensAndSample(
    inputIds: Array<number>, curPos: number, isPrefill: boolean
  ): Promise<number> {
    // 1. Convert input to NDArray
    const tstart = performance.now();
    this.tvm.beginScope();
    const inputData = this.tvm.empty([1, inputIds.length], "int32", this.device);
    inputData.copyFrom(inputIds);

    // 2. Forward tokens and get logits
    const logitsOnGPU: tvmjs.NDArray = this.forward(inputData, curPos);
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
    const inputData = this.tvm.empty([1, tokens.length], "int32", this.device);
    inputData.copyFrom(tokens);
    const prefillStart = performance.now();
    this.forward(inputData, tokens.length);
    this.tvm.endScope();
    await this.device.sync();

    const decodingStart = performance.now();

    this.tvm.beginScope();
    const firstSampleToken = this.tvm.empty([1, 1], "int32", this.device).copyFrom([6234]);
    const logitsOnCPU = this.updateLogitsOnCPU(this.forward(firstSampleToken, tokens.length + 1));
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
