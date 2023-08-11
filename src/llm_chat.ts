import * as tvmjs from "tvmjs";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig } from "./config";
import { getConversation, Conversation } from "./conversation";


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

  // parameter states
  private params: tvmjs.TVMObject;
  private kvCache: tvmjs.TVMObject;
  private logitsOnCPU?: tvmjs.NDArray = undefined;
  private filledKVCacheLength = 0;

  // meta data
  // TODO(tvm-team): remove hard-coded bos from config, likely can be part of conv template
  private bosTokenId = 1;
  private maxWindowLength;
  private resetStatsPerPrefill = true;
  private stopStr: string;
  private stopTokens: Array<number>;

  // states
  private outputMessage = "";
  private outputIds: Array<number> = [];
  private stopTriggered = false;
  private appearedTokens = new Set<number>();
  private conversation: Conversation;
  // Total amount of seq len prefilled so far

  // stats
  private decodingTotalTime = 0;
  private decodingTotalTokens = 0;
  private prefillTotalTime = 0;
  private prefillTotalTokens = 0;

  // logger
  private logger = console.log;

  constructor(tvm: tvmjs.Instance, tokenizer: Tokenizer, config: ChatConfig) {
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;

    this.conversation = getConversation(config.conv_template, config.conv_config);
    this.stopStr = this.conversation.getStopStr();

    this.device = this.tvm.webgpu();
    tvm.beginScope();

    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );
    this.prefill = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("prefill")
    );
    this.decoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("decode")
    );
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCache("param", -1)
    );
    const fgetMetadata = this.vm.getFunction("get_metadata");
    const ret_value = fgetMetadata();
    const metadataStr = this.tvm.detachFromCurrentScope(ret_value).toString();
    const metadata = JSON.parse(metadataStr);
    this.maxWindowLength = metadata.max_window_size;
    // TODO(tvm-team): move to conv template
    this.stopTokens = metadata.stop_tokens;

    const fcreateCache = this.vm.getFunction("create_kv_cache");
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.attention_kv_cache_array_clear")
    );

    // use extern config for now
    this.kvCache = this.tvm.detachFromCurrentScope(fcreateCache());
    this.filledKVCacheLength = 0;
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
  resetChat() {
    this.conversation.reset();
    this.resetRuntimeStats();
    this.fclearKVCaches(this.kvCache);
    this.filledKVCacheLength = 0;
  }

  /**
   * @returns Whether stop is triggered.
   */
  stopped(): boolean {
    return this.stopTriggered;
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

  async asyncLoadWebGPUPiplines() {
    await this.tvm.asyncLoadWebGPUPiplines(this.vm.getInternalModule());
  }

  /**
   * Generate the first token given input prompt
   */
  async prefillStep(inp: string): Promise<void> {
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }

    // cleanup the per convo states
    this.outputIds = [];
    this.appearedTokens.clear();
    this.outputMessage = "";
    this.stopTriggered = false;
    const conversation = this.conversation;

    // initialize
    conversation.appendMessage(conversation.config.roles[0], inp);
    conversation.appendReplyHeader(conversation.config.roles[1]);
    const promptTokens = this.getInputTokens();

    const tstart = performance.now();
    this.tvm.beginScope();
    const inputData = this.tvm.empty([1, promptTokens.length], "int32", this.device);
    inputData.copyFrom(promptTokens);

    const newSeqLen = this.filledKVCacheLength + promptTokens.length;
    const logits = this.tvm.detachFromCurrentScope(
      this.forward(inputData, newSeqLen)
    );
    this.filledKVCacheLength = newSeqLen;
    this.tvm.endScope();

    const nextToken = await this.sampleTokenFromLogits(
      logits, this.config.temperature, this.config.top_p);
    logits.dispose();
    const tend = performance.now();

    this.prefillTotalTime += (tend - tstart) / 1e3;
    this.prefillTotalTokens += promptTokens.length;

    this.processNextToken(nextToken);
  }

  async decodeStep(): Promise<void> {
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
    this.tvm.endScope();

    // sample from logits
    const nextToken = await this.sampleTokenFromLogits(
      logits, this.config.temperature, this.config.top_p);
    logits.dispose();
    const tend = performance.now();

    this.decodingTotalTime += (tend - tstart) / 1e3;
    this.decodingTotalTokens += 1;

    this.processNextToken(nextToken);
  }

  /**
   * Manually trigger stop if it is not stopped.
   */
  triggerStop() {
    if (this.stopTriggered) {
      return;
    }
    this.stopTriggered = true;
    this.conversation.finishReply(this.outputMessage);
  }

  /**
   * Add a generated token and check for stop.
   *
   * @param nextToken The next token.
   */
  private processNextToken(nextToken: number): void {
    if (this.stopTriggered) {
      throw Error("Cannot call process when it is stoppped");
    }

    this.outputIds.push(nextToken);
    this.appearedTokens.add(nextToken);

    // if there is a stop token
    if (this.stopTokens.includes(nextToken)) {
      this.stopTriggered = true;
    }

    let outputMessage = this.tokenizer.decode(new Int32Array(this.outputIds));
    const stopPos = outputMessage.lastIndexOf(this.stopStr);
    if (stopPos !== -1) {
      outputMessage = outputMessage.substring(0, stopPos);
      this.stopTriggered = true;
    }
    this.outputMessage = outputMessage;

    if (this.stopTriggered) {
      this.conversation.finishReply(this.outputMessage);
    }
  }

  private forward(inputs: tvmjs.NDArray, curPos: number): tvmjs.NDArray {
    this.tvm.beginScope();
    let retValue;
    const seqLenShape = this.tvm.makeShapeTuple([curPos]);
    if (inputs.shape[1] > 1) {
      retValue = this.prefill(
        inputs, seqLenShape, this.kvCache, this.params
      );
    } else {
      retValue = this.decoding(
        inputs, seqLenShape, this.kvCache, this.params
      );
    }
    const logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  // NOTE: caller must call device.sync()
  private updateLogitsOnCPU(logits: tvmjs.NDArray): tvmjs.NDArray {
    if (!this.logitsOnCPU) {
      this.logitsOnCPU = this.tvm.detachFromCurrentScope(
        this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu())
      );
    } else {
      if (logits.shape[0] !== this.logitsOnCPU.shape[0]) {
        throw Error("We expect the size of logits to remain unchanged");
      }
    }
    this.logitsOnCPU.copyFrom(logits);
    return this.logitsOnCPU;
  }

  private async sampleTokenFromLogits(
    logitsOnGPU: tvmjs.NDArray,
    temperature: number,
    top_p: number
  ) {
    this.tvm.beginScope();
    const logitsOnCPU = this.updateLogitsOnCPU(logitsOnGPU);
    this.tvm.endScope();
    await this.device.sync();

    if (!this.logitsOnCPU) {
      throw Error("logits should be assigned");
    }

    if (this.config.repetition_penalty < 1.0 + 1e-6) {
      return this.tvm.sampleTopPFromLogits(logitsOnCPU, temperature, top_p);
    } else {
      this.tvm.beginScope();
      const appeared_tokens_ndarray = this.tvm.empty(
        [1, this.appearedTokens.size], "int32", this.tvm.cpu());
      appeared_tokens_ndarray.copyFrom(Array.from(this.appearedTokens));
      this.tvm.applyRepetitionPenalty(
        this.logitsOnCPU, appeared_tokens_ndarray, this.config.repetition_penalty);
      this.tvm.endScope();
      return this.tvm.sampleTopPFromLogits(this.logitsOnCPU, temperature, top_p);
    }
  }

  private getInputTokens(): Array<number> {
    let tokens: Array<number> = [];
    let prompts;
    // beginning of the conversation
    if (this.conversation.messages.length <= 2) {
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
      if (this.filledKVCacheLength + ctxLength + this.config.mean_gen_len >= this.maxWindowLength) {
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

    // need shift window and re-encode
    this.logger("need shift window")
    this.filledKVCacheLength = 0;
    this.fclearKVCaches(this.kvCache);

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
      if (ctxLength >= this.config.shift_fill_factor * this.maxWindowLength && i + 2 < all_prompts.length) {
        break;
      }
      context.unshift(encoded);
    }
    for (const ctx of context) {
      tokens.push(...ctx);
    }
    if (tokens.length + this.config.mean_gen_len >= this.maxWindowLength) {
      throw Error("Exceed max window length curr=" + tokens.length);
    }
    return tokens;
  }

  async evaluate() {
    // run a canonical evaluation of the flow
    this.fclearKVCaches(this.kvCache);
    this.filledKVCacheLength = 0;

    const testPrompt = "The capital of Canada is";
    const ids = await this.tokenizer.encode(testPrompt);
    const tokens = Array.from(ids);
    tokens.unshift(this.bosTokenId);
    if (tokens.length === 0) {
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
