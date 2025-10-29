import * as tvmjs from "@mlc-ai/web-runtime";
import log from "loglevel";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig } from "./config";
import {
  EmbeddingChunkingUnsupportedError,
  EmbeddingExceedContextWindowSizeError,
  EmbeddingInputEmptyError,
  EmbeddingSlidingWindowError,
  MinValueError,
} from "./error";

export class EmbeddingPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private params: tvmjs.TVMObject;

  // metadata
  private contextWindowSize = -1;
  private prefillChunkSize = -1;
  private maxBatchSize = -1;

  // performance
  private curRoundEmbedTotalTokens = 0; // excludes padded tokens for batching
  private curRoundEmbedTotalTime = 0;

  constructor(tvm: tvmjs.Instance, tokenizer: Tokenizer, config: ChatConfig) {
    // 0. Setting attributes
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;
    this.device = this.tvm.webgpu();

    // 1. Create VM and get the core functions
    tvm.beginScope();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device),
    );
    this.prefill = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("prefill"),
    );

    // 2. Get json stored in the vm's metadata function
    const fgetMetadata = this.vm.getFunction("_metadata");
    const ret_value = fgetMetadata();
    const metadataStr = ret_value.toString();
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
    // We use context window size max batch size to check validity of the model
    // We assume prefillChunkSize is the same as contextWindowSize for embedding model for now
    this.maxBatchSize = metadata.max_batch_size;
    this.contextWindowSize = this.config.context_window_size;
    this.prefillChunkSize = metadata.prefill_chunk_size;
    log.info("Using maxBatchSize: ", this.maxBatchSize);
    log.info("Using contextWindowSize: ", this.contextWindowSize);
    log.info("Using prefillChunkSize: ", this.prefillChunkSize);

    if (this.config.sliding_window_size !== -1) {
      throw new EmbeddingSlidingWindowError(this.config.sliding_window_size);
    }
    if (this.maxBatchSize <= 0) {
      throw new MinValueError("maxBatchSize", 0);
    }
    if (this.contextWindowSize <= 0) {
      throw new MinValueError("contextWindowSize", 0);
    }
    if (this.prefillChunkSize <= 0) {
      throw new MinValueError("prefillChunkSize", 0);
    }
    if (this.prefillChunkSize !== this.contextWindowSize) {
      throw new EmbeddingChunkingUnsupportedError(
        this.contextWindowSize,
        this.prefillChunkSize,
      );
    }
    tvm.endScope();
  }

  async embedStep(
    input: string | Array<string> | Array<number> | Array<Array<number>>,
  ): Promise<Array<Array<number>>> {
    // 0. Reset performance metrics
    this.curRoundEmbedTotalTokens = 0;
    this.curRoundEmbedTotalTime = 0;
    let totalNumTokens = 0;
    const embedStart = performance.now();
    let tokenizedInputs: Array<Array<number>> = [];
    const tempInputs: Array<number> = [];
    // 1. Convert all possible input types to Array<Array<number>>, tokenize if not already
    // Cannot use input.every to match type, which leads to TS compilation error
    // https://github.com/microsoft/TypeScript/issues/33591
    if (input.length === 0) {
      throw new EmbeddingInputEmptyError();
    }
    if (typeof input === "string") {
      // string
      tokenizedInputs = [Array.from(this.tokenizer.encode(input))];
    } else {
      for (let i = 0; i < input.length; i++) {
        const curInput = input[i];
        if (Array.isArray(curInput)) {
          // Array<Array<number>>
          tokenizedInputs.push(curInput);
        } else if (typeof curInput === "string") {
          // Array<string>
          tokenizedInputs.push(Array.from(this.tokenizer.encode(curInput)));
        } else {
          // Array<number>
          tempInputs.push(curInput);
        }
      }
    }
    if (tempInputs.length > 0) {
      tokenizedInputs.push(tempInputs);
    }

    // 2. Check each input is not larger than the context window size
    // TODO: tokenizer.encode seems to implicitly truncates to contextWindowSize, confirm behavior
    // and decide whether to warn user
    for (let i = 0; i < tokenizedInputs.length; i++) {
      const curInputSize = tokenizedInputs[i].length;
      totalNumTokens += curInputSize;
      if (curInputSize > this.contextWindowSize) {
        throw new EmbeddingExceedContextWindowSizeError(
          this.contextWindowSize,
          curInputSize,
        );
      }
    }
    if (tokenizedInputs.length === 0) {
      throw new Error("InternalError: batch size is zero.");
    }

    // 3. Forward each batch
    const batchSize = tokenizedInputs.length;
    const result: Array<Array<number>> = [];
    for (let begin = 0; begin < batchSize; begin += this.maxBatchSize) {
      this.tvm.beginScope();
      // 3.1 Get current batch
      const end = Math.min(batchSize, begin + this.maxBatchSize);
      const curBatch: Array<Array<number>> = tokenizedInputs.slice(begin, end);
      const curBatchSize = curBatch.length;
      // 3.2 Max input size of current batch
      let maxInputSize = 0;
      for (let i = 0; i < curBatchSize; i++) {
        const curInputSize = curBatch[i].length;
        if (curInputSize > maxInputSize) {
          maxInputSize = curInputSize;
        }
      }
      // 3.3 Create inputs and attention mask
      // Padded with zeros and flattened, of size curBatchSize * maxInputSize
      const curBatchPaddedFlatten: Array<number> = [];
      // 1 for non-pad, 0 otherwise, also of size curBatchSize * maxInputSize
      const curAttnMask: Array<number> = [];
      const flattenedInputSize = curBatchSize * maxInputSize;
      for (let i = 0; i < curBatchSize; i++) {
        const padding = Array(maxInputSize - curBatch[i].length).fill(0);
        const ones = Array(curBatch[i].length).fill(1);
        curBatchPaddedFlatten.push(...curBatch[i]);
        curAttnMask.push(...ones);
        curBatchPaddedFlatten.push(...padding);
        curAttnMask.push(...padding);
      }
      if (
        curBatchPaddedFlatten.length !== flattenedInputSize ||
        curAttnMask.length !== flattenedInputSize
      ) {
        throw new Error(
          `InternalError: Expect input array to be ${flattenedInputSize}, ` +
            `but got ${curBatchPaddedFlatten.length}`,
        );
      }
      // 3.4 Convert inputs and attention mask to tvm ndarray on GPU, of shape (curBatchSize, maxInputSize)
      let inputNDArray = this.tvm.empty(
        [flattenedInputSize],
        "int32",
        this.device,
      );
      inputNDArray.copyFrom(curBatchPaddedFlatten);
      inputNDArray = inputNDArray.view([curBatchSize, maxInputSize]);
      let maskNDArray = this.tvm.empty(
        [flattenedInputSize],
        "int32",
        this.device,
      );
      maskNDArray.copyFrom(curAttnMask);
      maskNDArray = maskNDArray.view([curBatchSize, maxInputSize]);

      // 3.5 Actual forwarding on GPU, logits of shape (curBatchSize, maxInputSize, hidden_size)
      const logitsCurBatchOnGPU: tvmjs.Tensor = this.prefill(
        inputNDArray,
        maskNDArray,
        this.params,
      );
      await this.device.sync();

      // 3.6 Copy logits to CPU, flatten to curBatchSize * maxInputSize * hidden_size
      const hidden_size = logitsCurBatchOnGPU.shape[2];
      let logitsCurBatchOnCPU: tvmjs.Tensor = this.tvm.empty(
        logitsCurBatchOnGPU.shape,
        logitsCurBatchOnGPU.dtype,
        this.tvm.cpu(),
      );
      logitsCurBatchOnCPU.copyFrom(logitsCurBatchOnGPU);
      logitsCurBatchOnCPU = logitsCurBatchOnCPU.view([
        curBatchSize * maxInputSize * hidden_size,
      ]);
      await this.device.sync();
      const logitsCurBatchOnCPUArray: Float32Array = <Float32Array>(
        logitsCurBatchOnCPU.toArray()
      );

      // 3.7 Update final result. For each sentence, get [0,:], i.e. only the first token's output
      // That is, we are doing result.push(logits[:,0,:]) here.
      // TODO: check if all models only use [0,:]. If it is snowflake-specific, need to specify
      // this in mlc-chat-config.json
      for (let i = 0; i < curBatchSize; i++) {
        const b = i * maxInputSize * hidden_size;
        const e = b + hidden_size;
        result.push(Array.from(logitsCurBatchOnCPUArray.slice(b, e)));
      }
      this.tvm.endScope();
    }
    if (result.length !== batchSize) {
      throw new Error(`
        InternalError: expect result.length to be ${batchSize}, but got ${result.length}`);
    }
    const embedEnd = performance.now();
    this.curRoundEmbedTotalTokens = totalNumTokens;
    this.curRoundEmbedTotalTime = (embedEnd - embedStart) / 1e3;

    return result;
  }

  dispose() {
    this.params.dispose();
    this.prefill.dispose();
    this.vm.dispose();
    this.tvm.dispose();
    this.tokenizer.dispose();
  }

  /**
   * Synchronize the device.
   */
  async sync(): Promise<void> {
    // Is it equivalent to this.tvm.sync()?
    await this.device.sync();
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  // Performance APIs below

  /**
   * Get the time it took the last `embedStep()` in seconds.
   */
  getCurRoundEmbedTotalTime(): number {
    return this.curRoundEmbedTotalTime;
  }

  /**
   * Get the number of tokens embedded in the last `embedStep()`. This excludes the padded tokens.
   */
  getCurRoundEmbedTotalTokens(): number {
    return this.curRoundEmbedTotalTokens;
  }

  /**
   * @returns Prefill tokens per second, starting from the last prefill performed.
   */
  getCurRoundEmbedTokensPerSec(): number {
    return this.curRoundEmbedTotalTokens / this.curRoundEmbedTotalTime;
  }
}
