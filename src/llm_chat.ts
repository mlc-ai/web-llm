import * as tvmjs from "@mlc-ai/web-runtime";
import {
  createSampledTokenReadbackRing,
  type SampledTokenReadbackBatch,
  type SampledTokenReadbackRing,
  ensureNonblockingWebGPUCommandSubmission,
  ensureSampledTokenReadbackRingRuntime,
  SAMPLED_TOKEN_READBACK_BATCH_RUNTIME_HINT,
  submitPendingWebGPUCommands,
  submitSampledTokenReadbackBatch,
  submitTensorReadbackWords,
  supportsNonblockingWebGPUCommandSubmission,
  supportsSampledTokenReadbackBatchSubmit,
  supportsSampledTokenReadbackRingRuntime,
} from "./runtime_readback";
import * as xgr from "@mlc-ai/web-xgrammar";
import log from "loglevel";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import { ChatConfig, GenerationConfig, Role } from "./config";
import {
  getConversation,
  Conversation,
  ArtifactPromptSegment,
} from "./conversation";
import { decodeAudioInput } from "./audio";
import {
  ModelPackageManifest,
  parseCompiledProgramArtifact,
  ResolvedChatCompletionArtifact,
  resolveChatCompletionArtifact,
} from "./artifact_manifest";
import { LogitProcessor, LatencyBreakdown } from "./types";
import {
  getChunkedPrefillInputData,
  getImageDataFromURL,
  getResizedRGBArrayFromURL,
  getRGBArrayFromImageData,
  getTokenTableFromTokenizer,
  getTopProbs,
} from "./support";
import {
  ChatCompletionFinishReason,
  ChatCompletionTokenLogprob,
  TopLogprob,
  ResponseFormat,
  ChatCompletionContentPartImage,
  ChatCompletionContentPart,
  ChatCompletionContentPartInputAudio,
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
  CannotFindImageEmbedError,
  GrammarMatcherInitError,
  ArtifactManifestError,
} from "./error";

type ImageURL = ChatCompletionContentPartImage.ImageURL;

type KVStateKind = "kv_cache" | "rnn_state" | "hybrid" | "none";
type ComputeABIKind = "single" | "batch";

type VMFunctionAvailability = {
  prefill: boolean;
  batch_prefill: boolean;
  decode: boolean;
  batch_decode: boolean;
  create_tir_paged_kv_cache: boolean;
  create_rnn_state: boolean;
};

type VMFunctionRegistry = Record<string, tvmjs.PackedFunc | undefined>;

type SampleReadbackMode = "baseline" | "ring_vector" | "lookahead";
// "auto" uses the ring when the runtime has it and the request allows it.
type SampleReadbackRequestMode = "auto" | "baseline" | "ring_vector";
type SampleReadbackFallbackReason =
  | "runtime"
  | "recurrent_state"
  | "logit_processor"
  | "grammar"
  | "logprobs"
  | "penalties";

interface SampleReadbackPathSelection {
  requestedMode: SampleReadbackRequestMode;
  activeMode: SampleReadbackMode;
  fallbackReason: SampleReadbackFallbackReason | null;
}

interface StagedLogitBias {
  key: string;
  pos2seqIds: tvmjs.Tensor;
  tokenIds: tvmjs.Tensor;
  biases: tvmjs.Tensor;
}

interface StagedSamplerInputs {
  temperature: number;
  topP: number;
  logitBias?: StagedLogitBias;
}

interface StagedUniformSample {
  tensor: tvmjs.Tensor;
  rngStateAfterSample: unknown;
}

interface RingVectorReadbackSession {
  ring: SampledTokenReadbackRing;
  batchSize: number;
  computeSubmitInterval: number;
  decodeTokensSinceComputeSubmit: number;
  maxInFlightBatches: number;
  maxSpeculativeTokens: number;
  speculativeTokenLimit: number;
  slotCount: number;
  readyTokens: number[];
  pendingBatchCount: number;
  pendingTokenCount: number;
  tokenDevicePool: tvmjs.Tensor[];
  stagedTokenDeviceTensors: tvmjs.Tensor[];
  uniformSamplePool: StagedUniformSample[];
  uncommittedSampleRNGStates: unknown[];
  committedRNGState: unknown;
  nextDecodeInputTokenDevice?: tvmjs.Tensor;
}

type ArtifactDeviceCapabilities = {
  features: { has(feature: string): boolean };
  maxStorageBufferBindingSize: number;
};

type ArtifactPrefillChunk = {
  tokenIds: number[];
  modalityIds: number[];
  embeddings?: tvmjs.Tensor;
};

type ResolvedModelABI = {
  kvStateKind: Exclude<KVStateKind, "none">;
  prefillABI: ComputeABIKind;
  decodeABI: ComputeABIKind;
  prefillFunctionName: string;
  decodeFunctionName: string;
  needsKVCache: boolean;
  needsRNNState: boolean;
};

type CommitTokenSource = "prefill" | "decode";

const KV_CHECKPOINT_FUNC_NAMES = [
  "vm.builtin.attention_kv_cache_get_checkpoint_metadata_at",
  "vm.builtin.attention_kv_cache_export_page_range",
  "vm.builtin.attention_kv_cache_prepare_import",
  "vm.builtin.attention_kv_cache_import_page_range",
  "vm.builtin.attention_kv_cache_finish_import",
  "vm.builtin.attention_kv_cache_get_sequence_length",
] as const;

// Checkpoint pages are copied into mappable readback slots of up to this many
// bytes. The slots map asynchronously in queue order.
const KV_CHECKPOINT_READBACK_BATCH_BYTES = 8 * 1024 * 1024;
const KV_CHECKPOINT_READBACK_SLOTS = 2;

function byteLength(shape: number[], dtype: string): number {
  const bits = Number(/\d+$/.exec(dtype)?.[0]);
  if (!Number.isInteger(bits) || bits % 8 !== 0) {
    throw new Error(`Unsupported KV checkpoint dtype: ${dtype}`);
  }
  return shape.reduce((size, dim) => size * dim, bits / 8);
}

type ForwardPrefillResult = {
  logits: tvmjs.Tensor;
  promptLen: number;
  promptTokenIds?: number[];
  assistantPrefixTokenIds: number[];
};

export interface KVCheckpointPageGroupData {
  groupId: number;
  layerStart: number;
  layerEnd: number;
  data: Uint8Array;
}

export interface KVCheckpointLogitsData {
  shape: number[];
  dtype: string;
  data: Uint8Array;
}

/**
 * KV pages [pageBegin, ceil(processedSeqLen / page_size)) of every page group.
 * A checkpoint with pageBegin > 0 is a delta: the pages before it come from
 * the checkpoints it extends.
 */
export interface KVCheckpointData {
  processedSeqLen: number;
  layoutHash?: string;
  metadata: Record<string, unknown>;
  pageBegin: number;
  pageGroups: KVCheckpointPageGroupData[];
  nextLogits?: KVCheckpointLogitsData;
}

/**
 * Receives a checkpoint's page bytes as they are read back: each page group's
 * bytes in file order, one group after another. The readback waits for each
 * call before it reuses the slot, so at most two batches are held in memory.
 */
export type KVCheckpointSink = (
  group: number,
  data: Uint8Array,
) => Promise<void>;

/**
 * A KV checkpoint being read back while decoding continues. Pages below
 * processedSeqLen do not change until the sequence is reset, so they are
 * copied straight from the live KV cache into mappable readback slots, a batch
 * whenever a slot frees up, and passed to the sink. There is no copy of the
 * whole checkpoint on the GPU or in memory: the sequence must not be reset
 * before the readback ends.
 */
export interface KVCheckpointSnapshot {
  processedSeqLen: number;
  layoutHash?: string;
  metadata: Record<string, unknown>;
  pageBegin: number;
  /** The page groups, in the order their bytes reach the sink. */
  pageGroups: Array<Omit<KVCheckpointPageGroupData, "data">>;
  /** Size of the page ranges and logits once read back. */
  byteLength: number;
  /** GPU memory the readback holds: the copy buffer and readback slots. */
  stagingBytes: number;
  /**
   * Resolves once every page has reached the sink, with the logits copied at
   * capture if any, or with undefined if the snapshot was discarded first.
   * Map completions advance the readback, so it does not wait for decode
   * steps.
   */
  done: Promise<{ nextLogits?: KVCheckpointLogitsData } | undefined>;
  /**
   * Start copying the pages into `sink`; logits are copied at capture. The
   * caller picks the moment, to keep the copies off a latency-critical step.
   */
  start(sink: KVCheckpointSink): void;
  /** Stop the readback. Resetting the sequence discards it. */
  discard(): void;
}

export interface KVCheckpointReplayResult {
  replayedTokens: number;
  sampledFromCheckpointLogits: boolean;
  sampledToken?: SampledGenerationStep;
  committedToken?: CommittedGenerationStep;
}

export interface SamplePrefillOptions {
  capturePromptCheckpoint?: boolean;
  storeCheckpointLogits?: boolean;
}

interface KVCheckpointGroupInfo {
  groupId: number;
  layerStart: number;
  layerEnd: number;
  shape: number[];
  dtype: string;
}

/** Pages of one group copied into a readback batch. */
interface KVCheckpointPageSegment {
  group: number;
  page: number;
  count: number;
}

interface KVCheckpointReadback {
  snapshot: KVCheckpointSnapshot;
  groups: KVCheckpointGroupInfo[];
  pageEnd: number;
  pageBytes: number;
  /** Next page to copy: an index into groups and a logical page. */
  nextGroup: number;
  nextPage: number;
  ring: SampledTokenReadbackRing;
  staging?: tvmjs.Tensor;
  stagingPages: number;
  /** Host arrays page batches are read into, taken from the pipeline's pool. */
  hostBuffers?: Int32Array[];
  pageBatches: number;
  /** Submitted batches in ring order: page segments, or the logits. */
  inflight: Array<KVCheckpointPageSegment[] | KVCheckpointLogitsData>;
  sink?: KVCheckpointSink;
  nextLogits?: KVCheckpointLogitsData;
  resolve: (done: { nextLogits?: KVCheckpointLogitsData } | undefined) => void;
  reject: (err: unknown) => void;
  /**
   * Settles when the readback is discarded so pending mappings cannot delay
   * its cleanup.
   */
  disposed: Promise<SampledTokenReadbackBatch[]>;
  markDisposed: () => void;
}

export interface SampledGenerationStep {
  source: CommitTokenSource;
  tokenId: number;
  globalTokenPos: number;
  promptLen?: number;
  promptTokenIds?: number[];
  assistantPrefixTokenIds?: number[];
  promptCheckpoint?: KVCheckpointSnapshot;
  decodeCheckpoint?: KVCheckpointSnapshot;
  logprob?: ChatCompletionTokenLogprob;
}

export interface CommittedGenerationStep {
  source: CommitTokenSource;
  tokenId: number;
  globalTokenPos: number;
  textDelta: string;
  textPrefixLength: number;
  outputMessage: string;
  stopped: boolean;
  finishReason?: ChatCompletionFinishReason;
  logprob?: ChatCompletionTokenLogprob;
}

export interface ReplayedGenerationToken {
  globalTokenPos: number;
  tokenId: number;
  textDelta: string;
  textPrefixLength?: number;
  rngState?: unknown;
}

function commonPrefixLength(lhs: string, rhs: string): number {
  const limit = Math.min(lhs.length, rhs.length);
  let prefixLength = 0;
  while (prefixLength < limit && lhs[prefixLength] === rhs[prefixLength]) {
    prefixLength++;
  }
  return prefixLength;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export class LLMChatPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private decoding: tvmjs.PackedFunc;
  private resolvedModelABI!: ResolvedModelABI;
  private kvStateKind: KVStateKind = "kv_cache";
  private image_embed: tvmjs.PackedFunc | undefined;
  private embed: tvmjs.PackedFunc;
  private artifact?: ResolvedChatCompletionArtifact;
  private artifactAudioAdapter?: tvmjs.PackedFunc;
  private artifactImageAdapter?: tvmjs.PackedFunc;
  private fTensorCreateView?: tvmjs.PackedFunc;
  private fapplyBitmask: tvmjs.PackedFunc;
  private fapplyPenalty: tvmjs.PackedFunc;
  private fapplyLogitBias: tvmjs.PackedFunc;
  private fsoftmaxWithTemperature: tvmjs.PackedFunc;
  private fsampleWithTopP: tvmjs.PackedFunc;
  private fargsortProbs: tvmjs.PackedFunc;

  // Functions related to PagedKVCache
  private fclearKVCaches: tvmjs.PackedFunc;
  private fKVCacheAddSequence: tvmjs.PackedFunc;
  private fKVCacheRemoveSequence: tvmjs.PackedFunc;
  private fKVCacheBeginForward: tvmjs.PackedFunc;
  private fKVCacheEndForward: tvmjs.PackedFunc;
  private fKVCacheEnableSlidingWindowForSeq: tvmjs.PackedFunc;

  // parameter states
  private params: tvmjs.TVMObject;
  private kvCache?: tvmjs.TVMObject = undefined;
  private rnnState?: tvmjs.TVMObject = undefined;
  private kvCheckpointFuncs = new Map<string, tvmjs.PackedFunc>();
  private kvCheckpointUnavailableReason?: string;
  private kvCheckpointReadback?: KVCheckpointReadback;
  // Host arrays for checkpoint page batches, reused across checkpoints: a new
  // array per 8 MiB batch is freed only by garbage collection, which pauses
  // decoding.
  private kvCheckpointHostBuffers: Int32Array[] = [];
  private prefetchedDecodeLogits?: Promise<tvmjs.Tensor>;
  private decodePrefetchUnavailable = false;
  private prefillLogitPositions?: tvmjs.Tensor = undefined;
  private prefillLogitPositionHost = new Int32Array(1);
  private maxHistorySize = 1;
  private logitsOnCPU?: tvmjs.Tensor = undefined;
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
  private assistantPrefixLength = 0;
  private stopTriggered = false;
  private finishReason: ChatCompletionFinishReason | undefined = undefined;
  // frequency of appeared token ids till now (refresh after PrefillStep); token_id mapped to freq
  private appearedTokensFreq = new Map<number, number>();
  private imageDataCache = new Map<string, ImageData>();
  // Resized pixels for manifest image inputs, reused within one prefill.
  private artifactImagePixelCache = new Map<string, Uint8ClampedArray>();
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
  private curRoundCompletionTotalTokens = 0;
  private curRoundPrefillTotalTokens = 0;
  private curRoundDecodingTotalTime = 0;
  private curRoundPrefillTotalTime = 0;
  private pendingSampleOutputTokenBegin?: number = undefined;

  // additional stats, reset at every prefillStep()
  public curRoundLatencyBreakdown: LatencyBreakdown = {
    logitProcessorTime: [],
    logitBiasTime: [],
    penaltyTime: [],
    sampleTime: [],
    totalTime: [],
    grammarBitmaskTime: [],
  };

  // LogitProcessor
  private logitProcessor?: LogitProcessor = undefined;

  // Grammar-related
  // A grammar matcher for this current round if response_format is set. Reinitialized upon
  // each step regardless of whether the chat is multi-round or not.
  private grammarMatcher?: xgr.GrammarMatcher = undefined;
  // Cache key of the current response_format (schema / grammar / structural tag). If undefined,
  // grammarMatcher is simply using JSON mode. We use this field to determine whether we re-initiate
  // a GrammarMatcher or simply reset during each round (i.e. during prefillStep).
  private responseFormatCacheKey?: string = undefined;
  // A string list of tokens ordered by their token id, post-processed. Once initialized, will not
  // be reinitialized since `this.tokenizer` does not change throughout the lifetime of LLMChatPipeline.
  private xgTokenizerInfo?: xgr.TokenizerInfo = undefined;
  // Compiler for grammar. It is persistent since it specializes on xgTokenizerInfo.
  private grammarCompiler?: xgr.GrammarCompiler = undefined;
  // Size of the bitmask for grammar, determined by fullVocabSize
  private bitmaskSize: number;
  // `vocab_size` read from `config.json`. Can be different from the size of the tokenTable for some
  // models due to dummy padded tokens.
  private fullVocabSize: number;
  // Method to post process the token for grammar; either "byte_level" or default "byte_fallback".
  private token_postproc_method: string;
  // Whether to prepend space for grammar
  private prepend_space_in_encode: boolean;
  // stats for grammar-related overhead
  // Time to initialize grammar matcher in seconds
  private curRoundGrammarInitTotalTime = 0;
  // Total time of getting next bitmask and accepting token in seconds
  private curRoundGrammarPerTokenTotalTime = 0;
  // Instance variables for supporting sampling on WebGPU
  private sampleIndices: Int32Array;
  private sampleIndicesDevice: tvmjs.Tensor;
  private topPDevice: tvmjs.Tensor;
  private temperaturesDevice: tvmjs.Tensor;
  private fKVStatePopN: tvmjs.PackedFunc;
  // Sampled-token readback ring (sample_readback_mode "ring_vector")
  private stagedSamplerInputs?: StagedSamplerInputs;
  private hasSampledTokenReadbackRingRuntimeSupport: boolean;
  private hasWarnedRingVectorFallback = false;
  private activeSampleReadbackMode: SampleReadbackMode = "baseline";
  private sampleReadbackPathSelection: SampleReadbackPathSelection = {
    requestedMode: "baseline",
    activeMode: "baseline",
    fallbackReason: null,
  };
  private ringVectorReadbackSession?: RingVectorReadbackSession = undefined;
  private ringVectorPendingKVRollbackTokens = 0;

  constructor(
    tvm: tvmjs.Instance,
    tokenizer: Tokenizer,
    config: ChatConfig,
    logitProcessor?: LogitProcessor,
    modelPackage?: ModelPackageManifest,
    artifactDeviceCapabilities?: ArtifactDeviceCapabilities,
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
      this.prepend_space_in_encode =
        config.tokenizer_info.prepend_space_in_encode;
    } else if (config.token_table_postproc_method !== undefined) {
      this.token_postproc_method = config.token_table_postproc_method;
      this.prepend_space_in_encode = false;
    } else {
      log.warn(
        "Cannot find `tokenizer_info` or `token_table_postproc_method` in `mlc-chat-config.json`, " +
          "using default token_postproc_method `raw`.\n" +
          "This field is only used for json mode.",
      );
      this.token_postproc_method = "raw";
      this.prepend_space_in_encode = false;
    }
    log.info("token_postproc_method: ", this.token_postproc_method);
    log.info("prepend_space_in_encode: ", this.prepend_space_in_encode);

    this.device = this.tvm.webgpu();
    this.hasSampledTokenReadbackRingRuntimeSupport =
      supportsSampledTokenReadbackRingRuntime(this.device);

    // 1. Create VM and read model metadata
    tvm.beginScope();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device),
    );

    const fgetMetadata = this.vm.getFunction("_metadata");
    const ret_value = fgetMetadata();
    const metadataStr = ret_value.toString();
    const metadata = JSON.parse(metadataStr);
    if (modelPackage !== undefined) {
      if (metadata.artifact === undefined) {
        throw new ArtifactManifestError(
          "mlc-model-manifest.json is present, but _metadata.artifact is missing",
        );
      }
      this.artifact = resolveChatCompletionArtifact(
        modelPackage,
        parseCompiledProgramArtifact(metadata.artifact),
      );
      if (artifactDeviceCapabilities !== undefined) {
        for (const feature of this.artifact.compiled.resources
          .required_features) {
          if (!artifactDeviceCapabilities.features.has(feature)) {
            throw new ArtifactManifestError(
              `Compiled artifact requires unsupported WebGPU feature ${JSON.stringify(feature)}`,
            );
          }
        }
        const requiredBuffer =
          this.artifact.compiled.resources.max_storage_buffer_binding_size;
        if (
          requiredBuffer >
          artifactDeviceCapabilities.maxStorageBufferBindingSize
        ) {
          throw new ArtifactManifestError(
            `Compiled artifact requires maxStorageBufferBindingSize ${requiredBuffer}, ` +
              `but this device exposes ${artifactDeviceCapabilities.maxStorageBufferBindingSize}`,
          );
        }
      }
    }

    const functionNames = [
      "prefill",
      "batch_prefill",
      "decode",
      "batch_decode",
      "create_tir_paged_kv_cache",
      "create_rnn_state",
      "sample_with_top_p",
      "argsort_probs",
      "image_embed",
      "embed",
      "apply_bitmask_inplace",
      "apply_penalty_inplace",
      "apply_logit_bias_inplace",
      "softmax_with_temperature",
    ];
    if (this.artifact !== undefined) {
      functionNames.push(
        ...Object.values(this.artifact.program.exports),
        ...Object.values(this.artifact.program.adapters),
      );
    }
    const vmFunctionRegistry = LLMChatPipeline.loadVMFunctionRegistry(
      this.vm,
      Array.from(new Set(functionNames)),
    );
    this.kvStateKind = this.parseKVStateKind(metadata.kv_state_kind);

    if (this.artifact !== undefined) {
      // The roles decide the state set: create_kv_cache alone is a KV cache
      // model, with create_rnn_state it is a hybrid one.
      const needsRNNState =
        this.artifact.program.exports.create_rnn_state !== undefined;
      const kvStateKind = needsRNNState ? "hybrid" : "kv_cache";
      if (this.kvStateKind !== kvStateKind) {
        throw new ArtifactManifestError(
          `the manifest roles imply kv_state_kind=${JSON.stringify(kvStateKind)}, ` +
            `but the library reports ${JSON.stringify(this.kvStateKind)}`,
        );
      }
      this.resolvedModelABI = {
        kvStateKind,
        prefillABI: "single",
        decodeABI: "single",
        prefillFunctionName: this.artifact.generation.prefill,
        decodeFunctionName: this.artifact.generation.decode,
        needsKVCache: true,
        needsRNNState,
      };
    } else {
      const vmFunctionAvailability =
        LLMChatPipeline.getVMFunctionAvailability(vmFunctionRegistry);
      this.resolvedModelABI = LLMChatPipeline.resolveModelABI(
        this.kvStateKind,
        vmFunctionAvailability,
      );
    }
    const stateKinds: string[] = [];
    if (this.resolvedModelABI.needsKVCache) {
      stateKinds.push("kv_cache");
    }
    if (this.resolvedModelABI.needsRNNState) {
      stateKinds.push("rnn_state");
    }
    log.info(
      "Resolved model ABI:",
      JSON.stringify({
        kv_state_kind: this.kvStateKind,
        prefill: this.resolvedModelABI.prefillFunctionName,
        decode: this.resolvedModelABI.decodeFunctionName,
        states: stateKinds,
      }),
    );

    // 2. Look up the VM functions, by manifest role when the model has a manifest.
    this.prefill = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        this.resolvedModelABI.prefillFunctionName,
        vmFunctionRegistry,
      ),
    );
    this.decoding = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        this.resolvedModelABI.decodeFunctionName,
        vmFunctionRegistry,
      ),
    );
    if (this.resolvedModelABI.prefillABI === "batch") {
      log.info("Using batch_prefill kernel.");
    }
    if (this.resolvedModelABI.decodeABI === "batch") {
      log.info("Using batch_decode kernel.");
    }

    const embedSymbol = this.artifact?.program.exports.embed_tokens ?? "embed";
    this.embed = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        embedSymbol,
        vmFunctionRegistry,
      ),
    );
    this.fapplyBitmask = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "apply_bitmask_inplace",
        vmFunctionRegistry,
      ),
    );
    this.fapplyPenalty = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "apply_penalty_inplace",
        vmFunctionRegistry,
      ),
    );
    this.fapplyLogitBias = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "apply_logit_bias_inplace",
        vmFunctionRegistry,
      ),
    );
    this.fsoftmaxWithTemperature = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "softmax_with_temperature",
        vmFunctionRegistry,
      ),
    );
    this.fsampleWithTopP = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "sample_with_top_p",
        vmFunctionRegistry,
      ),
    );
    this.fargsortProbs = this.tvm.detachFromCurrentScope(
      LLMChatPipeline.getRequiredVMFunctionByName(
        "argsort_probs",
        vmFunctionRegistry,
      ),
    );
    // A manifest model reaches its vision function only through the adapter
    // the manifest names, which may be this same image_embed.
    const imageEmbed = vmFunctionRegistry.image_embed;
    if (imageEmbed === undefined) {
      log.info("Cannot find function image_embed.");
    } else if (this.artifact === undefined) {
      this.image_embed = this.tvm.detachFromCurrentScope(imageEmbed);
    }
    if (this.artifact !== undefined) {
      const { audioInput, imageInput, program } = this.artifact;
      const bindAdapter = (name: string) =>
        this.tvm.detachFromCurrentScope(
          LLMChatPipeline.getRequiredVMFunctionByName(
            program.adapters[name],
            vmFunctionRegistry,
          ),
        );
      if (audioInput !== undefined) {
        this.artifactAudioAdapter = bindAdapter(audioInput.adapter);
      }
      if (imageInput !== undefined) {
        this.artifactImageAdapter = bindAdapter(imageInput.adapter);
      }
      if (audioInput !== undefined || imageInput !== undefined) {
        this.fTensorCreateView = this.tvm.detachFromCurrentScope(
          this.tvm.getGlobalFunc("runtime.TensorCreateView"),
        );
      }
    }

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
    if (
      config.max_history_size !== undefined &&
      config.max_history_size !== null
    ) {
      if (config.max_history_size <= 0) {
        throw new MinValueError("max_history_size", 0);
      }
      this.maxHistorySize = config.max_history_size;
    } else if (this.resolvedModelABI.needsRNNState) {
      // Hybrid/recurrent models can over-allocate RNN state if we use context window directly.
      // Keep browser default conservative unless user explicitly overrides max_history_size.
      log.info("max_history_size is not set. Using browser-safe default: 1");
    }

    // 5. Create cache
    // Load cache functions and instantiate KVCache
    this.fKVStatePopN = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_popn"),
    );
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

    const defaultPageSize = 16;
    const defaultMaxNumSequence = 1;
    const maxTotalSeqLen =
      this.slidingWindowSize != -1
        ? this.slidingWindowSize
        : this.contextWindowSize;

    if (this.resolvedModelABI.needsKVCache) {
      const createKVCache = LLMChatPipeline.getRequiredVMFunctionByName(
        this.artifact?.program.exports.create_kv_cache ??
          "create_tir_paged_kv_cache",
        vmFunctionRegistry,
      );
      this.kvCache = this.tvm.detachFromCurrentScope(
        createKVCache(
          this.tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
          this.tvm.makeShapeTuple([maxTotalSeqLen]), // max_total_sequence_length
          this.tvm.makeShapeTuple([this.prefillChunkSize]), // prefill_chunk_size
          this.tvm.makeShapeTuple([defaultPageSize]), // page_size, hard coded for now
          this.tvm.makeShapeTuple([this.slidingWindowSize != -1 ? 1 : 0]),
        ),
      );
    }

    if (this.resolvedModelABI.needsRNNState) {
      const createRNNState = LLMChatPipeline.getRequiredVMFunctionByName(
        this.artifact?.program.exports.create_rnn_state ?? "create_rnn_state",
        vmFunctionRegistry,
      );
      this.rnnState = this.tvm.detachFromCurrentScope(
        createRNNState(
          this.tvm.makeShapeTuple([defaultMaxNumSequence]),
          this.tvm.makeShapeTuple([this.maxHistorySize]),
        ),
      );
      log.info("Using maxHistorySize for RNN state: ", this.maxHistorySize);
    }

    if (this.resolvedModelABI.prefillABI === "batch") {
      this.prefillLogitPositions = this.tvm.detachFromCurrentScope(
        this.tvm.empty([defaultMaxNumSequence], "int32", this.device),
      );
    }

    this.filledKVCacheLength = 0;
    this.resetChat(); // especially needed for PagedKVCache as we need to call fKVCacheAddSequence

    // Initialize WebGPU sampling related device tensors
    const numSamples = 1;
    const numProbs = 1;

    this.sampleIndices = new Int32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      this.sampleIndices[i] = i;
    }
    this.sampleIndicesDevice = this.tvm.detachFromCurrentScope(
      this.tvm
        .empty([numSamples], "int32", this.device)
        .copyFrom(this.sampleIndices),
    );

    this.topPDevice = this.tvm.detachFromCurrentScope(
      this.tvm.empty([numProbs], "float32", this.device),
    );
    this.temperaturesDevice = this.tvm.detachFromCurrentScope(
      this.tvm.empty([numSamples], "float32", this.device),
    );

    tvm.endScope();
  }

  getTokenizer(): Tokenizer {
    return this.tokenizer;
  }

  private canDecodeLookahead(): boolean {
    if (!this.hasRingVectorRuntime() || this.logitProcessor !== undefined) {
      return false;
    }
    // A stop discards the one forward that ran ahead.
    return this.rnnState === undefined || this.maxHistorySize - 1 >= 1;
  }

  private ensureLookaheadRing(): SampledTokenReadbackRing {
    if (this.lookaheadRing === undefined) {
      this.lookaheadRing = createSampledTokenReadbackRing(this.device, {
        slotCount: 2,
        maxTokensPerBatch: 1,
      });
    }
    return this.lookaheadRing;
  }

  private ensureLookaheadProbsRing(): SampledTokenReadbackRing {
    if (this.lookaheadProbsRing === undefined) {
      // The ring copies 4-byte elements, so it carries float32 as well.
      this.lookaheadProbsRing = createSampledTokenReadbackRing(this.device, {
        slotCount: 2,
        maxTokensPerBatch: this.fullVocabSize,
      });
    }
    return this.lookaheadProbsRing;
  }

  private disposeLookahead(): void {
    if (this.lookaheadLogits !== undefined) {
      this.lookaheadLogits.dispose();
      this.lookaheadLogits = undefined;
      // The forward that ran ahead fed the last committed token.
      this.ringVectorPendingKVRollbackTokens += 1;
    }
  }

  private async decodeStepLookahead(
    genConfig?: GenerationConfig,
  ): Promise<CommittedGenerationStep> {
    let logits = this.lookaheadLogits;
    this.lookaheadLogits = undefined;
    if (logits === undefined) {
      logits = await this.forwardDecodeToken(
        this.outputIds[this.outputIds.length - 1],
      );
    }
    const maxTokens = this.getGenerationMaxTokens(genConfig);
    let forwardedAhead = false;
    let nextToken: number;
    try {
      nextToken = await this.sampleFromRawLogits(
        logits,
        genConfig,
        async (sampledTokensDevice, probsDevice) => {
          const probsRing =
            probsDevice !== undefined
              ? this.ensureLookaheadProbsRing()
              : undefined;
          if (probsRing !== undefined) {
            submitTensorReadbackWords(
              probsRing,
              probsDevice!,
              this.fullVocabSize,
            );
          }
          const ring = this.ensureLookaheadRing();
          ring.submit(sampledTokensDevice, 1, 0);
          // Skip the forward when this token ends the reply by length.
          if (
            this.getGeneratedTokenCount() + 1 < maxTokens &&
            (this.slidingWindowSize !== -1 ||
              this.filledKVCacheLength < this.contextWindowSize)
          ) {
            this.lookaheadLogits =
              await this.decodeFromInputTokenDevice(sampledTokensDevice);
            forwardedAhead = true;
            submitPendingWebGPUCommands(this.device);
          }
          const batches = await ring.waitReady();
          const token = batches[0].tokens[0];
          if (probsRing === undefined) {
            return { token };
          }
          const bits = (await probsRing.waitReady())[0].tokens;
          return {
            token,
            probs: new Float32Array(bits.buffer, bits.byteOffset, bits.length),
          };
        },
      );
    } finally {
      logits.dispose();
    }
    const committed = this.commitToken(
      nextToken,
      genConfig,
      "decode",
      this.filledKVCacheLength - (forwardedAhead ? 1 : 0),
    );
    if (this.stopTriggered) {
      this.disposeLookahead();
      await this.applyPendingRingVectorKVRollback();
    }
    return committed;
  }

  private lookaheadLogits?: tvmjs.Tensor = undefined;

  private lookaheadRing?: SampledTokenReadbackRing = undefined;

  private lookaheadProbsRing?: SampledTokenReadbackRing = undefined;

  private ensureRingVectorReadbackRuntime(): void {
    if (!this.hasSampledTokenReadbackRingRuntimeSupport) {
      ensureSampledTokenReadbackRingRuntime(this.device);
    }
  }

  private warnRingVectorFallback(reason: SampleReadbackFallbackReason): void {
    if (!this.hasWarnedRingVectorFallback) {
      const reasonMessage: Record<SampleReadbackFallbackReason, string> = {
        runtime: "the web runtime has no sampled-token readback ring",
        recurrent_state:
          "the model's recurrent state keeps too little history to roll back the speculative tail",
        logit_processor: "logitProcessor is active",
        grammar: "grammar-constrained sampling is active",
        logprobs: "logprobs is requested",
        penalties: "sampling penalties are active",
      };
      log.warn(
        `sample_readback_mode="ring_vector" requested but ${reasonMessage[reason]}; falling back to baseline sampled-token readback for this request.`,
      );
      this.hasWarnedRingVectorFallback = true;
    }
  }

  private resolveSampleReadbackConfig(genConfig?: GenerationConfig): {
    mode: SampleReadbackRequestMode;
    batchSize: number;
    computeSubmitInterval: number;
    maxSpeculativeTokens: number;
  } {
    return {
      mode: genConfig?.sample_readback_mode ?? "auto",
      batchSize: genConfig?.sample_readback_batch_size ?? 1,
      computeSubmitInterval:
        genConfig?.sample_readback_compute_submit_interval ?? 1,
      maxSpeculativeTokens:
        genConfig?.sample_readback_max_speculative_tokens ?? 4,
    };
  }

  private resolveSamplerInputs(
    genConfig?: GenerationConfig,
  ): StagedSamplerInputs {
    let temperature = this.config.temperature;
    let topP = this.config.top_p;
    if (
      genConfig?.temperature !== undefined &&
      genConfig.temperature !== null
    ) {
      temperature = genConfig.temperature;
    }
    if (genConfig?.top_p !== undefined && genConfig.top_p !== null) {
      topP = genConfig.top_p;
    }
    if (topP === undefined || topP === null) {
      topP = 1.0;
    }
    if (!(topP > 0 && topP <= 1)) {
      throw new RangeError("top_p", 0, 1);
    }
    if (!(temperature >= 0)) {
      throw new MinValueError("temperature", 0);
    }
    return {
      temperature: Math.max(1e-6, temperature),
      topP: Math.max(topP, 1e-5),
    };
  }

  private stageSamplerInputs(
    genConfig?: GenerationConfig,
  ): StagedSamplerInputs {
    const inputs = this.resolveSamplerInputs(genConfig);
    this.temperaturesDevice.copyFrom(new Float32Array([inputs.temperature]));
    this.topPDevice.copyFrom(new Float32Array([inputs.topP]));
    this.disposeStagedLogitBias();
    inputs.logitBias = this.stageLogitBias(genConfig);
    this.stagedSamplerInputs = inputs;
    return inputs;
  }

  private logitBiasKey(genConfig?: GenerationConfig): string {
    const bias = genConfig?.logit_bias;
    return bias === undefined || bias === null ? "" : JSON.stringify(bias);
  }

  /** Upload a request's logit_bias once, since it is the same for every token. */
  private stageLogitBias(
    genConfig?: GenerationConfig,
  ): StagedLogitBias | undefined {
    const key = this.logitBiasKey(genConfig);
    const entries = Object.entries(genConfig?.logit_bias ?? {});
    if (entries.length === 0) {
      return undefined;
    }
    this.tvm.beginScope();
    const upload = (data: Int32Array | Float32Array, dtype: string) =>
      this.tvm.detachFromCurrentScope(
        this.tvm.empty([data.length], dtype, this.device).copyFrom(data),
      );
    const staged = {
      key,
      pos2seqIds: upload(new Int32Array(entries.length), "int32"),
      tokenIds: upload(
        Int32Array.from(entries, ([token]) => parseInt(token)),
        "int32",
      ),
      biases: upload(
        Float32Array.from(entries, ([, bias]) => bias),
        "float32",
      ),
    };
    this.tvm.endScope();
    return staged;
  }

  private disposeStagedLogitBias(): void {
    const staged = this.stagedSamplerInputs?.logitBias;
    if (staged === undefined) {
      return;
    }
    staged.pos2seqIds.dispose();
    staged.tokenIds.dispose();
    staged.biases.dispose();
    this.stagedSamplerInputs!.logitBias = undefined;
  }

  private ensureSamplerInputsStaged(
    genConfig?: GenerationConfig,
  ): StagedSamplerInputs {
    const expected = this.resolveSamplerInputs(genConfig);
    if (
      this.stagedSamplerInputs?.temperature !== expected.temperature ||
      this.stagedSamplerInputs.topP !== expected.topP ||
      (this.stagedSamplerInputs.logitBias?.key ?? "") !==
        this.logitBiasKey(genConfig)
    ) {
      return this.stageSamplerInputs(genConfig);
    }
    return this.stagedSamplerInputs;
  }

  private disposeRingVectorReadbackSession(): void {
    const session = this.ringVectorReadbackSession;
    if (session === undefined) {
      return;
    }
    // Keep encoded speculative passes ahead of any later KV metadata rewind or
    // buffer disposal in WebGPU queue order without waiting for completion.
    this.submitPendingRingVectorCompute(session, true);
    session.ring.dispose();
    const retainedTensors = new Set(session.tokenDevicePool);
    for (const tensor of session.stagedTokenDeviceTensors) {
      retainedTensors.add(tensor);
    }
    if (session.nextDecodeInputTokenDevice !== undefined) {
      retainedTensors.add(session.nextDecodeInputTokenDevice);
    }
    for (const tensor of retainedTensors) {
      tensor.dispose();
    }
    for (const sample of session.uniformSamplePool) {
      sample.tensor.dispose();
    }
    const discardedRNGSamples =
      session.uncommittedSampleRNGStates.length +
      session.uniformSamplePool.length;
    if (discardedRNGSamples > 0) {
      this.setRNGState(session.committedRNGState);
    }
    this.ringVectorReadbackSession = undefined;
  }

  private resetSampleReadbackSession(mode: SampleReadbackMode): void {
    this.disposeRingVectorReadbackSession();
    this.activeSampleReadbackMode = mode;
  }

  private stageRingVectorKVRollbackAndResetSession(): void {
    const session = this.ringVectorReadbackSession;
    if (session === undefined) {
      return;
    }
    this.assertRingVectorRNGAccounting(session);
    this.ringVectorPendingKVRollbackTokens +=
      this.getRingVectorUncommittedTokenCount(session);
    this.disposeRingVectorReadbackSession();
  }

  private async applyPendingRingVectorKVRollback(): Promise<void> {
    const rollbackTokens = this.ringVectorPendingKVRollbackTokens;
    if (!(rollbackTokens > 0)) {
      return;
    }
    if (rollbackTokens > this.filledKVCacheLength) {
      throw new Error(
        `Cannot roll back ${rollbackTokens} speculative tokens from a KV state ` +
          `with length ${this.filledKVCacheLength}.`,
      );
    }

    const states = this.getActiveKVStates();
    // PopN only rewinds CPU-side sequence/page metadata. Speculative GPU work
    // has already been encoded, and the next BeginForward uploads the dirty
    // auxiliary data after that work in WebGPU queue order. Waiting for the
    // entire device here adds a stop-path bubble without protecting any GPU
    // data that PopN mutates.
    const seqId = new tvmjs.Scalar(0, "int64");
    const popCount = new tvmjs.Scalar(rollbackTokens, "int32");
    for (const state of states) {
      this.fKVStatePopN(state, seqId, popCount);
    }
    this.filledKVCacheLength -= rollbackTokens;
    this.ringVectorPendingKVRollbackTokens = 0;
  }

  private resolveEffectivePenaltyConfig(genConfig?: GenerationConfig): {
    repetition: number;
    frequency: number;
    presence: number;
  } {
    const repetition =
      genConfig?.repetition_penalty ?? this.config.repetition_penalty;
    const frequency =
      genConfig?.frequency_penalty ?? this.config.frequency_penalty;
    const presence =
      genConfig?.presence_penalty ?? this.config.presence_penalty;
    return {
      repetition,
      frequency,
      presence,
    };
  }

  private resolveRingVectorFallbackReason(
    genConfig?: GenerationConfig,
  ): SampleReadbackFallbackReason | null {
    if (this.logitProcessor !== undefined) {
      return "logit_processor";
    }
    if (
      genConfig?.response_format?.type === "json_object" ||
      genConfig?.response_format?.type === "grammar" ||
      genConfig?.response_format?.type === "structural_tag"
    ) {
      return "grammar";
    }
    if (genConfig?.logprobs || (genConfig?.top_logprobs ?? 0) > 0) {
      return "logprobs";
    }
    // Recurrent state can roll back only as many tokens as its history keeps,
    // and a stop inside the speculative tail rolls back up to the whole tail.
    if (
      this.rnnState !== undefined &&
      this.maxHistorySize - 1 < this.resolveRingVectorTailBound(genConfig)
    ) {
      return "recurrent_state";
    }
    const penalties = this.resolveEffectivePenaltyConfig(genConfig);
    if (
      penalties.repetition !== 1.0 ||
      penalties.frequency !== 0.0 ||
      penalties.presence !== 0.0
    ) {
      return "penalties";
    }
    return null;
  }

  /** Largest speculative tail the ring session can hold for this request. */
  private resolveRingVectorShape(genConfig?: GenerationConfig): {
    maxSpeculativeTokens: number;
    maxInFlightBatches: number;
  } {
    const config = this.resolveSampleReadbackConfig(genConfig);
    const maxSpeculativeTokens = Math.max(
      config.batchSize,
      config.maxSpeculativeTokens,
    );
    return {
      maxSpeculativeTokens,
      maxInFlightBatches: Math.ceil(maxSpeculativeTokens / config.batchSize),
    };
  }

  private getGeneratedTokenCount(): number {
    return this.outputIds.length - this.assistantPrefixLength;
  }

  private resolveRingVectorTailBound(genConfig?: GenerationConfig): number {
    return this.resolveRingVectorShape(genConfig).maxSpeculativeTokens;
  }

  private hasRingVectorRuntime(): boolean {
    if (!this.hasSampledTokenReadbackRingRuntimeSupport) {
      return false;
    }
    // Submitting decode work between readbacks needs nonblocking submission.
    return supportsNonblockingWebGPUCommandSubmission(this.device);
  }

  private resolveSampleReadbackPath(
    genConfig?: GenerationConfig,
  ): SampleReadbackPathSelection {
    const requestedMode = this.resolveSampleReadbackConfig(genConfig).mode;
    if (requestedMode === "baseline") {
      return {
        requestedMode,
        activeMode: "baseline",
        fallbackReason: null,
      };
    }
    if (requestedMode === "auto" && !this.hasRingVectorRuntime()) {
      return {
        requestedMode,
        activeMode: "baseline",
        fallbackReason: "runtime",
      };
    }

    const fallbackReason = this.resolveRingVectorFallbackReason(genConfig);
    if (fallbackReason !== null && this.canDecodeLookahead()) {
      return {
        requestedMode,
        activeMode: "lookahead",
        fallbackReason,
      };
    }
    if (fallbackReason !== null) {
      if (requestedMode === "ring_vector") {
        this.warnRingVectorFallback(fallbackReason);
      }
      return {
        requestedMode,
        activeMode: "baseline",
        fallbackReason,
      };
    }
    return {
      requestedMode,
      activeMode: "ring_vector",
      fallbackReason: null,
    };
  }

  private selectSampleReadbackPath(genConfig?: GenerationConfig): void {
    this.hasWarnedRingVectorFallback = false;
    if (this.resolveSampleReadbackConfig(genConfig).mode === "ring_vector") {
      this.ensureRingVectorReadbackRuntime();
    }
    this.sampleReadbackPathSelection =
      this.resolveSampleReadbackPath(genConfig);
    this.resetSampleReadbackSession(
      this.sampleReadbackPathSelection.activeMode,
    );
    if (this.sampleReadbackPathSelection.activeMode === "ring_vector") {
      this.stageSamplerInputs(genConfig);
    }
  }

  private getGenerationMaxTokens(genConfig?: GenerationConfig): number {
    if (genConfig?.max_tokens !== undefined && genConfig.max_tokens !== null) {
      return genConfig.max_tokens;
    }
    return Infinity;
  }

  private getRingVectorUncommittedTokenCount(
    session: RingVectorReadbackSession,
  ): number {
    return (
      session.pendingTokenCount +
      session.readyTokens.length +
      session.stagedTokenDeviceTensors.length
    );
  }

  private assertRingVectorRNGAccounting(
    session: RingVectorReadbackSession,
  ): void {
    const uncommittedTokens = this.getRingVectorUncommittedTokenCount(session);
    if (session.uncommittedSampleRNGStates.length !== uncommittedTokens) {
      throw new Error(
        "Ring-vector RNG accounting does not match the speculative token tail: " +
          `${session.uncommittedSampleRNGStates.length} RNG states for ` +
          `${uncommittedTokens} tokens.`,
      );
    }
  }

  private hasRingVectorUncommittedTokens(
    session: RingVectorReadbackSession,
  ): boolean {
    return this.getRingVectorUncommittedTokenCount(session) > 0;
  }

  private ensureRingVectorReadbackSession(
    genConfig?: GenerationConfig,
  ): RingVectorReadbackSession {
    const readbackConfig = this.resolveSampleReadbackConfig(genConfig);
    const { maxSpeculativeTokens, maxInFlightBatches } =
      this.resolveRingVectorShape(genConfig);
    if (this.ringVectorReadbackSession !== undefined) {
      if (
        this.ringVectorReadbackSession.maxInFlightBatches !==
          maxInFlightBatches ||
        this.ringVectorReadbackSession.maxSpeculativeTokens !==
          maxSpeculativeTokens ||
        this.ringVectorReadbackSession.batchSize !== readbackConfig.batchSize ||
        this.ringVectorReadbackSession.computeSubmitInterval !==
          readbackConfig.computeSubmitInterval
      ) {
        this.disposeRingVectorReadbackSession();
      } else {
        return this.ringVectorReadbackSession;
      }
    }
    const ring = createSampledTokenReadbackRing(this.device, {
      slotCount: maxInFlightBatches,
      maxTokensPerBatch: readbackConfig.batchSize,
    });
    if (
      readbackConfig.batchSize > 1 &&
      !supportsSampledTokenReadbackBatchSubmit(ring)
    ) {
      ring.dispose();
      throw new Error(SAMPLED_TOKEN_READBACK_BATCH_RUNTIME_HINT);
    }
    if (
      readbackConfig.batchSize > 1 &&
      !supportsNonblockingWebGPUCommandSubmission(this.device)
    ) {
      ring.dispose();
      ensureNonblockingWebGPUCommandSubmission(this.device);
    }
    this.tvm.beginScope();
    const tokenDevicePool: tvmjs.Tensor[] = [];
    for (let i = 0; i < readbackConfig.batchSize; ++i) {
      tokenDevicePool.push(
        this.tvm.detachFromCurrentScope(
          this.tvm.empty([1], "int32", this.device),
        ),
      );
    }
    this.tvm.endScope();
    this.ringVectorReadbackSession = {
      ring,
      batchSize: readbackConfig.batchSize,
      computeSubmitInterval: readbackConfig.computeSubmitInterval,
      decodeTokensSinceComputeSubmit: 0,
      maxInFlightBatches,
      maxSpeculativeTokens,
      speculativeTokenLimit: 1,
      slotCount: maxInFlightBatches,
      readyTokens: [],
      pendingBatchCount: 0,
      pendingTokenCount: 0,
      tokenDevicePool,
      stagedTokenDeviceTensors: [],
      uniformSamplePool: [],
      uncommittedSampleRNGStates: [],
      committedRNGState: this.getRNGState(),
    };
    return this.ringVectorReadbackSession;
  }

  private stageRingVectorUniformSampleBatch(
    session: RingVectorReadbackSession,
    genConfig?: GenerationConfig,
  ): void {
    if (session.uniformSamplePool.length > 0) {
      return;
    }
    const uncommitted = this.getRingVectorUncommittedTokenCount(session);
    const remainingSpeculativeTokens =
      session.speculativeTokenLimit - uncommitted;
    const remainingGenerationTokens =
      this.getGenerationMaxTokens(genConfig) -
      this.getGeneratedTokenCount() -
      uncommitted;
    const sampleCount = Math.min(
      session.batchSize,
      remainingSpeculativeTokens,
      remainingGenerationTokens,
    );
    if (sampleCount <= 0) {
      return;
    }
    this.tvm.beginScope();
    for (let i = 0; i < sampleCount; ++i) {
      const tensor = this.tvm.detachFromCurrentScope(
        this.tvm.uniform([1], 0.0, 1.0, this.device),
      );
      session.uniformSamplePool.push({
        tensor,
        rngStateAfterSample: this.getRNGState(),
      });
    }
    this.tvm.endScope();
  }

  private drainRingVectorReadbackReadyTokens(
    session: RingVectorReadbackSession,
    readyBatches: Array<SampledTokenReadbackBatch>,
  ): void {
    for (const batch of readyBatches) {
      session.pendingBatchCount = Math.max(0, session.pendingBatchCount - 1);
      session.pendingTokenCount = Math.max(
        0,
        session.pendingTokenCount - batch.tokenCount,
      );
      for (let i = 0; i < batch.tokenCount; ++i) {
        session.readyTokens.push(batch.tokens[i]);
      }
    }
  }

  private async waitRingVectorReadbackReadyTokens(
    session: RingVectorReadbackSession,
  ): Promise<void> {
    this.drainRingVectorReadbackReadyTokens(
      session,
      await session.ring.waitReady(),
    );
  }

  private getOrCreateRingVectorDecodeInputTokenDevice(
    session: RingVectorReadbackSession,
  ): tvmjs.Tensor {
    if (session.nextDecodeInputTokenDevice !== undefined) {
      return session.nextDecodeInputTokenDevice;
    }
    const lastToken = this.outputIds[this.outputIds.length - 1];
    const inputTokenDevice = session.tokenDevicePool[session.batchSize - 1];
    inputTokenDevice.copyFrom([lastToken]);
    session.nextDecodeInputTokenDevice = inputTokenDevice;
    return inputTokenDevice;
  }

  private async decodeFromInputTokenDevice(
    inputTokenDevice: tvmjs.Tensor,
  ): Promise<tvmjs.Tensor> {
    if (this.artifact !== undefined) {
      return this.artifactDecodeFromInputTokenDevice(inputTokenDevice);
    }
    this.tvm.beginScope();
    const embeddings = this.embed(inputTokenDevice, this.params);
    const allEmbeddings = embeddings.view([1].concat(embeddings.shape));
    const inputLenShape = this.tvm.makeShapeTuple([1]);
    const seqIdsTuple = this.tvm.makeShapeTuple([0]);
    const forwardStates = this.getActiveKVStates();
    for (const state of forwardStates) {
      this.fKVCacheBeginForward(state, seqIdsTuple, inputLenShape);
    }
    const retValue = this.invokeDecode(allEmbeddings);
    for (let i = forwardStates.length - 1; i >= 0; --i) {
      this.fKVCacheEndForward(forwardStates[i]);
    }
    this.filledKVCacheLength += 1;
    const logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    this.tvm.endScope();
    return logits;
  }

  private sampleTokenDeviceFromLogitsRingVectorPath(
    logitsOnGPU: tvmjs.Tensor,
    uniformSamplesDevice: tvmjs.Tensor,
    destination: tvmjs.Tensor,
    genConfig?: GenerationConfig,
  ): tvmjs.Tensor {
    const samplerInputs = this.ensureSamplerInputsStaged(genConfig);

    const sampleBegin = performance.now();

    const numSeqs = 1;
    const numProbs = 1;
    this.tvm.beginScope();
    const logits = logitsOnGPU.view([numSeqs, numProbs, this.fullVocabSize]);
    const logitBias = samplerInputs.logitBias;
    if (logitBias !== undefined) {
      this.fapplyLogitBias(
        logitsOnGPU.view([1, this.fullVocabSize]),
        logitBias.pos2seqIds,
        logitBias.tokenIds,
        logitBias.biases,
      );
    }
    let probs = this.fsoftmaxWithTemperature(logits, this.temperaturesDevice);
    probs = probs.view([numProbs, this.fullVocabSize]);
    const argsortResults = this.fargsortProbs(probs);
    const sampledTokensDevice = this.fsampleWithTopP(
      argsortResults.get(0),
      argsortResults.get(1),
      uniformSamplesDevice,
      this.sampleIndicesDevice,
      this.topPDevice,
    );
    destination.copyFrom(sampledTokensDevice);
    this.tvm.endScope();

    if (genConfig?.enable_latency_breakdown) {
      const sampleEnd = performance.now();
      const sampleTimeSpent = (sampleEnd - sampleBegin) / 1e3;
      this.curRoundLatencyBreakdown.sampleTime.push(sampleTimeSpent);
    }
    return destination;
  }

  private canScheduleMoreRingVectorDecodeTokens(
    session: RingVectorReadbackSession,
    genConfig?: GenerationConfig,
  ): boolean {
    const maxTokens = this.getGenerationMaxTokens(genConfig);
    const uncommitted = this.getRingVectorUncommittedTokenCount(session);
    if (this.getGeneratedTokenCount() + uncommitted >= maxTokens) {
      return false;
    }
    if (uncommitted >= session.speculativeTokenLimit) {
      return false;
    }
    if (
      this.slidingWindowSize === -1 &&
      this.filledKVCacheLength >= this.contextWindowSize
    ) {
      return false;
    }
    if (session.stagedTokenDeviceTensors.length >= session.batchSize) {
      return false;
    }
    return (
      session.stagedTokenDeviceTensors.length > 0 ||
      session.pendingBatchCount < session.maxInFlightBatches
    );
  }

  private growRingVectorSpeculativeTokenLimit(
    session: RingVectorReadbackSession,
  ): void {
    const previousLimit = session.speculativeTokenLimit;
    const activeLimit = Math.min(
      session.maxSpeculativeTokens,
      previousLimit * 2,
    );
    if (activeLimit === previousLimit) {
      return;
    }
    session.speculativeTokenLimit = activeLimit;
  }

  private submitPendingRingVectorCompute(
    session: RingVectorReadbackSession,
    force = false,
  ): void {
    const decodedTokens = session.decodeTokensSinceComputeSubmit ?? 0;
    if (decodedTokens === 0) {
      return;
    }
    const computeSubmitInterval = session.computeSubmitInterval ?? 1;
    if (!force && decodedTokens < computeSubmitInterval) {
      return;
    }
    submitPendingWebGPUCommands(this.device);
    session.decodeTokensSinceComputeSubmit = 0;
  }

  private async submitStagedRingVectorBatch(
    session: RingVectorReadbackSession,
  ): Promise<void> {
    const tokenTensors = session.stagedTokenDeviceTensors.slice();
    if (tokenTensors.length === 0) {
      return;
    }
    try {
      submitSampledTokenReadbackBatch(session.ring, tokenTensors);
    } catch (err: any) {
      const msg = String(err ?? "");
      if (!msg.includes("ring") || !msg.includes("full")) {
        throw err;
      }
      this.drainRingVectorReadbackReadyTokens(
        session,
        session.ring.pollReady(),
      );
      if (session.readyTokens.length === 0 && session.pendingBatchCount > 0) {
        await this.waitRingVectorReadbackReadyTokens(session);
      }
      submitSampledTokenReadbackBatch(session.ring, tokenTensors);
    }

    session.decodeTokensSinceComputeSubmit = 0;
    session.stagedTokenDeviceTensors = [];
    session.pendingBatchCount += 1;
    session.pendingTokenCount += tokenTensors.length;
  }

  private async scheduleRingVectorDecodeToken(
    session: RingVectorReadbackSession,
    genConfig?: GenerationConfig,
  ): Promise<void> {
    this.stageRingVectorUniformSampleBatch(session, genConfig);
    const uniformSample = session.uniformSamplePool[0];
    if (uniformSample === undefined) {
      throw new Error("Ring-vector sampling has no staged uniform sample.");
    }
    const inputTokenDevice =
      this.getOrCreateRingVectorDecodeInputTokenDevice(session);
    const logits = await this.decodeFromInputTokenDevice(inputTokenDevice);
    const destination =
      session.tokenDevicePool[session.stagedTokenDeviceTensors.length];
    const sampledTokenDevice = this.sampleTokenDeviceFromLogitsRingVectorPath(
      logits,
      uniformSample.tensor,
      destination,
      genConfig,
    );
    session.uniformSamplePool.shift();
    uniformSample.tensor.dispose();
    logits.dispose();
    session.stagedTokenDeviceTensors.push(sampledTokenDevice);
    session.uncommittedSampleRNGStates.push(uniformSample.rngStateAfterSample);
    session.decodeTokensSinceComputeSubmit =
      (session.decodeTokensSinceComputeSubmit ?? 0) + 1;
    session.nextDecodeInputTokenDevice = sampledTokenDevice;
    const readbackRequiredForProgress =
      session.readyTokens.length === 0 &&
      session.pendingTokenCount === 0 &&
      !this.canScheduleMoreRingVectorDecodeTokens(session, genConfig);
    if (
      session.stagedTokenDeviceTensors.length === session.batchSize ||
      readbackRequiredForProgress
    ) {
      await this.submitStagedRingVectorBatch(session);
    } else {
      const readbackBacklogLow =
        session.computeSubmitInterval > 1 &&
        session.pendingTokenCount < session.batchSize;
      this.submitPendingRingVectorCompute(session, readbackBacklogLow);
    }
  }

  private async refillRingVectorSpeculativeWindow(
    session: RingVectorReadbackSession,
    genConfig?: GenerationConfig,
  ): Promise<void> {
    let scheduledTokens = 0;
    while (this.canScheduleMoreRingVectorDecodeTokens(session, genConfig)) {
      await this.scheduleRingVectorDecodeToken(session, genConfig);
      scheduledTokens += 1;
    }
    if (
      session.readyTokens.length === 0 &&
      session.pendingTokenCount === 0 &&
      session.stagedTokenDeviceTensors.length > 0
    ) {
      await this.submitStagedRingVectorBatch(session);
    }
    if (scheduledTokens === 0) {
      return;
    }
  }

  private async decodeStepRingVector(
    genConfig?: GenerationConfig,
  ): Promise<CommittedGenerationStep> {
    const outputTokenBegin = performance.now();
    const session = this.ensureRingVectorReadbackSession(genConfig);

    this.drainRingVectorReadbackReadyTokens(session, session.ring.pollReady());
    if (session.readyTokens.length === 0) {
      await this.refillRingVectorSpeculativeWindow(session, genConfig);
      this.drainRingVectorReadbackReadyTokens(
        session,
        session.ring.pollReady(),
      );
    }
    if (session.readyTokens.length === 0 && session.pendingTokenCount > 0) {
      await this.waitRingVectorReadbackReadyTokens(session);
    }
    if (session.readyTokens.length === 0) {
      throw new Error("Ring-vector sampling did not produce a readable token.");
    }

    this.assertRingVectorRNGAccounting(session);
    const nextToken = session.readyTokens.shift()!;
    session.committedRNGState = session.uncommittedSampleRNGStates.shift()!;
    // Forwards for the remaining tail have fed one token each beyond this one,
    // so the context-window check must see the length before them.
    const committed = this.commitToken(
      nextToken,
      genConfig,
      "decode",
      this.filledKVCacheLength -
        this.getRingVectorUncommittedTokenCount(session),
    );
    if (!this.stopTriggered) {
      this.growRingVectorSpeculativeTokenLimit(session);
      await this.refillRingVectorSpeculativeWindow(session, genConfig);
    }
    if (genConfig?.enable_latency_breakdown) {
      const outputTokenEnd = performance.now();
      const outputTokenTimeSpent = (outputTokenEnd - outputTokenBegin) / 1e3;
      this.curRoundLatencyBreakdown.totalTime.push(outputTokenTimeSpent);
    }
    if (this.stopTriggered) {
      if (this.hasRingVectorUncommittedTokens(session)) {
        this.stageRingVectorKVRollbackAndResetSession();
        await this.applyPendingRingVectorKVRollback();
      } else {
        this.disposeRingVectorReadbackSession();
      }
    }
    return committed;
  }

  dispose() {
    // TODO: Do we need to dispose all PackedFuncs here?
    this.disposeRingVectorReadbackSession();
    this.disposeStagedLogitBias();
    this.lookaheadLogits?.dispose();
    this.lookaheadRing?.dispose();
    this.lookaheadProbsRing?.dispose();
    this.grammarMatcher?.dispose();
    this.params.dispose();
    this.decoding.dispose();
    this.prefill.dispose();
    this.embed.dispose();
    this.artifactAudioAdapter?.dispose();
    this.artifactImageAdapter?.dispose();
    this.fTensorCreateView?.dispose();
    this.image_embed?.dispose();
    this.prefillLogitPositions?.dispose();
    this.rnnState?.dispose();
    this.kvCache?.dispose();
    this.fsampleWithTopP.dispose();
    this.fargsortProbs.dispose();
    for (const func of this.kvCheckpointFuncs.values()) {
      func.dispose();
    }
    this.prefetchedDecodeLogits = undefined;
    this.discardCheckpointReadback();
    this.kvCheckpointFuncs.clear();
    this.sampleIndicesDevice?.dispose();
    this.topPDevice?.dispose();
    this.temperaturesDevice?.dispose();
    this.vm.dispose();
    this.fclearKVCaches.dispose();
    this.fKVStatePopN.dispose();
    this.logitsOnCPU?.dispose();
    this.tvm.dispose();
    this.tokenizer.dispose();
    this.xgTokenizerInfo?.dispose();
    this.grammarCompiler?.dispose();
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
    this.discardPrefetchedDecode();
    this.discardCheckpointReadback();
    this.tvm.beginScope();
    this.conversation.reset();
    this.finishReason = undefined;
    if (!keepStats) {
      this.resetRuntimeStats();
    }
    this.disposeRingVectorReadbackSession();
    this.sampleReadbackPathSelection = {
      requestedMode: "baseline",
      activeMode: "baseline",
      fallbackReason: null,
    };
    this.disposeStagedLogitBias();
    this.lookaheadLogits?.dispose();
    this.lookaheadLogits = undefined;
    this.stagedSamplerInputs = undefined;
    this.activeSampleReadbackMode = "baseline";
    this.hasWarnedRingVectorFallback = false;
    this.ringVectorPendingKVRollbackTokens = 0;
    this.resetKVCache();
    this.filledKVCacheLength = 0;
    this.logitProcessor?.resetState();
    this.tvm.endScope();
  }

  /**
   * Reset KV Cache
   */
  resetKVCache() {
    const states = this.getActiveKVStates();
    for (const state of states) {
      this.fclearKVCaches(state);
      this.fKVCacheAddSequence!(state, new tvmjs.Scalar(0, "int64"));
    }
    if (this.slidingWindowSize != -1 && this.kvCache !== undefined) {
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
   * Number of tokens committed to the output this round: the token sampled
   * after prefill, every decoded token, and a terminating stop token.
   */
  getCurRoundCompletionTotalTokens(): number {
    return this.curRoundCompletionTotalTokens;
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
   * @returns the time (seconds) spent on for initializing grammar matcher for a single request.
   */
  getCurRoundGrammarInitTotalTime(): number {
    return this.curRoundGrammarInitTotalTime;
  }

  /**
   * @returns the total time (seconds) spent on creating bitmask and accepting token grammar matcher
   * for all the generated tokens in a single request.
   */
  getCurRoundGrammarPerTokenTotalTime(): number {
    return this.curRoundGrammarPerTokenTotalTime;
  }

  /**
   * @returns the breakdown of latencies for sampling each token for a single request.
   */
  getCurRoundLatencyBreakdown(): LatencyBreakdown {
    return this.curRoundLatencyBreakdown;
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
   * @returns Decode tokens per second, or zero when no decoding time has elapsed.
   */
  getCurRoundDecodingTokensPerSec(): number {
    return this.curRoundDecodingTotalTime > 0
      ? this.curRoundDecodingTotalTokens / this.curRoundDecodingTotalTime
      : 0;
  }

  /**
   * Set the seed for the RNG `this.tvm.rng`.
   */
  setSeed(seed: number): void {
    this.tvm.setSeed(seed);
  }

  private getResponseFormatKey(
    responseFormat?: ResponseFormat | null,
  ): string | undefined {
    if (!responseFormat) {
      return undefined;
    }
    if (responseFormat.type === "json_object") {
      return responseFormat.schema ?? undefined;
    }
    if (responseFormat.type === "grammar") {
      return responseFormat.grammar ?? undefined;
    }
    if (responseFormat.type === "structural_tag") {
      const structuralTag = responseFormat.structural_tag;
      if (structuralTag === undefined || structuralTag === null) {
        return undefined;
      }
      return typeof structuralTag === "string"
        ? structuralTag
        : JSON.stringify(structuralTag);
    }
    return undefined;
  }

  private isGrammarConstrained(
    responseFormat?: ResponseFormat | null,
  ): boolean {
    return (
      responseFormat?.type === "json_object" ||
      responseFormat?.type === "grammar" ||
      responseFormat?.type === "structural_tag"
    );
  }

  private resetGenerationRoundState(): void {
    this.outputIds = [];
    this.assistantPrefixLength = 0;
    this.appearedTokensFreq.clear();
    this.outputMessage = "";
    this.tokenLogprobArray = [];
    this.curRoundDecodingTotalTokens = 0;
    this.curRoundCompletionTotalTokens = 0;
    this.curRoundPrefillTotalTokens = 0;
    this.curRoundPrefillTotalTime = 0;
    this.curRoundDecodingTotalTime = 0;
    this.curRoundGrammarInitTotalTime = 0;
    this.curRoundGrammarPerTokenTotalTime = 0;
    this.pendingSampleOutputTokenBegin = undefined;

    this.curRoundLatencyBreakdown = {
      logitProcessorTime: [],
      logitBiasTime: [],
      penaltyTime: [],
      sampleTime: [],
      totalTime: [],
      grammarBitmaskTime: [],
    };
  }

  private prepareGrammarMatcherForSampling(
    genConfig?: GenerationConfig,
  ): Promise<void> | undefined {
    const responseFormat = genConfig?.response_format;
    if (
      responseFormat === undefined ||
      responseFormat === null ||
      !this.isGrammarConstrained(responseFormat)
    ) {
      return undefined;
    }

    const curResponseFormatKey = this.getResponseFormatKey(responseFormat);
    if (
      curResponseFormatKey === this.responseFormatCacheKey &&
      this.grammarMatcher
    ) {
      // If we did not change the schema and have instantiated a GrammarMatcher, we reuse it.
      const tGrammarInitStart = performance.now();
      log.info("Reuse grammar matcher.");
      this.grammarMatcher.reset();
      this.curRoundGrammarInitTotalTime =
        (performance.now() - tGrammarInitStart) / 1e3;
      return undefined;
    }

    // Else dispose current grammarMatcher, reinitialize, and update this.schema.
    const initialization = (async () => {
      const tGrammarInitStart = performance.now();
      log.info("Initialize new grammar matcher.");
      try {
        if (this.grammarMatcher) {
          this.grammarMatcher.dispose();
          this.grammarMatcher = undefined;
        }
        if (this.xgTokenizerInfo === undefined) {
          log.info("Initialize token table.");
          // Post process entire table
          const rawTokenTable = getTokenTableFromTokenizer(this.tokenizer);
          this.xgTokenizerInfo = await xgr.TokenizerInfo.createTokenizerInfo(
            rawTokenTable,
            this.token_postproc_method,
            this.prepend_space_in_encode,
            this.fullVocabSize,
            this.stopTokens,
          );
          this.grammarCompiler =
            await xgr.GrammarCompiler.createGrammarCompiler(
              this.xgTokenizerInfo,
            );
        }
        const grammar: xgr.CompiledGrammar =
          responseFormat.type === undefined
            ? await this.grammarCompiler!.compileBuiltinJSONGrammar()
            : responseFormat.type === "json_object"
              ? await this.grammarCompiler!.compileJSONSchema(
                  responseFormat.schema!,
                  true,
                  undefined,
                  undefined,
                  true,
                  16, // Bound whitespace between JSON elements to prevent generation loops.
                )
              : responseFormat.type === "grammar"
                ? await this.grammarCompiler!.compileGrammar(
                    responseFormat.grammar!,
                  )
                : await this.grammarCompiler!.compileStructuralTag(
                    responseFormat.structural_tag!,
                  );
        try {
          this.grammarMatcher =
            await xgr.GrammarMatcher.createGrammarMatcher(grammar);
        } finally {
          grammar.dispose();
        }
        this.responseFormatCacheKey = curResponseFormatKey;
        this.curRoundGrammarInitTotalTime =
          (performance.now() - tGrammarInitStart) / 1e3;
      } catch (err) {
        throw new GrammarMatcherInitError(
          responseFormat.type ?? "json_object",
          err,
        );
      }
    })();
    // Observe failures immediately while prefill is still in flight. Awaiting the
    // original promise below still propagates the initialization error.
    void initialization.catch(() => undefined);
    return initialization;
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
    this.stopStr = this.conversation.getStopStr();
    this.stopTokens = this.conversation.getStopTokens();
  }

  /** Input kinds other than text that the loaded model accepts. */
  getSupportedInputKinds(): ReadonlySet<"image" | "audio"> | undefined {
    const result = new Set<"image" | "audio">();
    if (this.artifact !== undefined) {
      if (this.artifact.audioInput !== undefined) {
        result.add("audio");
      }
      if (this.artifact.imageInput !== undefined) {
        result.add("image");
      }
      return result;
    }
    if (this.image_embed !== undefined) {
      result.add("image");
      return result;
    }
    // Text-only models keep the existing checks and error messages.
    return undefined;
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  /**
   * Generate the first token given input prompt
   */
  async prefillStep(
    inp: string | ChatCompletionContentPart[],
    msgRole: Role, // either user or tool
    inp_role_str?: string,
    genConfig?: GenerationConfig,
  ): Promise<void> {
    if (msgRole !== Role.user && msgRole !== Role.tool) {
      throw new MessageOrderError(
        "The last message should be from `user` or `tool`.",
      );
    }

    const step = await this.samplePrefillStep(
      inp,
      msgRole,
      inp_role_str,
      genConfig,
    );
    this.commitSampledStep(step, genConfig);
  }

  async samplePrefillStep(
    inp: string | ChatCompletionContentPart[],
    msgRole: Role,
    inp_role_str?: string,
    genConfig?: GenerationConfig,
    opts: SamplePrefillOptions = {},
  ): Promise<SampledGenerationStep> {
    if (msgRole !== Role.user && msgRole !== Role.tool) {
      throw new MessageOrderError(
        "The last message should be from `user` or `tool`.",
      );
    }
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }

    const tstart = performance.now();
    this.discardPrefetchedDecode();
    await this.applyPendingRingVectorKVRollback();
    this.resetGenerationRoundState();
    this.stopTriggered = false;
    this.selectSampleReadbackPath(genConfig);

    // -1. Instantiate grammar matcher according to generation config. This step is overlapped
    // with prefilling the prompt to hide overhead by using this promise.
    const grammarMatcherInitPromise =
      this.prepareGrammarMatcherForSampling(genConfig);
    const { logits, promptLen, promptTokenIds, assistantPrefixTokenIds } =
      await this.forwardPrefill(inp, msgRole, inp_role_str, genConfig);
    let promptCheckpoint: KVCheckpointSnapshot | undefined;
    let nextToken: number;
    try {
      promptCheckpoint =
        opts.capturePromptCheckpoint === true
          ? this.captureCheckpoint(
              this.filledKVCacheLength,
              0,
              opts.storeCheckpointLogits !== false ? logits : undefined,
            )
          : undefined;

      // 4. Sample, stats, post process token sampled.
      // We wait for prefill and grammar matcher init to finish
      await this.device.sync();
      await grammarMatcherInitPromise;
      nextToken = await this.sampleFromRawLogits(logits, genConfig);
    } finally {
      logits.dispose();
    }
    const tend = performance.now();

    this.prefillTotalTime += (tend - tstart) / 1e3;
    this.prefillTotalTokens += promptLen;
    this.curRoundPrefillTotalTokens += promptLen;
    this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;

    return {
      source: "prefill",
      tokenId: nextToken,
      globalTokenPos: this.filledKVCacheLength,
      promptLen,
      promptTokenIds,
      assistantPrefixTokenIds,
      promptCheckpoint,
      logprob: this.tokenLogprobArray.at(-1),
    };
  }

  private async forwardPrefill(
    inp: string | ChatCompletionContentPart[],
    msgRole: Role,
    inpRoleStr?: string,
    genConfig?: GenerationConfig,
  ): Promise<ForwardPrefillResult> {
    const conversation = this.conversation;
    const assistantPrefixTokenIds: number[] = [];

    // 0. Get inputData from conversation
    if (conversation.isTextCompletion) {
      if (typeof inp !== "string") {
        throw new TypeError("Text completion requires a string prompt");
      }
      conversation.prompt = inp;
    } else {
      conversation.appendMessage(msgRole, inp, inpRoleStr);
      if (genConfig?.enable_thinking === false) {
        // TODO(Charlie): In future we should make emptyThinkingBlockStr configurable.
        const emptyThinkingBlockStr = "<think>\n\n</think>\n\n";
        const encoded = Array.from(
          this.tokenizer.encode(emptyThinkingBlockStr),
        );
        this.outputIds.push(...encoded);
        this.assistantPrefixLength = encoded.length;
        assistantPrefixTokenIds.push(...encoded);
        conversation.appendEmptyThinkingReplyHeader(
          Role.assistant,
          emptyThinkingBlockStr,
        );
      } else {
        conversation.appendReplyHeader(Role.assistant);
      }
    }
    let promptLen: number;
    let promptTokenIds: number[] | undefined;
    let logits: tvmjs.Tensor | undefined;
    this.tvm.beginScope();
    try {
      if (this.artifact !== undefined) {
        const [chunks, artifactPromptLen] =
          await this.getArtifactPrefillChunks();
        promptLen = artifactPromptLen;
        if (
          chunks.every((chunk) => chunk.modalityIds.every((kind) => kind === 0))
        ) {
          promptTokenIds = chunks.flatMap((chunk) => chunk.tokenIds);
        }
        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          const previousLength = this.filledKVCacheLength;
          const embeddings =
            chunk.embeddings ?? this.getArtifactTextEmbeddings(chunk.tokenIds);
          const chunkLogits = this.artifactPrefillAndForward(
            embeddings,
            chunk.tokenIds,
            chunk.modalityIds,
          );
          if (
            this.filledKVCacheLength !==
            previousLength + chunk.tokenIds.length
          ) {
            throw new Error(
              "Internal Error: filledKVCacheLength does not match expected value.",
            );
          }
          if (i === chunks.length - 1) {
            logits = this.tvm.detachFromCurrentScope(chunkLogits);
          }
        }
      } else {
        const [inputData, legacyPromptLen, getEmbedSize] =
          await this.getInputData();
        promptLen = legacyPromptLen;
        promptTokenIds = inputData.every((data) => Array.isArray(data))
          ? inputData.flatMap((data) => data as number[])
          : undefined;
        const hasImageInput = inputData.some((data) => !Array.isArray(data));
        if (hasImageInput && this.image_embed === undefined) {
          throw new CannotFindImageEmbedError();
        }
        const [chunks, chunkLens] = getChunkedPrefillInputData(
          inputData,
          this.prefillChunkSize,
          getEmbedSize,
        );
        for (let i = 0; i < chunks.length; i++) {
          const previousLength = this.filledKVCacheLength;
          const chunkLogits = await this.embedAndForward(
            chunks[i],
            chunkLens[i],
          );
          if (this.filledKVCacheLength !== previousLength + chunkLens[i]) {
            throw new Error(
              "Internal Error: filledKVCacheLength does not match expected value.",
            );
          }
          if (i === chunks.length - 1) {
            logits = this.tvm.detachFromCurrentScope(chunkLogits);
          }
        }
      }
    } finally {
      this.imageDataCache.clear();
      this.artifactImagePixelCache.clear();
      this.tvm.endScope();
    }
    if (logits === undefined) {
      throw new Error("Internal Error: prefill did not produce logits.");
    }
    return { logits, promptLen, promptTokenIds, assistantPrefixTokenIds };
  }

  private async forwardDecodeToken(tokenId: number): Promise<tvmjs.Tensor> {
    if (tokenId === undefined) {
      throw new Error("Internal Error: missing decode token.");
    }
    this.tvm.beginScope();
    try {
      const previousLength = this.filledKVCacheLength;
      const logits =
        this.artifact === undefined
          ? await this.embedAndForward([[tokenId]], 1)
          : this.artifactDecodeAndForward(tokenId);
      if (this.filledKVCacheLength !== previousLength + 1) {
        throw new Error(
          "Internal Error: filledKVCacheLength does not match expected value.",
        );
      }
      return this.tvm.detachFromCurrentScope(logits);
    } finally {
      this.tvm.endScope();
    }
  }

  /** Decode and commit the next token, on the readback path chosen at prefill. */
  async decodeStep(
    genConfig?: GenerationConfig,
  ): Promise<CommittedGenerationStep> {
    if (this.stopTriggered) {
      throw Error("Cannot run decode when stopped");
    }

    const mode = this.sampleReadbackPathSelection?.activeMode;
    if (mode === "ring_vector" || mode === "lookahead") {
      this.activeSampleReadbackMode = mode;
      const tstart = performance.now();
      const committed =
        mode === "ring_vector"
          ? await this.decodeStepRingVector(genConfig)
          : await this.decodeStepLookahead(genConfig);
      const elapsed = (performance.now() - tstart) / 1e3;
      this.decodingTotalTime += elapsed;
      this.decodingTotalTokens += 1;
      this.curRoundDecodingTotalTokens += 1;
      this.curRoundDecodingTotalTime += elapsed;
      return committed;
    }
    this.resetSampleReadbackSession("baseline");
    this.activeSampleReadbackMode = "baseline";
    const step = await this.sampleDecodeStep(genConfig);
    return this.commitSampledStep(step, genConfig);
  }

  async sampleDecodeStep(
    genConfig?: GenerationConfig,
  ): Promise<SampledGenerationStep> {
    if (this.stopTriggered) {
      throw Error("Cannot run decode when stopped");
    }

    const tstart = performance.now();

    const lastToken = this.outputIds[this.outputIds.length - 1];
    const prefetched = this.prefetchedDecodeLogits;
    this.prefetchedDecodeLogits = undefined;
    const logits = await (prefetched ?? this.forwardDecodeToken(lastToken));
    let nextToken: number;
    try {
      nextToken = await this.sampleFromRawLogits(logits, genConfig);
    } finally {
      logits.dispose();
    }
    const tend = performance.now();

    this.decodingTotalTime += (tend - tstart) / 1e3;
    this.decodingTotalTokens += 1;
    this.curRoundDecodingTotalTokens += 1;
    this.curRoundDecodingTotalTime += (tend - tstart) / 1e3;

    return {
      source: "decode",
      tokenId: nextToken,
      globalTokenPos: this.filledKVCacheLength,
      logprob: this.tokenLogprobArray.at(-1),
    };
  }

  /**
   * Forward the last committed token now, before the caller awaits other work
   * such as a durable journal append, so the GPU runs the pass meanwhile. The
   * next sampleDecodeStep samples from it. Only a pure KV cache can undo the
   * pass (by popping one entry) if generation stops instead.
   */
  prefetchDecodeForward(): void {
    if (
      this.stopTriggered ||
      // The ring and lookahead paths already run the next forward ahead.
      this.sampleReadbackPathSelection.activeMode !== "baseline" ||
      this.prefetchedDecodeLogits !== undefined ||
      this.decodePrefetchUnavailable ||
      this.kvCache === undefined ||
      this.kvStateKind !== "kv_cache" ||
      this.artifact !== undefined
    ) {
      return;
    }
    try {
      this.getKVCheckpointFunc("vm.builtin.kv_state_popn");
    } catch {
      this.decodePrefetchUnavailable = true;
      return;
    }
    const logits = this.forwardDecodeToken(this.outputIds.at(-1)!);
    // sampleDecodeStep awaits the pass and surfaces its error.
    logits.catch(() => undefined);
    this.prefetchedDecodeLogits = logits;
    // tvmjs submits batched dispatches at the next readback, CPU write, or
    // buffer free. Freeing a 4-byte buffer submits the pass without waiting.
    this.tvm.beginScope();
    this.tvm.empty([1], "int32", this.device);
    this.tvm.endScope();
  }

  private discardPrefetchedDecode(): void {
    const logits = this.prefetchedDecodeLogits;
    if (logits === undefined) {
      return;
    }
    this.prefetchedDecodeLogits = undefined;
    void logits.then(
      (tensor) => tensor.dispose(),
      () => undefined,
    );
    const popN = this.getKVCheckpointFunc("vm.builtin.kv_state_popn");
    popN(
      this.kvCache,
      new tvmjs.Scalar(0, "int64"),
      new tvmjs.Scalar(1, "int32"),
    );
    this.filledKVCacheLength -= 1;
  }

  /**
   * Manually trigger stop if it is not stopped.
   */
  triggerStop() {
    this.discardPrefetchedDecode();
    if (this.stopTriggered) {
      return;
    }
    if (this.ringVectorReadbackSession !== undefined) {
      if (this.hasRingVectorUncommittedTokens(this.ringVectorReadbackSession)) {
        this.stageRingVectorKVRollbackAndResetSession();
      } else {
        this.disposeRingVectorReadbackSession();
      }
    }
    this.disposeLookahead();
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
    processedSeqLen = this.filledKVCacheLength,
  ): void {
    if (this.stopTriggered) {
      throw Error("Cannot call process when it is stoppped");
    }

    // Get max_tokens from generationConfig (specified by user in completion request)
    // If not specified, do not set a limit
    const max_tokens = genConfig?.max_tokens ?? Infinity;
    if (!(max_tokens > 0)) {
      throw new MinValueError("max_tokens", 0);
    }

    // Get ignore_eos from generationConfig (specified by user in completion request)
    let ignore_eos = false;
    if (
      genConfig !== undefined &&
      genConfig.ignore_eos !== undefined &&
      genConfig.ignore_eos !== null
    ) {
      ignore_eos = genConfig.ignore_eos;
    }

    // Get stopStrs, possibly overridden by genConfig for this round
    let stopStrs = this.stopStr;
    if (genConfig !== undefined && genConfig.stop) {
      stopStrs = stopStrs.concat(genConfig.stop);
    }

    let stopTokens = this.stopTokens;
    if (ignore_eos) {
      stopTokens = [];
      stopStrs = [];
    }

    // Stop condition 1: stop token; otherwise, append to `this.outputIds`
    if (stopTokens.includes(nextToken)) {
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
    let outputMessage = this.tokenizer.decode(
      new Int32Array(this.outputIds.slice(this.assistantPrefixLength)),
    );
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
    if (this.getGeneratedTokenCount() >= max_tokens) {
      this.stopTriggered = true;
      this.finishReason = "length";
      log.info("Generation stopped due to exceeding max_tokens.");
    }

    // Stop condition 4: exceed KVCache's context window size
    if (
      this.slidingWindowSize == -1 &&
      processedSeqLen == this.contextWindowSize
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

  private commitSampledToken(
    nextToken: number,
    genConfig: GenerationConfig | undefined,
    source: CommitTokenSource,
    processedSeqLen = this.filledKVCacheLength,
  ): void {
    switch (source) {
      case "prefill":
      case "decode":
        this.commitSamplerState(nextToken, genConfig);
        this.processNextToken(nextToken, genConfig, processedSeqLen);
        this.curRoundCompletionTotalTokens += 1;
        return;
    }
  }

  commitSampledStep(
    step: SampledGenerationStep,
    genConfig?: GenerationConfig,
  ): CommittedGenerationStep {
    return this.commitToken(
      step.tokenId,
      genConfig,
      step.source,
      step.globalTokenPos,
    );
  }

  /** Commit a token fed at globalTokenPos and describe the text it added. */
  private commitToken(
    tokenId: number,
    genConfig: GenerationConfig | undefined,
    source: CommitTokenSource,
    globalTokenPos: number,
  ): CommittedGenerationStep {
    const prevOutputMessage = this.outputMessage;
    this.commitSampledToken(tokenId, genConfig, source, globalTokenPos);
    const textPrefixLength = commonPrefixLength(
      prevOutputMessage,
      this.outputMessage,
    );
    return {
      source,
      tokenId,
      globalTokenPos,
      textDelta: this.outputMessage.slice(textPrefixLength),
      textPrefixLength,
      outputMessage: this.outputMessage,
      stopped: this.stopTriggered,
      finishReason: this.finishReason,
      logprob: this.tokenLogprobArray.at(-1),
    };
  }

  getRNGState(): unknown {
    return (this.tvm as any).getRNGState?.();
  }

  /**
   * RNG state after the last committed token. The ring draws the uniform
   * samples of its speculative tail ahead of time; replaying from this state
   * draws them again.
   */
  getCommittedRNGState(): unknown {
    return (
      this.ringVectorReadbackSession?.committedRNGState ?? this.getRNGState()
    );
  }

  setRNGState(state: unknown): boolean {
    const setRNGState = (this.tvm as any).setRNGState;
    if (setRNGState === undefined) {
      return false;
    }
    setRNGState.call(this.tvm, state);
    return true;
  }

  private getKVCheckpointFunc(name: string): tvmjs.PackedFunc {
    const cached = this.kvCheckpointFuncs.get(name);
    if (cached !== undefined) {
      return cached;
    }
    this.tvm.beginScope();
    try {
      const func = this.tvm.detachFromCurrentScope(
        this.tvm.getGlobalFunc(name),
      );
      this.kvCheckpointFuncs.set(name, func);
      return func;
    } finally {
      this.tvm.endScope();
    }
  }

  private resolveKVCheckpointFuncs(): void {
    for (const name of KV_CHECKPOINT_FUNC_NAMES) {
      this.getKVCheckpointFunc(name);
    }
  }

  /**
   * Start reading back the KV cache's first processedSeqLen positions as a
   * checkpoint. Only pages from the one holding position fromSeqLen onward are
   * copied: a delta over the checkpoint taken at fromSeqLen (0 for a full
   * checkpoint). Logits, if given, are copied now, behind the forward that
   * produced them; the pages after the snapshot is started, as readback slots
   * free up. Returns undefined if the model cannot checkpoint.
   */
  captureCheckpoint(
    processedSeqLen: number,
    fromSeqLen = 0,
    logits?: tvmjs.Tensor,
  ): KVCheckpointSnapshot | undefined {
    if (
      this.kvCache === undefined ||
      this.kvStateKind !== "kv_cache" ||
      this.kvCheckpointUnavailableReason !== undefined
    ) {
      return undefined;
    }
    try {
      this.resolveKVCheckpointFuncs();
      return this.startCheckpointReadback(processedSeqLen, fromSeqLen, logits);
    } catch (err) {
      this.kvCheckpointUnavailableReason = errorMessage(err);
      log.warn(
        `KV checkpoint capture disabled for this model: ${this.kvCheckpointUnavailableReason}`,
      );
      return undefined;
    }
  }

  private checkpointGroupInfo(group: unknown): KVCheckpointGroupInfo {
    const value = group as Record<string, unknown>;
    const groupId = value.group_index ?? value.groupIndex;
    const layerStart = value.layer_begin ?? value.layerBegin;
    const layerEnd = value.layer_end ?? value.layerEnd;
    const shape = value.shape;
    const dtype = value.dtype;
    if (
      typeof groupId !== "number" ||
      !Number.isInteger(groupId) ||
      typeof layerStart !== "number" ||
      !Number.isInteger(layerStart) ||
      typeof layerEnd !== "number" ||
      !Number.isInteger(layerEnd) ||
      !Array.isArray(shape) ||
      shape.length !== 6 ||
      !shape.every(Number.isInteger) ||
      typeof dtype !== "string"
    ) {
      throw new Error("Invalid KV checkpoint group metadata.");
    }
    return {
      groupId,
      layerStart,
      layerEnd,
      shape: shape as number[],
      dtype,
    };
  }

  private checkpointGroups(
    metadata: Record<string, unknown>,
  ): KVCheckpointGroupInfo[] {
    if (!Array.isArray(metadata.groups)) {
      throw new Error("KV checkpoint metadata is missing page groups.");
    }
    return metadata.groups.map((group) => this.checkpointGroupInfo(group));
  }

  private startCheckpointReadback(
    processedSeqLen: number,
    fromSeqLen: number,
    logits?: tvmjs.Tensor,
  ): KVCheckpointSnapshot {
    this.discardCheckpointReadback();
    const seqLen = new tvmjs.Scalar(processedSeqLen, "int64");
    const getMetadata = this.getKVCheckpointFunc(
      "vm.builtin.attention_kv_cache_get_checkpoint_metadata_at",
    );
    const metadata = JSON.parse(
      String(getMetadata(this.kvCache, new tvmjs.Scalar(0, "int64"), seqLen)),
    ) as Record<string, unknown>;
    const groups = this.checkpointGroups(metadata);
    const pageSize = Number(metadata.page_size);
    // Page group layout: (1, num_logical_pages, 2, kv_heads, page_size, dim).
    const [, pageEnd, ...pageShape] = groups[0].shape;
    const pageBytes = byteLength(pageShape, groups[0].dtype);
    const pageBegin =
      fromSeqLen > 0 && fromSeqLen < processedSeqLen
        ? Math.floor(fromSeqLen / pageSize)
        : 0;
    const totalPages = groups.length * (pageEnd - pageBegin);
    const stagingPages = Math.min(
      totalPages,
      Math.max(1, Math.floor(KV_CHECKPOINT_READBACK_BATCH_BYTES / pageBytes)),
    );
    const logitsBytes =
      logits === undefined ? 0 : byteLength(logits.shape, logits.dtype);
    // The ring reads 4-byte elements; float32 logits always qualify.
    const readLogits = logitsBytes > 0 && logitsBytes % 4 === 0;
    const slotBytes = Math.max(stagingPages * pageBytes, logitsBytes);
    const ring = createSampledTokenReadbackRing(this.device, {
      slotCount: KV_CHECKPOINT_READBACK_SLOTS,
      maxTokensPerBatch: slotBytes / 4,
    });

    let resolve!: KVCheckpointReadback["resolve"];
    let reject!: KVCheckpointReadback["reject"];
    const done = new Promise<
      { nextLogits?: KVCheckpointLogitsData } | undefined
    >((res, rej) => {
      resolve = res;
      reject = rej;
    });
    // A failure can settle the readback before its owner awaits it.
    done.catch(() => undefined);
    let markDisposed!: () => void;
    const disposed = new Promise<SampledTokenReadbackBatch[]>((res) => {
      markDisposed = () => res([]);
    });
    let started = false;
    const seqLength = metadata.seq_length ?? metadata.seqLength;
    const layoutHash = metadata.layout_hash ?? metadata.layoutHash;
    const readback: KVCheckpointReadback = {
      snapshot: {
        processedSeqLen:
          typeof seqLength === "number" ? seqLength : processedSeqLen,
        layoutHash: typeof layoutHash === "string" ? layoutHash : undefined,
        metadata,
        pageBegin,
        pageGroups: groups.map((group) => ({
          groupId: group.groupId,
          layerStart: group.layerStart,
          layerEnd: group.layerEnd,
        })),
        byteLength: totalPages * pageBytes + (readLogits ? logitsBytes : 0),
        stagingBytes:
          stagingPages * pageBytes + KV_CHECKPOINT_READBACK_SLOTS * slotBytes,
        done,
        start: (sink) => {
          if (this.kvCheckpointReadback === readback && !started) {
            started = true;
            readback.sink = sink;
            void this.readBackCheckpoint(readback);
          }
        },
        discard: () => {
          if (this.kvCheckpointReadback === readback) {
            this.discardCheckpointReadback();
          }
        },
      },
      groups,
      pageEnd,
      pageBytes,
      nextGroup: 0,
      nextPage: pageBegin,
      ring,
      stagingPages,
      pageBatches: 0,
      inflight: [],
      resolve,
      reject,
      disposed,
      markDisposed,
    };
    this.kvCheckpointReadback = readback;
    if (readLogits) {
      // Copied now, behind the forward that produced them.
      submitTensorReadbackWords(ring, logits!, logitsBytes / 4);
      readback.inflight.push({
        shape: [...logits!.shape],
        dtype: logits!.dtype,
        data: new Uint8Array(),
      });
    }
    return readback.snapshot;
  }

  /**
   * Keep both readback slots busy with pages and pass each batch to the sink
   * as its slot maps, until every page is read back or the readback is
   * discarded. Slot maps drive it, not decode steps, so it also finishes
   * after the reply ends and while a stream is paused. A slot is refilled
   * only after the sink has taken its batch.
   */
  private async readBackCheckpoint(
    readback: KVCheckpointReadback,
  ): Promise<void> {
    try {
      for (;;) {
        while (
          readback.nextGroup < readback.groups.length &&
          readback.inflight.length < KV_CHECKPOINT_READBACK_SLOTS
        ) {
          this.submitCheckpointPages(readback);
        }
        if (readback.inflight.length === 0) {
          break;
        }
        const batches = await Promise.race([
          readback.ring.waitReady(),
          readback.disposed,
        ]);
        if (this.kvCheckpointReadback !== readback) {
          return;
        }
        for (const batch of batches) {
          await this.collectCheckpointBatch(readback, batch);
          if (this.kvCheckpointReadback !== readback) {
            return;
          }
        }
      }
    } catch (err) {
      if (this.kvCheckpointReadback === readback) {
        this.disposeCheckpointReadback(readback);
        readback.reject(err);
      }
      return;
    } finally {
      // No sink call holds the arrays now, and a disposed ring no longer
      // writes into them.
      if (readback.hostBuffers !== undefined) {
        this.kvCheckpointHostBuffers = readback.hostBuffers;
      }
    }
    this.disposeCheckpointReadback(readback);
    readback.resolve({ nextLogits: readback.nextLogits });
  }

  /** Export the next pages into the copy buffer and queue one readback. */
  private submitCheckpointPages(readback: KVCheckpointReadback): void {
    const { groups, pageEnd, pageBytes, stagingPages } = readback;
    if (readback.staging === undefined) {
      this.tvm.beginScope();
      try {
        readback.staging = this.tvm.detachFromCurrentScope(
          this.tvm.empty(
            [1, stagingPages, ...groups[0].shape.slice(2)],
            groups[0].dtype,
            this.device,
          ),
        );
      } finally {
        this.tvm.endScope();
      }
    }
    const exportPageRange = this.getKVCheckpointFunc(
      "vm.builtin.attention_kv_cache_export_page_range",
    );
    const seqId = new tvmjs.Scalar(0, "int64");
    const seqLen = new tvmjs.Scalar(readback.snapshot.processedSeqLen, "int64");
    const segments: KVCheckpointPageSegment[] = [];
    let pages = 0;
    while (pages < stagingPages && readback.nextGroup < groups.length) {
      const count = Math.min(stagingPages - pages, pageEnd - readback.nextPage);
      exportPageRange(
        this.kvCache,
        seqId,
        new tvmjs.Scalar(groups[readback.nextGroup].groupId, "int64"),
        seqLen,
        new tvmjs.Scalar(readback.nextPage, "int64"),
        new tvmjs.Scalar(readback.nextPage + count, "int64"),
        readback.staging,
        new tvmjs.Scalar(pages, "int64"),
      );
      segments.push({
        group: readback.nextGroup,
        page: readback.nextPage,
        count,
      });
      pages += count;
      readback.nextPage += count;
      if (readback.nextPage === pageEnd) {
        readback.nextGroup += 1;
        readback.nextPage = readback.snapshot.pageBegin;
      }
    }
    // Batches complete in submission order and a slot is refilled only after
    // the sink has taken its batch, so alternating arrays never overlap.
    readback.hostBuffers ??= this.kvCheckpointHostBuffers.splice(0);
    const index = readback.pageBatches++ % KV_CHECKPOINT_READBACK_SLOTS;
    const tokenCount = (pages * pageBytes) / 4;
    if ((readback.hostBuffers[index]?.length ?? 0) < tokenCount) {
      readback.hostBuffers[index] = new Int32Array(
        (stagingPages * pageBytes) / 4,
      );
    }
    // The slot copy is queued before the next export reuses the buffer.
    submitTensorReadbackWords(
      readback.ring,
      readback.staging,
      tokenCount,
      readback.hostBuffers[index],
    );
    readback.inflight.push(segments);
  }

  private async collectCheckpointBatch(
    readback: KVCheckpointReadback,
    batch: SampledTokenReadbackBatch,
  ): Promise<void> {
    const bytes = new Uint8Array(
      batch.tokens.buffer,
      batch.tokens.byteOffset,
      batch.tokenCount * 4,
    );
    const entry = readback.inflight.shift();
    if (entry === undefined) {
      throw new Error("KV checkpoint readback received an unexpected batch.");
    }
    if (!Array.isArray(entry)) {
      readback.nextLogits = { ...entry, data: bytes };
      return;
    }
    // Segments follow group order and page order, so each group's bytes
    // reach the sink in file order.
    let offset = 0;
    for (const { group, count } of entry) {
      const length = count * readback.pageBytes;
      await readback.sink!(group, bytes.subarray(offset, offset + length));
      offset += length;
    }
  }

  // Slots still mapping are destroyed by the runtime once their map settles.
  private disposeCheckpointReadback(readback: KVCheckpointReadback): void {
    if (this.kvCheckpointReadback === readback) {
      this.kvCheckpointReadback = undefined;
    }
    readback.ring.dispose();
    readback.staging?.dispose();
    readback.staging = undefined;
    readback.markDisposed();
  }

  private discardCheckpointReadback(): void {
    const readback = this.kvCheckpointReadback;
    if (readback === undefined) {
      return;
    }
    this.disposeCheckpointReadback(readback);
    readback.resolve(undefined);
  }

  /**
   * Import a checkpoint chain: a full checkpoint followed by the deltas that
   * extend it, oldest first. Each delta overwrites the partial last page of
   * the checkpoint before it.
   */
  private async importCheckpointChain(
    chain: KVCheckpointData[],
  ): Promise<void> {
    if (this.kvCache === undefined || this.kvStateKind !== "kv_cache") {
      throw new Error("KV checkpoint import requires a pure KV cache.");
    }
    const tip = chain.at(-1);
    if (tip === undefined || chain[0].pageBegin !== 0) {
      throw new Error("KV checkpoint chain does not start at page 0.");
    }
    const seqId = new tvmjs.Scalar(0, "int64");
    const prepareImport = this.getKVCheckpointFunc(
      "vm.builtin.attention_kv_cache_prepare_import",
    );
    prepareImport(this.kvCache, seqId, JSON.stringify(tip.metadata));

    const groups = new Map(
      this.checkpointGroups(tip.metadata).map((info) => [info.groupId, info]),
    );
    const pageSize = Number(tip.metadata.page_size);
    const importPageRange = this.getKVCheckpointFunc(
      "vm.builtin.attention_kv_cache_import_page_range",
    );
    for (const checkpoint of chain) {
      const pageEnd = Math.ceil(checkpoint.processedSeqLen / pageSize);
      for (const group of checkpoint.pageGroups) {
        const info = groups.get(group.groupId);
        if (info === undefined) {
          throw new Error(
            `Missing metadata for KV checkpoint group ${group.groupId}.`,
          );
        }
        this.tvm.beginScope();
        try {
          const src = this.tvm.empty(
            [1, pageEnd - checkpoint.pageBegin, ...info.shape.slice(2)],
            info.dtype,
            this.device,
          );
          src.copyFromRawBytes(group.data);
          importPageRange(
            this.kvCache,
            seqId,
            new tvmjs.Scalar(group.groupId, "int64"),
            new tvmjs.Scalar(checkpoint.pageBegin, "int64"),
            new tvmjs.Scalar(pageEnd, "int64"),
            src,
            new tvmjs.Scalar(0, "int64"),
          );
        } finally {
          this.tvm.endScope();
        }
      }
    }
    const finishImport = this.getKVCheckpointFunc(
      "vm.builtin.attention_kv_cache_finish_import",
    );
    finishImport(this.kvCache, seqId);
    const getSequenceLength = this.getKVCheckpointFunc(
      "vm.builtin.attention_kv_cache_get_sequence_length",
    );
    this.filledKVCacheLength = Number(getSequenceLength(this.kvCache, seqId));
    await this.device.sync();
  }

  /**
   * Restore a checkpoint chain (full checkpoint, then deltas; see
   * importCheckpointChain) and replay the journaled tokens after its tip.
   */
  async replayFromPromptCheckpoint(
    chain: KVCheckpointData[],
    assistantPrefixTokenIds: number[],
    coveredGeneratedTokens: ReplayedGenerationToken[],
    tailGeneratedTokens: ReplayedGenerationToken[],
    genConfig?: GenerationConfig,
  ): Promise<KVCheckpointReplayResult> {
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }
    this.resetChat();
    this.setConversation(
      getConversation(this.config.conv_template, this.config.conv_config, true),
    );
    this.resetGenerationRoundState();
    this.stopTriggered = false;
    this.finishReason = undefined;
    this.selectSampleReadbackPath(genConfig);

    const checkpoint = chain[chain.length - 1];
    const grammarMatcherInitPromise =
      this.prepareGrammarMatcherForSampling(genConfig);
    await this.importCheckpointChain(chain);
    await Promise.all([this.device.sync(), grammarMatcherInitPromise]);
    if (this.filledKVCacheLength !== checkpoint.processedSeqLen) {
      throw new Error(
        `KV checkpoint sequence length mismatch: expected ${checkpoint.processedSeqLen}, got ${this.filledKVCacheLength}.`,
      );
    }

    this.outputIds.push(...assistantPrefixTokenIds);
    this.assistantPrefixLength = assistantPrefixTokenIds.length;
    for (const token of coveredGeneratedTokens) {
      // The imported cache already covers these tokens. Rebuild output and
      // penalties using each token's original position for the context limit.
      this.commitSamplerState(token.tokenId, genConfig);
      this.processNextToken(token.tokenId, genConfig, token.globalTokenPos);
    }

    for (let i = 0; i < tailGeneratedTokens.length; i++) {
      if (i > 0) {
        const logits = await this.forwardDecodeToken(
          tailGeneratedTokens[i - 1].tokenId,
        );
        logits.dispose();
      }
      this.commitSampledToken(
        tailGeneratedTokens[i].tokenId,
        genConfig,
        "decode",
      );
    }

    if (tailGeneratedTokens.length > 0) {
      return {
        replayedTokens: tailGeneratedTokens.length,
        sampledFromCheckpointLogits: false,
      };
    }
    if (checkpoint.nextLogits === undefined) {
      throw new Error("Prompt checkpoint is missing next-token logits.");
    }
    this.tvm.beginScope();
    try {
      const logits = this.tvm.empty(
        checkpoint.nextLogits.shape,
        checkpoint.nextLogits.dtype,
        this.device,
      );
      logits.copyFromRawBytes(checkpoint.nextLogits.data);
      const tokenId = await this.sampleFromRawLogits(logits, genConfig);
      const sampledToken: SampledGenerationStep = {
        source: "prefill",
        tokenId,
        globalTokenPos: checkpoint.processedSeqLen,
      };
      const committedToken = this.commitSampledStep(sampledToken, genConfig);
      return {
        replayedTokens: 0,
        sampledFromCheckpointLogits: true,
        sampledToken,
        committedToken,
      };
    } finally {
      this.tvm.endScope();
    }
  }

  private commitSamplerState(
    sampledToken: number,
    genConfig?: GenerationConfig,
  ): void {
    this.logitProcessor?.processSampledToken(sampledToken);

    if (this.isGrammarConstrained(genConfig?.response_format)) {
      if (this.grammarMatcher === undefined) {
        throw Error("Expect grammar matcher to be initialized.");
      }
      const tAcceptStart = performance.now();
      const accepted = this.grammarMatcher.acceptToken(sampledToken);
      this.curRoundGrammarPerTokenTotalTime +=
        (performance.now() - tAcceptStart) / 1e3;
      if (!accepted) {
        throw Error("Grammar matcher rejected the newly sampled token.");
      }
    }

    if (
      genConfig?.enable_latency_breakdown &&
      this.pendingSampleOutputTokenBegin !== undefined
    ) {
      const outputTokenEnd = performance.now();
      const outputTokenTimeSpent =
        (outputTokenEnd - this.pendingSampleOutputTokenBegin) / 1e3;
      this.curRoundLatencyBreakdown.totalTime.push(outputTokenTimeSpent);
    }
    this.pendingSampleOutputTokenBegin = undefined;
  }

  private getArtifactTextEmbeddings(inputTokens: number[]): tvmjs.Tensor {
    if (
      inputTokens.length === 0 ||
      inputTokens.length > this.prefillChunkSize
    ) {
      throw new Error(
        `Artifact text chunks must contain 1..${this.prefillChunkSize} tokens`,
      );
    }
    this.tvm.beginScope();
    const inputData = this.tvm.empty(
      [inputTokens.length],
      "int32",
      this.device,
    );
    inputData.copyFrom(inputTokens);
    const embeddings = this.tvm.detachFromCurrentScope(
      this.embed(inputData, this.params) as tvmjs.Tensor,
    );
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embeddings);
    return embeddings;
  }

  private getArtifactAudioEmbeddings(
    part: ChatCompletionContentPartInputAudio,
  ): tvmjs.Tensor {
    if (
      this.artifact?.audioInput === undefined ||
      this.artifactAudioAdapter === undefined
    ) {
      throw new ArtifactManifestError(
        "input_audio was provided, but the artifact has no bound audio adapter",
      );
    }
    const samples = decodeAudioInput(
      part.input_audio,
      this.artifact.audioInput.processor,
    );
    let embeddings: tvmjs.Tensor;
    this.tvm.beginScope();
    try {
      const sampleTensor = this.tvm
        .empty([samples.length], "float32", this.device)
        .copyFrom(samples);
      embeddings = this.tvm.detachFromCurrentScope(
        this.artifactAudioAdapter(sampleTensor, this.params) as tvmjs.Tensor,
      );
    } finally {
      this.tvm.endScope();
    }
    // The caller's scope owns the result, so it is freed there if it is rejected.
    this.tvm.attachToCurrentScope(embeddings);
    if (
      embeddings.shape.length !== 2 ||
      embeddings.shape[0] <= 0 ||
      embeddings.shape[1] <= 0
    ) {
      throw new ArtifactManifestError(
        `audio adapter must return [audio_tokens, hidden_size], got [${embeddings.shape.join(", ")}]`,
      );
    }
    return embeddings;
  }

  private async getArtifactImageEmbeddings(
    part: ChatCompletionContentPartImage,
  ): Promise<tvmjs.Tensor> {
    if (
      this.artifact?.imageInput === undefined ||
      this.artifactImageAdapter === undefined
    ) {
      throw new ArtifactManifestError(
        "image_url was provided, but the artifact has no bound image adapter",
      );
    }
    const { processor, dtype } = this.artifact.imageInput;
    const { resize, num_embeddings } = processor;
    const cacheKey = `${resize.mode}:${resize.width}x${resize.height}:${part.image_url.url}`;
    let pixels = this.artifactImagePixelCache.get(cacheKey);
    if (pixels === undefined) {
      pixels = await getResizedRGBArrayFromURL(part.image_url.url, resize);
      this.artifactImagePixelCache.set(cacheKey, pixels);
    }
    let embeddings: tvmjs.Tensor;
    this.tvm.beginScope();
    try {
      // The values stay 0..255 whichever width the compiled adapter takes.
      const pixelTensor = this.tvm.empty(
        [1, resize.height, resize.width, 3],
        dtype,
        this.device,
      );
      if (dtype === "uint32") {
        pixelTensor.copyFromRawBytes(
          new Uint8Array(Uint32Array.from(pixels).buffer),
        );
      } else {
        pixelTensor.copyFrom(pixels);
      }
      embeddings = this.tvm.detachFromCurrentScope(
        this.artifactImageAdapter(pixelTensor, this.params) as tvmjs.Tensor,
      );
    } finally {
      this.tvm.endScope();
    }
    // The caller's scope owns the result, so it is freed there if it is rejected.
    this.tvm.attachToCurrentScope(embeddings);
    if (
      embeddings.shape.length !== 2 ||
      embeddings.shape[0] !== num_embeddings ||
      embeddings.shape[1] <= 0
    ) {
      throw new ArtifactManifestError(
        `image adapter must return [${num_embeddings}, hidden_size], got [${embeddings.shape.join(", ")}]`,
      );
    }
    return embeddings;
  }

  private static dtypeBytes(dtype: string): number {
    if (dtype === "bfloat16") {
      return 2;
    }
    const match = /^(?:float|int|uint)(\d+)$/.exec(dtype);
    if (match === null || Number(match[1]) % 8 !== 0) {
      throw new Error(`Cannot determine element size for dtype ${dtype}`);
    }
    return Number(match[1]) / 8;
  }

  private sliceTensorRows(
    tensor: tvmjs.Tensor,
    rowStart: number,
    rowCount: number,
  ): tvmjs.Tensor {
    if (rowStart === 0 && rowCount === tensor.shape[0]) {
      return tensor;
    }
    if (this.fTensorCreateView === undefined || tensor.shape.length !== 2) {
      throw new ArtifactManifestError(
        "runtime tensor-view support is required to chunk adapter embeddings",
      );
    }
    const hiddenSize = tensor.shape[1];
    const relativeByteOffset =
      rowStart * hiddenSize * LLMChatPipeline.dtypeBytes(tensor.dtype);
    const view = this.fTensorCreateView(
      tensor,
      this.tvm.makeShapeTuple([rowCount, hiddenSize]),
      tensor.dtype,
      new tvmjs.Scalar(relativeByteOffset, "int"),
    ) as tvmjs.Tensor;
    if (relativeByteOffset === 0) {
      return view;
    }
    // Compiled prefill kernels need inputs with no byte offset, so chunks after
    // the first are copied on the GPU.
    return this.tvm
      .empty([rowCount, hiddenSize], tensor.dtype, this.device)
      .copyFrom(view);
  }

  private getArtifactPromptSegments(): ArtifactPromptSegment[] {
    if (this.conversation.isTextCompletion) {
      if (this.filledKVCacheLength !== 0) {
        throw new TextCompletionExpectsKVEmptyError();
      }
      return this.conversation.getPromptArrayTextCompletion();
    }
    return this.filledKVCacheLength === 0
      ? this.conversation.getArtifactPromptSegments()
      : this.conversation.getArtifactPromptSegmentsLastRound();
  }

  private async getArtifactPrefillChunks(): Promise<
    [ArtifactPrefillChunk[], number]
  > {
    if (this.artifact === undefined) {
      throw new Error("InternalError: artifact prompt path has no artifact");
    }
    const chunks: ArtifactPrefillChunk[] = [];
    let pendingTokens: number[] = [];
    const flushTokens = () => {
      if (pendingTokens.length !== 0) {
        chunks.push({
          tokenIds: pendingTokens,
          modalityIds: new Array(pendingTokens.length).fill(0),
        });
        pendingTokens = [];
      }
    };
    const appendTokens = (tokenIds: Iterable<number>) => {
      for (const tokenId of tokenIds) {
        pendingTokens.push(tokenId);
        if (pendingTokens.length === this.prefillChunkSize) {
          flushTokens();
        }
      }
    };

    if (
      this.filledKVCacheLength === 0 &&
      this.conversation.config.system_prefix_token_ids !== undefined &&
      this.conversation.config.system_prefix_token_ids !== null
    ) {
      appendTokens(this.conversation.config.system_prefix_token_ids);
    }
    for (const segment of this.getArtifactPromptSegments()) {
      if (typeof segment === "string") {
        appendTokens(this.tokenizer.encode(segment));
      } else {
        const isImage = segment.type === "image_url";
        const input = isImage
          ? this.artifact.imageInput
          : this.artifact.audioInput;
        if (input === undefined) {
          throw new ArtifactManifestError(
            `The loaded artifact does not declare an ${isImage ? "image" : "audio"} input`,
          );
        }
        appendTokens(input.prompt.prefix_token_ids);
        flushTokens();

        const embeddings =
          segment.type === "image_url"
            ? await this.getArtifactImageEmbeddings(segment)
            : this.getArtifactAudioEmbeddings(segment);
        const adapterTokenCount = embeddings.shape[0];
        // Reject an input that cannot fit before slicing it.
        const lengthSoFar =
          this.filledKVCacheLength +
          pendingTokens.length +
          adapterTokenCount +
          chunks.reduce((total, chunk) => total + chunk.tokenIds.length, 0);
        if (
          this.slidingWindowSize === -1 &&
          lengthSoFar > this.contextWindowSize
        ) {
          embeddings.dispose();
          throw new ContextWindowSizeExceededError(
            lengthSoFar - this.filledKVCacheLength,
            this.contextWindowSize,
          );
        }
        for (let start = 0; start < adapterTokenCount; ) {
          const count = Math.min(
            this.prefillChunkSize,
            adapterTokenCount - start,
          );
          chunks.push({
            tokenIds: new Array(count).fill(input.prompt.placeholder_token_id),
            modalityIds: new Array(count).fill(1),
            embeddings: this.sliceTensorRows(embeddings, start, count),
          });
          start += count;
        }
        appendTokens(input.prompt.suffix_token_ids);
      }
    }
    flushTokens();
    const promptLength = chunks.reduce(
      (total, chunk) => total + chunk.tokenIds.length,
      0,
    );
    if (promptLength === 0) {
      throw new Error("Prompt must contain at least one token");
    }
    if (
      this.slidingWindowSize === -1 &&
      promptLength + this.filledKVCacheLength > this.contextWindowSize
    ) {
      throw new ContextWindowSizeExceededError(
        promptLength,
        this.contextWindowSize,
      );
    }
    return [chunks, promptLength];
  }

  private artifactPrefillAndForward(
    embeddings: tvmjs.Tensor,
    tokenIds: number[],
    modalityIds: number[],
  ): tvmjs.Tensor {
    if (tokenIds.length === 0) {
      throw new Error("Artifact prefill chunk has no tokens");
    }
    if (
      tokenIds.length !== modalityIds.length ||
      embeddings.shape.length !== 2 ||
      embeddings.shape[0] !== tokenIds.length
    ) {
      throw new Error("Artifact prompt bundle dimensions do not match");
    }
    this.tvm.beginScope();
    const inputLength = tokenIds.length;
    const inputEmbeddings = embeddings.view([
      1,
      inputLength,
      embeddings.shape[1],
    ]);
    const idTensors =
      this.artifact?.generation.inputs === "embeds"
        ? []
        : [tokenIds, modalityIds].map((ids) =>
            this.tvm
              .empty([1, inputLength], "int32", this.device)
              .copyFrom(ids),
          );
    const states = this.beginArtifactForward(inputLength);
    const result = this.prefill(
      inputEmbeddings,
      ...idTensors,
      ...states,
      this.params,
    );
    this.endArtifactForward(states);
    this.filledKVCacheLength += inputLength;
    const logits = this.tvm.detachFromCurrentScope(result.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  /** Manifest decode from a token id that is already on the GPU. */
  private artifactDecodeFromInputTokenDevice(
    inputTokenDevice: tvmjs.Tensor,
  ): tvmjs.Tensor {
    this.tvm.beginScope();
    let input: tvmjs.Tensor;
    if (this.artifact?.generation.inputs === "embeds") {
      const embedding = this.embed(
        inputTokenDevice,
        this.params,
      ) as tvmjs.Tensor;
      input = embedding.view([1, 1, embedding.shape[1]]);
    } else {
      input = inputTokenDevice.view([1, 1]);
    }
    const states = this.beginArtifactForward(1);
    const result = this.decoding(input, ...states, this.params);
    this.endArtifactForward(states);
    this.filledKVCacheLength += 1;
    const logits = this.tvm.detachFromCurrentScope(result.get(0));
    this.tvm.endScope();
    return logits;
  }

  private artifactDecodeAndForward(tokenId: number): tvmjs.Tensor {
    this.tvm.beginScope();
    let input: tvmjs.Tensor;
    if (this.artifact?.generation.inputs === "embeds") {
      const embedding = this.getArtifactTextEmbeddings([tokenId]);
      input = embedding.view([1, 1, embedding.shape[1]]);
    } else {
      input = this.tvm.empty([1, 1], "int32", this.device).copyFrom([tokenId]);
    }
    const states = this.beginArtifactForward(1);
    const result = this.decoding(input, ...states, this.params);
    this.endArtifactForward(states);
    this.filledKVCacheLength += 1;
    const logits = this.tvm.detachFromCurrentScope(result.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  /** Begin a forward of `inputLength` positions on every state the model keeps, in the order the functions take them. */
  private beginArtifactForward(inputLength: number): tvmjs.TVMObject[] {
    const inputLenShape = this.tvm.makeShapeTuple([inputLength]);
    const seqIdsTuple = this.tvm.makeShapeTuple([0]);
    const states = this.getActiveKVStates();
    for (const state of states) {
      this.fKVCacheBeginForward(state, seqIdsTuple, inputLenShape);
    }
    return states;
  }

  private endArtifactForward(states: tvmjs.TVMObject[]): void {
    for (let i = states.length - 1; i >= 0; --i) {
      this.fKVCacheEndForward(states[i]);
    }
  }

  /**
   * Given input tokens, return embeddings of them by calling embed kernel.
   *
   * @note precondition: inputTokens.length <= prefillChunkSize, since we take care of
   * chunking in `getChunkedPrefillInputData()`.
   */
  private getTokensEmbeddings(inputTokens: number[]): tvmjs.Tensor {
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
    const embed: tvmjs.Tensor = this.tvm.detachFromCurrentScope(
      this.embed!(inputData, this.params),
    );
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embed); // tracked by scope of embedAndForward
    return embed;
  }

  /**
   * Compute the number of embedding tokens an image will produce.
   * Must match the model's image_embed output size.
   * Based on mlc_llm/serve/data.py _compute_embed_size.
   */
  private computeImageEmbedSize(
    imageHeight: number,
    imageWidth: number,
  ): number {
    const modelType = this.config.model_type;
    if (modelType === "phi3_v") {
      const [cropH, cropW] = this.calculateCropShape(imageHeight, imageWidth);
      const subTokens = cropH * 12 * (cropW * 12 + 1);
      const glbTokens = 12 * (12 + 1);
      return subTokens + 1 + glbTokens;
    }
    // For models with fixed embed size (e.g. Gemma3V)
    const mmTokens = this.config.model_config?.mm_tokens_per_image;
    if (mmTokens !== undefined) {
      return mmTokens;
    }
    throw new Error(
      `Cannot determine image embed size for model type "${modelType}". ` +
        "Please add mm_tokens_per_image to model_config in mlc-chat-config.json.",
    );
  }

  /**
   * Calculate resize dimensions per model type.
   * Based on vlm_utils.cc CalculateResizeShape
   */
  private calculateResizeShape(
    imageHeight: number,
    imageWidth: number,
  ): [number, number] {
    switch (this.config.model_type) {
      case "phi3_v": {
        const hdNum = 16;
        const ratio = imageWidth / imageHeight;
        let scale = 1;
        while (scale * Math.ceil(scale / ratio) <= hdNum) {
          scale += 1;
        }
        scale -= 1;
        const newW = scale * 336;
        const newH = Math.floor(newW / ratio);
        return [newH, newW];
      }
      default:
        throw new Error(
          `Unsupported model type "${this.config.model_type}" for image resize.`,
        );
    }
  }

  /**
   * Calculate crop dimensions per model type.
   * Based on vlm_utils.cc CalculateCropShape / CalculatePadShape
   */
  private calculateCropShape(
    imageHeight: number,
    imageWidth: number,
  ): [number, number] {
    switch (this.config.model_type) {
      case "phi3_v": {
        const [resizedHeight, resizedWidth] = this.calculateResizeShape(
          imageHeight,
          imageWidth,
        );
        const padH = Math.ceil(resizedHeight / 336) * 336;
        return [Math.floor(padH / 336), Math.floor(resizedWidth / 336)];
      }
      default:
        throw new Error(
          `Unsupported model type "${this.config.model_type}" for image crop.`,
        );
    }
  }

  /**
   * Embed an image input.
   */
  private async getImageEmbeddings(
    inputImage: ImageURL,
  ): Promise<tvmjs.Tensor> {
    this.tvm.beginScope();
    // 1. Transform ImageURL into image input in TVMArray
    const url = inputImage.url;
    const imgData: ImageData =
      this.imageDataCache.get(url) ?? (await getImageDataFromURL(url));
    const pixelValues: Uint8ClampedArray = getRGBArrayFromImageData(imgData);
    const pixelArray = this.tvm
      .empty([imgData.height, imgData.width, 3], "uint32", this.device)
      .copyFrom(pixelValues)
      .view([1, imgData.height, imgData.width, 3]); // NHWC

    // 2. Calculate resize and crop dimensions
    const [resizeH, resizeW] = this.calculateResizeShape(
      imgData.height,
      imgData.width,
    );
    const [cropH, cropW] = this.calculateCropShape(
      imgData.height,
      imgData.width,
    );
    const resizeHeightShape = this.tvm.makeShapeTuple([resizeH]);
    const resizeWidthShape = this.tvm.makeShapeTuple([resizeW]);
    const cropHeightShape = this.tvm.makeShapeTuple([cropH]);
    const cropWidthShape = this.tvm.makeShapeTuple([cropW]);

    // 3. embed kernel
    const embed: tvmjs.Tensor = this.tvm.detachFromCurrentScope(
      this.image_embed!(
        pixelArray,
        resizeHeightShape,
        resizeWidthShape,
        cropHeightShape,
        cropWidthShape,
        this.params,
      ),
    );
    const expectedSize = this.computeImageEmbedSize(
      imgData.height,
      imgData.width,
    );
    if (embed.shape[0] !== expectedSize) {
      throw new Error(
        `InternalError: expect embed.shape[0] to be ${expectedSize}, ` +
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
   * @returns The logits returned by this forward as tvmjs.Tensor on GPU.
   *
   * @note Precondition: inputData's data length is smaller than prefill chunk size
   */
  private async embedAndForward(
    inputData: Array<Array<number> | ImageURL>,
    inputDataLen: number,
  ): Promise<tvmjs.Tensor> {
    if (inputDataLen > this.prefillChunkSize) {
      throw new Error(
        "InternalError: expect inputDataLen <= this.prefillChunkSize.",
      );
    }
    // TODO: we should combine string data to embed once, then rearrange the embeddings; currently
    // ["hi", imageUrl, "hi"] would call embed kernels 3 times, while 2 would suffice.

    // 1. Embed all inputData
    this.tvm.beginScope();
    const embeddings: tvmjs.Tensor[] = [];
    for (let i = 0; i < inputData.length; i++) {
      const data = inputData[i];
      if (Array.isArray(data)) {
        embeddings.push(this.getTokensEmbeddings(data));
      } else {
        embeddings.push(await this.getImageEmbeddings(data));
      }
    }

    // 2. Concatenate embeddings
    let allEmbeddings: tvmjs.Tensor;
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
    const forwardStates = this.getActiveKVStates();
    for (const state of forwardStates) {
      this.fKVCacheBeginForward!(state, seqIdsTuple, inputLenShape);
    }
    let retValue;
    if (inputDataLen > 1) {
      retValue = this.invokePrefill(allEmbeddings, inputDataLen);
    } else {
      retValue = this.invokeDecode(allEmbeddings);
    }

    // Epilogue
    for (let i = forwardStates.length - 1; i >= 0; --i) {
      this.fKVCacheEndForward!(forwardStates[i]);
    }
    this.filledKVCacheLength += inputDataLen;
    const logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  private static loadVMFunctionRegistry(
    vm: tvmjs.VirtualMachine,
    names: string[],
  ): VMFunctionRegistry {
    const registry: VMFunctionRegistry = {};
    for (const name of names) {
      try {
        const func = vm.getFunction(name) as unknown;
        if (typeof func === "function") {
          registry[name] = func as tvmjs.PackedFunc;
        }
      } catch {
        // no-op for unexported symbols
      }
    }
    return registry;
  }

  private static getRequiredVMFunctionByName(
    name: string,
    registry: VMFunctionRegistry,
  ): tvmjs.PackedFunc {
    const func = registry[name];
    if (func !== undefined) {
      return func;
    }

    const availableNames = Object.entries(registry)
      .filter((entry) => entry[1] !== undefined)
      .map((entry) => entry[0])
      .sort();
    const availableStr =
      availableNames.length === 0 ? "(none)" : availableNames.join(", ");
    throw new Error(
      `Cannot find required VM function \`${name}\`. Available candidate functions: ${availableStr}`,
    );
  }

  private static getVMFunctionAvailability(
    registry: VMFunctionRegistry,
  ): VMFunctionAvailability {
    return {
      prefill: registry.prefill !== undefined,
      batch_prefill: registry.batch_prefill !== undefined,
      decode: registry.decode !== undefined,
      batch_decode: registry.batch_decode !== undefined,
      create_tir_paged_kv_cache:
        registry.create_tir_paged_kv_cache !== undefined,
      create_rnn_state: registry.create_rnn_state !== undefined,
    };
  }

  private parseKVStateKind(kvStateKindRaw: unknown): KVStateKind {
    if (kvStateKindRaw === undefined || kvStateKindRaw === null) {
      return "kv_cache";
    }
    if (typeof kvStateKindRaw !== "string") {
      throw new Error(
        `Invalid kv_state_kind in model metadata: expected string, got ${typeof kvStateKindRaw}`,
      );
    }
    const kvStateKind = kvStateKindRaw as KVStateKind;
    if (
      kvStateKind === "kv_cache" ||
      kvStateKind === "rnn_state" ||
      kvStateKind === "hybrid" ||
      kvStateKind === "none"
    ) {
      return kvStateKind;
    }
    throw new Error(
      `Unsupported kv_state_kind in model metadata: ${kvStateKindRaw}`,
    );
  }

  private static resolveModelABI(
    kvStateKind: KVStateKind,
    availability: VMFunctionAvailability,
  ): ResolvedModelABI {
    const hasSingleKernelPair = availability.prefill && availability.decode;
    const hasBatchKernelPair =
      availability.batch_prefill && availability.batch_decode;
    const availableNames = Object.entries(availability)
      .filter((entry) => entry[1])
      .map((entry) => entry[0])
      .sort();
    const availableStr =
      availableNames.length === 0 ? "(none)" : availableNames.join(", ");

    if (kvStateKind === "none") {
      throw new Error(
        "kv_state_kind=`none` is not supported in LLMChatPipeline chat runtime.",
      );
    }

    if (kvStateKind === "hybrid") {
      const missingFunctions: string[] = [];
      if (!availability.batch_prefill) {
        missingFunctions.push("batch_prefill");
      }
      if (!availability.batch_decode) {
        missingFunctions.push("batch_decode");
      }
      if (!availability.create_tir_paged_kv_cache) {
        missingFunctions.push("create_tir_paged_kv_cache");
      }
      if (!availability.create_rnn_state) {
        missingFunctions.push("create_rnn_state");
      }
      if (missingFunctions.length !== 0) {
        throw new Error(
          "Invalid hybrid ABI. Missing required functions: " +
            `${missingFunctions.join(", ")}. ` +
            `Available candidate functions: ${availableStr}`,
        );
      }
      return {
        kvStateKind,
        prefillABI: "batch",
        decodeABI: "batch",
        prefillFunctionName: "batch_prefill",
        decodeFunctionName: "batch_decode",
        needsKVCache: true,
        needsRNNState: true,
      };
    }

    if (kvStateKind === "rnn_state") {
      if (!availability.create_rnn_state) {
        throw new Error(
          "Invalid rnn_state ABI. Missing required function: create_rnn_state.",
        );
      }
      if (hasSingleKernelPair) {
        return {
          kvStateKind,
          prefillABI: "single",
          decodeABI: "single",
          prefillFunctionName: "prefill",
          decodeFunctionName: "decode",
          needsKVCache: false,
          needsRNNState: true,
        };
      }
      if (hasBatchKernelPair) {
        return {
          kvStateKind,
          prefillABI: "batch",
          decodeABI: "batch",
          prefillFunctionName: "batch_prefill",
          decodeFunctionName: "batch_decode",
          needsKVCache: false,
          needsRNNState: true,
        };
      }
      throw new Error(
        "Invalid rnn_state ABI. Require either `prefill`+`decode` or " +
          "`batch_prefill`+`batch_decode`. " +
          `Available candidate functions: ${availableStr}`,
      );
    }

    // kv_cache
    if (hasSingleKernelPair) {
      return {
        kvStateKind,
        prefillABI: "single",
        decodeABI: "single",
        prefillFunctionName: "prefill",
        decodeFunctionName: "decode",
        needsKVCache: true,
        needsRNNState: false,
      };
    }
    if (hasBatchKernelPair) {
      return {
        kvStateKind,
        prefillABI: "batch",
        decodeABI: "batch",
        prefillFunctionName: "batch_prefill",
        decodeFunctionName: "batch_decode",
        needsKVCache: true,
        needsRNNState: false,
      };
    }
    throw new Error(
      "Invalid kv_cache ABI. Require either `prefill`+`decode` or " +
        "`batch_prefill`+`batch_decode`. " +
        `Available candidate functions: ${availableStr}`,
    );
  }

  private requireKVCache(): tvmjs.TVMObject {
    if (this.kvCache === undefined) {
      throw new Error("InternalError: kv cache is not initialized.");
    }
    return this.kvCache;
  }

  private requireRNNState(): tvmjs.TVMObject {
    if (this.rnnState === undefined) {
      throw new Error("InternalError: rnn state is not initialized.");
    }
    return this.rnnState;
  }

  private getSingleStateForABI(): tvmjs.TVMObject {
    if (
      this.resolvedModelABI.needsKVCache &&
      !this.resolvedModelABI.needsRNNState
    ) {
      return this.requireKVCache();
    }
    if (
      !this.resolvedModelABI.needsKVCache &&
      this.resolvedModelABI.needsRNNState
    ) {
      return this.requireRNNState();
    }
    throw new Error(
      "InternalError: single-state ABI requested for a hybrid/non-single state model.",
    );
  }

  private getActiveKVStates(): tvmjs.TVMObject[] {
    const states: tvmjs.TVMObject[] = [];
    if (this.kvCache !== undefined) {
      states.push(this.kvCache);
    }
    if (this.rnnState !== undefined) {
      states.push(this.rnnState);
    }
    if (states.length === 0) {
      throw new Error("InternalError: no kv states initialized.");
    }
    return states;
  }

  private invokePrefill(
    allEmbeddings: tvmjs.Tensor,
    inputDataLen: number,
  ): any {
    if (this.resolvedModelABI.prefillABI === "single") {
      return this.prefill(
        allEmbeddings,
        this.getSingleStateForABI(),
        this.params,
      );
    }

    if (this.prefillLogitPositions === undefined) {
      throw new Error(
        "InternalError: batch prefill ABI requires prefillLogitPositions tensor.",
      );
    }
    this.prefillLogitPositionHost[0] = inputDataLen - 1;
    this.prefillLogitPositions.copyFrom(this.prefillLogitPositionHost);

    if (
      this.resolvedModelABI.needsKVCache &&
      this.resolvedModelABI.needsRNNState
    ) {
      return this.prefill(
        allEmbeddings,
        this.prefillLogitPositions,
        this.requireKVCache(),
        this.requireRNNState(),
        this.params,
      );
    }
    return this.prefill(
      allEmbeddings,
      this.prefillLogitPositions,
      this.getSingleStateForABI(),
      this.params,
    );
  }

  private invokeDecode(allEmbeddings: tvmjs.Tensor): any {
    if (this.resolvedModelABI.decodeABI === "single") {
      return this.decoding(
        allEmbeddings,
        this.getSingleStateForABI(),
        this.params,
      );
    }
    if (
      this.resolvedModelABI.needsKVCache &&
      this.resolvedModelABI.needsRNNState
    ) {
      return this.decoding(
        allEmbeddings,
        this.requireKVCache(),
        this.requireRNNState(),
        this.params,
      );
    }
    return this.decoding(
      allEmbeddings,
      this.getSingleStateForABI(),
      this.params,
    );
  }

  // NOTE: caller must call device.sync()
  private updateLogitsOnCPU(logits: tvmjs.Tensor | Float32Array): tvmjs.Tensor {
    const shape =
      logits instanceof Float32Array ? [1, logits.length] : logits.shape;
    if (this.logitsOnCPU == undefined) {
      this.logitsOnCPU = this.tvm.detachFromCurrentScope(
        this.tvm.empty(shape, "float32", this.tvm.cpu()),
      );
    } else {
      if (shape[0] != this.logitsOnCPU.shape[0]) {
        throw Error("We expect the size of logits to remain unchanged");
      }
    }
    this.logitsOnCPU.copyFrom(logits);
    return this.logitsOnCPU;
  }

  /**
   * @param readSampledToken Reads the sampled token from the GPU tensor instead
   * of a blocking copy, so the caller can queue more GPU work first. Not used
   * when logprobs are requested, since those read the probabilities as well.
   */
  private async sampleFromRawLogits(
    logitsOnGPU: tvmjs.Tensor,
    genConfig?: GenerationConfig,
    readSampledToken?: (
      sampledTokensDevice: tvmjs.Tensor,
      probsDevice?: tvmjs.Tensor,
    ) => Promise<{ token: number; probs?: Float32Array }>,
  ): Promise<number> {
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
      // TODO: setting top_p to 1.0 by default might run into issues since
      // top_p masking in relax uses < instead of <=
      // Set default top_p to 1.0 if not set
      if (!_hasValue(top_p)) {
        top_p = 1.0;
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
      // If only one of frequency or presence penalty is set, make the other one 0.0
      if (_hasValue(frequency_penalty) && !_hasValue(presence_penalty)) {
        presence_penalty = 0.0;
      }
      if (_hasValue(presence_penalty) && !_hasValue(frequency_penalty)) {
        frequency_penalty = 0.0;
      }
      if (!_hasValue(frequency_penalty)) {
        frequency_penalty = 0.0;
      }
      if (!_hasValue(presence_penalty)) {
        presence_penalty = 0.0;
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
    if (!(top_p > 0 && top_p <= 1)) {
      throw new RangeError("top_p", 0, 1);
    }
    if (!(temperature >= 0)) {
      throw new MinValueError("temperature", 0);
    }
    if (!(repetition_penalty > 0)) {
      throw new MinValueError("repetition_penalty", 0);
    }
    if (!(frequency_penalty >= -2.0 && frequency_penalty <= 2.0)) {
      throw new RangeError("frequency_penalty", -2.0, 2.0);
    }
    if (!(presence_penalty >= -2.0 && presence_penalty <= 2.0)) {
      throw new RangeError("presence_penalty", -2.0, 2.0);
    }

    const outputTokenBegin = performance.now();
    this.pendingSampleOutputTokenBegin = outputTokenBegin;
    const grammarConstrained = this.isGrammarConstrained(response_format);

    // 0. Update logitsOnGPU with on-GPU grammar bitmasking
    if (grammarConstrained) {
      const grammarBitmaskBegin = performance.now();

      this.tvm.beginScope();
      if (this.grammarMatcher === undefined) {
        throw Error("Expect grammar matcher to be initialized.");
      }

      const tBitmaskStart = performance.now();
      const bitMaskOnCPU: Int32Array =
        await this.grammarMatcher.getNextTokenBitmask();
      this.curRoundGrammarPerTokenTotalTime +=
        (performance.now() - tBitmaskStart) / 1e3;

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

      if (genConfig?.enable_latency_breakdown) {
        const grammarBitmaskEnd = performance.now();
        const grammarBitmaskTimeSpent =
          (grammarBitmaskEnd - grammarBitmaskBegin) / 1e3;
        this.curRoundLatencyBreakdown.grammarBitmaskTime.push(
          grammarBitmaskTimeSpent,
        );
      }
    }

    // 1. Apply logitProcessor on CPU
    if (this.logitProcessor !== undefined) {
      // Move logits to CPU
      this.tvm.beginScope();
      this.updateLogitsOnCPU(logitsOnGPU);
      this.tvm.endScope();
      await this.device.sync();

      const logitProcessorBegin = performance.now();

      if (this.logitsOnCPU == undefined) {
        throw Error("logits should be assigned");
      }
      let logitsOnCPUArray: Float32Array = <Float32Array>(
        this.logitsOnCPU.toArray()
      );
      logitsOnCPUArray = this.logitProcessor.processLogits(logitsOnCPUArray);
      logitsOnGPU.copyFrom(logitsOnCPUArray);
      this.logitsOnCPU.copyFrom(logitsOnCPUArray);

      if (genConfig?.enable_latency_breakdown) {
        const logitProcessorEnd = performance.now();
        const logitProcessorTimeSpent =
          (logitProcessorEnd - logitProcessorBegin) / 1e3;
        this.curRoundLatencyBreakdown.logitProcessorTime.push(
          logitProcessorTimeSpent,
        );
      }
    }

    // 2. Apply logit_bias on GPU
    if (_hasValue(logit_bias)) {
      const logitBiasBegin = performance.now();

      const numTokens = Object.keys(logit_bias ?? {}).length;
      const pos2seqIds = new Int32Array(numTokens).fill(0);
      const tokenIds = new Int32Array(numTokens);
      const tokenLogitBias = new Float32Array(numTokens);

      const logitBiasKeys = Object.keys(logit_bias ?? {});
      for (let index = 0; index < numTokens; index++) {
        const tokenId = parseInt(logitBiasKeys[index]);
        tokenIds[index] = tokenId;
        tokenLogitBias[index] = logit_bias![tokenId];
      }

      this.tvm.beginScope();

      const pos2seqIdsDevice = this.tvm
        .empty([numTokens], "int32", this.device)
        .copyFrom(pos2seqIds);

      const tokenIdsDevice = this.tvm
        .empty([numTokens], "int32", this.device)
        .copyFrom(tokenIds);

      const tokenLogitBiasDevice = this.tvm
        .empty([numTokens], "float32", this.device)
        .copyFrom(tokenLogitBias);

      this.fapplyLogitBias(
        logitsOnGPU.view([1, this.fullVocabSize]),
        pos2seqIdsDevice,
        tokenIdsDevice,
        tokenLogitBiasDevice,
      );

      this.tvm.endScope();

      if (genConfig?.enable_latency_breakdown) {
        const logitBiasEnd = performance.now();
        const logitBiasTimeSpent = (logitBiasEnd - logitBiasBegin) / 1e3;
        this.curRoundLatencyBreakdown.logitBiasTime.push(logitBiasTimeSpent);
      }
    }

    // 3. Apply penalties to logits on GPU
    if (
      frequency_penalty != 0.0 ||
      presence_penalty != 0.0 ||
      repetition_penalty != 1.0
    ) {
      const appearedTokens = [...this.appearedTokensFreq.keys()];
      const appearedTokensFreqs = [...this.appearedTokensFreq.values()];

      const numTokens = appearedTokens.length;

      if (numTokens > 0) {
        const penaltyBegin = performance.now();

        const pos2seqIds = new Int32Array(numTokens).fill(0);
        const tokenIds = new Int32Array(numTokens).fill(0);
        const tokenCnt = new Int32Array(numTokens).fill(0);
        const penalties = new Float32Array([
          presence_penalty,
          frequency_penalty,
          repetition_penalty,
        ]);

        tokenIds.set(appearedTokens);
        tokenCnt.set(appearedTokensFreqs);

        this.tvm.beginScope();
        const seqIdsArray = this.tvm
          .empty([1], "int32", this.device)
          .copyFrom([0]);

        const pos2seqIdsDevice = this.tvm
          .empty([numTokens], "int32", this.device)
          .copyFrom(pos2seqIds);

        const tokenIdsDevice = this.tvm
          .empty([numTokens], "int32", this.device)
          .copyFrom(tokenIds);

        const tokenCntDevice = this.tvm
          .empty([numTokens], "int32", this.device)
          .copyFrom(tokenCnt);

        const penaltiesDevice = this.tvm
          .empty([1, 3], "float32", this.device)
          .copyFrom(penalties);

        this.fapplyPenalty(
          logitsOnGPU.view([1, this.fullVocabSize]),
          seqIdsArray,
          pos2seqIdsDevice,
          tokenIdsDevice,
          tokenCntDevice,
          penaltiesDevice,
        );

        this.tvm.endScope();

        if (genConfig?.enable_latency_breakdown) {
          const penaltyEnd = performance.now();
          const penaltyTimeSpent = (penaltyEnd - penaltyBegin) / 1e3;
          this.curRoundLatencyBreakdown.penaltyTime.push(penaltyTimeSpent);
        }
      }
    }

    // TODO: Explore usage of multinomial sampling kernel (currently blocked due to usage
    // of i8) for cases where top_p is not set
    // 4. Sample token from logits
    const sampleBegin = performance.now();

    const numSeqs = 1;
    const numProbs = 1;
    const wantLogprobs = logprobs === true;
    // Inplace transform logitsOnCPU to a distribution
    temperature = Math.max(1e-6, temperature); // to prevent division by zero

    const temperatures = new Float32Array([temperature]);

    let sampledToken = -1;
    this.tvm.beginScope();
    const temperaturesDevice = this.tvm
      .empty([numSeqs], "float32", this.device)
      .copyFrom(temperatures);

    let probs = this.fsoftmaxWithTemperature(
      logitsOnGPU.view([numSeqs, numProbs, this.fullVocabSize]),
      temperaturesDevice,
    );
    probs = probs.view([numProbs, this.fullVocabSize]);

    const topPValue = Math.max(top_p, 1e-5);
    const argsortResults = this.fargsortProbs(probs);
    const sortedProbsDevice = argsortResults.get(0);
    const sortedIndicesDevice = argsortResults.get(1);
    const uniformSamplesDevice = this.tvm.uniform([1], 0.0, 1.0, this.device);

    const topPHost = new Float32Array(numProbs).fill(-1);
    this.sampleIndices.forEach((row) => {
      topPHost[row] = topPValue;
    });
    this.topPDevice.copyFrom(topPHost);

    const sampledTokensDevice = this.fsampleWithTopP(
      sortedProbsDevice,
      sortedIndicesDevice,
      uniformSamplesDevice,
      this.sampleIndicesDevice,
      this.topPDevice,
    );
    if (readSampledToken !== undefined) {
      const deferredTokens =
        this.tvm.detachFromCurrentScope(sampledTokensDevice);
      const deferredProbs = wantLogprobs
        ? this.tvm.detachFromCurrentScope(probs)
        : undefined;
      this.tvm.endScope();
      try {
        const sampled = await readSampledToken(deferredTokens, deferredProbs);
        sampledToken = sampled.token;
        if (sampled.probs !== undefined) {
          this.updateLogitsOnCPU(sampled.probs);
        }
      } finally {
        deferredTokens.dispose();
        deferredProbs?.dispose();
      }
    } else {
      const sampledTokensHost = this.tvm.detachFromCurrentScope(
        this.tvm
          .empty([numSeqs], "int32", this.tvm.cpu())
          .copyFrom(sampledTokensDevice),
      );
      if (wantLogprobs) {
        this.updateLogitsOnCPU(probs);
      }
      this.tvm.endScope();
      await this.device.sync();

      sampledToken = sampledTokensHost.toArray()[0];
      sampledTokensHost.dispose();
    }
    if (sampledToken < 0) {
      this.pendingSampleOutputTokenBegin = undefined;
      throw new Error("InternalError: failed to sample a valid token.");
    }

    if (wantLogprobs) {
      this.tokenLogprobArray.push(
        this.getTokenLogprob(sampledToken, top_logprobs ?? 0),
      );
    }

    if (genConfig?.enable_latency_breakdown) {
      const sampleEnd = performance.now();
      const sampleTimeSpent = (sampleEnd - sampleBegin) / 1e3;
      this.curRoundLatencyBreakdown.sampleTime.push(sampleTimeSpent);
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
  private async getInputData(): Promise<
    [Array<Array<number> | ImageURL>, number, (image: ImageURL) => number]
  > {
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
        prompts = this.conversation.getPromptArray(this.config);
      } else {
        prompts = this.conversation.getPromptArrayLastRound(this.config);
      }
    }

    // 1.5. Preload image dimensions to compute per-image embed sizes
    const imageDimensions = new Map<string, [number, number]>();
    const uniqueImageUrls = new Set<string>();
    for (const prompt of prompts) {
      if (typeof prompt !== "string") {
        for (const content of prompt) {
          if (typeof content !== "string") {
            uniqueImageUrls.add(content.url);
          }
        }
      }
    }
    if (uniqueImageUrls.size > 0 && this.image_embed === undefined) {
      throw new CannotFindImageEmbedError();
    }
    this.imageDataCache.clear();
    await Promise.all(
      Array.from(uniqueImageUrls).map(async (url) => {
        const imgData = await getImageDataFromURL(url);
        this.imageDataCache.set(url, imgData);
        imageDimensions.set(url, [imgData.height, imgData.width]);
      }),
    );
    const getEmbedSize = (image: ImageURL): number => {
      const dims = imageDimensions.get(image.url);
      if (!dims) {
        throw new Error("InternalError: image dimensions not preloaded");
      }
      return this.computeImageEmbedSize(dims[0], dims[1]);
    };

    // 2. Resolve BOI/EOI tokens
    const cfg = this.config.model_config;
    const boiToken = cfg?.vision_start_token_id ?? cfg?.boi_token_index;
    const eoiToken = cfg?.vision_end_token_id ?? cfg?.eoi_token_index;

    // 3. Encode all prompts. Iterate through each message in the prompt array, where each
    // prompt can either be a string, or an array of a mixture of string and ImageURLs.
    let numPromptTokens = 0;
    for (let i = 0; i < prompts.length; i++) {
      const curPrompt = prompts[i];
      if (typeof curPrompt === "string") {
        const encoded = this.tokenizer.encode(curPrompt);
        numPromptTokens += encoded.length;
        for (const tokenId of encoded) {
          curTokens.push(tokenId);
        }
      } else {
        for (let j = 0; j < curPrompt.length; j++) {
          const curPromptContent: string | ImageURL = curPrompt[j];
          if (typeof curPromptContent === "string") {
            const encoded = this.tokenizer.encode(curPromptContent);
            numPromptTokens += encoded.length;
            for (const tokenId of encoded) {
              curTokens.push(tokenId);
            }
          } else {
            // Insert BOI token before image if configured
            if (boiToken !== undefined) {
              curTokens.push(boiToken);
              numPromptTokens += 1;
            }
            // push curTokens to ret, push imageUrl, create a new curTokens
            ret.push([...curTokens]);
            ret.push(curPromptContent);
            numPromptTokens += getEmbedSize(curPromptContent);
            curTokens = [];
            // Insert EOI token after image if configured
            if (eoiToken !== undefined) {
              curTokens.push(eoiToken);
              numPromptTokens += 1;
            }
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
    return [ret, numPromptTokens, getEmbedSize];
  }

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
  ): Promise<number> {
    const tstart = performance.now();
    this.tvm.beginScope();
    if (this.artifact !== undefined) {
      if (inputIds.length === 0) {
        throw new Error("inputIds must not be empty");
      }
      let logitsOnGPU: tvmjs.Tensor;
      if (isPrefill) {
        for (let start = 0; start < inputIds.length; ) {
          const tokenIds = inputIds.slice(start, start + this.prefillChunkSize);
          logitsOnGPU = this.artifactPrefillAndForward(
            this.getArtifactTextEmbeddings(tokenIds),
            tokenIds,
            new Array(tokenIds.length).fill(0),
          );
          start += tokenIds.length;
        }
      } else {
        if (inputIds.length !== 1) {
          throw new Error("Artifact decode expects exactly one token ID");
        }
        logitsOnGPU = this.artifactDecodeAndForward(inputIds[0]);
      }
      const nextToken = await this.sampleFromRawLogits(logitsOnGPU!);
      this.commitSamplerState(nextToken, undefined);
      this.tvm.endScope();

      const tend = performance.now();
      if (isPrefill) {
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

    // 1. Chunk inputData if needed
    const inputData: Array<Array<number>> = [inputIds];
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize,
      () => 0, // text-only path, no images
    );
    const chunks: Array<Array<number> | ImageURL>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    // 2. Prefill each chunk
    let logitsOnGPU: tvmjs.Tensor;
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
    const nextToken = await this.sampleFromRawLogits(logitsOnGPU!);
    this.commitSamplerState(nextToken);
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

  private async forwardKnownTokens(
    inputIds: Array<number>,
    isPrefill: boolean,
  ): Promise<tvmjs.Tensor | undefined> {
    if (inputIds.length === 0) {
      return undefined;
    }
    const tstart = performance.now();
    this.tvm.beginScope();
    const inputData: Array<Array<number>> = [inputIds];
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize,
      () => 0,
    );
    const chunks: Array<Array<number> | ImageURL>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    let lastLogits: tvmjs.Tensor | undefined;
    try {
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const chunkLen = chunkLens[i];
        const prevFilledLen = this.filledKVCacheLength;
        const tokenIds = (chunk as number[][]).flat();
        const logits =
          this.artifact === undefined
            ? await this.embedAndForward(chunk, chunkLen)
            : this.artifactPrefillAndForward(
                this.getArtifactTextEmbeddings(tokenIds),
                tokenIds,
                new Array(tokenIds.length).fill(0),
              );
        if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
          throw new Error(
            "Internal Error: filledKVCacheLength does not match expected value.",
          );
        }
        if (i === chunks.length - 1) {
          lastLogits = this.tvm.detachFromCurrentScope(logits);
        }
      }
    } finally {
      this.tvm.endScope();
    }

    const tend = performance.now();
    if (isPrefill) {
      this.prefillTotalTime += (tend - tstart) / 1e3;
      this.prefillTotalTokens += inputIds.length;
      this.curRoundPrefillTotalTokens += inputIds.length;
      this.curRoundPrefillTotalTime += (tend - tstart) / 1e3;
    }
    return lastLogits;
  }

  async replayGenerationTokens(
    promptTokenIds: number[],
    assistantPrefixTokenIds: number[],
    generatedTokens: ReplayedGenerationToken[],
    genConfig?: GenerationConfig,
  ): Promise<KVCheckpointReplayResult> {
    if (this.resetStatsPerPrefill) {
      this.resetRuntimeStats();
    }
    this.resetChat();
    this.setConversation(
      getConversation(this.config.conv_template, this.config.conv_config, true),
    );
    this.resetGenerationRoundState();
    this.stopTriggered = false;
    this.finishReason = undefined;
    this.selectSampleReadbackPath(genConfig);

    const grammarMatcherInitPromise =
      this.prepareGrammarMatcherForSampling(genConfig);
    const promptLogits = await this.forwardKnownTokens(promptTokenIds, true);
    try {
      await Promise.all([this.device.sync(), grammarMatcherInitPromise]);

      this.outputIds.push(...assistantPrefixTokenIds);
      this.assistantPrefixLength = assistantPrefixTokenIds.length;
      if (generatedTokens.length === 0) {
        if (promptLogits === undefined) {
          throw new Error("Cannot resume from an empty prompt token record.");
        }
        const tokenId = await this.sampleFromRawLogits(promptLogits, genConfig);
        const sampledToken: SampledGenerationStep = {
          source: "prefill",
          tokenId,
          globalTokenPos: promptTokenIds.length,
        };
        const committedToken = this.commitSampledStep(sampledToken, genConfig);
        return {
          replayedTokens: 0,
          sampledFromCheckpointLogits: false,
          sampledToken,
          committedToken,
        };
      }
      for (let i = 0; i < generatedTokens.length; i++) {
        if (i > 0) {
          const logits = await this.forwardDecodeToken(
            generatedTokens[i - 1].tokenId,
          );
          logits.dispose();
        }
        this.commitSampledToken(
          generatedTokens[i].tokenId,
          genConfig,
          "decode",
        );
      }
      return {
        replayedTokens: generatedTokens.length,
        sampledFromCheckpointLogits: false,
      };
    } finally {
      promptLogits?.dispose();
    }
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
    const prefillChunk: Array<Array<number>> = [tokens] as Array<Array<number>>;
    const prefillChunkLen = tokens.length;
    const prefillStart = performance.now();
    if (this.artifact === undefined) {
      await this.embedAndForward(prefillChunk, prefillChunkLen);
    } else {
      this.artifactPrefillAndForward(
        this.getArtifactTextEmbeddings(tokens),
        tokens,
        new Array(tokens.length).fill(0),
      );
    }
    this.tvm.endScope();
    await this.device.sync();

    const decodingStart = performance.now();

    this.tvm.beginScope();
    const decodeChunk: Array<Array<number>> = [[6234]];
    const decodeChunkLen = 1;
    const logitsOnCPU = this.updateLogitsOnCPU(
      this.artifact === undefined
        ? await this.embedAndForward(decodeChunk, decodeChunkLen)
        : this.artifactDecodeAndForward(6234),
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
