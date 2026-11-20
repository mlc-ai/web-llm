/// <reference types="@webgpu/types" preserve="true" />
/// <reference types="chrome" preserve="true" />

export {
  ModelRecord,
  AppConfig,
  OPFSAccessMode,
  ChatOptions,
  MLCEngineConfig,
  GenerationConfig,
  ModelType,
  prebuiltAppConfig,
  modelVersion,
  modelLibURLPrefix,
  functionCallingModelIds,
} from "./config";

export {
  verifyIntegrity,
  isValidSRI,
  type ModelIntegrity,
  type SRIString,
  type FileIntegrityMap,
} from "./integrity";

export { IntegrityError } from "./error";

export {
  MODEL_PACKAGE_MANIFEST_FILENAME,
  TENSOR_CACHE_MANIFEST_FILENAME,
  ARTIFACT_SCHEMA_VERSION,
  type ModelPackageManifest,
  type CompiledProgramArtifact,
} from "./artifact_manifest";

export {
  InitProgressCallback,
  InitProgressReport,
  MLCEngineInterface,
  LogitProcessor,
  LogLevel,
  ResumableGenerationConfig,
  ResumeChatCompletionOptions,
  ResumeProbeResult,
  ResumeResult,
} from "./types";

export { MLCEngine, CreateMLCEngine } from "./engine";

export {
  hasModelInCache,
  deleteChatConfigInCache,
  deleteModelAllInfoInCache,
  deleteModelWasmInCache,
  deleteModelInCache,
} from "./cache_util";

export {
  WebWorkerMLCEngineHandler,
  WebWorkerMLCEngine,
  CreateWebWorkerMLCEngine,
} from "./web_worker";

export { WorkerRequest, WorkerResponse, CustomRequestParams } from "./message";

export {
  ServiceWorkerMLCEngineHandler,
  ServiceWorkerMLCEngine,
  CreateServiceWorkerMLCEngine,
} from "./service_worker";

export {
  ServiceWorkerMLCEngineHandler as ExtensionServiceWorkerMLCEngineHandler,
  ServiceWorkerMLCEngine as ExtensionServiceWorkerMLCEngine,
  CreateServiceWorkerMLCEngine as CreateExtensionServiceWorkerMLCEngine,
} from "./extension_service_worker";

export * from "./openai_api_protocols/index";

export * from "./artifact_manifest_v2";

export * from "./artifact_graph_executor";

export {
  createAudioEncoderTVMFactory,
  createWhisperAudioTVMFactory,
  type WhisperAudioTVMOptions,
  type AudioEncoderTVMOptions,
} from "./artifact_graph_tvm";

export {
  createThinkerTVMFactory,
  type ThinkerTVMOptions,
} from "./artifact_graph_thinker_tvm";

export {
  createWaveformDecoderTVMFactory,
  WaveformDecoderTVMOptions,
} from "./artifact_graph_waveform_tvm";

export {
  createReferenceConditionerTVMFactory,
  ReferenceConditionerTVMOptions,
} from "./artifact_graph_reference_tvm";

export {
  createFlowPredictorTVMFactory,
  integrateFlowRK4,
  FlowPredictorTVMOptions,
  FlowSamplingOptions,
  FlowInputs,
  FlowPrediction,
  FlowLimits,
} from "./artifact_graph_flow_tvm";

export {
  createTalkerTVMFactory,
  TalkerTVMOptions,
} from "./artifact_graph_talker_tvm";

export {
  createThinkerSpeechTVMFactory,
  EmptyThinkerResponseError,
  type ThinkerSpeechTVMOptions,
} from "./artifact_graph_thinker_speech_tvm";
export {
  alignThinkerSpeechTrace,
  type ThinkerSpeechTrace,
} from "./omni_thinker_alignment";
export {
  createOmniSpeechGraph,
  type OmniSpeechGraph,
  type OmniSpeechOptions,
  type OmniSpeechParts,
  type OmniSpeechPart,
} from "./omni_speech";
export { renderOmniPrompt } from "./omni_prompt";
