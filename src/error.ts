export class ModelNotFoundError extends Error {
  constructor(modelId: string) {
    super(
      `Cannot find model record in appConfig for ${modelId}. Please check if the model ID is correct and included in the model_list configuration.`,
    );
    this.name = "ModelNotFoundError";
  }
}

export class ConfigValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigValueError";
  }
}

export class MinValueError extends ConfigValueError {
  constructor(paramName: string, minValue: number) {
    super(`Make sure \`${paramName}\` > ${minValue}.`);
    this.name = "MinValueError";
  }
}

export class RangeError extends ConfigValueError {
  constructor(
    paramName: string,
    minValue: number,
    maxValue: number,
    additionalMessage?: string,
  ) {
    super(
      `Make sure ${minValue} < ${paramName} <= ${maxValue}.${additionalMessage ? " " + additionalMessage : ""}`,
    );
    this.name = "RangeError";
  }
}

export class NonNegativeError extends ConfigValueError {
  constructor(paramName: string) {
    super(`Make sure ${paramName} >= 0.`);
    this.name = "NonNegativeError";
  }
}

export class InvalidNumberStringError extends ConfigValueError {
  constructor(paramName: string, actualValue?: string) {
    super(
      `Make sure ${paramName} to be number represented in string.${actualValue ? " Got " + actualValue : ""}`,
    );
    this.name = "InvalidNumberStringError";
  }
}

export class DependencyError extends ConfigValueError {
  constructor(
    dependentParam: string,
    requiredParam: string,
    requiredValue: any,
  ) {
    super(
      `${dependentParam} requires ${requiredParam} to be ${requiredValue}.`,
    );
    this.name = "DependencyError";
  }
}

export class WebGPUNotAvailableError extends Error {
  constructor() {
    super(
      "WebGPU is not supported in your current environment, but it is necessary to run the WebLLM engine. " +
        "Please make sure that your browser supports WebGPU and that it is enabled in your browser settings. " +
        "You can also consult your browser's compatibility chart to see if it supports WebGPU. " +
        "For more information about WebGPU support in your browser, visit https://webgpureport.org/",
    );
    this.name = "WebGPUNotAvailableError";
  }
}

export class WebGPUNotFoundError extends Error {
  constructor() {
    super("Cannot find WebGPU in the environment");
    this.name = "WebGPUNotFoundError";
  }
}

export class ModelNotLoadedError extends Error {
  constructor() {
    super(
      "Model not loaded before calling chatCompletion(). Please ensure you have called `MLCEngine.reload(model)` to load the model before initiating chat operations, or initialize your engine using `CreateMLCEngine()` with a valid model configuration.",
    );
    this.name = "ModelNotLoadedError";
  }
}

export class WorkerEngineModelNotLoadedError extends Error {
  constructor(engineName: string) {
    super(
      `${engineName} is not loaded with a model. Did you call \`engine.reload()\`?`,
    );
    this.name = "WorkerEngineModelNotLoadedError";
  }
}

export class MessageOrderError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MessageOrderError";
  }
}

export class SystemMessageOrderError extends Error {
  constructor() {
    super("System prompt should always be the first message in `messages`.");
    this.name = "SystemMessageOrderError";
  }
}

export class ContentTypeError extends Error {
  constructor(name: string) {
    super(`${name} should have string content.`);
    this.name = "ContentTypeError";
  }
}

export class UserMessageContentError extends Error {
  constructor(content: any) {
    super(
      `User message only supports string content for now, but received: ${content}`,
    );
    this.name = "UserMessageContentError";
  }
}

export class UnsupportedRoleError extends Error {
  constructor(role: string) {
    super(`Unsupported role of message: ${role}`);
    this.name = "UnsupportedRoleError";
  }
}

export class ToolCallOutputParseError extends Error {
  constructor(outputMessage: string, error: Error) {
    super(
      `Internal error: error encountered when parsing outputMessage for function ` +
        `calling. Got outputMessage: ${outputMessage}\nGot error: ${error}`,
    );
    this.name = "ToolCallOutputParseError";
  }
}

export class ToolCallOutputInvalidTypeError extends Error {
  constructor(expectedType: string) {
    super(
      `Internal error: expect output of function calling to be an ${expectedType}`,
    );
    this.name = "ToolCallOutputInvalidTypeError";
  }
}

export class ToolCallOutputMissingFieldsError extends Error {
  constructor(missingFields: string[], object: any) {
    super(
      `Expect generated tool call to have fields ${missingFields.map((field) => `"\`${field}\`"`).join(", ")}, but got object: ${JSON.stringify(object)}`,
    );
    this.name = "JSONFieldError";
  }
}

export class ConfigurationNotInitializedError extends Error {
  constructor() {
    super(
      "Configuration not initialized. Ensure you have called `reload()` function first.",
    );
    this.name = "ConfigurationNotInitializedError";
  }
}

export class MissingModelWasmError extends Error {
  constructor(modelId: string) {
    super(
      `Missing \`model_lib\` for the model with ID "${modelId}". Please ensure that \`model_lib\` is provided in \`model_list\` for each model. This URL is essential for downloading the WASM library necessary to run the model.`,
    );
    this.name = "MissingModelError";
  }
}

export class FeatureSupportError extends Error {
  constructor(feature: string) {
    super(
      `This model requires feature ${feature}, which is not yet supported by this browser.`,
    );
    this.name = "FeatureSupportError";
  }
}

export class UnsupportedFieldsError extends Error {
  constructor(unsupportedFields: string[]) {
    super(
      "The following fields in ChatCompletionRequest are not yet supported: \n" +
        unsupportedFields.join(", "),
    );
    this.name = "UnsupportedFieldsError";
  }
}

export class ShaderF16SupportError extends FeatureSupportError {
  constructor() {
    super(
      "This model requires WebGPU extension shader-f16, which is not enabled in this browser. " +
        'You can try to launch Chrome Canary in command line with flag "--enable-dawn-features=allow_unsafe_apis".',
    );
    this.name = "ShaderF16SupportError";
  }
}
export class DeviceLostError extends Error {
  constructor() {
    super(
      "The WebGPU device was lost while loading the model. This issue often occurs due to running out of memory (OOM). To resolve this, try reloading with a model that has fewer parameters or uses a smaller context length.",
    );
    this.name = "DeviceLostError";
  }
}

export class InvalidToolChoiceError extends Error {
  constructor(toolChoice: string) {
    super(
      `Invalid tool choice value: '${toolChoice}'. Please check your input and try again.`,
    );
    this.name = "InvalidToolChoiceError";
  }
}

export class UnsupportedToolChoiceTypeError extends Error {
  constructor() {
    super(
      "Unsupported tool choice type. Only tool choices of type 'function' are supported.",
    );
    this.name = "UnsupportedToolChoiceTypeError";
  }
}

export class FunctionNotFoundError extends Error {
  constructor(functionName: string) {
    super(
      `The tool choice function ${functionName} is not found in the tools list`,
    );
    this.name = "FunctionNotFoundError";
  }
}

export class UnsupportedToolTypeError extends Error {
  constructor() {
    super("Only 'function' tool type is supported");
    this.name = "UnsupportedToolTypeError";
  }
}
export class ChatModuleNotInitializedError extends Error {
  constructor() {
    super(
      "Chat module not yet initialized. Ensure you initialize the chat module by calling `chat.reload()` first.",
    );
    this.name = "ChatModuleNotInitializedError";
  }
}
export class UnsupportedTokenizerFilesError extends Error {
  constructor(files: string[]) {
    super(`Cannot handle tokenizer files ${files}`);
    this.name = "UnsupportedTokenizerFilesError";
  }
}

export class WindowSizeConfigurationError extends Error {
  constructor(contextWindowSize: number, slidingWindowSize: number) {
    super(
      `Only one of context_window_size and sliding_window_size can be positive. Got: ` +
        `context_window_size: ${contextWindowSize}, sliding_window_size: ${slidingWindowSize}\n` +
        `Consider modifying ModelRecord.overrides to set one of them to -1.`,
    );
    this.name = "WindowSizeConfigurationError";
  }
}

export class AttentionSinkSizeError extends Error {
  constructor() {
    super(
      "Need to specify non-negative attention_sink_size if using sliding window. " +
        "Consider modifying ModelRecord.overrides. " +
        "Use `attention_sink_size=0` for default sliding window.",
    );
    this.name = "AttentionSinkSizeError";
  }
}

export class WindowSizeSpecificationError extends Error {
  constructor() {
    super(
      "Need to specify either sliding_window_size or max_window_size.\n" +
        "Consider modifying ModelRecord.overrides to set one of them to positive.",
    );
    this.name = "WindowSizeSpecificationError";
  }
}

export class ContextWindowSizeExceededError extends Error {
  constructor(numPromptTokens: number, contextWindowSize: number) {
    super(
      `Prompt tokens exceed context window size: number of prompt tokens: ${numPromptTokens}; ` +
        `context window size: ${contextWindowSize}\nConsider shortening the prompt, or increase ` +
        "`context_window_size`, or using sliding window via `sliding_window_size`.",
    );
    this.name = "ContextWindowSizeExceededError";
  }
}

export class NonWorkerEnvironmentError extends Error {
  constructor(className: string) {
    super(`${className} must be created in the service worker script.`);
    this.name = "NonWorkerEnvironmentError";
  }
}

export class NoServiceWorkerAPIError extends Error {
  constructor() {
    super(
      "Service worker API is not available in your browser. Please ensure that your browser supports service workers and that you are using a secure context (HTTPS). " +
        "Check the browser compatibility and ensure that service workers are not disabled in your browser settings.",
    );
    this.name = "NoServiceWorkerAPIError";
  }
}

export class ServiceWorkerInitializationError extends Error {
  constructor() {
    super(
      "Service worker failed to initialize. This could be due to a failure in the service worker registration process or because the service worker is not active. " +
        "Please refresh the page to retry initializing the service worker.",
    );
    this.name = "ServiceWorkerInitializationError";
  }
}

export class StreamingCountError extends Error {
  constructor() {
    super("When streaming, `n` cannot be > 1.");
    this.name = "StreamingCountError";
  }
}

export class SeedTypeError extends Error {
  constructor(seed: any) {
    super("`seed` should be an integer, but got " + seed);
    this.name = "SeedTypeError";
  }
}
export class InvalidResponseFormatError extends Error {
  constructor() {
    super("JSON schema is only supported with `json_object` response format.");
    this.name = "InvalidResponseFormatError";
  }
}

export class CustomResponseFormatError extends Error {
  constructor(currentFormat: any) {
    super(
      "When using Hermes-2-Pro function calling via ChatCompletionRequest.tools, " +
        "cannot specify customized response_format. We will set it for you internally. Currently " +
        "set to: " +
        JSON.stringify(currentFormat),
    );
    this.name = "CustomResponseFormatError";
  }
}
export class UnsupportedModelIdError extends Error {
  constructor(currentModelId: string, supportedModelIds: string[]) {
    super(
      `${currentModelId} is not supported for ChatCompletionRequest.tools. Currently, models ` +
        `that support function calling are: ${supportedModelIds.join(", ")}`,
    );
    this.name = "UnsupportedModelIdError";
  }
}
export class CustomSystemPromptError extends Error {
  constructor() {
    super(
      "When using Hermes-2-Pro function calling via ChatCompletionRequest.tools, cannot specify customized system prompt.",
    );
    this.name = "CustomSystemPromptError";
  }
}

export class InvalidStreamOptionsError extends Error {
  constructor() {
    super("Only specify stream_options when stream=True.");
    this.name = "InvalidStreamOptionsError";
  }
}
export class UnknownMessageKindError extends Error {
  constructor(msgKind: string, msgContent: any) {
    super(`Unknown message kind, msg: [${msgKind}] ${msgContent}`);
    this.name = "UnknownMessageKindError";
  }
}
