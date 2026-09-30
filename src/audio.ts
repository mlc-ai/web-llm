import { AudioDecodeProcessor } from "./artifact_manifest";
import {
  ChatCompletionAudioInput,
  ChatCompletionAudioInputPCM,
} from "./openai_api_protocols";
import { AudioInputError } from "./error";

interface DecodedWav {
  samples: Float32Array;
  sampleRate: number;
}

function audioError(message: string): never {
  throw new AudioInputError(message);
}

function decodeBase64Wav(data: string): Uint8Array {
  let encoded = data;
  if (data.startsWith("data:")) {
    const match = /^data:(audio\/(?:wav|wave|x-wav));base64,([\s\S]*)$/i.exec(
      data,
    );
    if (match === null) {
      audioError("input_audio data URLs must contain base64-encoded WAV audio");
    }
    encoded = match[2];
  }
  if (/^https?:/i.test(encoded)) {
    audioError("audio URLs are not supported");
  }
  let binary: string;
  try {
    binary = globalThis.atob(encoded);
  } catch {
    audioError("input_audio.data is not valid base64");
  }
  const bytes = new Uint8Array(binary!.length);
  for (let i = 0; i < binary!.length; ++i) {
    bytes[i] = binary!.charCodeAt(i);
  }
  return bytes;
}

function fourCC(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

const WAVE_FORMAT_EXTENSIBLE = 0xfffe;
// Bytes 4..15 of the PCM and IEEE float subformat GUIDs. Bytes 0..3 hold
// the format tag.
const SUBFORMAT_GUID_TAIL = [
  0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

// Returns the format tag carried by an extensible fmt chunk. Subformats
// outside the known GUID family map to the extensible tag itself so the
// caller rejects them.
function readExtensibleFormat(
  view: DataView,
  payload: number,
  size: number,
  bitsPerSample: number,
): number {
  if (size < 40 || view.getUint16(payload + 16, true) < 22) {
    audioError("WAV extensible fmt chunk is truncated");
  }
  const validBitsPerSample = view.getUint16(payload + 18, true);
  if (validBitsPerSample < 1 || validBitsPerSample > bitsPerSample) {
    audioError("WAV valid bits per sample must fit the container bit depth");
  }
  const guid = payload + 24;
  for (let i = 0; i < SUBFORMAT_GUID_TAIL.length; ++i) {
    if (view.getUint8(guid + 4 + i) !== SUBFORMAT_GUID_TAIL[i]) {
      return WAVE_FORMAT_EXTENSIBLE;
    }
  }
  return view.getUint32(guid, true);
}

/** Number of samples a recording has once resampled to the target rate. */
function canonicalLength(
  frameCount: number,
  sourceRate: number,
  targetRate: number,
): number {
  if (frameCount === 0) {
    return 0;
  }
  return sourceRate === targetRate
    ? frameCount
    : Math.max(1, Math.round((frameCount * targetRate) / sourceRate));
}

function checkSampleRange(length: number, processor: AudioDecodeProcessor) {
  if (length < processor.min_samples || length > processor.max_samples) {
    audioError(
      `canonical audio must contain ${processor.min_samples}..${processor.max_samples} samples; got ${length}`,
    );
  }
}

function decodeWav(
  bytes: Uint8Array,
  processor: AudioDecodeProcessor,
): DecodedWav {
  if (bytes.byteLength < 12) {
    audioError("WAV input is truncated");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (fourCC(view, 0) !== "RIFF" || fourCC(view, 8) !== "WAVE") {
    audioError("input_audio is not a RIFF/WAVE file");
  }

  let format: number | undefined;
  let channels: number | undefined;
  let sampleRate: number | undefined;
  let bitsPerSample: number | undefined;
  let blockAlign: number | undefined;
  let dataOffset: number | undefined;
  let dataSize: number | undefined;
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = fourCC(view, offset);
    const size = view.getUint32(offset + 4, true);
    const payload = offset + 8;
    if (payload + size > view.byteLength) {
      audioError(`WAV ${id} chunk is truncated`);
    }
    if (id === "fmt ") {
      if (size < 16) {
        audioError("WAV fmt chunk is truncated");
      }
      format = view.getUint16(payload, true);
      channels = view.getUint16(payload + 2, true);
      sampleRate = view.getUint32(payload + 4, true);
      blockAlign = view.getUint16(payload + 12, true);
      bitsPerSample = view.getUint16(payload + 14, true);
      if (format === WAVE_FORMAT_EXTENSIBLE) {
        format = readExtensibleFormat(view, payload, size, bitsPerSample);
      }
    } else if (id === "data") {
      dataOffset = payload;
      dataSize = size;
    }
    offset = payload + size + (size & 1);
  }
  if (
    format === undefined ||
    channels === undefined ||
    sampleRate === undefined ||
    bitsPerSample === undefined ||
    blockAlign === undefined ||
    dataOffset === undefined ||
    dataSize === undefined
  ) {
    audioError("WAV input requires fmt and data chunks");
  }
  if (format !== 1 && format !== 3) {
    audioError(
      `WAV codec ${format} is unsupported; use PCM integer or IEEE float WAV`,
    );
  }
  if (channels < 1 || sampleRate < 1 || blockAlign < 1) {
    audioError(
      "WAV channel count, sample rate, and block alignment must be positive",
    );
  }
  const bytesPerSample = bitsPerSample / 8;
  const validIntegerDepth =
    format === 1 && [8, 16, 24, 32].includes(bitsPerSample);
  const validFloatDepth = format === 3 && [32, 64].includes(bitsPerSample);
  if (
    !Number.isInteger(bytesPerSample) ||
    (!validIntegerDepth && !validFloatDepth)
  ) {
    audioError(
      `unsupported WAV sample format: codec=${format}, bits=${bitsPerSample}`,
    );
  }
  if (blockAlign !== channels * bytesPerSample || dataSize % blockAlign !== 0) {
    audioError("WAV block alignment does not match its sample format");
  }

  const frameCount = dataSize / blockAlign;
  // Reject a recording that is too long before its samples are allocated.
  checkSampleRange(
    canonicalLength(frameCount, sampleRate, processor.sample_rate_hz),
    processor,
  );
  const samples = new Float32Array(frameCount);
  const readSample = (sampleOffset: number): number => {
    if (format === 3) {
      return bitsPerSample === 32
        ? view.getFloat32(sampleOffset, true)
        : view.getFloat64(sampleOffset, true);
    }
    if (bitsPerSample === 8) {
      return (view.getUint8(sampleOffset) - 128) / 128;
    }
    if (bitsPerSample === 16) {
      return view.getInt16(sampleOffset, true) / 32768;
    }
    if (bitsPerSample === 24) {
      let value =
        view.getUint8(sampleOffset) |
        (view.getUint8(sampleOffset + 1) << 8) |
        (view.getUint8(sampleOffset + 2) << 16);
      if ((value & 0x800000) !== 0) {
        value |= 0xff000000;
      }
      return value / 8388608;
    }
    return view.getInt32(sampleOffset, true) / 2147483648;
  };

  for (let frame = 0; frame < frameCount; ++frame) {
    let mixed = 0;
    for (let channel = 0; channel < channels; ++channel) {
      const sampleOffset =
        dataOffset + frame * blockAlign + channel * bytesPerSample;
      const sample = readSample(sampleOffset);
      if (!Number.isFinite(sample)) {
        audioError("WAV input contains a non-finite sample");
      }
      mixed += sample;
    }
    samples[frame] = mixed / channels;
  }
  return { samples, sampleRate };
}

// Hann windowed sinc low-pass applied before downsampling. The cutoff sits
// at 90% of the target Nyquist and the kernel spans 32 target samples on
// each side, so the transition band ends near the target Nyquist for any
// ratio. Samples past either edge count as zero.
/** Highest sample rate accepted. Above this the filter would be pointlessly large. */
const MAX_SAMPLE_RATE = 384000;

/**
 * Hann windowed sinc kernel with its cutoff a little under the target Nyquist.
 * Returns the taps on one side, so the kernel has 2 * half + 1 entries.
 */
function lowPassKernel(
  sourceRate: number,
  targetRate: number,
  maxHalf: number,
) {
  const step = sourceRate / targetRate;
  const cutoff = 0.45 / step;
  const half = Math.min(Math.ceil(32 * step), maxHalf);
  const kernel = new Float32Array(2 * half + 1);
  for (let t = -half; t <= half; ++t) {
    const x = 2 * Math.PI * cutoff * t;
    const sinc = x === 0 ? 1 : Math.sin(x) / x;
    kernel[t + half] =
      sinc * (0.5 + 0.5 * Math.cos((Math.PI * t) / (half + 1)));
  }
  return { kernel, half };
}

/**
 * The low-passed value at one source index, normalized by the taps that fall
 * inside the recording so that a constant signal stays constant at the edges.
 */
function filteredAt(
  samples: Float32Array,
  kernel: Float32Array,
  half: number,
  n: number,
): number {
  const first = Math.max(0, n - half);
  const last = Math.min(samples.length - 1, n + half);
  let sum = 0;
  let gain = 0;
  for (let k = first; k <= last; ++k) {
    const weight = kernel[k - n + half];
    sum += samples[k] * weight;
    gain += weight;
  }
  return sum / gain;
}

export function resampleLinear(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number,
): Float32Array {
  if (!Number.isSafeInteger(sourceRate) || sourceRate <= 0) {
    audioError("input_audio.sample_rate must be a positive integer");
  }
  if (sourceRate > MAX_SAMPLE_RATE) {
    audioError(`input_audio.sample_rate must not exceed ${MAX_SAMPLE_RATE}`);
  }
  if (samples.length === 0) {
    return new Float32Array();
  }
  if (sourceRate === targetRate) {
    return samples.slice();
  }
  const outputLength = Math.max(
    1,
    Math.round((samples.length * targetRate) / sourceRate),
  );
  // When downsampling, the source is low-passed at the positions the
  // interpolation reads, so the work grows with the output, not the input.
  const filter =
    targetRate < sourceRate
      ? lowPassKernel(sourceRate, targetRate, samples.length)
      : undefined;
  let cachedIndex = -1;
  let cachedValue = 0;
  const source = (index: number): number => {
    if (filter === undefined) {
      return samples[index];
    }
    if (index !== cachedIndex) {
      cachedIndex = index;
      cachedValue = filteredAt(samples, filter.kernel, filter.half, index);
    }
    return cachedValue;
  };
  const output = new Float32Array(outputLength);
  let previousIndex = -1;
  let previousValue = 0;
  for (let i = 0; i < outputLength; ++i) {
    const sourcePosition = (i * sourceRate) / targetRate;
    const left = Math.min(Math.floor(sourcePosition), samples.length - 1);
    const right = Math.min(left + 1, samples.length - 1);
    const leftValue = left === previousIndex ? previousValue : source(left);
    const rightValue = right === left ? leftValue : source(right);
    previousIndex = right;
    previousValue = rightValue;
    output[i] = leftValue + (rightValue - leftValue) * (sourcePosition - left);
  }
  return output;
}

function isPCMInput(
  input: ChatCompletionAudioInput,
): input is ChatCompletionAudioInputPCM {
  return input.format === "pcm_f32";
}

export function decodeAudioInput(
  input: ChatCompletionAudioInput,
  processor: AudioDecodeProcessor,
): Float32Array {
  let samples: Float32Array;
  let sourceRate: number;
  if (isPCMInput(input)) {
    if (!(input.data instanceof Float32Array)) {
      audioError("pcm_f32 input_audio.data must be a Float32Array");
    }
    checkSampleRange(
      canonicalLength(
        input.data.length,
        input.sample_rate,
        processor.sample_rate_hz,
      ),
      processor,
    );
    for (const sample of input.data) {
      if (!Number.isFinite(sample)) {
        audioError("pcm_f32 input_audio contains a non-finite sample");
      }
    }
    samples = input.data;
    sourceRate = input.sample_rate;
  } else {
    if (typeof input.data !== "string") {
      audioError(
        "WAV input_audio.data must be a base64 string or WAV data URL",
      );
    }
    const decoded = decodeWav(decodeBase64Wav(input.data), processor);
    samples = decoded.samples;
    sourceRate = decoded.sampleRate;
  }

  const canonical = resampleLinear(
    samples,
    sourceRate,
    processor.sample_rate_hz,
  );
  checkSampleRange(canonical.length, processor);
  return canonical;
}
