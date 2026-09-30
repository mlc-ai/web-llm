import { decodeAudioInput, resampleLinear } from "../src/audio";
import { AudioDecodeProcessor } from "../src/artifact_manifest";

const processor: AudioDecodeProcessor = {
  kind: "audio_decode",
  format: "pcm_f32",
  sample_rate_hz: 16000,
  channels: 1,
  min_samples: 1,
  max_samples: 480000,
};

interface WavSpec {
  format: 1 | 3;
  sampleRate: number;
  channels: number;
  interleaved: number[];
  // Write a WAVE_FORMAT_EXTENSIBLE header. A subformat tag other than
  // `format` stands in for an unsupported codec GUID.
  extensible?: { subformat?: number };
}

const SUBFORMAT_GUID_TAIL = [
  0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

function wav(spec: WavSpec): Uint8Array {
  const { format, sampleRate, channels, interleaved, extensible } = spec;
  const bytesPerSample = format === 1 ? 2 : 4;
  const fmtSize = extensible === undefined ? 16 : 40;
  const dataSize = interleaved.length * bytesPerSample;
  const bytes = new Uint8Array(28 + fmtSize + dataSize);
  const view = new DataView(bytes.buffer);
  const write = (offset: number, value: string) => {
    for (let i = 0; i < value.length; ++i) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };
  write(0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, fmtSize, true);
  view.setUint16(20, extensible === undefined ? format : 0xfffe, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  if (extensible !== undefined) {
    view.setUint16(36, 22, true);
    view.setUint16(38, bytesPerSample * 8, true);
    view.setUint32(40, 0, true);
    view.setUint32(44, extensible.subformat ?? format, true);
    SUBFORMAT_GUID_TAIL.forEach((byte, i) => view.setUint8(48 + i, byte));
  }
  const dataOffset = 20 + fmtSize;
  write(dataOffset, "data");
  view.setUint32(dataOffset + 4, dataSize, true);
  interleaved.forEach((sample, index) => {
    const offset = dataOffset + 8 + index * bytesPerSample;
    if (format === 1) {
      view.setInt16(offset, sample, true);
    } else {
      view.setFloat32(offset, sample, true);
    }
  });
  return bytes;
}

function pcm16Wav(
  interleaved: number[],
  sampleRate: number,
  channels: number,
): Uint8Array {
  return wav({ format: 1, sampleRate, channels, interleaved });
}

function base64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

test("accepts native Float32 PCM and resamples it", () => {
  const actual = decodeAudioInput(
    {
      format: "pcm_f32",
      data: new Float32Array([0, 1, 0, -1]),
      sample_rate: 8000,
    },
    processor,
  );
  expect(actual).toHaveLength(8);
  expect(Array.from(actual.slice(0, 5))).toEqual([0, 0.5, 1, 0.5, 0]);
});

test("decodes raw-base64 and data-URL PCM16 WAV", () => {
  const wav = pcm16Wav([32767, -32768, 16384, -16384], 16000, 2);
  const encoded = base64(wav);
  for (const data of [encoded, `data:audio/wav;base64,${encoded}`]) {
    const actual = decodeAudioInput({ format: "wav", data }, processor);
    expect(actual).toHaveLength(2);
    expect(actual[0]).toBeCloseTo(-1 / 65536, 6);
    expect(actual[1]).toBeCloseTo(0, 6);
  }
});

test("decodes WAVE_FORMAT_EXTENSIBLE PCM and float like the plain tags", () => {
  const decode = (bytes: Uint8Array) =>
    Array.from(
      decodeAudioInput({ format: "wav", data: base64(bytes) }, processor),
    );
  const pcm = {
    format: 1 as const,
    sampleRate: 16000,
    channels: 2,
    interleaved: [32767, -32768, 16384, -16384, 1000, 1000],
  };
  expect(decode(wav({ ...pcm, extensible: {} }))).toEqual(decode(wav(pcm)));
  const float = {
    format: 3 as const,
    sampleRate: 16000,
    channels: 1,
    interleaved: [0.25, -0.5, 1, 0],
  };
  const decodedFloat = decode(wav({ ...float, extensible: {} }));
  expect(decodedFloat).toEqual(decode(wav(float)));
  expect(decodedFloat).toEqual(float.interleaved);
  expect(() =>
    decode(wav({ ...pcm, extensible: { subformat: 0x55 } })),
  ).toThrow(/codec 85 is unsupported/);
});

test("linear resampling is deterministic at the boundary", () => {
  expect(Array.from(resampleLinear(new Float32Array([1, 3]), 2, 4))).toEqual([
    1, 2, 3, 3,
  ]);
});

function tone(frequency: number, rate: number, length: number): Float32Array {
  const samples = new Float32Array(length);
  for (let n = 0; n < length; ++n) {
    samples[n] = Math.sin((2 * Math.PI * frequency * n) / rate);
  }
  return samples;
}

// Peak amplitude of one DFT bin, so a unit sine reads as 1.
function amplitudeAt(
  samples: Float32Array,
  rate: number,
  frequency: number,
): number {
  let re = 0;
  let im = 0;
  for (let n = 0; n < samples.length; ++n) {
    const phase = (2 * Math.PI * frequency * n) / rate;
    re += samples[n] * Math.cos(phase);
    im -= samples[n] * Math.sin(phase);
  }
  return (2 * Math.hypot(re, im)) / samples.length;
}

function rms(samples: Float32Array): number {
  let sum = 0;
  for (const sample of samples) {
    sum += sample * sample;
  }
  return Math.sqrt(sum / samples.length);
}

test("downsampling removes content above the target Nyquist", () => {
  const aliased = resampleLinear(tone(12000, 48000, 48000), 48000, 16000);
  expect(aliased).toHaveLength(16000);
  expect(amplitudeAt(aliased, 16000, 4000)).toBeLessThan(0.001);
  expect(amplitudeAt(aliased, 16000, 12000)).toBeLessThan(0.001);
  expect(rms(aliased)).toBeLessThan(0.01);

  const kept = resampleLinear(tone(1000, 48000, 48000), 48000, 16000);
  expect(kept).toHaveLength(16000);
  expect(amplitudeAt(kept, 16000, 1000)).toBeCloseTo(1, 2);
  expect(rms(kept)).toBeCloseTo(Math.SQRT1_2, 2);

  expect(resampleLinear(new Float32Array(1001), 44100, 16000)).toHaveLength(
    363,
  );
});

test("rejects URLs, malformed WAV, non-finite PCM, and sample bounds", () => {
  expect(() =>
    decodeAudioInput(
      { format: "wav", data: "https://example.com/a.wav" },
      processor,
    ),
  ).toThrow(/URLs are not supported/);
  expect(() =>
    decodeAudioInput(
      { format: "wav", data: base64(new Uint8Array([1, 2])) },
      processor,
    ),
  ).toThrow(/truncated/);
  expect(() =>
    decodeAudioInput(
      {
        format: "pcm_f32",
        data: new Float32Array([Number.NaN]),
        sample_rate: 16000,
      },
      processor,
    ),
  ).toThrow(/non-finite/);
  expect(() =>
    decodeAudioInput(
      { format: "pcm_f32", data: new Float32Array([0]), sample_rate: 16000 },
      { ...processor, min_samples: 2 },
    ),
  ).toThrow(/2..480000/);
});

test("rejects a recording that is too long before decoding its samples", () => {
  const bytes = wav({
    format: 1,
    sampleRate: 48000,
    channels: 1,
    interleaved: new Array(30).fill(0),
  });
  const allocations: number[] = [];
  const original = globalThis.Float32Array;
  const spy = jest
    .spyOn(globalThis, "Float32Array")
    .mockImplementation(function (length?: unknown) {
      if (typeof length === "number") allocations.push(length);
      return new original(length as number);
    } as never);
  expect(() =>
    decodeAudioInput(
      { format: "wav", data: base64(bytes) },
      { ...processor, max_samples: 5 },
    ),
  ).toThrow(/1..5 samples; got 10/);
  spy.mockRestore();
  expect(allocations).toEqual([]);

  expect(() =>
    decodeAudioInput(
      { format: "pcm_f32", data: new Float32Array(31), sample_rate: 1 },
      processor,
    ),
  ).toThrow(/got 496000/);
});

test("downsampling keeps a constant signal constant at the edges", () => {
  expect(resampleLinear(new Float32Array([1]), 48000, 16000)[0]).toBeCloseTo(
    1,
    6,
  );
  const constant = resampleLinear(new Float32Array(300).fill(1), 48000, 16000);
  expect(constant.length).toBe(100);
  expect(constant[0]).toBeCloseTo(1, 6);
  expect(constant[constant.length - 1]).toBeCloseTo(1, 6);
});

test("rejects sample rates the filter cannot handle", () => {
  expect(() =>
    resampleLinear(new Float32Array([0]), 4_000_000_000, 16000),
  ).toThrow(/must not exceed 384000/);
  const started = performance.now();
  expect(resampleLinear(new Float32Array(1), 384000, 16000).length).toBe(1);
  expect(performance.now() - started).toBeLessThan(100);
});
