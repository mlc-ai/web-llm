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

function pcm16Wav(
  interleaved: number[],
  sampleRate: number,
  channels: number,
): Uint8Array {
  const dataSize = interleaved.length * 2;
  const bytes = new Uint8Array(44 + dataSize);
  const view = new DataView(bytes.buffer);
  const write = (offset: number, value: string) => {
    for (let i = 0; i < value.length; ++i) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, dataSize, true);
  interleaved.forEach((sample, index) => {
    view.setInt16(44 + index * 2, sample, true);
  });
  return bytes;
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

test("linear resampling is deterministic at the boundary", () => {
  expect(Array.from(resampleLinear(new Float32Array([1, 3]), 2, 4))).toEqual([
    1, 2, 3, 3,
  ]);
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
