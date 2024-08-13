import {
  EmbeddingInputEmptyError,
  EmbeddingUnsupportedEncodingFormatError,
} from "../src/error";
import {
  EmbeddingCreateParams,
  postInitAndCheckFields,
} from "../src/openai_api_protocols/embedding";
import { describe, expect, test } from "@jest/globals";

describe("Check embeddings supported requests", () => {
  test("Supported embedding request float", () => {
    const request: EmbeddingCreateParams = {
      input: ["Hello", "Hi"],
      encoding_format: "float",
    };
    postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
  });

  test("Supported embedding request, unspecified format", () => {
    const request: EmbeddingCreateParams = {
      input: ["Hello", "Hi"],
    };
    postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
  });

  test("Supported embedding request, single string", () => {
    const request: EmbeddingCreateParams = {
      input: "Hello",
    };
    postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
  });

  test("Supported embedding request, single token array", () => {
    const request: EmbeddingCreateParams = {
      input: [0, 1],
    };
    postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
  });

  test("Supported embedding request, array of token arrays", () => {
    const request: EmbeddingCreateParams = {
      input: [
        [0, 1],
        [0, 1],
      ],
    };
    postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
  });
});

describe("Invalid embedding input", () => {
  test("Empty string", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: "",
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow(new EmbeddingInputEmptyError());
  });

  test("Contains empty string", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: ["Hi", "hello", ""],
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow(new EmbeddingInputEmptyError());
  });

  test("Empty token array", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: [],
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow(new EmbeddingInputEmptyError());
  });

  test("Contains empty token array", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: [[1, 2], [3], [], [4]],
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow(new EmbeddingInputEmptyError());
  });
});

describe("Check embeddings unsupported requests", () => {
  test("base64 encoding_format", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: ["Hello", "Hi"],
        encoding_format: "base64",
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow(new EmbeddingUnsupportedEncodingFormatError());
  });

  test("user", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: ["Hello", "Hi"],
        encoding_format: "float",
        user: "Bob",
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow("The following fields in");
  });

  test("dimensions", () => {
    expect(() => {
      const request: EmbeddingCreateParams = {
        input: ["Hello", "Hi"],
        encoding_format: "float",
        dimensions: 2048,
      };
      postInitAndCheckFields(request, "snowflake-arctic-embed-m-q0f32-MLC");
    }).toThrow("The following fields in");
  });
});
