import * as webllm from "@mlc-ai/web-llm";

// Define LogitProcessor
export class MyLogitProcessor implements webllm.LogitProcessor {
  private tokenSequence: Array<number> = [];

  processLogits(logits: Float32Array): Float32Array {
    logits[0] = 100.0; // should be enough so that we always sample token 0 below
    return logits;
  }

  processSampledToken(token: number): void {
    this.tokenSequence.push(token);
    console.log("processSampledToken: " + this.tokenSequence.length);
  }

  resetState(): void {
    this.tokenSequence = [];
    console.log("resetState");
  }
}
