import { MLCEngineInterface } from "../types";
import { Completions } from "./chat_completion";

export class Chat {
  private engine: MLCEngineInterface;
  completions: Completions;

  constructor(engine: MLCEngineInterface) {
    this.engine = engine;
    this.completions = new Completions(this.engine);
  }
}
