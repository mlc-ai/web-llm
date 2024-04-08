import { EngineInterface } from "../types";
import { Completions } from "./chat_completion";

export class Chat {
    private engine: EngineInterface;
    completions: Completions;

    constructor(engine: EngineInterface) {
        this.engine = engine;
        this.completions = new Completions(this.engine);
    }
}
