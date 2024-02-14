import { GenerationConfig, postInitAndCheckGenerationConfigValues } from "../src/config"
import { describe, expect, test } from '@jest/globals';

describe('Check generation config illegal values', () => {
    test('High-level unsupported fields', () => {
        expect(() => {
            const genConfig: GenerationConfig = {
                max_gen_len: 0,
            }
            postInitAndCheckGenerationConfigValues(genConfig)
        }).toThrow("`max_gen_len` should be greater than zero.");
    });
    test('Do not specify both reptition and frequency penalty', () => {
        expect(() => {
            const genConfig: GenerationConfig = {
                repetition_penalty: 0,
                frequency_penalty: 0,
            }
            postInitAndCheckGenerationConfigValues(genConfig)
        }).toThrow("If `frequency_penalty` or `presence_penalty`");
    });
});

describe('Check generation post init', () => {
    test('Only set one of presence or frequency penalty', () => {

        const genConfig: GenerationConfig = {
            frequency_penalty: 1.5,
        };
        postInitAndCheckGenerationConfigValues(genConfig);
        expect(genConfig.presence_penalty).toBe(0.0);
    });
});
