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

    test('logit_bias exceeds range', () => {
        expect(() => {
            const genConfig: GenerationConfig = {
                max_gen_len: 10,
                logit_bias: {
                    "1355": 155
                }
            };
            postInitAndCheckGenerationConfigValues(genConfig)
        }).toThrow("logit_bias should be in range [-100, 100];");
    });

    test('logit_bias invalid key', () => {
        expect(() => {
            const genConfig: GenerationConfig = {
                max_gen_len: 10,
                logit_bias: {
                    "thisRaisesError": 50
                }
            };
            postInitAndCheckGenerationConfigValues(genConfig)
        }).toThrow("Expect logit_bias's keys to be number represented in string");
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
