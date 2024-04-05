import { getTopProbs, postProcessToken } from "../src/support"

describe('Check getTopLogprobs correctness', () => {
    test('Correctness test 1', () => {
        const logitsOnCPUArray = new Float32Array([0.05, 0.15, 0.3, 0.16, 0.04, 0.2, 0.1]);
        const actual = getTopProbs(3, logitsOnCPUArray);
        const expected: Array<[number, number]> = [[2, 0.3], [5, 0.2], [3, 0.16]];
        expect(actual.length).toBe(expected.length);
        for (let i = 0; i < actual.length; i++) {
            expect(actual[i][0]).toBe(expected[i][0]);
            expect(actual[i][1]).toBeCloseTo(expected[i][1], 4);
        }
    });

    test('Zero top_logprobs', () => {
        const logitsOnCPUArray = new Float32Array([0.05, 0.15, 0.3, 0.16, 0.04, 0.2, 0.1]);
        const topLogProbs = getTopProbs(0, logitsOnCPUArray);
        expect(topLogProbs).toEqual([]);
    });
});


describe('Check postProcessToken correctness', () => {
    test('Token represents a byte', () => {
        const tokens = ["<0x00>", "<0x16>", "<0xB5>", "<0x8E>", "<0xDB>", "<0xFF>"];
        const expectedCharCode = [0, 22, 181, 142, 219, 255];
        for (let i = 0; i < tokens.length; i++) {
            const actual = postProcessToken(tokens[i]);
            const expected = String.fromCharCode(expectedCharCode[i]);
            expect(actual).toBe(expected);
        }
    });

    test('Token contains a space', () => {
        const tokens = ["▁response", "▁▁▁▁"];
        const expectedString = [" response", "    "];
        for (let i = 0; i < tokens.length; i++) {
            const actual = postProcessToken(tokens[i]);
            const expected = expectedString[i];
            expect(actual).toBe(expected);
        }
    });

    test('Regular tokens', () => {
        const tokens = ["es", "m", "Comment", "ho"];
        for (let i = 0; i < tokens.length; i++) {
            const actual = postProcessToken(tokens[i]);
            expect(actual).toBe(tokens[i]);
        }
    });
});