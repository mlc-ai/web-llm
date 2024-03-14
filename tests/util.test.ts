import { getTopProbs } from "../src/support"

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
