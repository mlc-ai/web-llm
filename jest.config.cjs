module.exports = {
    preset: "ts-jest",
    testEnvironment: "node",
    roots: ["<rootDir>/tests", "<rootDir>/src"],
    modulePathIgnorePatterns: ["<rootDir>/examples/"],
    collectCoverageFrom: ["src/**/*.{ts,tsx}", "!src/**/*.d.ts"],
    coverageThreshold: {
        global: {
            statements: 25,
            branches: 20,
            functions: 20,
            lines: 25,
        },
        "./src/engine.ts": {
            statements: 35,
            branches: 25,
            functions: 40,
            lines: 35,
        },
    },
};
