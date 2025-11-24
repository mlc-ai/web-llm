module.exports = {
    preset: "ts-jest",
    testEnvironment: "node",
    roots: ["<rootDir>/tests", "<rootDir>/src"],
    modulePathIgnorePatterns: ["<rootDir>/examples/"],
};
