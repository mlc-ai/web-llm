const {
    defineConfig,
    globalIgnores,
} = require("eslint/config");

const tsParser = require("@typescript-eslint/parser");
const typescriptEslint = require("@typescript-eslint/eslint-plugin");
const js = require("@eslint/js");

const {
    FlatCompat,
} = require("@eslint/eslintrc");

const compat = new FlatCompat({
    baseDirectory: __dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all
});

module.exports = defineConfig([{
    extends: compat.extends(
        "eslint:recommended",
        "plugin:@typescript-eslint/recommended",
        "plugin:prettier/recommended",
    ),

    languageOptions: {
        parser: tsParser,
    },

    plugins: {
        "@typescript-eslint": typescriptEslint,
    },

    rules: {
        "@typescript-eslint/no-explicit-any": "off",
        "@typescript-eslint/no-empty-function": "off",
        "@typescript-eslint/no-non-null-assertion": "off",
    },
}, {
    files: ["examples/**/*.js", "examples/**/*.ts"],

    "rules": {
        "no-undef": "off",
        "@typescript-eslint/no-unused-vars": "off",
    },
}, globalIgnores([
    "**/dist",
    "**/debug",
    "**/lib",
    "**/build",
    "**/node_modules",
    "**/3rdparty",
    "**/.eslintrc.cjs",
    "**/.next",
])]);
