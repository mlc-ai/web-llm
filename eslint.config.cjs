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
}, {
    files: [
        "examples/**/*.js",
        "examples/**/*.jsx",
        "examples/**/*.ts",
        "examples/**/*.tsx",
    ],

    "rules": {
        "no-restricted-syntax": ["error", {
            selector: "AssignmentExpression[left.property.name='innerHTML']",
            message: "Render example text with textContent or DOM nodes instead of innerHTML.",
        }, {
            selector: "AssignmentExpression[left.property.name='outerHTML']",
            message: "Render example text with textContent or DOM nodes instead of outerHTML.",
        }, {
            selector: "CallExpression[callee.property.name='insertAdjacentHTML']",
            message: "Build example DOM nodes directly instead of parsing strings with insertAdjacentHTML.",
        }, {
            selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
            message: "Render example text as JSX text or DOM nodes instead of dangerouslySetInnerHTML.",
        }],
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
