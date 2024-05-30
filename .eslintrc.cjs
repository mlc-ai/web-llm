module.exports = {
    extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended', 'plugin:prettier/recommended'],
    parser: '@typescript-eslint/parser',
    plugins: ['@typescript-eslint'],
    root: true,
    rules: {
        "@typescript-eslint/no-explicit-any": "off",
        "@typescript-eslint/no-empty-function": "off",
        "@typescript-eslint/no-non-null-assertion": "off",
    },
    overrides: [
        {
          "files": ["examples/**/*.js", "examples/**/*.ts"],
          "rules": {
            "no-undef": "off",
            "@typescript-eslint/no-unused-vars": "off"
          }
        }
    ]
};
