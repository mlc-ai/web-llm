# Contributing to WebLLM

Thank you for your interest in contributing to WebLLM. This guide helps contributors get set up quickly and make high-impact changes that are easy to review and merge.

## Ways To Contribute

We welcome contributions across the project, including:

- Bug reports with clear reproduction steps
- Bug fixes and reliability improvements
- New features and API improvements
- Performance and memory optimizations
- Tests and test coverage improvements
- Documentation updates and tutorials
- New or improved examples in `examples/`
- Model integration and configuration improvements
- Code review and issue triage support

If you are unsure where to start, look for open issues in the repository and propose a plan in the issue thread before implementation.

## Community Principles

WebLLM is part of a broader open-source ecosystem and follows collaborative, public-first development norms.

- Keep technical discussion in public, archivable channels (issues and pull requests)
- Use clear technical reasoning and seek consensus on non-trivial changes
- For major design changes, start with an issue or RFC-style proposal before coding
- Review other contributors' PRs when possible

Additional reference: Apache TVM community guidelines

- https://tvm.apache.org/docs/contribute/community.html

## Development Setup

### Prerequisites

- Node.js (see `.nvmrc` for the required version)
- npm
- Git

Optional:

- Python 3 (for docs build)
- Emscripten/toolchain setup

### Local Setup

```bash
git clone https://github.com/mlc-ai/web-llm.git
cd web-llm
npm install
```

### Build, Lint, and Test

```bash
npm run build
npm run lint
npm test
```

Notes:

- `npm test` runs Jest with coverage thresholds.
- For quick iteration on a single test file, you can run:

```bash
npx jest --coverage=false tests/<file>.test.ts
```

### Auto-formatting

If lint or style checks fail, run:

```bash
npm run format
```

Pre-commit hooks (Husky + lint-staged) are configured in this repo.

## Testing Changes In Examples

To test local package changes inside an example app:

1. Edit `examples/<example>/package.json` and set `"@mlc-ai/web-llm"` to `"../.."` (or `"file:../.."` if needed).
2. Install and run the example.

```bash
cd examples/<example>
npm install
npm run start
```

## Documentation Contributions

Docs are in `docs/` and built with Sphinx.

```bash
cd docs
pip3 install -r requirements.txt
make html
```

Open the built docs from `docs/_build/html`.

## Pull Request Guidelines

Before opening a PR:

1. Keep the change scoped to one problem or feature.
2. Add or update tests for behavior changes.
3. Update docs/examples for user-facing changes.
4. Run `npm run lint` and `npm test` locally.
5. Include a clear PR description with:
   - Problem statement
   - Proposed solution
   - Validation steps and results
   - Backward-compatibility considerations

During review:

- Respond to comments with concrete follow-ups
- Prefer additional tests over assumptions
- Keep commit history understandable (small, logical commits)

## Reporting Bugs and Requesting Features

- Use GitHub Issues for bug reports and feature requests.
- Include environment details, expected vs. actual behavior, and minimal reproduction steps.
- For substantial feature additions, open an issue first to align on design and scope.

## Security Reporting

Please do not report security vulnerabilities in public issues. Report vulnerabilities via email to `mlc-llm-private@googlegroups.com`.

Reference:

- https://github.com/mlc-ai/web-llm/blob/main/SECURITY.md

## License

By contributing, you agree that your contributions are provided under the repository's Apache-2.0 license.
