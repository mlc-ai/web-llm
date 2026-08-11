# Contribution Guidelines

## Development Setup

1. Clone repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Setup backend and frontend (see SETUP.md)
4. Make changes
5. Test locally
6. Commit with clear messages
7. Push and create Pull Request

## Code Style

### Python (Backend)
- Use Black for formatting
- Follow PEP 8
- Type hints required
- Docstrings for functions

### JavaScript/TypeScript (Frontend)
- Use ESLint and Prettier
- Prefer TypeScript
- Functional components
- Clear variable names

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/image-search

# Make changes and commit
git add .
git commit -m "feat: implement image search with CLIP"

# Push to remote
git push origin feature/image-search

# Create Pull Request on GitHub
```

## Testing

### Backend
```bash
pytest              # Run all tests
pytest -v          # Verbose output
pytest -k test_    # Run specific tests
```

### Frontend
```bash
npm test            # Run tests
npm run type-check # Type checking
npm run lint       # Linting
```

## Pull Request Guidelines

- Use descriptive titles
- Link related issues
- Describe changes clearly
- Add screenshots for UI changes
- Ensure CI passes
- Request review from maintainers

## Issue Guidelines

- Use issue templates
- Provide minimal reproduction
- Include error logs
- Specify environment details

## Commit Message Format

```
type(scope): subject

body

footer
```

Types: feat, fix, docs, style, refactor, test, chore

Example:
```
feat(products): add image search functionality

Implement CLIP model for image similarity search
- Add image upload endpoint
- Extract and store embeddings
- Add search matching algorithm

Closes #42
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/)
- [Python Async](https://docs.python.org/3/library/asyncio.html)
