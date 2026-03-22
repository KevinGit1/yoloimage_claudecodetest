# CLAUDE.md — streetwatch

## Project
Person and car detection in images using ultralytics YOLOv8.

## Language & Runtime
- Python 3.11+
- Type hints required on all functions (parameters and return types)
- Manage dependencies with `uv`

## Project Structure
```
src/          # Source code
tests/        # Test files
output/       # Output images (gitignored)
```

## Commands
- Run tests: `uv run pytest`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`

## Conventions
- Use `loguru` for all logging. Never use `print()`.
- Output images go to the `output/` directory.
- Always run `uv run ruff check` before committing.
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, etc.
