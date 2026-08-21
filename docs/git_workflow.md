# Git Workflow & Branching Strategy

## Branches

- `main`: default branch representing the clean, reviewed repository
- `prod`: stable production release branch
- `test`: release candidate validation branch
- `dev`: active integration branch

## Feature branches

Create feature branches from `dev`:

```bash
git checkout dev
git pull
git checkout -b feature/target-definition
```

## Promotion Flow

```text
feature/* ──► dev ──► test ──► prod ──► main
```

Each promotion should have clean commits, passing tests, and a clear operational summary.

## Commit style

Prefer small commits with clear intent:

```text
feat: add leakage-safe rolling features
fix: correct future-window target construction
test: add temporal split coverage
docs: document false alarm evaluation
refactor: isolate model training pipeline
```
