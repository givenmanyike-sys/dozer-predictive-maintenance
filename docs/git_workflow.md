# Git Workflow

## Branches

Use three long-lived branches:

- `dev`: active integration branch
- `test`: release candidate validation
- `prod`: stable branch representing the reviewed solution

## Feature branches

Create feature branches from `dev`:

```bash
git checkout dev
git pull
git checkout -b feature/target-definition
```

Keep each feature branch focused. Good examples:

- `feature/data-validation`
- `feature/target-definition`
- `feature/rolling-features`
- `feature/temporal-validation`
- `feature/model-selection`
- `feature/event-level-metrics`
- `feature/robustness-analysis`
- `feature/production-design`

## Promotion

```text
feature/* -> dev -> test -> prod
```

Each promotion should have a pull request, passing tests and a concise explanation of the change.

## Commit style

Prefer small commits with clear intent:

```text
feat: add leakage-safe rolling features
fix: correct future-window target construction
test: add temporal split coverage
docs: document false alarm evaluation
refactor: isolate model training pipeline
```
