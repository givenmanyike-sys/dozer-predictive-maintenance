# Contributing

## Development principles

- Keep functions small and single-purpose.
- Put reusable logic in `src/`.
- Keep scripts thin and easy to run.
- Add docstrings to public functions.
- Explain non-obvious assumptions in comments.
- Never introduce future information into a feature.
- Preserve machine boundaries during feature engineering.
- Prefer temporal validation over random row splitting.
- Add or update tests when behaviour changes.
- Update documentation when modelling assumptions change.

## Pull requests

Every pull request should explain:

1. What changed.
2. Why it changed.
3. How it was validated.
4. Whether the target, features, validation strategy or model behaviour changed.
5. Any known limitations.

For modelling changes, include the relevant before and after metrics and explain why the change improves the operational objective.
