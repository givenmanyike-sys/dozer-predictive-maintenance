# Code Documentation Standards

This repository is designed so that another data scientist can take ownership of the project without needing the original author's memory of why each decision was made.

## Documentation principles

### 1. Explain the reason, not the syntax

Comments should answer questions such as:

- Why is the current observation excluded from a rolling feature?
- Why is validation chronological rather than random?
- Why are false alarms excluded from the target?
- Why is imputation performed inside the model pipeline?
- Why is a simple forecasting baseline being tested before a more complex model?

Avoid comments that simply translate code into English.

Poor:

```python
# Sort the dataframe
```

Better:

```python
# Sort by machine and time so rolling windows cannot cross from one dozer
# into another and all temporal features are calculated in chronological order.
```

## 2. Use docstrings for reusable functions

Public functions should explain:

- what the function does
- important parameters
- important assumptions
- what it returns
- leakage or operational considerations when relevant

Docstrings do not need to repeat every obvious implementation detail.

## 3. Keep scripts thin

Files under `scripts/` should orchestrate pipeline stages. Reusable logic belongs under `src/` so that it can be imported by tests and other pipeline stages.

## 4. Document assumptions close to the code

If a calculation depends on an assessment-specific assumption, document it beside the calculation. This is preferable to hiding an important assumption in a separate document that a future maintainer may not find.

Examples include:

- telemetry is treated as observed operating rows
- the prediction horizon is measured in subsequent operating observations
- genuine failure is represented by `Unplanned Failure`
- OEM thresholds are features, not labels
- rolling features exclude the current observation

## 5. Do not document unsupported conclusions

Comments and docstrings should describe what the code actually guarantees. They should not claim that a model is production-ready, that a feature causes failure, or that a validation result generalises beyond the available failure events unless the analysis supports that conclusion.

## 6. Maintainability is part of the model quality

Predictive maintenance systems are likely to be maintained by people other than the original author. Clear naming, small functions, tests, comments and documentation therefore form part of the technical deliverable rather than being cosmetic additions.
