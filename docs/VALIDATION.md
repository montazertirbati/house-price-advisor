# Validation status

Date: 2026-10-28.

## Completed locally

- Syntax checks passed for the revised Python source and tests on Python 3.13.3.
- Five standalone recommendation tests passed: exact filtering, fixed-budget relaxation, missing-field exclusion, empty affordable sets, and invalid-budget rejection.
- The recommender ran on the supplied legacy prediction CSV (1,605 rows) using a budget of 198,700 and the default preferences. It identified 1,158 eligible affordable rows, found zero exact matches, excluded one row with invalid required values, and returned five ranked alternatives. These results use legacy price estimates and do not validate their accuracy.
- Original input inventory: 1,314 labeled training rows and 1,605 unlabeled raw test rows. No training duplicates were detected; one duplicate raw test predictor row is preserved.

## Pending

Local model execution is blocked because scikit-learn and its dependencies could not be installed: files.pythonhosted.org failed DNS resolution. New Ames model training, holdout metrics and raw-test predictions have not been verified. No historical scores are presented as new results.

A GitHub Actions workflow runs all behavioral tests, a synthetic training demo, standalone recommendations, and model reload. Its execution result will be linked here after verification. Synthetic scores must never be presented as Ames benchmark scores.
