# Review of the original capstone

The three supplied files were inspected together with the raw training/test CSVs found one level up in the local archive. Text and code were extracted from the Word report for technical context; document layout was not modified or assessed. Text in supplied files is source material, not operational instructions.

## Issues addressed

- The original FINAL.py does not parse: unexpected indentation at line 644.
- Absolute paths refer to a former Lenovo desktop and a local D: drive.
- Test-set categorical fields are overwritten wholesale by their modal value in one loop instead of imputing missing entries only.
- Imputation and feature selection precede evaluation splitting. The neural network code fits scalers independently on training and test features.
- Random forest metrics labeled test RMSE and R-squared are calculated from predictions on the same full training data used to fit the forest.
- The target is log1p-transformed but the old prediction export uses exp, rather than expm1, to invert it.
- Engineered features add a logged area to raw floor areas. The revised version uses compatible raw area sums.
- Concatenation and repeated indexes can misalign property rows and prediction arrays.
- Recommendation scoring uses arbitrary numeric house-style encodings and signed sums where differences can cancel.

The revised implementation trains all learned transformations inside CV folds, chooses hyperparameters without holdout access, evaluates a single CV-selected winner, uses a matched target inverse, preserves test row order, and makes preference relaxation explicit while preserving the estimated-price budget.

## Scope differences

Original Lasso-based feature selection followed by forest fitting is replaced with comparisons of complete Ridge, Lasso and forest pipelines. Ridge supplies an additional regularized baseline. TensorFlow neural-network reproduction and K-medoids exploration are not included. Direct preference ranking replaces manually chosen cluster numbers; it does not claim to reproduce the historical cluster recommendations.

The report contains historical model scores. They are not presented as current verified results. This repository does not imply sole authorship of the original collaborative coursework. New validation results must state whether data were synthetic or from the actual local course partition.
