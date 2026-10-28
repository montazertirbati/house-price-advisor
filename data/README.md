# Input data

The local archive contains `project/train_data.csv` (1,314 rows, 79 predictors plus SalePrice) and `project/test_data.csv` (1,605 rows, 79 predictors). Their columns match Ames-style housing data. The exact origin of this course-specific train/test partition and redistribution terms have not been established; files are excluded from Git.

The provided `test_data_with_prediction.csv` has 1,605 rows and a `y_test_predict` column, but no ground-truth SalePrice. It is an old prediction artifact, not a test-label dataset. Its features were modified by the original script. Do not use it as training data or as the raw test input to `train`.

Required predictors: YearBuilt, YrSold, 1stFlrSF, 2ndFlrSF, TotalBsmtSF, GrLivArea, HouseStyle, BedroomAbvGr, OverallQual, OverallCond. Additional matching numeric/categorical fields are accepted. Training and raw test predictors must have the same column names. `Id`, `source_row`, SalePrice and known prediction columns are excluded from model predictors.

Prices retain the source dataset's currency units; floor areas are square feet. One duplicate predictor row exists in the local test file; all input rows and their original order are preserved in predictions. Training duplicates are rejected before splitting to avoid duplicate records crossing the holdout boundary.

Run `python housing.py demo` for synthetic input generation. The synthetic records are deliberately artificial and are not licensed copies of the original dataset.
