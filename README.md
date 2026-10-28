# House Price Prediction and Property Recommendations

A data science capstone that combines house price regression with transparent, budget-constrained property recommendations. It modernizes a collaborative 2020 coursework project on housing records from Ames, Iowa.

The project answers two questions: how can we estimate a home's sale price from its characteristics, and which properties best match a buyer's budget and preferences?

## What this project demonstrates

- Mixed numeric and categorical preprocessing inside cross-validation pipelines.
- Feature engineering for age at sale and compatible floor-area measurements.
- Ridge, Lasso and random forest model selection using training-fold RMSLE.
- A separate holdout evaluation against a median-price baseline.
- Recommendations that keep the estimated-price budget fixed and explain any relaxed preferences.
- A reproducible CLI, automated behavioral tests, and a synthetic demo requiring no private dataset.

## Quick start on Windows

Install Python 3.11 or newer. Download this repository using **Code > Download ZIP** and extract it, or clone it with Git. Open PowerShell in the extracted repository folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe housing.py demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These commands do not require activating the environment or changing PowerShell execution policy. If `py` is unavailable but Python is installed, use `python -m venv .venv` for the first command.

On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python` for the remaining commands.

**The demo uses artificial data. Its scores are not evidence of performance on Ames housing.** See [validation](docs/VALIDATION.md) for actual verification status.

## Train on your local housing data

The raw training file must contain `SalePrice`; the raw test file must not contain prices or previous predictions. Original datasets and the supplied legacy prediction export are not redistributed. See [data notes](data/README.md).

```powershell
.\.venv\Scripts\python.exe housing.py train --train "D:/data science class/project/train_data.csv" --test "D:/data science class/project/test_data.csv" --output outputs/ames
```

Add `--quick` for a three-fold smoke run with smaller grids. The default uses five folds. Paths above refer to the original local archive; other users should provide their own permitted CSV files.

Outputs include `metrics.json`, `holdout_predictions.csv`, `test_predictions.csv`, and a locally saved `model.joblib`. Predictions are in the original price scale. `source_row` is the zero-based row position in the input CSV, not a property identifier.

## Recommend properties

```powershell
.\.venv\Scripts\python.exe recommend.py --catalog outputs/ames/test_predictions.csv --budget 198700 --bedrooms 4 --max-age 20 --min-basement 960 --house-style 2Story
```

The standalone recommendation CLI requires only NumPy and pandas.

The budget is a hard filter on estimated prices. If exact matches exist, rank them by quality, condition and price. Otherwise, rank affordable alternatives by an explicit preference-distance score and export the bedroom difference, excess age, basement shortfall and style mismatch. No artificial numeric ordering is assigned to house styles.

`--min-basement` is a minimum area in square feet. `--max-age` refers to age at the historical sale (`YrSold - YearBuilt`), not age today. The normalization and ranking are explained in the output `summary.json`. This is a transparent equal-weight heuristic, not a claim of globally minimal changes.

To inspect the previously supplied prediction file without retraining:

```powershell
.\.venv\Scripts\python.exe recommend.py --catalog "D:/data science class/project/FINAL/ارسال/test_data_with_prediction.csv" --price-column y_test_predict --budget 198700
```

That command uses **legacy estimates** from the old script. Their accuracy has not been established and they are not new model results.

## Evaluation design

Reserve 20% of labeled rows using seed 42. Grid search only the remaining 80%; imputation, one-hot encoding, scaling and regression are fitted inside each fold. Select the winning candidate by CV RMSLE. Evaluate that candidate once on holdout and compare it with a median-price baseline. Finally, refit its configuration on all labeled rows to predict the unlabeled test file.

Report RMSE and MAE on the original price scale, R-squared, and RMSLE. Modeling uses `log1p(SalePrice)` and predictions use the matching `expm1` inverse. Negative predictions are clipped at zero. Test records lack ground truth, so test accuracy is not reported. A random holdout measures within-dataset generalization; it does not establish performance on future markets or other cities.

## Project history and attribution

The supplied report credits **Vajiheh Montazer Torbati, Amin Mirzaei, and Parastoo Fahimi** (English transliterations of the Persian names). The original script's author header says `Lenovo`, which is a machine/user label rather than reliable authorship evidence. The supplied report and original source are retained locally and are not published here.

The original work explored Lasso feature selection, random forest, a neural network, and clustering-based recommendations. This repository provides a revised classical ML implementation and direct preference ranking. Neural network and K-medoids replication are outside this version; their historical results are not claimed as reproduced. See [review notes](docs/REVIEW.md).

## Repository layout

```text
housing.py                   Training, prediction export and demo CLI
recommend.py                 Standalone recommendation CLI
requirements.txt             Runtime dependencies
tests/test_housing.py        Behavioral checks
data/README.md               Input schema and provenance notes
docs/REVIEW.md               Original issues and modernization scope
docs/VALIDATION.md           Verified results and remaining limitations
.github/workflows/verify.yml Automated tests and synthetic demo
```

Model files use joblib serialization: load only artifacts you created or trust. No license for the original collaborative work or data has been inferred.
