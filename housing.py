"""House price prediction and budget-constrained property recommendations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recommend import recommend

SEED = 42
TARGET = "SalePrice"
PRICE = "predicted_price"
REQUIRED = ["YearBuilt", "YrSold", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "GrLivArea", "HouseStyle", "BedroomAbvGr", "OverallQual", "OverallCond"]
FORBIDDEN = {TARGET, PRICE, "y_test_predict", "price predict ", "source_row"}


class HousingFeatures(BaseEstimator, TransformerMixin):
    """Stateless features; category and imputation statistics remain in the pipeline."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        result = x.copy()
        for name in ["MSSubClass", "MoSold"]:
            if name in result:
                result[name] = result[name].map(lambda v: str(v) if pd.notna(v) else np.nan)
        result["AgeAtSale"] = result["YrSold"] - result["YearBuilt"]
        result["AboveGroundSF"] = result["1stFlrSF"] + result["2ndFlrSF"]
        # Basement and above-ground floor areas have compatible units; no logged areas are added.
        result["TotalFloorSF"] = result["TotalBsmtSF"] + result["AboveGroundSF"]
        return result


def read_frame(path):
    frame = pd.read_csv(path)
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Expected a nonempty CSV with unique columns")
    return frame


def predictors(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required housing fields: {sorted(missing)}")
    result = frame.drop(columns=list(FORBIDDEN | {"Id"}), errors="ignore").copy()
    for name in set(REQUIRED) - {"HouseStyle"}:
        result[name] = pd.to_numeric(result[name], errors="raise")
    numeric = result.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Numeric predictors contain infinity")
    return result


def build_estimator(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("numeric", numeric, make_column_selector(dtype_include=np.number)),
                                    ("categorical", categorical, make_column_selector(dtype_exclude=np.number))])
    pipeline = Pipeline([("features", HousingFeatures()), ("preprocess", preprocess), ("model", model)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def log_rmse(actual, predicted):
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))


def metrics(actual, predicted):
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)), "rmsle": log_rmse(actual, predicted)}


def predict_prices(model, x):
    values = np.maximum(model.predict(x), 0)
    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite prices")
    return values


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(train_path, test_path, output, quick=False):
    frame, catalog = read_frame(train_path), read_frame(test_path)
    if TARGET not in frame:
        raise ValueError("Training CSV must include SalePrice")
    if set(catalog.columns) & FORBIDDEN:
        raise ValueError("Use the raw unlabeled test CSV, not a file containing targets or predictions")
    if frame.duplicated().any():
        raise ValueError("Training rows contain duplicates; resolve them before splitting")
    x, y = predictors(frame), pd.to_numeric(frame[TARGET], errors="raise")
    if len(x) < 50 or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Need at least 50 training rows and finite, positive SalePrice values")
    xtest = predictors(catalog)
    if set(x.columns) != set(xtest.columns):
        raise ValueError("Training and test predictor schemas differ")
    xtest = xtest.loc[:, x.columns]
    xfit, xhold, yfit, yhold = train_test_split(x, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=3 if quick else 5, shuffle=True, random_state=SEED)
    scorer = make_scorer(log_rmse, greater_is_better=False)
    candidates = {
        "ridge": (Ridge(), {"regressor__model__alpha": [1.0] if quick else [0.1, 1.0, 10.0, 100.0]}),
        "lasso": (Lasso(max_iter=30000, random_state=SEED), {"regressor__model__alpha": [0.001] if quick else [0.0001, 0.001, 0.01]}),
        "random_forest": (RandomForestRegressor(n_estimators=50 if quick else 250, random_state=SEED, n_jobs=1),
                          {"regressor__model__min_samples_leaf": [1] if quick else [1, 3], "regressor__model__max_features": [1.0] if quick else [0.7, 1.0]})}
    searches, scores = {}, []
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(build_estimator(estimator), grid, scoring=scorer, cv=cv, n_jobs=1, error_score="raise")
        search.fit(xfit, yfit)
        searches[name] = search
        scores.append({"model": name, "cv_rmsle": float(-search.best_score_), "parameters": search.best_params_})
    winner = min(scores, key=lambda row: row["cv_rmsle"])["model"]
    selected = searches[winner].best_estimator_
    predicted = predict_prices(selected, xhold)
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(yfit), 1)), yfit)
    baseline_pred = baseline.predict(np.zeros((len(yhold), 1)))
    report = {"seed": SEED, "mode": "quick" if quick else "full", "training_rows": len(frame), "test_rows": len(catalog),
              "fit_rows": len(xfit), "holdout_rows": len(xhold), "cv_folds": cv.n_splits,
              "selection_metric": "training CV RMSLE (lower is better)", "selected_model": winner,
              "candidates": scores, "holdout": metrics(yhold, predicted), "median_baseline_holdout": metrics(yhold, baseline_pred),
              "train_sha256": file_hash(train_path), "test_sha256": file_hash(test_path),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
              "note": "Only the CV-selected candidate is evaluated on holdout. Final predictions use its clone refitted on all labeled rows. Test rows have no ground truth; no test accuracy is claimed."}
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source_row": xhold.index, "actual_price": yhold.to_numpy(), PRICE: predicted}).to_csv(out / "holdout_predictions.csv", index=False)
    final = clone(selected).fit(x, y)
    results = catalog.copy()
    results.insert(0, "source_row", np.arange(len(catalog)))
    results[PRICE] = predict_prices(final, xtest)
    results.to_csv(out / "test_predictions.csv", index=False)
    joblib.dump({"model": final, "columns": list(x.columns)}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report



def demo_data(directory, seed=SEED):
    """Generate artificial records for reproducible execution without original data."""
    rng = np.random.default_rng(seed); n = 200
    df = pd.DataFrame({"YearBuilt": rng.integers(1950, 2009, n), "YrSold": np.full(n, 2010),
                       "1stFlrSF": rng.uniform(700, 1600, n), "2ndFlrSF": rng.uniform(0, 900, n),
                       "TotalBsmtSF": rng.uniform(400, 1500, n), "GrLivArea": rng.uniform(900, 2600, n),
                       "HouseStyle": rng.choice(["1Story", "2Story"], n), "BedroomAbvGr": rng.integers(2, 5, n),
                       "OverallQual": rng.integers(3, 10, n), "OverallCond": rng.integers(3, 9, n),
                       "MSSubClass": rng.choice([20, 60, 70], n), "Neighborhood": rng.choice(["DemoA", "DemoB"], n)})
    df[TARGET] = np.exp(10.4 + 0.10 * df.OverallQual + 0.0003 * df.GrLivArea + rng.normal(0, 0.06, n))
    df.loc[[1, 20, 37], "TotalBsmtSF"] = np.nan
    folder = Path(directory); folder.mkdir(parents=True, exist_ok=True)
    train_path, test_path = folder / "synthetic_train.csv", folder / "synthetic_test.csv"
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].drop(columns=TARGET).to_csv(test_path, index=False)
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Select a model, evaluate holdout, and predict raw test rows")
    t.add_argument("--train", type=Path, required=True); t.add_argument("--test", type=Path, required=True)
    t.add_argument("--output", type=Path, default=Path("outputs/full")); t.add_argument("--quick", action="store_true")
    d = sub.add_parser("demo", help="Run a synthetic demonstration; not an Ames benchmark")
    d.add_argument("--output", type=Path, default=Path("outputs/demo"))
    q = sub.add_parser("recommend", help="Rank a local prediction catalog within budget")
    q.add_argument("--catalog", type=Path, required=True); q.add_argument("--budget", type=float, required=True)
    q.add_argument("--bedrooms", type=int, default=4); q.add_argument("--max-age", type=float, default=20)
    q.add_argument("--min-basement", type=float, default=960); q.add_argument("--house-style", default="2Story")
    q.add_argument("--top", type=int, default=5); q.add_argument("--price-column", default=PRICE)
    q.add_argument("--output", type=Path, default=Path("outputs/recommendations"))
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.train, args.test, args.output, args.quick)
        elif args.command == "demo":
            a, b = demo_data(args.output / "data")
            result = train(a, b, args.output, quick=True)
            result["data_kind"] = "synthetic demonstration, not an Ames benchmark"
            (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        else:
            ranked, result = recommend(read_frame(args.catalog), args.budget, args.bedrooms, args.max_age, args.min_basement, args.house_style, args.top, args.price_column)
            args.output.mkdir(parents=True, exist_ok=True)
            ranked.to_csv(args.output / "recommendations.csv", index=False)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    # Keep serialized transformer classes importable as housing.HousingFeatures.
    from housing import main as run_main
    run_main()
