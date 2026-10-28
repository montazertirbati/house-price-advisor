"""Behavior checks for leakage boundaries and recommendation constraints."""
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from housing import HousingFeatures, build_estimator, predictors, recommend, demo_data
from pathlib import Path
from tempfile import TemporaryDirectory


class HousingTests(unittest.TestCase):
    def catalog(self):
        return pd.DataFrame({"predicted_price": [100000, 150000, 900000], "YearBuilt": [2000, 1960, 2005],
                             "YrSold": [2010]*3, "BedroomAbvGr": [4, 3, 4], "TotalBsmtSF": [1000, 800, 1200],
                             "HouseStyle": ["2Story", "1Story", "2Story"], "OverallQual": [6, 9, 10], "OverallCond": [5, 6, 8]})

    def test_budget_remains_hard_when_preferences_relax(self):
        result, meta = recommend(self.catalog(), 200000, bedrooms=5)
        self.assertEqual(meta["mode"], "relaxed")
        self.assertTrue((result.predicted_price <= 200000).all())
        self.assertNotIn(2, result.source_row.tolist())
        self.assertGreater(result.preference_distance.min(), 0)

    def test_exact_matches_are_prioritized(self):
        result, meta = recommend(self.catalog(), 200000)
        self.assertEqual(meta["mode"], "exact")
        self.assertEqual(result.source_row.tolist(), [0])

    def test_empty_affordable_set_and_missing_prices(self):
        df = self.catalog(); df.loc[0, "predicted_price"] = np.nan
        result, meta = recommend(df, 1000)
        self.assertTrue(result.empty)
        self.assertEqual(meta["mode"], "no_affordable_candidates")
        self.assertEqual(meta["excluded_invalid_rows"], 1)

    def test_preprocessing_is_not_refitted_on_unseen_categories(self):
        with TemporaryDirectory() as folder:
            a, b = demo_data(Path(folder))
            raw = pd.read_csv(a); x = predictors(raw)
            estimator = build_estimator(Ridge()).fit(x, raw.SalePrice)
            prep = estimator.regressor_.named_steps["preprocess"]
            imputer = prep.named_transformers_["numeric"].named_steps["impute"]
            before = imputer.statistics_.copy()
            unseen = predictors(pd.read_csv(b)); unseen["Neighborhood"] = "NeverSeen"
            unseen["TotalBsmtSF"] = 1000000
            predicted = estimator.predict(unseen)
            self.assertTrue(np.isfinite(predicted).all())
            np.testing.assert_array_equal(before, imputer.statistics_)

    def test_targets_and_predictions_are_never_predictors(self):
        with TemporaryDirectory() as folder:
            a, _ = demo_data(Path(folder)); raw = pd.read_csv(a)
            raw["predicted_price"] = raw.SalePrice
            raw["y_test_predict"] = raw.SalePrice
            x = predictors(raw)
            self.assertFalse({"SalePrice", "predicted_price", "y_test_predict"} & set(x.columns))

    def test_feature_engineering_preserves_raw_input(self):
        with TemporaryDirectory() as folder:
            a, _ = demo_data(Path(folder)); raw = pd.read_csv(a); before = raw.copy(deep=True)
            features = HousingFeatures().transform(raw)
            pd.testing.assert_frame_equal(raw, before)
            np.testing.assert_allclose(features.AboveGroundSF, raw["1stFlrSF"] + raw["2ndFlrSF"])


if __name__ == "__main__":
    unittest.main()
