import unittest
import numpy as np
import pandas as pd
from recommend import recommend

class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"predicted_price":[100000,150000,900000], "YearBuilt":[2000,1960,2005], "YrSold":[2010]*3, "BedroomAbvGr":[4,3,4], "TotalBsmtSF":[1000,800,1200], "HouseStyle":["2Story","1Story","2Story"], "OverallQual":[6,9,10], "OverallCond":[5,6,8]})

    def test_exact_filter_excludes_unaffordable_high_quality_house(self):
        rows, meta = recommend(self.data, 200000)
        self.assertEqual(meta["mode"], "exact")
        self.assertEqual(rows.source_row.tolist(), [0])

    def test_relaxation_keeps_budget_and_exposes_differences(self):
        rows, meta = recommend(self.data, 200000, bedrooms=5)
        self.assertEqual(meta["mode"], "relaxed")
        self.assertTrue((rows.predicted_price <= 200000).all())
        self.assertTrue((rows.bedroom_difference > 0).all())

    def test_missing_values_are_not_filled_into_recommendations(self):
        self.data.loc[0,"YearBuilt"] = np.nan
        rows, meta = recommend(self.data, 200000)
        self.assertEqual(meta["excluded_invalid_rows"], 1)
        self.assertNotIn(0, rows.source_row.tolist())

    def test_no_affordable_rows_returns_explicit_empty_result(self):
        rows, meta = recommend(self.data, 1)
        self.assertTrue(rows.empty)
        self.assertEqual(meta["mode"], "no_affordable_candidates")

    def test_invalid_budget_is_rejected(self):
        for budget in [0,-1,float("nan")]:
            with self.assertRaises(ValueError):
                recommend(self.data, budget)

if __name__ == "__main__":
    unittest.main()
