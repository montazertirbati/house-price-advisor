"""Rank properties by budget and explicit preference differences."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PRICE = "predicted_price"

def recommend(catalog, budget, bedrooms=4, max_age=20, min_basement=960, house_style="2Story", top=5, price_column=PRICE):
    if not np.isfinite([budget, max_age, min_basement]).all() or budget <= 0 or max_age < 0 or min_basement < 0 or bedrooms < 0 or top < 1:
        raise ValueError("Use a positive budget/top count and nonnegative preference values")
    needed = [price_column, "YearBuilt", "YrSold", "BedroomAbvGr", "TotalBsmtSF", "HouseStyle", "OverallQual", "OverallCond"]
    missing = set(needed) - set(catalog.columns)
    if missing:
        raise ValueError(f"Recommendation catalog lacks: {sorted(missing)}")
    df = catalog.copy()
    if "source_row" not in df:
        df.insert(0, "source_row", np.arange(len(df)))
    numeric = [price_column, "YearBuilt", "YrSold", "BedroomAbvGr", "TotalBsmtSF", "OverallQual", "OverallCond"]
    for name in numeric:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    valid = np.isfinite(df[numeric]).all(axis=1) & df.HouseStyle.notna()
    df = df.loc[valid].copy()
    df["age_at_sale"] = df.YrSold - df.YearBuilt
    df = df.loc[(df[price_column] > 0) & (df[price_column] <= budget) & (df.age_at_sale >= 0)].copy()
    # Budget is a hard filter. The remaining preferences are soft only when no exact match exists.
    df["bedroom_difference"] = (df.BedroomAbvGr - bedrooms).abs()
    df["age_excess"] = (df.age_at_sale - max_age).clip(lower=0)
    df["basement_shortfall"] = (min_basement - df.TotalBsmtSF).clip(lower=0)
    df["style_mismatch"] = (df.HouseStyle != house_style).astype(int)
    df["preference_distance"] = (df.bedroom_difference / max(bedrooms, 1) + df.age_excess / max(max_age, 1)
                                  + df.basement_shortfall / max(min_basement, 1) + df.style_mismatch)
    exact = df.loc[df.preference_distance == 0]
    mode = "exact" if len(exact) else "relaxed" if len(df) else "no_affordable_candidates"
    pool = exact if len(exact) else df
    ranked = pool.sort_values(["preference_distance", "OverallQual", "OverallCond", price_column, "source_row"],
                              ascending=[True, False, False, True, True]).head(top).copy()
    meta = {"mode": mode, "budget": budget, "affordable_rows": len(df), "exact_matches": len(exact), "returned": len(ranked),
            "excluded_invalid_rows": int((~valid).sum()), "price_column": price_column,
            "preferences": {"bedrooms": bedrooms, "max_age_at_sale": max_age, "minimum_basement_sqft": min_basement, "house_style": house_style},
            "distance": "abs(bedrooms-target)/max(target,1) + excess_age/max(max_age,1) + basement_shortfall/max(min_basement,1) + style_mismatch",
            "note": "Budget is applied to model estimates, not guaranteed sale prices. Age means age at historical sale. Relaxation is transparent equal-weight heuristic ranking, not a globally optimal or learned recommendation model."}
    return ranked, meta



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--bedrooms", type=int, default=4)
    parser.add_argument("--max-age", type=float, default=20)
    parser.add_argument("--min-basement", type=float, default=960)
    parser.add_argument("--house-style", default="2Story")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--price-column", default=PRICE)
    parser.add_argument("--output", type=Path, default=Path("outputs/recommendations"))
    args = parser.parse_args()
    try:
        ranked, result = recommend(pd.read_csv(args.catalog), args.budget, args.bedrooms, args.max_age, args.min_basement, args.house_style, args.top, args.price_column)
        args.output.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.output / "recommendations.csv", index=False)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
