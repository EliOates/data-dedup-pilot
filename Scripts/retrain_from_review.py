#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# Allow imports from src/
sys.path.append("src")

import dedupe_pipeline
from active_learning_pipeline import (
    load_manual_labels,
    integrate_manual_labels,
    compute_ml_confidence_and_mismatch,
    retrain_model_with_labels
)

def main():
    # 1) Load your reviewed labels
    review_df = load_manual_labels(Path("review/review_needed.xlsx"))

    # 2) Read the rule-only results and recompute ML preds
    df = pd.read_excel("output/temp_results.xlsx", engine="openpyxl")
    df_ml = compute_ml_confidence_and_mismatch(
        df,
        model_path=Path("models/rf_model.joblib"),
        encoder_path=Path("models/label_encoder.joblib"),
        feature_extractor=dedupe_pipeline.extract_features
    )

    # 3) Merge your manual labels back into the ML table by Contact Id
    merged = integrate_manual_labels(
        df_ml,
        review_df,
        key_columns=["Contact Id"]
    )

    # 4) Retrain and overwrite the model + encoder
    retrain_model_with_labels(
        merged,
        feature_extractor=dedupe_pipeline.extract_features,
        model_path=Path("models/rf_model.joblib"),
        encoder_path=Path("models/label_encoder.joblib")
    )

    print("Retraining complete — model updated in /models")

if __name__ == "__main__":
    main()
