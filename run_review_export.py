#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# Make your modules importable
sys.path.append("src")

import dedupe_pipeline
from active_learning_pipeline import (
    compute_ml_confidence_and_mismatch,
    export_uncertain_cases
)

def main():
    # 1) Read your rule-based output
    df = pd.read_excel("output/temp_results.xlsx", engine="openpyxl")

    # 2) Compute ML predictions + confidence + mismatches
    df_ml = compute_ml_confidence_and_mismatch(
        df,
        model_path=Path("models/rf_model.joblib"),
        encoder_path=Path("models/label_encoder.joblib"),
        feature_extractor=dedupe_pipeline.extract_features
    )

    # 3) Export only low-confidence or mismatched rows
    review_df = export_uncertain_cases(
        df_ml,
        output_path=Path("review/review_needed.xlsx"),
        confidence_threshold=0.80
    )

    print(f"Exported {len(review_df)} rows for manual review → review/review_needed.xlsx")

if __name__ == "__main__":
    main()
