#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# Make your modules importable
sys.path.append("src")
import dedupe_pipeline
from active_learning_pipeline import compute_ml_confidence_and_mismatch

def main():
    # 1) Load the rule-based results
    df = pd.read_excel("output/temp_results.xlsx", engine="openpyxl")

    # 2) Apply your RF model to get ml_pred, ml_confidence, ml_mismatch
    df_ml = compute_ml_confidence_and_mismatch(
        df,
        model_path=Path("models/rf_model.joblib"),
        encoder_path=Path("models/label_encoder.joblib"),
        feature_extractor=dedupe_pipeline.extract_features
    )

    # 3) (Optional) If you have LLM validations, merge them here
    # from llm_validation import validate_clusters_with_llm
    # clusters = df_ml['dupe_cluster_id'].unique().tolist()
    # llm_df = validate_clusters_with_llm(df_ml, clusters)
    # df_ml = df_ml.merge(llm_df, on="dupe_cluster_id", how="left")

    # 4) Export the *entire* annotated dataset for review
    out_path = Path("review/full_review.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_ml.to_excel(out_path, index=False)

    print(f"Exported full annotated dataset ({len(df_ml)} rows) → {out_path}")

if __name__ == "__main__":
    main()
