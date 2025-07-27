#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Full Pipeline Invocation
# ----------------------------------------------------------------------------

import sys
from pathlib import Path

# Let Python find your pipeline module
sys.path.append("src")
import dedupe_pipeline

def main():
    # 1) Where your raw Excel lives
    input_path = Path("Data/Duplicate Contact Scrub.xlsx")

    # 2) A temp output (rule-only) if you like
    temp_output = Path("output/temp_results.xlsx")

    # 3) Where to save model + encoder
    model_path   = Path("models/rf_model.joblib")
    encoder_path = Path("models/label_encoder.joblib")

    # 4) Run the pipeline (no ML inference/train)
    df = dedupe_pipeline.run_pipeline(
        input_path=input_path,
        output_path=temp_output,
        model_path=None,
        encoder_path=None,
        train_model_flag=False
    )

    # 5) Extract features & labels
    X, y, encoder = dedupe_pipeline.extract_features(df)

    # 6) Train & save the Random Forest
    dedupe_pipeline.train_and_save_model(
        X, y, encoder,
        model_path=model_path,
        encoder_path=encoder_path
    )

    print("Training complete — models written to /models")

if __name__ == "__main__":
    main()
