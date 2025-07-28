#!/usr/bin/env python3
import sys
from pathlib import Path

# allow imports from your src/
sys.path.append("src")
import dedupe_pipeline

def main():
    # 1) Source data
    input_path   = Path("Data/Duplicate Contact Scrub.xlsx")
    # 2) Final output (rules + ML)
    output_path  = Path("output/dedup_results_with_ml.xlsx")
    # 3) Pre-trained model + encoder

    model_path   = Path("models/rf_model.joblib")
    encoder_path = Path("models/label_encoder.joblib")

    # 4) Run pipeline: NO training, just inference
    df = dedupe_pipeline.run_pipeline(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        encoder_path=encoder_path,
        train_model_flag=False
    )

    print(f"✅ Dedupe + ML results written to {output_path}")

if __name__ == "__main__":
    main()
