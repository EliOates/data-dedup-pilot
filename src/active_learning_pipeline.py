#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
from openai import OpenAI

# Confirm module load
print("📥 active_learning_pipeline.py loaded")

# Reuse the same OpenAI client
openai_client = OpenAI()

def load_manual_labels(
    review_path: Path,
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Loads the reviewed Excel, expecting a column like 'manual_label' (case-insensitive)
    with the true resolution_status values (e.g., 'keep','merge','inactivate').
    Normalizes that column name to 'manual_label'.
    """
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}")
    df = pd.read_excel(review_path, engine='openpyxl', dtype=str)
    # Find the actual label column case-insensitively
    cols_map = {col.lower(): col for col in df.columns}
    if label_column not in cols_map:
        print("Available columns in review file:", df.columns.tolist())
        raise ValueError(f"Column '{label_column}' not found (case-insensitive) in {review_path}")
    real_col = cols_map[label_column]
    # Normalize blanks to NA
    df[real_col] = df[real_col].replace("", pd.NA)
    # Rename to consistent lowercase label_column
    if real_col != label_column:
        df = df.rename(columns={real_col: label_column})
    return df


def integrate_manual_labels(
    df_full: pd.DataFrame,
    df_labeled: pd.DataFrame,
    key_columns: list[str],
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Merges the manual_label back into the full ML DataFrame on key_columns.
    """
    return df_full.merge(
        df_labeled[[*key_columns, label_column]],
        on=key_columns,
        how='left'
    )


def retrain_model_with_labels(
    merged_df: pd.DataFrame,
    feature_extractor,
    model_path: Path,
    encoder_path: Path
) -> None:
    """
    Retrains the RandomForest on only the manually labeled rows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    from sklearn.metrics import classification_report

    # 1) Filter to rows you actually labeled
    train_df = merged_df[merged_df['manual_label'].notna()].copy()
    if train_df.empty:
        print("⚠️ No manual labels found; skipping retrain.")
        return

    # 2) Use manual_label as the true resolution_status
    train_df['resolution_status'] = train_df['manual_label']

    # 3) Extract features + labels
    X, y, encoder = feature_extractor(train_df)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4) Attempt stratified split; fallback on full set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)
        print("Retrain evaluation:\n", classification_report(y_test, clf.predict(X_test)))
    except ValueError as e:
        print(f"⚠️ Stratified split failed ({e}); training on full manual set.")
        clf.fit(X, y)

    # 5) Save model + encoder
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"✅ Retrained model saved to {model_path} and encoder to {encoder_path}")


def main():
    print("🔄 Starting Active Learning Pipeline…")
    # Project root is two levels up from this script
    ROOT = Path(__file__).parent.parent
    OUTPUT_FILE = ROOT / "output" / "Contacts_LLM_Test1.xlsx"

    # 1) Load your ML output (includes resolution_status + manual_label)
    df_full = pd.read_excel(OUTPUT_FILE, engine="openpyxl", dtype=str)

    # 2) Load manual labels from the same file (will normalize 'Manual_label' -> 'manual_label')
    df_reviewed = load_manual_labels(OUTPUT_FILE, label_column="manual_label")

    # 3) Merge manual labels back onto the full frame
    merged = integrate_manual_labels(
        df_full,
        df_reviewed,
        key_columns=["Contact Id"],
        label_column="manual_label"
    )

    # 4) Retrain the RF on just the labeled rows (import feature_extractor from pipeline)
    from src.dedupe_pipeline import feature_extractor
    retrain_model_with_labels(
        merged_df=merged,
        feature_extractor=feature_extractor,
        model_path=ROOT / "models" / "rf_model.joblib",
        encoder_path=ROOT / "models" / "label_encoder.joblib"
    )

    print("🏁 Active Learning Pipeline complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
