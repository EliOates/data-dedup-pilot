#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
from openai import OpenAI

# Ensure local src folder is on the import path (this script lives in src/)
sys.path.insert(0, str(Path(__file__).parent))

# Confirm module load
print("📥 active_learning_pipeline.py loaded")

# Reuse the same OpenAI client
openai_client = OpenAI()

def load_manual_labels(
    review_path: Path,
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Loads the reviewed Excel, finding the manual_label column case-insensitively,
    normalizing it to 'manual_label'.
    """
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}")
    df = pd.read_excel(review_path, engine='openpyxl', dtype=str)
    # locate the manual label column regardless of case
    cols_map = {col.lower(): col for col in df.columns}
    if label_column not in cols_map:
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"No column matching '{label_column}' found in {review_path}")
    real_col = cols_map[label_column]
    # convert blank strings to NA
    df[real_col] = df[real_col].replace("", pd.NA)
    # rename to consistent label
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
    Merge the manual_label back into the full ML DataFrame on key_columns.
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
    Retrain the RandomForest on only the manually labeled rows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    from sklearn.metrics import classification_report

    # select only labeled rows
    train_df = merged_df[merged_df['manual_label'].notna()].copy()
    if train_df.empty:
        print("⚠️ No manual labels found; skipping retrain.")
        return

    # set the true label
    train_df['resolution_status'] = train_df['manual_label']

    # coerce any boolean-like string fields to float
    for col in ['is_privileged']:
        if col in train_df.columns:
            train_df[col] = train_df[col].map({'True':1.0,'False':0.0}).fillna(0.0)

    # extract features and labels
    X, y, encoder = feature_extractor(train_df)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # try a stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)
        print("Retrain evaluation:\n", classification_report(y_test, clf.predict(X_test)))
    except ValueError as e:
        print(f"⚠️ Stratified split failed ({e}); training on full set.")
        clf.fit(X, y)

    # save
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"✅ Retrained model saved to {model_path} and encoder to {encoder_path}")


def main():
    print("🔄 Starting Active Learning Pipeline…")
    # determine paths
    ROOT = Path(__file__).parent.parent
    OUTPUT = ROOT / 'output' / 'Contacts_LLM_Test1.xlsx'

    # load full dataset (must include resolution_status)
    df_full = pd.read_excel(OUTPUT, engine='openpyxl', dtype=str)

    # load manual labels from that same file
    df_labels = load_manual_labels(OUTPUT, label_column='manual_label')

    # merge labels back
    merged = integrate_manual_labels(
        df_full, df_labels, key_columns=['Contact Id'], label_column='manual_label'
    )

    # import feature extractor from dedupe_pipeline
    import dedupe_pipeline
    extract_fn = dedupe_pipeline.extract_features

    # retrain model
    retrain_model_with_labels(
        merged_df=merged,
        feature_extractor=extract_fn,
        model_path=ROOT / 'models' / 'rf_model.joblib',
        encoder_path=ROOT / 'models' / 'label_encoder.joblib'
    )

    print("🏁 Active Learning Pipeline complete.")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
