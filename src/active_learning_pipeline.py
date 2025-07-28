import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# Reuse the same client you configured
openai_client = OpenAI()

def load_manual_labels(
    review_path: Path,
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Loads the reviewed Excel, expecting a column 'manual_label'
    with values 'correct' or 'incorrect'.
    """
    df = pd.read_excel(review_path, engine='openpyxl', dtype=str)
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in {review_path}")
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

def compute_ml_confidence_and_mismatch(
    df: pd.DataFrame,
    model_path: Path,
    encoder_path: Path,
    feature_extractor
) -> pd.DataFrame:
    """
    Applies trained ML model to DataFrame, computes prediction, confidence,
    and flags mismatches vs. rule-based resolution_status.
    """
    import joblib

    clf = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    X, _, _ = feature_extractor(df)
    proba = clf.predict_proba(X)
    confidences = proba.max(axis=1)
    preds = clf.predict(X)
    labels = encoder.inverse_transform(preds)

    df = df.copy()
    df['ml_pred'] = labels
    df['ml_confidence'] = confidences
    df['ml_mismatch'] = df['ml_pred'] != df['resolution_status']
    return df

def export_uncertain_cases(
    df: pd.DataFrame,
    output_path: Path,
    confidence_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Filters DataFrame for review and writes to Excel.
    """
    df = df.copy()
    df['low_confidence'] = df['ml_confidence'] < confidence_threshold
    review_df = df[df['ml_mismatch'] | df['low_confidence']]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_excel(output_path, index=False)
    return review_df

def retrain_model_with_labels(
    merged_df,
    feature_extractor,
    model_path: Path,
    encoder_path: Path
) -> None:
    """
    Retrains the RandomForest on only the manually labeled rows.
    If train_test_split fails (too few samples), falls back to training
    on the entire manual set.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    from sklearn.metrics import classification_report

    # 1) Filter to rows you actually labeled
    train_df = merged_df[merged_df["manual_label"].notna()].copy()
    if train_df.empty:
        print("No manual labels found; skipping retrain.")
        return

    # 2) Use your manual_label as the target
    train_df["resolution_status"] = train_df["manual_label"]

    # 3) Extract features + labels
    X, y, encoder = feature_extractor(train_df)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4) Attempt stratified split; on failure, train full set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)
        print("Retrain evaluation:\n",
              classification_report(y_test, clf.predict(X_test)))
    except ValueError as e:
        print(f"Stratified split failed ({e}); training on full manual set.")
        clf.fit(X, y)

    # 5) Save model + encoder
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"Retrained model saved to {model_path} and encoder to {encoder_path}")
