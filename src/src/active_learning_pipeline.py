import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

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
    clf = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Extract features
    X, _, _ = feature_extractor(df)

    # Predict
    proba = clf.predict_proba(X)
    confidences = proba.max(axis=1)
    preds = clf.predict(X)
    labels = encoder.inverse_transform(preds)

    # Attach to df
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
    Filters DataFrame for review:
      - mismatches between ML and rule-based
      - low-confidence predictions
    Exports these to an Excel file and returns the subset.
    """
    df = df.copy()
    df['low_confidence'] = df['ml_confidence'] < confidence_threshold
    review_df = df[df['ml_mismatch'] | df['low_confidence']]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_excel(output_path, index=False)
    print(f"Exported {len(review_df)} uncertain cases to {output_path}")
    return review_df

def load_manual_labels(
    review_path: Path,
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Loads Excel of reviewed cases, expecting a column 'manual_label'
    with values 'correct' or 'incorrect'.
    """
    df = pd.read_excel(review_path, engine='openpyxl', dtype=str)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {review_path}")
    return df

def integrate_manual_labels(
    df_full: pd.DataFrame,
    df_labeled: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Merges manual labels back into the full DataFrame.
    If key_columns is None, merges on index.
    """
    df_full = df_full.copy()
    if key_columns:
        merged = df_full.merge(
            df_labeled[[*key_columns, label_column]],
            how='left',
            on=key_columns
        )
    else:
        merged = df_full.merge(
            df_labeled[[label_column]],
            how='left',
            left_index=True,
            right_index=True
        )
    return merged

def retrain_model_with_labels(
    merged_df: pd.DataFrame,
    feature_extractor,
    model_path: Path,
    encoder_path: Path,
    positive_label: str = 'correct'
) -> None:
    """
    Retrains the RandomForest using only the manually labeled rows.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    # Filter to manually labeled rows
    train_df = merged_df[merged_df['manual_label'].notna()].copy()
    # Use manual_label as the new resolution_status
    train_df['resolution_status'] = train_df['manual_label']

    # Extract features & labels
    X, y, encoder = feature_extractor(train_df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"Retrained model saved to {model_path} and encoder to {encoder_path}")
