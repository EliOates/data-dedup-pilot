#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------------------------------------
# Logging configuration
# ---------------------------------------
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------
# Initialize OpenAI client
# ---------------------------------------
openai_client = OpenAI()

# ---------------------------------------
# Load reviewed labels
# ---------------------------------------
def load_manual_labels(
    review_path: Path,
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Load the reviewed Excel file and normalize the manual label column.
    """
    if not review_path.exists():
        logger.error("Review file not found: %s", review_path)
        raise FileNotFoundError(f"Review file not found: {review_path}")

    df = pd.read_excel(review_path, engine='openpyxl', dtype=str)
    # Case-insensitive lookup of label column
    mapping = {col.lower(): col for col in df.columns}
    matched = mapping.get(label_column.lower())
    if not matched:
        logger.error("Available columns in review file: %s", list(df.columns))
        raise ValueError(f"Column '{label_column}' not found in {review_path}")

    # Normalize and rename
    df = df.rename(columns={matched: label_column})
    df[label_column] = df[label_column].fillna(pd.NA)
    return df

# ---------------------------------------
# Merge labels into full dataset
# ---------------------------------------
def integrate_manual_labels(
    df_full: pd.DataFrame,
    df_labeled: pd.DataFrame,
    key_columns: list[str],
    label_column: str = 'manual_label'
) -> pd.DataFrame:
    """
    Merge manual labels into the full DataFrame on key_columns.
    """
    merged = df_full.merge(
        df_labeled[key_columns + [label_column]],
        on=key_columns,
        how='left'
    )
    return merged

# ---------------------------------------
# Retrain model with manual labels
# ---------------------------------------
def retrain_model_with_labels(
    merged_df: pd.DataFrame,
    feature_extractor,
    model_path: Path,
    encoder_path: Path
) -> None:
    """
    Retrain a RandomForest classifier on manually labeled rows.
    """
    # Filter to manually labeled rows
    train_df = merged_df.dropna(subset=['manual_label']).copy()
    if train_df.empty:
        logger.warning("No manual labels found; skipping retrain.")
        return

    # Ensure boolean is_canonical column for feature extraction
    if 'is_canonical' in train_df.columns:
        train_df['is_canonical'] = train_df['is_canonical'].apply(
            lambda x: str(x).lower() in ('true', '1', 'yes')
        )
    # Use manual_label as the target resolution_status
    train_df['resolution_status'] = train_df['manual_label']

    # Extract features and labels
    X, y, encoder = feature_extractor(train_df)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train/test split with stratify
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        logger.info("Retrain evaluation:\n%s", classification_report(y_test, y_pred))
    except ValueError as e:
        logger.warning("Stratified split failed (%s); training on full set.", e)
        clf.fit(X, y)

    # Save model and encoder
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    logger.info("Retrained model saved to %s and encoder to %s", model_path, encoder_path)

# ---------------------------------------
# Main pipeline
# ---------------------------------------
def main():
    logger.info("Starting Active Learning Pipeline…")
    root = Path(__file__).parent.parent
    output_file = root / 'output' / 'Contacts_LLM_Test1.xlsx'

    # 1) Load full ML output
    df_full = pd.read_excel(output_file, engine='openpyxl', dtype=str)

    # Coerce is_canonical to boolean if present (string -> bool)
    if 'is_canonical' in df_full.columns:
        df_full['is_canonical'] = df_full['is_canonical'].apply(
            lambda x: str(x).lower() in ('true','1','yes')
        )

    # 2) Load manual labels
    df_reviewed = load_manual_labels(output_file, label_column='manual_label')

    # 3) Merge
    merged = integrate_manual_labels(
        df_full, df_reviewed, key_columns=['Contact Id'], label_column='manual_label'
    )

        # 4) Retrain
    # Ensure that the parent directory (where dedupe_pipeline.py lives) is on the path
    sys.path.insert(0, str(Path(__file__).parent))
    from dedupe_pipeline import extract_features
    retrain_model_with_labels(
        merged_df=merged,
        feature_extractor=extract_features,
        model_path=root / 'models' / 'rf_model.joblib',
        encoder_path=root / 'models' / 'label_encoder.joblib'
    )
