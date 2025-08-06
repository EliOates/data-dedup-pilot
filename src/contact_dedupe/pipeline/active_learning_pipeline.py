#!/usr/bin/env python3
import logging
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import dedupe_pipeline  # your existing pipeline module

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Now reading manual labels from the pipeline output with Manual_label column
MANUAL_LABELS_FILE = Path("output/Contacts_LLM_Test1.xlsx")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "rf_model_active.joblib"  # separate name to distinguish

# Map human labels to numeric
LABEL_MAP = {
    "Keep": 2,
    "Inactive": 0,
    "TBM": 1,
    # TBM entries will be ignored
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------------
# Step 1: Load the labeled pipeline output
# ----------------------------------------------------------------------------
def load_labeled_data(path: Path) -> pd.DataFrame:
    """
    Reads the pipeline output Excel file containing a 'Manual_label' column.
    Expects at least: all columns needed by extract_features, plus 'Manual_label'.
    Filters out TBM entries and maps labels to numeric.
    Converts key columns to proper types (e.g., booleans for is_canonical) so that extract_features works.
    Returns DataFrame with an added 'label' column.
    """
    if not path.exists():
        raise FileNotFoundError(f"Labeled data not found: {path}")
    # Let pandas infer dtypes so booleans stay booleans
    df = pd.read_excel(path, engine="openpyxl")

    if "Manual_label" not in df.columns:
        raise ValueError("Excel must contain 'Manual_label' column")
    # Clean up
    df["Manual_label"] = df["Manual_label"].astype(str).str.strip()
    # Filter to Keep/Inactivate only
    df = df[df["Manual_label"].isin(LABEL_MAP.keys())].copy()
    df["label"] = df["Manual_label"].map(LABEL_MAP)
    logger.info("Loaded labeled data: %d records", len(df))

    # Convert potential string booleans to real bools for key cols
    if "is_canonical" in df.columns:
        df["is_canonical"] = df["is_canonical"].astype(str).str.lower().eq("true")
    # Ensure dtype of dupe_cluster_id is string
    if "dupe_cluster_id" in df.columns:
        df["dupe_cluster_id"] = df["dupe_cluster_id"].astype(str)
    return df


# ----------------------------------------------------------------------------
# Step 2: Train Random Forest on human-labeled records
# ----------------------------------------------------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("Active-learning training evaluation:\n%s", report)
    return clf


# ----------------------------------------------------------------------------
# Step 3: Save the trained model
# ----------------------------------------------------------------------------
def save_model(clf, model_path: Path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info("Saved active-learning model to %s", model_path)


# ----------------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------------
def main():
    # 1) Load human-labeled data
    df = load_labeled_data(MANUAL_LABELS_FILE)

    # 2) Feature extraction via existing pipeline
    #    extract_features returns (X, y_dummy, encoder)
    X, _, encoder = dedupe_pipeline.extract_features(df)

    # 3) Use human labels for training
    y = df["label"].astype(int)

    # 4) Train and save
    clf = train_model(X, y)
    save_model(clf, MODEL_PATH)

    # 5) Save the encoder
    joblib.dump(encoder, MODEL_DIR / "label_encoder_active.joblib")
    logger.info("Saved label encoder to %s", MODEL_DIR / "label_encoder_active.joblib")

    print(
        f"✅ Active-learning model trained on {len(df)} labeled records and saved to {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
