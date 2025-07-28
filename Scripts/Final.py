import os
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

# ----------------------------------------------------------------------------
# Configuration and Setup
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# tunable thresholds (same as before)
NAME_SIMILARITY_THRESHOLD = 95
EMAIL_EDIT_DISTANCE_THRESHOLD = 1

CONNECTLINK_STATUS_TO_TIER = {
    "A": "3",  # Active connections
    "I": "2",  # Inactive
    "U": "2",  # Unknown→Inactive
    "":  "1"   # Blank/other
}

# ----------------------------------------------------------------------------
# Step 1: Ingest & normalize (exactly as before)
# ----------------------------------------------------------------------------
def load_contacts(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, engine="openpyxl", dtype=str)
    # … [same header‐mapping, Full Name synthesis, date parsing, normalization …]
    # (copy the function body from your original load_contacts)
    return df

# ----------------------------------------------------------------------------
# Step 2: Build hierarchy tag (exactly as before)
# ----------------------------------------------------------------------------
def build_hierarchy_tag(df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    # … [copy your original build_hierarchy_tag body here]
    return df

# ----------------------------------------------------------------------------
# Helpers for merge/inactivate (you can copy these unchanged)
# ----------------------------------------------------------------------------
def prepare_normalized_fields(df: pd.DataFrame) -> pd.DataFrame:
    # … [copy your original prepare_normalized_fields]
    return df

def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    # … [copy original select_canonical]
    return df

def merge_or_inactivate(df: pd.DataFrame) -> pd.DataFrame:
    # … [copy original merge_or_inactivate]
    return df

def enforce_primary_merge_threshold(df: pd.DataFrame) -> pd.DataFrame:
    # … [copy original enforce_primary_merge_threshold]
    return df

def reassign_inactive_merges(df: pd.DataFrame) -> pd.DataFrame:
    # … [copy original reassign_inactive_merges]
    return df

def apply_one_char_off_inactivation(df: pd.DataFrame) -> pd.DataFrame:
    # … [copy original apply_one_char_off_inactivation]
    return df

# ----------------------------------------------------------------------------
# Full “no‐recluster” pipeline
# ----------------------------------------------------------------------------
def run_dedupe_only(
    input_path: Path,
    output_path: Path
) -> pd.DataFrame:
    # 1) Load & normalize
    df = load_contacts(input_path)

    # 2) Build hier tags
    df = build_hierarchy_tag(df)

    # 3) **Skip** clustering; assume `dupe_cluster_id` already in df

    # 4) Canonical selection
    df = select_canonical(df)

    # 5) Merge/inactivate steps
    df = merge_or_inactivate(df)
    df = enforce_primary_merge_threshold(df)
    df = reassign_inactive_merges(df)
    df = apply_one_char_off_inactivation(df)

    # 6) Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
    logger.info("Exported final deduped results to %s", output_path)
    return df

if __name__ == "__main__":
    base = Path(__file__).parent
    inp = base / "data" / "classification_results.xlsx"
    out = base / "data" / "final_results.xlsx"
    run_dedupe_only(inp, out)
