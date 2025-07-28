#!/usr/bin/env python3
"""
Script: apply_llm_suggestions_only.py

Loads a deduped Excel file with 'dupe_cluster_id' and LLM suggestion column,
applies sequential LLM suggestion fixes to update cluster IDs, and writes
out a cleaned Excel without running any additional dedupe logic.
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Configure logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Sequential LLM Suggestion Fix
# ----------------------------------------------------------------------------
def fix_llm_suggestions(
    df: pd.DataFrame,
    suggestion_col: str = 'llm_suggested_cluster',
    cluster_col: str = 'dupe_cluster_id'
) -> pd.DataFrame:
    """
    Sequentially apply LLM-based suggestions:
    For each row, if suggestion exists in current clusters, update it.

    Args:
        df: DataFrame with deduped data.
        suggestion_col: Column with LLM suggestions.
        cluster_col: Column with existing cluster IDs.

    Returns:
        DataFrame with updated cluster_col values.
    """
    df = df.copy()
    for idx, suggestion in df[suggestion_col].fillna('').items():
        if not suggestion:
            continue
        valid_clusters = set(df[cluster_col].dropna())
        if suggestion in valid_clusters:
            df.at[idx, cluster_col] = suggestion
    logger.info(
        "Applied LLM fixes: '%s' -> '%s' on %d rows",
        suggestion_col, cluster_col, len(df)
    )
    return df

# ----------------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------------
def main():
    # Set default paths in 'data' directory
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    default_input = data_dir / 'dedup_with_suggestions.xlsx'
    default_output = data_dir / 'classification_results.xlsx'

    parser = argparse.ArgumentParser(
        description='Apply only LLM suggestion fixes to existing clusters.'
    )
    parser.add_argument(
        '-i', '--input-file', dest='input_file', type=Path,
        default=default_input,
        help=f'Input Excel file (default: {default_input})'
    )
    parser.add_argument(
        '-o', '--output-file', dest='output_file', type=Path,
        default=default_output,
        help=f'Output Excel file (default: {default_output})'
    )
    parser.add_argument(
        '--suggestion-col', dest='suggestion_col', type=str,
        default='llm_suggested_cluster',
        help='Column with LLM suggested cluster IDs'
    )
    parser.add_argument(
        '--cluster-col', dest='cluster_col', type=str,
        default='dupe_cluster_id',
        help='Column with original cluster IDs'
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    # Load deduped data
    df = pd.read_excel(args.input_file)

    # Apply suggestion fixes
    df = fix_llm_suggestions(
        df,
        suggestion_col=args.suggestion_col,
        cluster_col=args.cluster_col
    )

    # Write cleaned data
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output_file, index=False)
    logger.info("Cleaned data saved to %s", args.output_file)

if __name__ == '__main__':
    main()
