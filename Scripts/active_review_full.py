#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
REQUEST_DELAY = 0.5  # seconds between API calls
MODEL_NAME = 'gpt-4o-mini'

# ─── OPENAI CLIENT SETUP ─────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Please set OPENAI_API_KEY in your environment", file=sys.stderr)
    sys.exit(1)
client = OpenAI(api_key=api_key)

# ─── PROMPT BUILDER FOR NEIGHBORHOOD REVIEW ─────────────────────────────────
def build_review_prompt(singleton: dict, neighbors: list) -> list:
    """
    Build messages asking the LLM to decide if a singleton should join one of its neighbors' clusters.
    """
    system_content = """
You are a human-like data deduplication assistant. A contact record appears alone in its cluster. 
Examine this record and its immediate neighbors (3 above and 3 below in sorted data).
Using common-sense name/email grouping (exact or near-exact email, 
exact or fuzzy name >=80%, nicknames, minor misspellings), 
choose whether the singleton belongs in any of the neighbor clusters or should truly stand alone.
Reply ONLY with a JSON object: {"suggested_cluster": "<cluster_id>"} or {"suggested_cluster": null}.
"""
    system_msg = {'role': 'system', 'content': system_content}

    user_payload = {'singleton': singleton, 'neighbors': neighbors}
    user_msg = {'role': 'user', 'content': json.dumps(user_payload, indent=2)}

    return [system_msg, user_msg]

# ─── LLM CALL WITH RETRIES ─────────────────────────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content

# ─── MAIN FUNCTION ───────────────────────────────────────────────────────────
def main():
    base = Path(__file__).resolve().parent.parent
    infile = base / 'Data' / 'dedup_results_with_ml.xlsx'
    outfile = base / 'Data' / 'dedup_with_suggestions.xlsx'

    if not infile.exists():
        print(f"❌ Input file not found: {infile}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(infile, engine='openpyxl', dtype=str)
    if 'dupe_cluster_id' not in df.columns:
        print("❌ Missing 'dupe_cluster_id' column", file=sys.stderr)
        sys.exit(1)

    # add suggestion column
    df['llm_suggested_cluster'] = None

    # foreach singleton, gather neighbor context
    for idx, row in df.iterrows():
        cid = row['dupe_cluster_id']
        # skip if cluster size >1
        if df[df['dupe_cluster_id'] == cid].shape[0] > 1:
            continue

        # singleton: get 3 above and 3 below indices
        start = max(0, idx - 3)
        end = min(len(df) - 1, idx + 3)
        neighbors = []
        for ni in range(start, end + 1):
            if ni == idx:
                continue
            nbr = df.iloc[ni]
            neighbors.append({
                'cluster_id': nbr['dupe_cluster_id'],
                'contact_id': nbr['Contact Id'],
                'full_name': nbr.get('Full Name', ''),
                'name_norm': nbr.get('name_norm', ''),
                'email_norm': nbr.get('email_norm', '')
            })

        singleton = {
            'contact_id': row['Contact Id'],
            'full_name': row.get('Full Name', ''),
            'name_norm': row.get('name_norm', ''),
            'email_norm': row.get('email_norm', '')
        }

        messages = build_review_prompt(singleton, neighbors)
        time.sleep(REQUEST_DELAY)
        suggestion = None
        try:
            raw = call_llm(messages).strip()
            # strip fences
            if raw.startswith('```'):
                raw = '\n'.join(l for l in raw.splitlines() if not l.startswith('```')).strip()
            parsed = json.loads(raw)
            suggestion = parsed.get('suggested_cluster')
        except Exception as e:
            print(f"❌ Error on singleton {singleton['contact_id']}: {e}")

        df.at[idx, 'llm_suggested_cluster'] = suggestion
        print(f"Processed singleton {singleton['contact_id']}: suggestion={suggestion}")

    # save results
    df.to_excel(outfile, index=False, engine='openpyxl')
    print(f"✅ Wrote suggestions to {outfile}")

if __name__ == '__main__':
    main()
