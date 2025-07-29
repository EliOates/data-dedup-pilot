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
    Build messages for the LLM to decide if a singleton record should join one of its neighbors' clusters
    using lenient common-sense criteria.
    """
    system_content = """
You are a human-like data deduplication assistant. A contact record appears alone in its cluster.
Review this singleton and its nearby records (3 above and 3 below). First, pick the most plausible
name variant (considering email formality and common usage). Then decide if the record should
join one of those nearby clusters or remain standalone.
Use lenient grouping: exact or near-exact email (allow one-character typos), exact or fuzzy name >=80%
(allow nicknames, minor misspellings), and your own nickname knowledge.
Ignore hierarchy tags and ML statuses. When undecided, favor merging to reduce singletons.
Reply only with a JSON object:
{"suggested_cluster": "<cluster_id>"}
or {"suggested_cluster": null}.
"""
    system_msg = {'role': 'system', 'content': system_content}
    user_msg = {
        'role': 'user',
        'content': json.dumps({'singleton': singleton, 'neighbors': neighbors}, indent=2)
    }
    return [system_msg, user_msg]

# ─── LLM CALL WITH RETRIES ─────────────────────────────────────────────────────
@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(messages: list) -> str:
    resp = client.chat.completions.create(model=MODEL_NAME, messages=messages)
    return resp.choices[0].message.content

# ─── MAIN FUNCTION ───────────────────────────────────────────────────────────
def main():
    base = Path(__file__).resolve().parent.parent
    infile = base / 'Data' / 'dedup_results_with_ml.xlsx'
    outfile = base / 'Data' / 'dedup_with_suggestions.xlsx'

    if not infile.exists():
        print(f"❌ Input not found: {infile}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(infile, engine='openpyxl', dtype=str)
    if 'dupe_cluster_id' not in df.columns:
        print("❌ Missing 'dupe_cluster_id' column", file=sys.stderr)
        sys.exit(1)

    # initialize suggestion column
    df['llm_suggested_cluster'] = None

    # iterate through all rows, process only singletons
    for idx in range(len(df)):
        cid = df.at[idx, 'dupe_cluster_id']
        group_size = (df['dupe_cluster_id'] == cid).sum()
        if group_size > 1:
            continue

        # collect neighbors indices 3 above/below
        start, end = max(0, idx-3), min(len(df)-1, idx+3)
        neighbors = []
        for ni in range(start, end+1):
            if ni == idx: continue
            nbr = df.iloc[ni]
            neighbors.append({
                'cluster_id': nbr['dupe_cluster_id'],
                'contact_id': nbr['Contact Id'],
                'name_norm': nbr.get('name_norm',''),
                'email_norm': nbr.get('email_norm','')
            })

        singleton = {
            'contact_id': df.at[idx,'Contact Id'],
            'name_norm': df.at[idx,'name_norm'],
            'email_norm': df.at[idx,'email_norm']
        }

        # call LLM
        msgs = build_review_prompt(singleton, neighbors)
        time.sleep(REQUEST_DELAY)
        raw = ''
        try:
            raw = call_llm(msgs).strip()
            if raw.startswith('```'):
                raw = '\n'.join(line for line in raw.splitlines() if not line.startswith('```')).strip()
            suggestion = json.loads(raw).get('suggested_cluster')
        except Exception as e:
            print(f"⚠️ LLM error at idx {idx}, id {singleton['contact_id']}: {e}")
            suggestion = None

        df.at[idx, 'llm_suggested_cluster'] = suggestion
        print(f"Processed {singleton['contact_id']}: suggestion={suggestion}")

    # post-process to flatten singleton chains
    # build mapping original singleton cluster → suggestion
    mapping = df[['dupe_cluster_id','llm_suggested_cluster']].dropna().astype(str).set_index('dupe_cluster_id')['llm_suggested_cluster'].to_dict()
    # resolve chains: if suggestion is itself a singleton and maps further, follow
    def resolve(s):
        seen=set()
        while s in mapping and s not in seen:
            seen.add(s)
            s2 = mapping[s]
            if s2 and s2 in mapping:
                s = mapping[s]
            else:
                return s2
        return s if s in mapping.values() else mapping.get(s)

    df['llm_suggested_cluster'] = df['llm_suggested_cluster'].apply(lambda x: resolve(x) if pd.notna(x) else x)

    # save
    df.to_excel(outfile, index=False, engine='openpyxl')
    print(f"✅ Suggestions written to {outfile}")

if __name__ == '__main__':
    main()