import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# Reuse the same client you configured
openai_client = OpenAI()

def validate_clusters_with_llm(
    df: pd.DataFrame,
    cluster_ids: list[str],
    model: str = "gpt-4",
    temperature: float = 0.0
) -> pd.DataFrame:
    """
    For each cluster_id, asks the LLM if all records in that cluster
    refer to the SAME person or DIFFERENT people.
    Returns a DataFrame of dupe_cluster_id, llm_verdict, llm_reason.
    """
    results = []
    for cid in cluster_ids:
        sub = df[df["dupe_cluster_id"] == cid]
        prompt = (
            "Here are contact records. Are they the SAME person or DIFFERENT people?\n\n"
            + "\n".join(
                f"- Name: {r['Full Name']}, Email: {r['Email']}, Account: {r['Account Name']}"
                for _, r in sub.iterrows()
            )
            + "\n\n"
            "Respond exactly as:\n"
            "Verdict: SAME or DIFFERENT\n"
            "Reason: <brief explanation>"
        )

        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        text = resp.choices[0].message.content.strip().splitlines()
        verdict = next((line.split(":",1)[1].strip() for line in text if line.lower().startswith("verdict:")), "UNKNOWN")
        reason  = next((line.split(":",1)[1].strip() for line in text if line.lower().startswith("reason:")),  "")
        results.append({
            "dupe_cluster_id": cid,
            "llm_verdict": verdict,
            "llm_reason": reason
        })

        time.sleep(0.2)  # avoid rate limits
    return pd.DataFrame(results)
