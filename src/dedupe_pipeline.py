import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
from rapidfuzz import fuzz, distance
from openai import OpenAI

# Additional imports for machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
# ----------------------------------------------------------------------------
# Configuration and Setup
# ----------------------------------------------------------------------------

# Configure logging to show messages at INFO level and higher
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client (for potential embedding-based features)
# TODO: Insert your OpenAI API key below
os.environ["OPENAI_API_KEY"] = "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"
openai_client = OpenAI()

# Tunable thresholds for rule-based matching
NAME_SIMILARITY_THRESHOLD = 95        # Minimum similarity ratio for name fuzzy matches
EMAIL_EDIT_DISTANCE_THRESHOLD = 1     # Maximum Levenshtein distance for email matches

# Mapping of ConnectLink Status codes to lexicographic tiers
CONNECTLINK_STATUS_TO_TIER: Dict[str, str] = {
    "A": "3",  # Active connections
    "I": "2",  # Inactive connections
    "U": "2",  # Unknown treated as Inactive
    "":  "1"   # Blank or other statuses
}
# ----------------------------------------------------------------------------
# Step 1: Ingestion and Header Mapping
# ----------------------------------------------------------------------------

def load_contacts(file_path: Path) -> pd.DataFrame:
    """
    Reads the raw Excel file of contacts, renames incoming columns to the canonical
    schema, synthesizes 'Full Name' if needed, parses date columns, normalizes
    key text fields, and validates that all required columns are present.

    Parameters:
        file_path (Path): Path to the input Excel file.

    Returns:
        pd.DataFrame: A DataFrame ready for deduplication steps.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If required columns are missing after processing.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    df = pd.read_excel(file_path, engine="openpyxl", dtype=str)

    # Rename upstream headers to our canonical column names
    rename_map = {
        "Account Name: Acct_ID_18": "Account Name",
        "Contact_id_18":           "Contact Id",
        "Primary Contact Any":     "Primary Contact",
        "Agile Contact Email":     "Connect Link Email",
        "# of Cases":              "# of cases",
        "# of Opps":               "# of opps"
    }
    df = df.rename(columns=rename_map)

    # Synthesize 'Full Name' if missing but First Name and Last Name exist
    if "Full Name" not in df.columns:
        if {"First Name", "Last Name"}.issubset(df.columns):
            df["Full Name"] = (
                df["First Name"].fillna("").str.strip() + " " +
                df["Last Name"].fillna("").str.strip()
            ).str.strip()
        else:
            raise ValueError("Missing both 'Full Name' and 'First Name'+'Last Name' columns")

    # Parse date columns into datetime, coercing errors to NaT
    for dt_col in ["Last Activity", "Created Date"]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # Validate presence of all required columns
    required = [
        "Account Name", "Full Name", "Email", "Contact Id",
        "Admin Role", "Primary Contact", "Active Contact",
        "ConnectLink Status", "Connect Link Email",
        "# of cases", "# of opps", "Last Activity", "Created Date"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after ingestion: {missing}")

    # Normalize text fields: fill missing with empty, collapse whitespace, strip edges
    for col in required:
        if df[col].dtype == object:
            df[col] = (
                df[col].fillna("")
                      .astype(str)
                      .str.strip()
                      .str.replace(r"\s+", " ", regex=True)
            )
    # Convert 'Primary Contact' to boolean
    df["Primary Contact"] = df["Primary Contact"].astype(str).str.lower().isin({"true","1","yes"})

    logger.info("Loaded and normalized %d rows", len(df))
    return df

# ----------------------------------------------------------------------------
# Step 2: Hierarchy Tag Construction
# ----------------------------------------------------------------------------

def build_hierarchy_tag(df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
    df = df.copy()
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    # parse dates
    df["Last Activity"]  = pd.to_datetime(df["Last Activity"], errors="coerce")
    df["Created Date"]   = pd.to_datetime(df["Created Date"],  errors="coerce")

    # 1. Privilege bit
    df["is_privileged"]      = df["Admin Role"].str.lower().isin({"owner","admin"})

    # 2. Primary contact
    df["primary_bit"]        = df["Primary Contact"].astype(bool).astype(int)

    # 3. Active contact
    df["active_bit"]         = df["Active Contact"].str.lower().eq("active").astype(int)

    # 4. Connection tier
    df["connect_tier"]       = df["ConnectLink Status"] \
                                 .fillna("") \
                                 .str.upper() \
                                 .map(CONNECTLINK_STATUS_TO_TIER) \
                                 .fillna("1")

    # 5. Opportunity bucket
    opps = df["# of opps"].fillna("0").astype(int)
    df["opps_bucket"]        = pd.cut(opps, [-1,0,3,float("inf")],
                                      labels=["Z","L","H"]).astype(str)

    # 6. Activity bucket (coarse):
    #    2 = <= 180 days, 1 = <= 548 days, 0 = older or blank
    days_act = (reference_date - df["Last Activity"]).dt.days.fillna(9999).astype(int)
    df["activity_bucket"]    = pd.cut(days_act,
                                      bins=[-1,180,548,float("inf")],
                                      labels=["2","1","0"]
                                     ).astype(str)

    # 7. Primary email presence
    df["primary_email_bit"]  = df["Email"].astype(str).str.strip().ne("").astype(int)

    # 8. Connect-link email presence
    df["connect_email_bit"]  = df["Connect Link Email"].astype(str).str.strip().ne("").astype(int)

    # 9. Days since last activity (exact, zero-padded)
    exact_days = days_act.clip(0,99999).astype(int)
    df["days_since_activity"] = exact_days.astype(str).str.zfill(5)

    # 10. Creation seniority rank (zero-padded days since creation)
    days_cr = (reference_date - df["Created Date"]).dt.days.fillna(0).astype(int)
    df["created_rank"]       = days_cr.clip(0,99999).astype(str).str.zfill(5)

    # Build hier_tag in the exact order you described:
    df["hier_tag"] = (
        df["is_privileged"].astype(int).astype(str) + "|" +
        df["primary_bit"].astype(str)        + "|" +
        df["active_bit"].astype(str)         + "|" +
        df["connect_tier"]                   + "|" +
        df["opps_bucket"]                    + "|" +
        df["activity_bucket"]                + "|" +  # coarse recency
        df["primary_email_bit"].astype(str)  + "|" +
        df["connect_email_bit"].astype(str)  + "|" +
        df["days_since_activity"]            + "|" +  # fine recency
        df["created_rank"]                      # fallback creation
    )

    # drop the helpers
    return df.drop(columns=[
        "primary_bit","active_bit","connect_tier",
        "opps_bucket","activity_bucket",
        "primary_email_bit","connect_email_bit",
        "days_since_activity","created_rank"
    ])

# ----------------------------------------------------------------------------
# Step 3: Normalized-Fields Preparation
# ----------------------------------------------------------------------------

def prepare_normalized_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds helper columns for blocking and fuzzy matching:
      • email_norm: lowercased, stripped primary email
      • connect_norm: lowercased, stripped Connect Link email
      • name_norm: lowercased, letters-and-spaces-only full name
      • sfi_key: blocking key of surname + '_' + first initial
    """
    df = df.copy()
    df["email_norm"]   = df["Email"].astype(str).str.lower().str.strip()
    df["connect_norm"] = df["Connect Link Email"].astype(str).str.lower().str.strip()
    df["name_norm"]    = (
        df["Full Name"].astype(str)
                      .str.lower()
                      .str.replace(r"[^a-z ]", "", regex=True)
                      .str.strip()
    )
    def make_sfi(name: str) -> str:
        parts = name.split()
        return "" if len(parts) < 2 else f"{parts[-1]}_{parts[0][0]}"
    df["sfi_key"]      = df["name_norm"].apply(make_sfi)
    logger.info("Prepared normalized fields for clustering")
    return df
# ----------------------------------------------------------------------------
# Step 4: Duplicate‑Candidate Clustering
# ----------------------------------------------------------------------------

class UnionFind:
    """
    Union‑Find (Disjoint Set) data structure to group record indices
    when they satisfy blocking or fuzzy‑match rules.
    """
    def __init__(self):
        self.parent: Dict[int,int] = {}
    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def cluster_records(
    df: pd.DataFrame,
    name_threshold: int = NAME_SIMILARITY_THRESHOLD,
    email_dist: int   = EMAIL_EDIT_DISTANCE_THRESHOLD
) -> pd.DataFrame:
    """
    Assigns each record a 'dupe_cluster_id' based on these rules within each
    Account Name:
      1) Exact match on primary email_norm (only non‑blank emails)
      2) One‑character‑off primary email local‑part (same domain)
      3) Exact match on name_norm
      4) Fuzzy match on name_norm (token_sort_ratio ≥ name_threshold)
    Ignores connect_norm clustering. Blank emails are skipped in step 1.
    Returns a DataFrame with new column 'dupe_cluster_id'.
    """
    df = prepare_normalized_fields(df)
    uf = UnionFind()

    # Cluster within each account group
    for acct, grp in df.groupby("Account Name"):
        indices = list(grp.index)

        # 1) Exact primary email (skip blanks)
        non_blank = grp[grp["email_norm"].ne("")]
        for _, block in non_blank.groupby("email_norm"):
            ids = list(block.index)
            for i in ids[1:]:
                uf.union(ids[0], i)

        # 2) One‑character‑off local‑part of primary email
        valid = non_blank[non_blank["email_norm"].str.contains("@", na=False)]
        parts = valid.assign(
            domain = valid["email_norm"].str.split("@").str[1],
            local  = valid["email_norm"].str.split("@").str[0]
        )
        for _, dgrp in parts.groupby("domain"):
            idxs = list(dgrp.index)
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    if distance.Levenshtein.distance(
                        dgrp.at[idxs[i], "local"],
                        dgrp.at[idxs[j], "local"]
                    ) <= email_dist:
                        uf.union(idxs[i], idxs[j])

        # 3) Exact full‑name match
        for _, block in grp.groupby("name_norm"):
            ids = list(block.index)
            for i in ids[1:]:
                uf.union(ids[0], i)

        # 4) Fuzzy full‑name matching
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                a, b = indices[i], indices[j]
                if uf.find(a) != uf.find(b):
                    if fuzz.token_sort_ratio(
                        df.at[a, "name_norm"],
                        df.at[b, "name_norm"]
                    ) >= name_threshold:
                        uf.union(a, b)

        # 5) Exact sfi_key match (surname + '_' + first initial)
        for _, block in grp.groupby("sfi_key"):
            ids = block.index.tolist()
        for i in ids[1:]:
            uf.union(ids[0], i)


    # Assign stable cluster IDs
    root_to_cid: Dict[int,str] = {}
    cluster_ids: List[str] = []
    counter = 1
    for idx in df.index:
        root = uf.find(idx)
        if root not in root_to_cid:
            root_to_cid[root] = f"C{counter:05d}"
            counter += 1
        cluster_ids.append(root_to_cid[root])
    df["dupe_cluster_id"] = cluster_ids

    logger.info("Assigned dupe_cluster_id to %d records", len(df))
    return df

# ----------------------------------------------------------------------------
# Step 5: Canonical Selection
# ----------------------------------------------------------------------------

def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each dupe_cluster_id group, identify the record(s) with the highest
    'hier_tag'. If there is exactly one such record, mark it as 'keep'. If there
    are multiple tied at the top tag, mark them 'keep_tie'. All other records
    get marked 'merge' and pointed at the chosen canonical Contact Id.

    Returns:
        pd.DataFrame: Copy of input with new columns:
          - is_canonical (bool)
          - canonical_contact_id (str)
          - resolution_status (str)
    """
    df = df.copy()
    df["is_canonical"] = False
    df["canonical_contact_id"] = None
    df["resolution_status"] = None

    def pick_top(sub: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sorted_sub = sub.sort_values("hier_tag", ascending=False)
        top_tag = sorted_sub.iloc[0]["hier_tag"]
        tied = sorted_sub[sorted_sub["hier_tag"] == top_tag]
        return tied, sorted_sub

    for cluster_id, group in df.groupby("dupe_cluster_id"):
        tied, sorted_group = pick_top(group)
        if len(tied) > 1:
            # Multiple top-tier records: all are kept, others merged
            for idx in tied.index:
                df.at[idx, "is_canonical"] = True
                df.at[idx, "canonical_contact_id"] = df.at[idx, "Contact Id"]
                df.at[idx, "resolution_status"] = "keep_tie"
            loser_idx = sorted_group.index.difference(tied.index)
            for idx in loser_idx:
                df.at[idx, "canonical_contact_id"] = tied.iloc[0]["Contact Id"]
                df.at[idx, "resolution_status"] = "merge"
        else:
            # Single winner: keep it, merge the rest
            winner_idx = tied.index[0]
            df.at[winner_idx, "is_canonical"] = True
            df.at[winner_idx, "canonical_contact_id"] = df.at[winner_idx, "Contact Id"]
            df.at[winner_idx, "resolution_status"] = "keep"
            loser_idx = sorted_group.index.difference([winner_idx])
            for idx in loser_idx:
                df.at[idx, "canonical_contact_id"] = df.at[winner_idx, "Contact Id"]
                df.at[idx, "resolution_status"] = "merge"
    logger.info("Selected canonical records")
    return df
# ----------------------------------------------------------------------------
# Step 6: Merge or Inactivate
# ----------------------------------------------------------------------------

def merge_or_inactivate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each record not marked canonical, applies merge/inactivate logic uniformly,
    with:
      • special handling when the keep has a blank primary email (unchanged),
      • and in the standard case (keep has a primary), merges only on primary→primary
        (exact or Levenshtein ≤ threshold) and immediately inactivates any other non-blank.
    """
    df = df.copy()

    # 1) Privileged safeguard
    priv = df["is_privileged"]
    df.loc[priv, "is_canonical"]          = True
    df.loc[priv, "canonical_contact_id"]  = df.loc[priv, "Contact Id"]
    df.loc[priv, "resolution_status"]     = "keep_privileged"

    # 2) Gather canonical info per cluster
    canonals: Dict[str, List[Dict[str,str]]] = {}
    for cid, grp in df[df["is_canonical"]].groupby("dupe_cluster_id"):
        canonals[cid] = [{
            "primary":      df.at[idx, "email_norm"],
            "connect":      df.at[idx, "connect_norm"],
            "name":         df.at[idx, "name_norm"],
            "contact_id":   df.at[idx, "Contact Id"],
            "hier_tag":     df.at[idx, "hier_tag"]
        } for idx in grp.index]

    # 3) Identify candidates
    candidates: Dict[str,List[int]] = {}
    for cid, grp in df.groupby("dupe_cluster_id"):
        lst = [i for i in grp.index
               if not df.at[i,"is_canonical"] and not df.at[i,"is_privileged"]]
        if lst:
            candidates[cid] = lst

    # 4) Special‑case: keep with blank primary email (unchanged)
    for cid, cans in candidates.items():
        keeps = canonals.get(cid, [])
        if len(keeps)==1 and keeps[0]["primary"] == "":
            keep_id = keeps[0]["contact_id"]
            # single candidate → merge
            if len(cans)==1:
                df.at[cans[0], "resolution_status"]        = "merge_exact"
                df.at[cans[0], "canonical_contact_id"]    = keep_id
                continue
            # multiple candidates → choose by connect match or hier_tag
            keep_con = keeps[0]["connect"]
            # 4a) primary==keep.connect
            prim = [i for i in cans if df.at[i,"email_norm"] == keep_con]
            if len(prim)==1:
                winner = prim[0]
            elif prim:
                winner = max(prim, key=lambda i: df.at[i,"hier_tag"])
            else:
                # 4b) connect==keep.connect
                conn = [i for i in cans if df.at[i,"connect_norm"] == keep_con]
                if len(conn)==1:
                    winner = conn[0]
                elif conn:
                    winner = max(conn, key=lambda i: df.at[i,"hier_tag"])
                else:
                    winner = max(cans, key=lambda i: df.at[i,"hier_tag"])
            # apply merges/inactivations
            for i in cans:
                if i == winner:
                    df.at[i, "resolution_status"]        = "merge_exact"
                    df.at[i, "canonical_contact_id"]    = keep_id
                else:
                    df.at[i, "resolution_status"]        = "inactive"
                    df.at[i, "canonical_contact_id"]    = None
            continue

    # 5) Standard logic: keep has a primary email
    for idx, row in df[(~df["is_canonical"]) & (~df["is_privileged"])].iterrows():
        # skip if already set
        if pd.notna(row["resolution_status"]):
            continue

        cid      = row["dupe_cluster_id"]
        S        = canonals.get(cid, [])
        if not S:
            continue

        keep_pri = S[0]["primary"]
        r_pri    = row["email_norm"]

        # 5a) If keep_pri is non-blank, only merge on primary→primary (exact or one-char-off)
        if keep_pri:
            if r_pri:
                dist = distance.Levenshtein.distance
                if r_pri == keep_pri or dist(r_pri, keep_pri) <= EMAIL_EDIT_DISTANCE_THRESHOLD:
                    df.at[idx, "resolution_status"]       = "merge_exact"
                    df.at[idx, "canonical_contact_id"]   = S[0]["contact_id"]
                else:
                    df.at[idx, "resolution_status"]       = "inactive"
                continue
            # else r_pri is blank → fall through to name logic

        # 5b) Name-based for truly blank primary
        matched = False
        for C in S:
            if row["name_norm"] == C["name"]:
                df.at[idx, "resolution_status"]       = "merge_name_exact"
                df.at[idx, "canonical_contact_id"]   = C["contact_id"]
                matched = True
                break
        if matched:
            continue

        for C in S:
            if fuzz.token_sort_ratio(row["name_norm"], C["name"]) >= NAME_SIMILARITY_THRESHOLD:
                df.at[idx, "resolution_status"]       = "merge_name_fuzzy"
                df.at[idx, "canonical_contact_id"]   = C["contact_id"]
                matched = True
                break
        if not matched:
            df.at[idx, "resolution_status"] = "WIP"

    logger.info("Applied updated merge/inactivate logic")
    return df

def enforce_primary_merge_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post‑check: for any row that was merged (merge_exact, merge_name_*, etc.),
    if its primary email_norm and its keep’s email_norm differ by >1 char,
    mark it inactive instead.
    """
    df = df.copy()
    # build lookup of Contact Id → keep’s primary email_norm
    keep_email = df[df["is_canonical"]].set_index("Contact Id")["email_norm"].to_dict()
    thresh = EMAIL_EDIT_DISTANCE_THRESHOLD
    dist   = distance.Levenshtein.distance

    for idx, row in df[df["resolution_status"].str.startswith("merge")].iterrows():
        r_pri = row["email_norm"]
        keep_id = row["canonical_contact_id"]
        c_pri = keep_email.get(keep_id, "")
        # only enforce on non‑blank primaries
        if r_pri and c_pri and dist(r_pri, c_pri) > thresh:
            df.at[idx, "resolution_status"] = "inactive"
    return df

# ----------------------------------------------------------------------------
# Step 6c: Re‑assign exact‑email inactives back to merge_exact
# ----------------------------------------------------------------------------

def reassign_inactive_merges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Any row marked 'inactive' but whose primary email_norm exactly matches
    a keep's primary email_norm in the same dupe_cluster_id will be flipped
    to 'merge_exact' against that keep.
    """
    df = df.copy()

    # Build lookup: (dupe_cluster_id, primary_email) -> keep Contact Id
    keep_lookup = (
        df[df["is_canonical"] & df["email_norm"].ne("")]
          .groupby(["dupe_cluster_id", "email_norm"])["Contact Id"]
          .first()
          .to_dict()
    )

    # Find all inactive rows with a non‑blank primary email
    mask = (df["resolution_status"] == "inactive") & df["email_norm"].ne("")
    for idx in df[mask].index:
        key = (df.at[idx, "dupe_cluster_id"], df.at[idx, "email_norm"])
        if key in keep_lookup:
            df.at[idx, "resolution_status"]      = "merge_exact"
            df.at[idx, "canonical_contact_id"]   = keep_lookup[key]

    return df

# ----------------------------------------------------------------------------
# Step 6d: Special one-char-off inactivation
# ----------------------------------------------------------------------------

def apply_one_char_off_inactivation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Any record whose resolution_status == 'merge_exact' solely because
    its email local-part is one Levenshtein edit away from the keep’s,
    and that single edit is one of:
      1) first_initial + surname_initial
      2) first_name + surname_initial
      3) insertion/removal of '@' in the full email
      4) insertion or deletion of one character at the beginning or end of the local-part
      5) insertion of a single numeric character in the local-part
    → mark as inactive instead.
    """
    from rapidfuzz.distance import Levenshtein

    df = df.copy()
    mask = df["resolution_status"] == "merge_exact"
    for idx in df[mask].index:
        row = df.loc[idx]
        keep = df[df["Contact Id"] == row["canonical_contact_id"]].iloc[0]

        # split local parts
        local_r = row["email_norm"].split("@")[0]
        local_k = keep["email_norm"].split("@")[0]

        # only proceed if exactly one edit in local-part
        if Levenshtein.distance(local_r, local_k) != 1:
            continue

        # pattern 1: initials match (fi + si)
        name_r = row["Full Name"].strip().lower().split()
        if len(name_r) >= 2:
            first_name, surname = name_r[0], name_r[-1]
            fi, si = first_name[0], surname[0]
            if local_r in {fi + si, first_name + si}:
                df.at[idx, "resolution_status"] = "inactive"
                continue

        # pattern 3: single '@' insertion/deletion in full email
        full_r, full_k = row["Email"], keep["Email"]
        if Levenshtein.distance(full_r, full_k) == 1:
            diffs = [i for i, (a, b) in enumerate(zip(full_r, full_k)) if a != b]
            if diffs and (full_r[diffs[0]] == "@" or full_k[diffs[0]] == "@"):
                df.at[idx, "resolution_status"] = "inactive"
                continue

        # pattern 4: one-char boundary insertion/deletion
        #    e.g. "jdoe" ↔ "ajdoe" or "jdoe" ↔ "jdoex"
        if abs(len(local_r) - len(local_k)) == 1:
            longer, shorter = (local_r, local_k) if len(local_r) > len(local_k) else (local_k, local_r)
            if longer[1:] == shorter or longer[:-1] == shorter:
                df.at[idx, "resolution_status"] = "inactive"
                continue

        # pattern 5: single numeric insertion anywhere
        #    e.g. "jdoe" ↔ "jdoe1"
        if len(local_r) - len(local_k) == 1:
            # row has the extra char
            for i, ch in enumerate(local_r):
                if ch.isdigit() and (local_r[:i] + local_r[i+1:]) == local_k:
                    df.at[idx, "resolution_status"] = "inactive"
                    break

    return df


# ----------------------------------------------------------------------------
# Step 7: Feature Engineering for Machine Learning
# ----------------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Ensure ConnectLink Status is always a string (empty if missing)
    df["ConnectLink Status"] = df["ConnectLink Status"].fillna("").astype(str)
    # Ensure Active Contact is always a string (empty if missing)
    df["Active Contact"]     = df["Active Contact"].fillna("").astype(str)

    # Now prep normalized fields as before
    df = prepare_normalized_fields(df)
    # Identify canonical per cluster…


    # Ensure ConnectLink Status is a string (empty if missing)
    df["ConnectLink Status"] = df["ConnectLink Status"].fillna("").astype(str)


    # Ensure name_norm and email_norm exist
    df = prepare_normalized_fields(df)
    # Identify canonical per cluster
    canon_map: Dict[str, Dict[str, str]] = {}
    for cid, group in df[df["is_canonical"]].groupby("dupe_cluster_id"):
        canon_map[cid] = {
            "email_norm": group.iloc[0]["email_norm"],
            "name_norm":  group.iloc[0]["name_norm"]
        }
    # Compute features
    feature_dicts: List[Dict[str, float]] = []
    labels: List[str] = []
    for idx, row in df.iterrows():
        cid = row["dupe_cluster_id"]
        canon_vals = canon_map.get(cid, {"email_norm":"","name_norm":""})
        # Hierarchy bits
        feat: Dict[str, float] = {}
        feat["is_privileged"] = float(row["is_privileged"])
        feat["primary_bit"]    = float(row["Primary Contact"])
        feat["active_bit"]     = float(row["Active Contact"].lower()=="active")
        # Connect tier numeric
        feat["connect_tier"]   = float(CONNECTLINK_STATUS_TO_TIER.get(row["ConnectLink Status"].upper(), "1"))
        # Opportunity bucket numeric
        opps = int(row["# of opps"] or 0)
        feat["opps_bucket"]    = float(0 if opps==0 else 1 if opps<=3 else 2)
        # Activity recency days
        days_act = (pd.Timestamp.today().normalize() - row["Last Activity"]).days if pd.notna(row["Last Activity"]) else 0
        feat["days_since_activity"] = float(days_act)
        # Creation seniority days
        days_cr = (pd.Timestamp.today().normalize() - row["Created Date"]).days if pd.notna(row["Created Date"]) else 0
        feat["days_since_created"] = float(days_cr)
        # Similarities to canonical
        feat["name_similarity"] = float(fuzz.token_sort_ratio(row["name_norm"], canon_vals["name_norm"]))
        feat["email_edit_dist"] = float(distance.Levenshtein.distance(row["email_norm"], canon_vals["email_norm"]))
        # Cluster size
        feat["cluster_size"]    = float(len(df[df["dupe_cluster_id"]==cid]))
        feature_dicts.append(feat)
        labels.append(row["resolution_status"])
    X = pd.DataFrame(feature_dicts)
    # Encode string labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    # Persist label encoder for inference
    df_label_enc = pd.DataFrame({"label": labels})
    return X, pd.Series(y), encoder

# ----------------------------------------------------------------------------
# Step 8: Model Training, Evaluation, Saving
# ----------------------------------------------------------------------------

def train_and_save_model(
    X: pd.DataFrame,
    y: pd.Series,
    encoder: LabelEncoder,
    model_path: Path,
    encoder_path: Path
) -> None:
    """
    Splits features and labels into train/test, trains a RandomForest classifier,
    evaluates performance, and saves both the trained model and label encoder.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("Model evaluation:\n%s", report)

    # Save model and encoder
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)
    logger.info("Saved trained model to %s and encoder to %s", model_path, encoder_path)

# ----------------------------------------------------------------------------
# Step 9: Model Loading and Inference
# ----------------------------------------------------------------------------

def load_and_apply_model(
    df: pd.DataFrame,
    model_path: Path,
    encoder_path: Path
) -> pd.DataFrame:
    """
    Loads a saved classifier and label encoder, computes features for the DataFrame,
    predicts resolution_status for each record, and appends a new column
    'ml_resolution_status' with the decoded predictions.
    """
    # Load model and encoder
    clf: RandomForestClassifier = joblib.load(model_path)
    encoder: LabelEncoder = joblib.load(encoder_path)

    # Extract features (we discard the returned encoder)
    X, _, _ = extract_features(df)
    y_pred = clf.predict(X)
    decoded = encoder.inverse_transform(y_pred)

    df = df.copy()
    df["ml_resolution_status"] = decoded
    logger.info("Applied ML model to %d records", len(df))
    return df
# ----------------------------------------------------------------------------
# Full Pipeline Invocation (Definition Only)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Full Pipeline Invocation
# ----------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    output_path: Path,
    model_path: Optional[Path]   = None,
    encoder_path: Optional[Path] = None,
    train_model_flag: bool       = False
) -> pd.DataFrame:
    """
    Executes the full deduplication pipeline:
      1) Load and normalize contacts
      2) Build hierarchy tags
      3) Cluster duplicate candidates
      4) Select canonical records
      5) Merge or inactivate other records
      6) Optional: train an RF model if train_model_flag=True
      7) Optional: apply an existing model for ML-based status
      8) Export results to Excel

    Returns:
        pd.DataFrame: DataFrame of final results.
    """
    # 1) Ingest
    df = load_contacts(input_path)

    # 2) Hierarchy tagging
    df = build_hierarchy_tag(df)

    # 3) Duplicate clustering
    df = cluster_records(df)

    # 4) Canonical selection
    df = select_canonical(df)

    # 5) Merge or inactivate
    df = merge_or_inactivate(df)
    df = enforce_primary_merge_threshold(df)
    df = reassign_inactive_merges(df)
    df = apply_one_char_off_inactivation(df)


    # 6) Optional training
    if train_model_flag and model_path and encoder_path:
        X, y, enc = extract_features(df)
        train_and_save_model(X, y, enc, model_path, encoder_path)

    # 7) Optional inference
    if model_path and encoder_path and not train_model_flag:
        if model_path.exists() and encoder_path.exists():
            df = load_and_apply_model(df, model_path, encoder_path)
        else:
            logger.warning("Model or encoder missing; skipping ML inference")

    # 8) Export to Excel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
    logger.info("Exported results to %s", output_path)

    return df
