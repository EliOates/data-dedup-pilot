import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd
from rapidfuzz import fuzz, distance
from itertools import combinations

from contact_dedupe.pipeline.llm_validation import ask_names_same

# ----------------------------------------------------------------------------
# Configuration and Setup
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thresholds for fuzzy/rule-based matching
NAME_SIMILARITY_THRESHOLD = 95
EMAIL_EDIT_DISTANCE_THRESHOLD = 0

CONNECTLINK_STATUS_TO_TIER: Dict[str, str] = {
    "A": "3",  # Active
    "I": "2",  # Inactive
    "U": "2",  # Unknown→Inactive
    "": "1",  # Blank/Other
}

# ----------------------------------------------------------------------------
# Step 1: Ingestion and Header Mapping
# ----------------------------------------------------------------------------


def load_contacts(file_path: Path) -> pd.DataFrame:
    """
    Reads the raw Excel file, normalizes headers/fields, and validates required columns.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl", dtype=str)

    # Rename upstream headers to our canonical column names
    rename_map = {
        "Account Name: Acct_ID_18": "Account Name",
        "Contact_id_18": "Contact Id",
        "Primary Contact Any": "Primary Contact",
        "Agile Contact Email": "Connect Link Email",
        "# of Cases": "# of cases",
        "# of Opps": "# of opps",
    }
    df = df.rename(columns=rename_map)

    # Synthesize 'Full Name' if missing but First/Last Name exist
    if "Full Name" not in df.columns:
        if {"First Name", "Last Name"}.issubset(df.columns):
            df["Full Name"] = (
                df["First Name"].fillna("").str.strip()
                + " "
                + df["Last Name"].fillna("").str.strip()
            ).str.strip()
        else:
            raise ValueError(
                "Missing both 'Full Name' and 'First Name'+'Last Name' columns"
            )

    # Parse date columns
    for dt_col in ["Last Activity", "Created Date"]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    required = [
        "Account Name",
        "Duplicate Record Set ID",
        "Full Name",
        "Email",
        "Contact Id",
        "Admin Role",
        "Primary Contact",
        "Active Contact",
        "ConnectLink Status",
        "Connect Link Email",
        "# of cases",
        "# of opps",
        "Last Activity",
        "Created Date",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after ingestion: {missing}")

    # Normalize text fields
    for col in required:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    # Convert 'Primary Contact' to boolean
    df["Primary Contact"] = (
        df["Primary Contact"].astype(str).str.lower().isin({"true", "1", "yes"})
    )

    logger.info("Loaded and normalized %d rows", len(df))
    return df


# ----------------------------------------------------------------------------
# Step 2: Hierarchy Tag Construction
# ----------------------------------------------------------------------------


def build_hierarchy_tag(
    df: pd.DataFrame, reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    df = df.copy()
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    # parse dates
    df["Last Activity"] = pd.to_datetime(df["Last Activity"], errors="coerce")
    df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")

    # 1. Privilege bit
    df["is_privileged"] = df["Admin Role"].str.lower().isin({"owner", "admin"})

    # 2. Primary contact
    df["primary_bit"] = df["Primary Contact"].astype(bool).astype(int)

    # 3. Active contact
    df["active_bit"] = df["Active Contact"].str.lower().eq("active").astype(int)

    # 4. Connection tier
    df["connect_tier"] = (
        df["ConnectLink Status"]
        .fillna("")
        .str.upper()
        .map(CONNECTLINK_STATUS_TO_TIER)
        .fillna("1")
    )

    # 5. Opportunity bucket
    opps = df["# of opps"].replace("", 0).astype(int)
    df["opps_bucket"] = pd.cut(
        opps, bins=[-1, 0, 3, 6, float("inf")], labels=["0", "1", "2", "3"]
    ).astype(str)

    # 6. Activity bucket (coarse)
    days_act = (reference_date - df["Last Activity"]).dt.days.fillna(9999).astype(int)
    df["activity_bucket"] = pd.cut(
        days_act, bins=[-1, 365, 1095, 1825, float("inf")], labels=["3", "2", "1", "0"]
    ).astype(str)

    # 7. Primary email presence
    df["primary_email_bit"] = df["Email"].astype(str).str.strip().ne("").astype(int)

    # 8. Connect-link email presence
    df["connect_email_bit"] = (
        df["Connect Link Email"].astype(str).str.strip().ne("").astype(int)
    )

    # 9. Days since last activity (inverted, zero-padded)
    exact_days = days_act.clip(0, 99999).astype(int)
    inv_days = (99999 - exact_days).astype(int)
    df["days_since_activity"] = inv_days.astype(str).str.zfill(5)

    # 10. Creation seniority (zero-padded)
    days_cr = (reference_date - df["Created Date"]).dt.days.fillna(0).astype(int)
    df["created_rank"] = days_cr.clip(0, 99999).astype(str).str.zfill(5)

    df["hier_tag"] = (
        df["is_privileged"].astype(int).astype(str)
        + "|"
        + df["primary_bit"].astype(str)
        + "|"
        + df["active_bit"].astype(str)
        + "|"
        + df["connect_tier"]
        + "|"
        + df["opps_bucket"]
        + "|"
        + df["activity_bucket"]
        + "|"
        + df["primary_email_bit"].astype(str)
        + "|"
        + df["connect_email_bit"].astype(str)
        + "|"
        + df["days_since_activity"]
        + "|"
        + df["created_rank"]
    )

    return df.drop(
        columns=[
            "primary_bit",
            "active_bit",
            "connect_tier",
            "opps_bucket",
            "activity_bucket",
            "primary_email_bit",
            "connect_email_bit",
            "days_since_activity",
            "created_rank",
        ]
    )


# ----------------------------------------------------------------------------
# Step 3: Normalized-Fields Preparation
# ----------------------------------------------------------------------------


def prepare_normalized_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["email_norm"] = df["Email"].astype(str).str.lower().str.strip()
    df["connect_norm"] = df["Connect Link Email"].astype(str).str.lower().str.strip()
    df["name_norm"] = (
        df["Full Name"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.strip()
    )

    def make_sfi(name: str) -> str:
        parts = name.split()
        return "" if len(parts) < 2 else f"{parts[-1]}_{parts[0][0]}"

    df["sfi_key"] = df["name_norm"].apply(make_sfi)
    logger.info("Prepared normalized fields for clustering")
    return df


# ----------------------------------------------------------------------------
# Step 4: Duplicate-Candidate Clustering + LLM override for singletons
# ----------------------------------------------------------------------------


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def cluster_records(
    df: pd.DataFrame,
    name_threshold: int = NAME_SIMILARITY_THRESHOLD,
    email_dist: int = EMAIL_EDIT_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    df = prepare_normalized_fields(df)

    uf = UnionFind()

    # 1) Force-union by Duplicate Record Set ID within an account
    for acct, grp in df.groupby("Account Name"):
        valid = grp["Duplicate Record Set ID"].notna() & (
            grp["Duplicate Record Set ID"] != ""
        )
        for drsid in grp.loc[valid, "Duplicate Record Set ID"].unique():
            idxs = grp[grp["Duplicate Record Set ID"] == drsid].index.tolist()
            for i, j in combinations(idxs, 2):
                uf.union(i, j)

    # 2) Seed cluster IDs
    root_to_cid: Dict[int, str] = {}
    seed_cids: List[str] = []
    cid_counter = 1
    for i in df.index:
        root = uf.find(i)
        if root not in root_to_cid:
            root_to_cid[root] = f"C{cid_counter:05d}"
            cid_counter += 1
        seed_cids.append(root_to_cid[root])
    df["seed_dupe_cluster_id"] = seed_cids

    # 3) Heuristic-based clustering per account
    for acct, grp in df.groupby("Account Name"):
        indices = list(grp.index)

        # 3a) Exact primary email
        non_blank = grp[grp["email_norm"].ne("")]
        for _, block in non_blank.groupby("email_norm"):
            ids = list(block.index)
            for idx in ids[1:]:
                uf.union(ids[0], idx)

        # 3b) Exact normalized name
        for _, block in grp.groupby("name_norm"):
            ids = list(block.index)
            for idx in ids[1:]:
                uf.union(ids[0], idx)

        # 3c) Fuzzy name
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if uf.find(a) != uf.find(b) and (
                    fuzz.token_sort_ratio(df.at[a, "name_norm"], df.at[b, "name_norm"])
                    >= name_threshold
                ):
                    uf.union(a, b)

        # 3d) SFI key
        for _, block in grp.groupby("sfi_key"):
            ids = list(block.index)
            for idx in ids[1:]:
                uf.union(ids[0], idx)

    # 4) Final cluster IDs
    root_to_cid.clear()
    final_cids: List[str] = []
    cid_counter = 1
    for i in df.index:
        root = uf.find(i)
        if root not in root_to_cid:
            root_to_cid[root] = f"C{cid_counter:05d}"
            cid_counter += 1
        final_cids.append(root_to_cid[root])
    df["dupe_cluster_id"] = final_cids

    # 5) LLM override for singletons (optional)
    counts = df["dupe_cluster_id"].value_counts()
    singleton_idxs = df[
        df["dupe_cluster_id"].isin(counts[counts == 1].index)
    ].index.tolist()

    llm_uf = UnionFind()
    logger.info("→ %d singletons to consider for LLM override", len(singleton_idxs))

    for idx in singleton_idxs:
        # check immediate neighbors in the singleton list
        pos = singleton_idxs.index(idx)
        neighbours = []
        if pos > 0:
            neighbours.append(singleton_idxs[pos - 1])
        if pos < len(singleton_idxs) - 1:
            neighbours.append(singleton_idxs[pos + 1])

        name_a = df.at[idx, "Full Name"]
        for n_idx in neighbours:
            name_b = df.at[n_idx, "Full Name"]
            if ask_names_same(name_a, name_b, lenient=True):
                llm_uf.union(idx, n_idx)
                break

    # fold LLM suggestions into the main clusters
    for idx in singleton_idxs:
        parent = llm_uf.find(idx)
        if parent != idx:
            df.at[idx, "dupe_cluster_id"] = df.at[parent, "dupe_cluster_id"]

    logger.info("Applied LLM overrides to singletons")
    return df


# ----------------------------------------------------------------------------
# Step 5: Canonical Record Selection
# ----------------------------------------------------------------------------


def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
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
            for idx in tied.index:
                df.at[idx, "is_canonical"] = True
                df.at[idx, "canonical_contact_id"] = df.at[idx, "Contact Id"]
                df.at[idx, "resolution_status"] = "keep_tie"
            loser_idx = sorted_group.index.difference(tied.index)
            for idx in loser_idx:
                df.at[idx, "canonical_contact_id"] = tied.iloc[0]["Contact Id"]
                df.at[idx, "resolution_status"] = "merge"
        else:
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
    df = df.copy()

    # 1) Privileged safeguard
    priv = df["is_privileged"]
    df.loc[priv, "is_canonical"] = True
    df.loc[priv, "canonical_contact_id"] = df.loc[priv, "Contact Id"]
    df.loc[priv, "resolution_status"] = "keep_privileged"

    # 2) Gather canonical info per cluster
    canonals: Dict[str, List[Dict[str, str]]] = {}
    for cid, grp in df[df["is_canonical"]].groupby("dupe_cluster_id"):
        canonals[cid] = [
            {
                "primary": df.at[idx, "email_norm"],
                "connect": df.at[idx, "connect_norm"],
                "name": df.at[idx, "name_norm"],
                "contact_id": df.at[idx, "Contact Id"],
                "hier_tag": df.at[idx, "hier_tag"],
            }
            for idx in grp.index
        ]

    # 3) Identify candidates
    candidates: Dict[str, List[int]] = {}
    for cid, grp in df.groupby("dupe_cluster_id"):
        lst = [
            i
            for i in grp.index
            if not df.at[i, "is_canonical"] and not df.at[i, "is_privileged"]
        ]
        if lst:
            candidates[cid] = lst

    # 4) Special-case: keep with blank primary email
    for cid, cans in candidates.items():
        keeps = canonals.get(cid, [])
        if len(keeps) == 1 and keeps[0]["primary"] == "":
            keep_id = keeps[0]["contact_id"]
            if len(cans) == 1:
                df.at[cans[0], "resolution_status"] = "merge_exact"
                df.at[cans[0], "canonical_contact_id"] = keep_id
                continue

            keep_con = keeps[0]["connect"]
            prim = [i for i in cans if df.at[i, "email_norm"] == keep_con]
            if len(prim) == 1:
                winner = prim[0]
            elif prim:
                winner = max(prim, key=lambda i: df.at[i, "hier_tag"])
            else:
                conn = [i for i in cans if df.at[i, "connect_norm"] == keep_con]
                if len(conn) == 1:
                    winner = conn[0]
                elif conn:
                    winner = max(conn, key=lambda i: df.at[i, "hier_tag"])
                else:
                    winner = max(cans, key=lambda i: df.at[i, "hier_tag"])

            for i in cans:
                if i == winner:
                    df.at[i, "resolution_status"] = "merge_exact"
                    df.at[i, "canonical_contact_id"] = keep_id
                else:
                    df.at[i, "resolution_status"] = "inactive"
                    df.at[i, "canonical_contact_id"] = None
            continue

    # 5) Standard logic: keep has a primary email
    for idx, row in df[(~df["is_canonical"]) & (~df["is_privileged"])].iterrows():
        if pd.notna(row["resolution_status"]):
            continue

        cid = row["dupe_cluster_id"]
        S = canonals.get(cid, [])
        if not S:
            continue

        keep_pri = S[0]["primary"]
        r_pri = row["email_norm"]

        if keep_pri:
            if r_pri:
                dist = distance.Levenshtein.distance
                if (
                    r_pri == keep_pri
                    or dist(r_pri, keep_pri) <= EMAIL_EDIT_DISTANCE_THRESHOLD
                ):
                    df.at[idx, "resolution_status"] = "merge_exact"
                    df.at[idx, "canonical_contact_id"] = S[0]["contact_id"]
                else:
                    df.at[idx, "resolution_status"] = "inactive"
                continue

        # Name-based fallback for blank primary
        matched = False
        for C in S:
            if row["name_norm"] == C["name"]:
                df.at[idx, "resolution_status"] = "merge_name_exact"
                df.at[idx, "canonical_contact_id"] = C["contact_id"]
                matched = True
                break
        if matched:
            continue

        for C in S:
            if (
                fuzz.token_sort_ratio(row["name_norm"], C["name"])
                >= NAME_SIMILARITY_THRESHOLD
            ):
                df.at[idx, "resolution_status"] = "merge_name_fuzzy"
                df.at[idx, "canonical_contact_id"] = C["contact_id"]
                matched = True
                break

        if not matched:
            df.at[idx, "resolution_status"] = "WIP"

    logger.info("Applied updated merge/inactivate logic")
    return df


def enforce_primary_merge_threshold(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep_email = df[df["is_canonical"]].set_index("Contact Id")["email_norm"].to_dict()
    thresh = EMAIL_EDIT_DISTANCE_THRESHOLD
    dist = distance.Levenshtein.distance

    for idx, row in df[df["resolution_status"].str.startswith("merge")].iterrows():
        r_pri = row["email_norm"]
        keep_id = row["canonical_contact_id"]
        c_pri = keep_email.get(keep_id, "")
        if r_pri and c_pri and dist(r_pri, c_pri) > thresh:
            df.at[idx, "resolution_status"] = "inactive"
    return df


def reassign_inactive_merges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep_lookup = (
        df[df["is_canonical"] & df["email_norm"].ne("")]
        .groupby(["dupe_cluster_id", "email_norm"])["Contact Id"]
        .first()
        .to_dict()
    )
    mask = (df["resolution_status"] == "inactive") & df["email_norm"].ne("")
    for idx in df[mask].index:
        key = (df.at[idx, "dupe_cluster_id"], df.at[idx, "email_norm"])
        if key in keep_lookup:
            df.at[idx, "resolution_status"] = "merge_exact"
            df.at[idx, "canonical_contact_id"] = keep_lookup[key]
    return df


def apply_one_char_off_inactivation(df: pd.DataFrame) -> pd.DataFrame:
    from rapidfuzz.distance import Levenshtein

    df = df.copy()
    mask = df["resolution_status"] == "merge_exact"
    for idx in df[mask].index:
        row = df.loc[idx]
        keep = df[df["Contact Id"] == row["canonical_contact_id"]].iloc[0]

        local_r = row["email_norm"].split("@")[0]
        local_k = keep["email_norm"].split("@")[0]

        if Levenshtein.distance(local_r, local_k) != 1:
            continue

        name_r = row["Full Name"].strip().lower().split()
        if len(name_r) >= 2:
            first_name, surname = name_r[0], name_r[-1]
            fi, si = first_name[0], surname[0]
            if local_r in {fi + si, first_name + si}:
                df.at[idx, "resolution_status"] = "inactive"
                continue

        full_r, full_k = row["Email"], keep["Email"]
        if Levenshtein.distance(full_r, full_k) == 1:
            diffs = [i for i, (a, b) in enumerate(zip(full_r, full_k)) if a != b]
            if diffs and (full_r[diffs[0]] == "@" or full_k[diffs[0]] == "@"):
                df.at[idx, "resolution_status"] = "inactive"
                continue

        if abs(len(local_r) - len(local_k)) == 1:
            longer, shorter = (
                (local_r, local_k)
                if len(local_r) > len(local_k)
                else (local_k, local_r)
            )
            if longer[1:] == shorter or longer[:-1] == shorter:
                df.at[idx, "resolution_status"] = "inactive"
                continue

        if len(local_r) - len(local_k) == 1:
            for i, ch in enumerate(local_r):
                if ch.isdigit() and (local_r[:i] + local_r[i + 1 :]) == local_k:
                    df.at[idx, "resolution_status"] = "inactive"
                    break

    return df


# ----------------------------------------------------------------------------
# Full Pipeline
# ----------------------------------------------------------------------------


def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Executes the dedupe pipeline (no ML):
      1) Load & normalize
      2) Build hierarchy tags
      3) Cluster duplicates (+ optional LLM override)
      4) Select canonical
      5) Merge/Inactivate passes
      6) Export
    """
    df = load_contacts(input_path)
    df = build_hierarchy_tag(df)
    df = cluster_records(df)
    df = select_canonical(df)
    df = merge_or_inactivate(df)
    df = enforce_primary_merge_threshold(df)
    df = reassign_inactive_merges(df)
    df = apply_one_char_off_inactivation(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
    logger.info("Exported results to %s", output_path)
    return df
