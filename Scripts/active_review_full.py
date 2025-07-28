def format_cluster_prompt(cluster_id: str, df: pd.DataFrame) -> str:
    # pick only the fields you need (including both rule‑ and ML‑statuses)
    records = df[
        [
            "Contact Id",
            "Full Name",
            "Email",
            "is_privileged",
            "hier_tag",
            "resolution_status",
            "ml_resolution_status",
        ]
    ].to_dict(orient="records")

    # pretty‑print the JSON
    json_records = json.dumps(records, indent=2)

    # example of the JSON schema we expect back
    example_response = {
        "cluster_id": cluster_id,
        "llm_resolution_status": "keep",   # or "merge" / "inactive"
        "explanation": "Your brief reason here."
    }
    json_example = json.dumps(example_response, indent=2)

    # stitch it all together
    prompt = (
        f"Cluster ID: {cluster_id}\n\n"
        f"Here are the {len(records)} records in this cluster as JSON:\n"
        "```json\n"
        f"{json_records}\n"
        "```\n\n"
        "Please reply with a JSON object in this format:\n"
        "```json\n"
        f"{json_example}\n"
        "```\n"
    )
    return prompt
