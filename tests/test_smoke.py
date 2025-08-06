from pathlib import Path
import pandas as pd
from contact_dedupe.pipeline.dedupe_pipeline import run_pipeline


def test_smoke(tmp_path: Path):
    df = pd.DataFrame(
        {
            "Account Name": ["A", "A"],
            "Duplicate Record Set ID": ["", ""],
            "Full Name": ["Jon Brady", "John Brede"],
            "Email": ["a@example.com", "a@example.com"],
            "Contact Id": ["1", "2"],
            "Admin Role": ["", ""],
            "Primary Contact": [True, False],
            "Active Contact": ["active", "inactive"],
            "ConnectLink Status": ["A", ""],
            "Connect Link Email": ["", ""],
            "# of cases": ["0", "0"],
            "# of opps": ["0", "0"],
            "Last Activity": ["2024-01-01", "2024-01-01"],
            "Created Date": ["2023-01-01", "2023-01-01"],
        }
    )
    # write input
    src = tmp_path / "in.xlsx"
    df.to_excel(src, index=False)
    # run pipeline
    out = tmp_path / "out.xlsx"
    result = run_pipeline(src, out)
    # assertions
    assert out.exists()
    assert len(result) == 2
