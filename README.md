\# Contact Dedupe



Rules-first contact deduplication with optional LLM assist for edge cases.



\## Quickstart (Windows)

```powershell

python -m venv .venv

. .\\.venv\\Scripts\\Activate.ps1

pip install -e .

python -m contact\_dedupe.cli run --input-path Data\\Contacts\_LLM\_Test.xlsx --output-path output\\Contacts\_Clean.xlsx
