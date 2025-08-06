from pathlib import Path
import typer

app = typer.Typer(help="Contact dedupe pipeline")


@app.command()
def run(
    input_path: Path = typer.Option(
        ..., exists=True, readable=True, help="Input Excel/CSV file"
    ),
    output_path: Path = typer.Option(
        Path("output/Contacts_Clean.xlsx"), help="Output file"
    ),
):
    """
    Run the full dedupe pipeline and write an output file.
    """
    # Import inside the command so the CLI can load even if heavy deps aren't ready
    from contact_dedupe.pipeline.dedupe_pipeline_final import run_pipeline

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = run_pipeline(input_path=input_path, output_path=output_path)
    typer.echo(f"✅ Wrote {output_path} with {len(df)} rows")


if __name__ == "__main__":
    app()
