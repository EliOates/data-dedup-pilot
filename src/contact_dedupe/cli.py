from pathlib import Path
import typer

app = typer.Typer(help="Contact dedupe & ML pipeline")


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
    from contact_dedupe.pipeline.dedupe_pipeline import run_pipeline

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = run_pipeline(
        input_path=input_path, output_path=output_path
    )  # adjust if your signature differs
    typer.echo(f"✅ Wrote {output_path} with {len(df)} rows")


@app.command()
def train(
    labels_path: Path = typer.Option(
        ..., exists=True, readable=True, help="Labeled Excel/CSV for training"
    ),
    model_out: Path = typer.Option(
        Path("models/rf_model_active.joblib"), help="Where to save the model"
    ),
):
    """
    Train/update the ML model (wraps your active-learning pipeline).
    """
    # If your training script exposes a function, prefer importing it here.
    try:
        from contact_dedupe.pipeline.active_learning_pipeline import train_model  # type: ignore

        train_model(
            labels_path, model_out
        )  # If your function has a different name/signature, tell me and I'll adjust.
    except Exception:
        # Fallback: call the script-style main if present
        from contact_dedupe.pipeline import active_learning_pipeline as alp  # type: ignore

        if hasattr(alp, "main"):
            alp.main()
        else:
            raise RuntimeError(
                "Couldn't find train_model(...) or main() in active_learning_pipeline.py. "
                "Share the top of that file and I'll wire it correctly."
            )
    typer.echo(f"✅ Trained model -> {model_out}")


if __name__ == "__main__":
    app()
