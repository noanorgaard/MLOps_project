"""Upload reference feature statistics to GCS.

This is used by the drift detection API as reference data.

Run:
    uv run python -m ai_vs_human.upload_reference_features
"""

from ai_vs_human.data_drif import upload_training_features


def main() -> None:
    """Upload reference features to the configured GCS bucket."""
    upload_training_features()


if __name__ == "__main__":
    main()
