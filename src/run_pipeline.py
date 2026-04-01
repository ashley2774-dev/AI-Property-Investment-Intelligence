from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_step(script: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script)] + (args or [])
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full property ML pipeline")
    parser.add_argument("--skip-scraping", action="store_true")
    args = parser.parse_args()

    if not args.skip_scraping:
        run_step("src/scraping/privateproperty_scraper_pipeline.py")
    run_step("src/data/clean_sales_listings_pipeline.py")
    run_step("src/data/clean_rentals_pipeline.py")
    run_step("src/data/property_dataset_expansion_pipeline.py")
    run_step("src/eda/eda_sales_pipeline.py")
    run_step("src/eda/eda_rentals_pipeline.py")
    run_step("src/features/financial_engine_pipeline.py")
    run_step("src/features/feature_engineering_pipeline.py")
    run_step("src/modeling/feature_selection_modeling_pipeline.py")
    print("\n✅ Full pipeline complete")


if __name__ == "__main__":
    main()
