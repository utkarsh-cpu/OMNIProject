"""
OMNI Solar Data Pipeline — Entry Point
=======================================
Usage:
    python run_pipeline.py                  # Full pipeline
    python run_pipeline.py --skip-extract   # Analysis only (data already downloaded)
    python run_pipeline.py --skip-analysis  # Download only (no FITS analysis)
    python run_pipeline.py --analyze FILE   # Analyse a single FITS file
"""

import argparse
import sys

from src.pipeline import SolarDataPipeline
from src.fits_analyzer import FITSAnalyzer
from src.config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="OMNI Solar Flare Data Pipeline — extract & analyse SDO data"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to pipeline_config.yaml (default: config/pipeline_config.yaml)"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip data extraction; only run FITS analysis on existing files"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip FITS analysis; only download data"
    )
    parser.add_argument(
        "--analyze", metavar="FILE",
        help="Analyse and print info for a single FITS file, then exit"
    )
    args = parser.parse_args()

    # Single-file analysis mode
    if args.analyze:
        FITSAnalyzer.print_info(args.analyze)
        config = PipelineConfig(args.config)
        analyzer = FITSAnalyzer(config)
        result = analyzer.analyze_file(args.analyze)
        print("\nStructured analysis:")
        import json
        print(json.dumps(result, indent=2, default=str))
        return

    # Full pipeline
    pipeline = SolarDataPipeline(config_path=args.config)
    pipeline.run(
        skip_extraction=args.skip_extract,
        skip_analysis=args.skip_analysis,
    )


if __name__ == "__main__":
    main()
