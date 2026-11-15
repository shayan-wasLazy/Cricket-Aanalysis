from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cricket_pressure.rr_intelligence import build_rr_intelligence, download_cricsheet_archive  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Rajasthan Royals pressure and squad-fit intelligence from Cricsheet IPL JSON."
    )
    parser.add_argument("--source", type=Path, default=None,
                        help="Cricsheet IPL JSON ZIP or extracted JSON directory.")
    parser.add_argument("--download", action="store_true",
                        help="Download the official Cricsheet IPL JSON archive first.")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--team", default="Rajasthan Royals")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "rr_intelligence")
    args = parser.parse_args()

    archive = ROOT / "dataset" / "rr_intelligence" / "ipl_json.zip"
    source = download_cricsheet_archive(args.source or archive) if args.download else (args.source or archive)
    if not source.exists():
        parser.error(f"Source not found: {source}. Use --download or provide --source.")
    build_rr_intelligence(
        source=source, output_dir=args.output_dir,
        start_year=args.start_year, end_year=args.end_year, rr_team=args.team,
    )


if __name__ == "__main__":
    main()
