from __future__ import annotations

from pathlib import Path

import pandas as pd

from cricket_pressure.player_rankings import build_player_rankings, compare_top10_rankings


def main() -> None:
    root = Path(__file__).resolve().parent
    processed_path = root / "data" / "processed" / "real_pressure_comparison.csv.gz"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed delivery data not found: {processed_path}. "
            "Run python run_analysis.py first to build it."
        )

    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(processed_path, low_memory=False)
    rankings = build_player_rankings(real)
    if rankings.empty:
        raise RuntimeError("No player rankings were generated from the processed data.")

    rankings.to_csv(output_dir / "player_rankings.csv", index=False)
    compare_top10_rankings(rankings, top_n=10).to_csv(
        output_dir / "player_top10_comparison.csv", index=False
    )

    reference = rankings["recency_reference_date"].dropna().astype(str).iloc[0]
    print(f"Player rankings refreshed as of {reference}.")
    print(f"Saved {len(rankings):,} ranking rows to {output_dir / 'player_rankings.csv'}.")


if __name__ == "__main__":
    main()
