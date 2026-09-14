"""Combine yearly ATP match CSV files into one chronological dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine_csv_files(
    data_dir: Path, start_year: int = 1968, end_year: int = 2024
) -> pd.DataFrame:
    """Load available ``atp_matches_YEAR.csv`` files and combine their rows."""
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        source = data_dir / f"atp_matches_{year}.csv"
        if not source.is_file():
            print(f"Skipping missing file: {source.name}")
            continue

        frame = pd.read_csv(source)
        frame["year"] = year
        frames.append(frame)
        print(f"Loaded {source.name}: {len(frame)} matches")

    if not frames:
        raise FileNotFoundError(
            f"No yearly ATP files found in {data_dir} for {start_year}-{end_year}"
        )

    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing atp_matches_YEAR.csv files (default: data)",
    )
    parser.add_argument("--start-year", type=int, default=1968)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: DATA_DIR/atp_matches_START_END_completo.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output = args.output or data_dir / (
        f"atp_matches_{args.start_year}_{args.end_year}_completo.csv"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    matches = combine_csv_files(data_dir, args.start_year, args.end_year)
    matches.to_csv(output, index=False)
    print(f"Saved {len(matches)} matches to {output}")


if __name__ == "__main__":
    main()
