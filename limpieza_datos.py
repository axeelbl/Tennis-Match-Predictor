"""Build leakage-aware model features from chronological ATP match data."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "tourney_date",
    "surface",
    "round",
    "best_of",
    "tourney_level",
    "winner_name",
    "winner_hand",
    "winner_age",
    "winner_rank",
    "loser_name",
    "loser_hand",
    "loser_age",
    "loser_rank",
    "w_ace",
    "w_df",
    "w_1stWon",
    "w_2ndWon",
    "l_ace",
    "l_df",
    "l_1stWon",
    "l_2ndWon",
]
NUMERIC_COLUMNS = [
    "winner_rank",
    "loser_rank",
    "winner_age",
    "loser_age",
    "w_ace",
    "w_df",
    "w_1stWon",
    "w_2ndWon",
    "l_ace",
    "l_df",
    "l_1stWon",
    "l_2ndWon",
]


def calculate_elo(
    elo_winner: float, elo_loser: float, k_factor: int = 32
) -> tuple[float, float]:
    """Return post-match Elo values for a winner and loser."""
    expected_win = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
    adjustment = k_factor * (1 - expected_win)
    return elo_winner + adjustment, elo_loser - adjustment


def _validate_columns(matches: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(matches.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def build_model_dataset(matches: pd.DataFrame, recent_window: int = 5) -> pd.DataFrame:
    """Create two labelled perspectives per match using only pre-match statistics."""
    if recent_window < 1:
        raise ValueError("recent_window must be at least 1")
    _validate_columns(matches)

    data = matches[REQUIRED_COLUMNS].copy()
    data = data.dropna(subset=["winner_name", "loser_name", "surface", "round"])
    if data.empty:
        raise ValueError("No usable matches remain after removing incomplete rows")

    data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    data["winner_hand"] = data["winner_hand"].fillna("U")
    data["loser_hand"] = data["loser_hand"].fillna("U")
    data["tourney_level"] = data["tourney_level"].fillna("Unknown")
    data = data.sort_values("tourney_date", kind="stable").reset_index(drop=True)
    data["match_id"] = data.index

    elo: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    elo_winner: list[float] = []
    elo_loser: list[float] = []
    for row in data.itertuples(index=False):
        winner, loser = row.winner_name, row.loser_name
        before_winner, before_loser = elo[winner], elo[loser]
        elo_winner.append(before_winner)
        elo_loser.append(before_loser)
        elo[winner], elo[loser] = calculate_elo(before_winner, before_loser)
    data["elo_winner"] = elo_winner
    data["elo_loser"] = elo_loser

    h2h: defaultdict[tuple[str, str], int] = defaultdict(int)
    h2h_winner_vs_loser: list[int] = []
    h2h_loser_vs_winner: list[int] = []
    for row in data.itertuples(index=False):
        winner, loser = row.winner_name, row.loser_name
        h2h_winner_vs_loser.append(h2h[(winner, loser)])
        h2h_loser_vs_winner.append(h2h[(loser, winner)])
        h2h[(winner, loser)] += 1
    data["h2h_winner_vs_loser"] = h2h_winner_vs_loser
    data["h2h_loser_vs_winner"] = h2h_loser_vs_winner

    recent: defaultdict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=recent_window)
    )
    recent_winner_wins: list[int] = []
    recent_loser_wins: list[int] = []
    for row in data.itertuples(index=False):
        winner, loser = row.winner_name, row.loser_name
        recent_winner_wins.append(sum(recent[winner]))
        recent_loser_wins.append(sum(recent[loser]))
        recent[winner].append(1)
        recent[loser].append(0)
    data["recent_winner_wins"] = recent_winner_wins
    data["recent_loser_wins"] = recent_loser_wins

    ace_stats: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"aces": 0.0, "matches": 0.0}
    )
    ace_winner: list[float] = []
    ace_loser: list[float] = []

    def average_aces(player: str) -> float:
        stats = ace_stats[player]
        return stats["aces"] / stats["matches"] if stats["matches"] else 0.0

    for row in data.itertuples(index=False):
        winner, loser = row.winner_name, row.loser_name
        ace_winner.append(average_aces(winner))
        ace_loser.append(average_aces(loser))
        ace_stats[winner]["aces"] += row.w_ace
        ace_stats[winner]["matches"] += 1
        ace_stats[loser]["aces"] += row.l_ace
        ace_stats[loser]["matches"] += 1
    data["ace_winner"] = ace_winner
    data["ace_loser"] = ace_loser

    surface_stats: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "total": 0}
    )
    surface_winner_wr: list[float] = []
    surface_loser_wr: list[float] = []

    def win_rate(player: str, surface: str) -> float:
        stats = surface_stats[(player, surface)]
        return stats["wins"] / stats["total"] if stats["total"] else 0.5

    for row in data.itertuples(index=False):
        surface, winner, loser = row.surface, row.winner_name, row.loser_name
        surface_winner_wr.append(win_rate(winner, surface))
        surface_loser_wr.append(win_rate(loser, surface))
        surface_stats[(winner, surface)]["wins"] += 1
        surface_stats[(winner, surface)]["total"] += 1
        surface_stats[(loser, surface)]["total"] += 1
    data["surface_winner_wr"] = surface_winner_wr
    data["surface_loser_wr"] = surface_loser_wr

    tournament_dummies = pd.get_dummies(
        data["tourney_level"], prefix="tourney", dtype=int
    )
    data = pd.concat([data, tournament_dummies], axis=1)
    tournament_columns = tournament_dummies.columns.tolist()

    def create_row(p1: str, p2: str, match: pd.Series, target: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "match_id": match["match_id"],
            "p1_name": match[f"{p1}_name"],
            "p2_name": match[f"{p2}_name"],
            "p1_rank": match[f"{p1}_rank"],
            "p2_rank": match[f"{p2}_rank"],
            "p1_age": match[f"{p1}_age"],
            "p2_age": match[f"{p2}_age"],
            "p1_hand": match[f"{p1}_hand"],
            "p2_hand": match[f"{p2}_hand"],
            "p1_ace": match[f"ace_{p1}"],
            "p2_ace": match[f"ace_{p2}"],
            "elo_p1": match[f"elo_{p1}"],
            "elo_p2": match[f"elo_{p2}"],
            "h2h_p1_vs_p2": match[f"h2h_{p1}_vs_{p2}"],
            "h2h_p2_vs_p1": match[f"h2h_{p2}_vs_{p1}"],
            "p1_recent_wins": match[f"recent_{p1}_wins"],
            "p2_recent_wins": match[f"recent_{p2}_wins"],
            "p1_surface_wr": match[f"surface_{p1}_wr"],
            "p2_surface_wr": match[f"surface_{p2}_wr"],
            "target": target,
        }
        result.update({column: match[column] for column in tournament_columns})
        return result

    rows: list[dict[str, Any]] = []
    for _, match in data.iterrows():
        rows.append(create_row("winner", "loser", match, 1))
        rows.append(create_row("loser", "winner", match, 0))
    return pd.DataFrame(rows).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Combined ATP match CSV produced by juntar_csv.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tennis_model_dataset.csv"),
        help="Feature dataset destination (default: tennis_model_dataset.csv)",
    )
    parser.add_argument("--recent-window", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model_data = build_model_dataset(pd.read_csv(source), args.recent_window)
    model_data.to_csv(output, index=False)
    print(f"Saved {len(model_data)} model rows to {output}")


if __name__ == "__main__":
    main()
