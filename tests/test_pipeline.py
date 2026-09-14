from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from juntar_csv import combine_csv_files
from limpieza_datos import REQUIRED_COLUMNS, build_model_dataset, calculate_elo
from Modelo_Random_Forest import prepare_features, train_and_evaluate


def match_frame(count: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(count):
        winner = "Player A" if index % 2 == 0 else "Player B"
        loser = "Player B" if index % 2 == 0 else "Player A"
        rows.append(
            {
                "tourney_date": 20240101 + index,
                "surface": "Hard",
                "round": "R32",
                "best_of": 3,
                "tourney_level": "A",
                "winner_name": winner,
                "winner_hand": "R",
                "winner_age": 25,
                "winner_rank": 10 + index,
                "loser_name": loser,
                "loser_hand": "L",
                "loser_age": 26,
                "loser_rank": 20 + index,
                "w_ace": 8 + index,
                "w_df": 2,
                "w_1stWon": 30,
                "w_2ndWon": 15,
                "l_ace": 4 + index,
                "l_df": 3,
                "l_1stWon": 24,
                "l_2ndWon": 10,
            }
        )
    return pd.DataFrame(rows)


def test_combine_csv_files_skips_missing_years(tmp_path: Path) -> None:
    pd.DataFrame({"winner_name": ["A"]}).to_csv(
        tmp_path / "atp_matches_2023.csv", index=False
    )
    pd.DataFrame({"winner_name": ["B", "C"]}).to_csv(
        tmp_path / "atp_matches_2025.csv", index=False
    )

    combined = combine_csv_files(tmp_path, 2023, 2025)

    assert combined["year"].tolist() == [2023, 2025, 2025]


def test_combine_csv_files_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No yearly ATP files"):
        combine_csv_files(tmp_path, 2020, 2021)


def test_build_model_dataset_uses_only_pre_match_statistics() -> None:
    dataset = build_model_dataset(match_frame(2))

    assert len(dataset) == 4
    assert dataset.groupby("match_id")["target"].agg(set).tolist() == [{0, 1}, {0, 1}]
    assert dataset.loc[dataset["match_id"] == 0, "p1_ace"].eq(0).all()
    second_match = dataset[dataset["match_id"] == 1]
    assert sorted(second_match["p1_ace"].tolist()) == [4.0, 8.0]
    assert "tourney_A" in dataset.columns


def test_grouped_training_keeps_match_perspectives_together() -> None:
    dataset = build_model_dataset(match_frame())

    _, accuracy, train_groups, test_groups = train_and_evaluate(
        dataset, test_size=0.25
    )

    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == set(dataset["match_id"])
    assert 0 <= accuracy <= 1


def test_prepare_features_rejects_missing_or_invalid_data() -> None:
    dataset = build_model_dataset(match_frame())
    dataset["p1_rank"] = dataset["p1_rank"].astype(object)
    dataset.loc[0, "p1_rank"] = "not-a-number"

    with pytest.raises(ValueError, match="non-numeric or missing"):
        prepare_features(dataset)


def test_required_column_validation_is_explicit() -> None:
    incomplete = match_frame().drop(columns=[REQUIRED_COLUMNS[0]])

    with pytest.raises(ValueError, match=REQUIRED_COLUMNS[0]):
        build_model_dataset(incomplete)


def test_elo_update_is_zero_sum() -> None:
    winner, loser = calculate_elo(1500, 1500)

    assert winner == 1516
    assert loser == 1484
