"""Train and evaluate a Random Forest tennis match classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

BASE_FEATURES = [
    "p1_rank",
    "p2_rank",
    "p1_age",
    "p2_age",
    "p1_hand",
    "p2_hand",
    "p1_ace",
    "p2_ace",
    "elo_p1",
    "elo_p2",
    "h2h_p1_vs_p2",
    "h2h_p2_vs_p1",
    "p1_recent_wins",
    "p2_recent_wins",
    "p1_surface_wr",
    "p2_surface_wr",
]
HAND_CODES = {"U": 0, "R": 1, "L": 2}


def prepare_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and convert a generated model dataset into numeric features."""
    required = set(BASE_FEATURES) | {"target", "match_id"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    tournament_features = sorted(
        column for column in dataset.columns if column.startswith("tourney_")
    )
    features = dataset[BASE_FEATURES + tournament_features].copy()
    for hand_column in ("p1_hand", "p2_hand"):
        features[hand_column] = (
            features[hand_column]
            .fillna("U")
            .astype(str)
            .str.upper()
            .map(HAND_CODES)
            .fillna(HAND_CODES["U"])
        )

    features = features.apply(pd.to_numeric, errors="coerce")
    invalid = features.columns[features.isna().any()].tolist()
    if invalid:
        raise ValueError(f"Features contain non-numeric or missing values: {invalid}")

    target = pd.to_numeric(dataset["target"], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("target must contain only binary values 0 and 1")
    if target.nunique() != 2:
        raise ValueError("target must contain examples from both classes")
    return features, target.astype(int)


def train_and_evaluate(
    dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[RandomForestClassifier, float, set[int], set[int]]:
    """Train with match-group isolation and return model, accuracy, and split groups."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    features, target = prepare_features(dataset)
    groups = pd.to_numeric(dataset["match_id"], errors="raise").astype(int)
    if groups.nunique() < 2:
        raise ValueError("At least two distinct match_id groups are required")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_index, test_index = next(splitter.split(features, target, groups))
    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features.iloc[train_index], target.iloc[train_index])
    predictions = model.predict(features.iloc[test_index])
    accuracy = accuracy_score(target.iloc[test_index], predictions)
    return model, accuracy, train_groups, test_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("tennis_model_dataset.csv"),
        help="Feature CSV produced by limpieza_datos.py",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.dataset.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source}")

    dataset = pd.read_csv(source)
    _, accuracy, train_groups, test_groups = train_and_evaluate(
        dataset, test_size=args.test_size
    )
    assert train_groups.isdisjoint(test_groups)
    print(f"Random Forest accuracy: {accuracy:.4f}")
    print(f"Training matches: {len(train_groups)}; test matches: {len(test_groups)}")


if __name__ == "__main__":
    main()
