import pandas as pd

from cricket_pressure.analysis import _excluded_national_team_player_ids


def test_excluded_national_team_ids_uses_pakistan_and_bangladesh_mapping_only():
    tables = {
        "teams": pd.DataFrame({
            "team_id": [1, 2, 3, 4],
            "team_name": ["Pakistan", "Bangladesh", "India", "Mumbai Indians"],
        }),
        "player_mapping": pd.DataFrame({
            "team_id": [1, 1, 2, 2, 3, 4],
            "player_id": [101, 102, 103, 104, 201, 301],
        }),
    }

    assert _pakistan_player_ids(tables) == {101, 102, 103, 104}


def test_excluded_national_team_ids_fails_closed_if_metadata_missing():
    tables = {
        "teams": pd.DataFrame({"team_id": [2], "team_name": ["India"]}),
        "player_mapping": pd.DataFrame({"team_id": [2], "player_id": [201]}),
    }

    try:
        _pakistan_player_ids(tables)
    except ValueError as exc:
        assert "national-team records" in str(exc)
    else:
        raise AssertionError("Expected missing Pakistan metadata to stop ranking generation")
