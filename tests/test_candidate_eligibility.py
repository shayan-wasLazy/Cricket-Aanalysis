import pandas as pd

from cricket_pressure.analysis import _pakistan_player_ids


def test_pakistan_player_ids_uses_national_team_mapping_only():
    tables = {
        "teams": pd.DataFrame({
            "team_id": [1, 2, 3],
            "team_name": ["Pakistan", "India", "Mumbai Indians"],
        }),
        "player_mapping": pd.DataFrame({
            "team_id": [1, 1, 2, 3],
            "player_id": [101, 102, 201, 301],
        }),
    }

    assert _pakistan_player_ids(tables) == {101, 102}


def test_pakistan_player_ids_fails_closed_if_team_metadata_missing():
    tables = {
        "teams": pd.DataFrame({"team_id": [2], "team_name": ["India"]}),
        "player_mapping": pd.DataFrame({"team_id": [2], "player_id": [201]}),
    }

    try:
        _pakistan_player_ids(tables)
    except ValueError as exc:
        assert "no Pakistan national-team record" in str(exc)
    else:
        raise AssertionError("Expected missing Pakistan metadata to stop ranking generation")
