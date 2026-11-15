import json
import zipfile

import numpy as np
import pandas as pd

from cricket_pressure.rr_intelligence import build_rr_intelligence, parse_cricsheet_match


def _delivery(batter, bowler, runs=0, wicket=False, extras=None):
    item = {
        "batter": batter, "bowler": bowler, "non_striker": "Partner",
        "runs": {"batter": runs, "extras": sum((extras or {}).values()),
                 "total": runs + sum((extras or {}).values())},
    }
    if extras:
        item["extras"] = extras
    if wicket:
        item["wickets"] = [{"kind": "bowled", "player_out": batter}]
    return item


def _innings(team, batter, bowler, runs_each=1, wickets_first=0, make_wide=False):
    overs = []
    for over in range(20):
        balls = []
        for ball in range(6):
            index = over * 6 + ball
            if make_wide and index == 0:
                item = _delivery(batter, bowler, 0, extras={"wides": 1})
            else:
                item = _delivery(batter, bowler, runs_each, wicket=index < wickets_first)
            balls.append(item)
        overs.append({"over": over, "deliveries": balls})
    return {"team": team, "overs": overs}


def _match(rr_match=True):
    if rr_match:
        teams = ["Chennai Super Kings", "Rajasthan Royals"]
        bat1, bowl1, bat2, bowl2 = "CSK Batter", "RR Bowler", "RR Batter", "CSK Bowler"
        first_runs, second_runs, wickets_first = 2, 0, 6
        filename = "rr_match.json"
    else:
        teams = ["Mumbai Indians", "Kolkata Knight Riders"]
        bat1, bowl1, bat2, bowl2 = "MI Batter", "KKR Bowler", "KKR Batter", "MI Bowler"
        first_runs, second_runs, wickets_first = 2, 1, 1
        filename = "other_match.json"
    payload = {
        "info": {
            "dates": ["2022-04-01"], "season": "2022", "overs": 20,
            "event": {"name": "Indian Premier League"}, "teams": teams,
            "venue": "Test Ground", "outcome": {"winner": teams[0]},
            "players": {teams[0]: [bat1, bowl2], teams[1]: [bat2, bowl1]},
            "registry": {"people": {
                bat1: "batter-id-1", bowl1: "bowler-id-1",
                bat2: "batter-id-2", bowl2: "bowler-id-2",
            }},
        },
        "innings": [
            _innings(teams[0], bat1, bowl1, runs_each=first_runs, make_wide=True),
            _innings(teams[1], bat2, bowl2, runs_each=second_runs, wickets_first=wickets_first),
        ],
    }
    return filename, payload


def test_parser_reconstructs_ball_state_and_wide_accounting():
    filename, payload = _match()
    frame = parse_cricsheet_match(payload, filename)
    first_ball = frame.iloc[0]
    assert first_ball["runs_total"] == 1
    assert not bool(first_ball["is_legal_ball"])
    assert not bool(first_ball["batter_ball_faced"])
    chase = frame.loc[frame["innings_number"].eq(2)]
    assert chase["target"].iloc[0] == 240
    assert chase["pressure_index"].notna().any()
    assert chase["is_bowling_high_pressure"].notna().all()


class DummyPressureXGB:
    def predict_proba(self, features):
        return np.tile(np.array([[0.05, 0.10, 0.25, 0.60]]), (len(features), 1))


def test_build_outputs_separate_rr_diagnostics_from_all_ipl_candidate_pool(tmp_path):
    archive = tmp_path / "ipl_json.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for rr_match in [True, False]:
            name, payload = _match(rr_match)
            zf.writestr(name, json.dumps(payload))
    paths = build_rr_intelligence(archive, tmp_path / "out", 2022, 2026, xgb_model=DummyPressureXGB())

    matches = pd.read_csv(paths["rr_matches"])
    all_deliveries = pd.read_csv(paths["all_ipl_deliveries"])
    rr_deliveries = pd.read_csv(paths["rr_deliveries"])
    pressure = pd.read_csv(paths["rr_pressure_points"])
    fit = pd.read_csv(paths["player_fit_shortlist"])

    assert len(matches) == 1
    assert all_deliveries["match_id"].nunique() == 2
    assert rr_deliveries["match_id"].nunique() == 1
    assert not pressure.empty
    assert pressure["pressure_index"].min() >= 60
    assert {"Batter", "Bowler"}.issubset(set(fit["role"]))
    assert paths["report"].exists()
    frequent_xi = pd.read_csv(paths["frequent_xi"])
    phase_rankings = pd.read_csv(paths["phase_player_rankings"])
    phase_options = pd.read_csv(paths["frequent_xi_phase_options"])
    assert frequent_xi["player_name"].tolist() == ["RR Batter", "RR Bowler"] or set(frequent_xi["player_name"]) == {"RR Batter", "RR Bowler"}
    assert {"pi_performance_score", "xgb_performance_score", "decision_score"}.issubset(phase_rankings.columns)
    assert len(phase_options) == len(frequent_xi) * 6
