from types import SimpleNamespace as N

import picks


def _pick(**kw):
    base = dict(market="spreads", sources="market,ratings",
                home="Dallas Cowboys", away="Philadelphia Eagles",
                pick="Philadelphia Eagles", point=4.5, fair_market=4.2,
                book="fanduel", notes="")
    return N(**{**base, **kw})


M = {"ratings": -2.0, "injuries": -0.8, "margin": -2.8, "hfa": 1.6,
     "pass_off": -1.9, "run_off": 0.3, "pass_def": -0.7, "run_def": 0.2,
     "qb": 1.5, "home_qb": "Dak Prescott", "away_qb": "Michael Penix Jr."}


def test_spread_reason_is_the_models_case_not_the_sharp_line():
    why = picks.explain_pick(_pick(), {"Dallas Cowboys": {"ratings": M}})
    assert "sharp" not in why
    assert why.startswith("The model has the Eagles winning by 2.8")
    assert "lose by 5 or more" in why
    assert "their passing game (+1.9)" in why
    assert "Prescott over Penix at quarterback (1.5)" in why


def test_line_shopping_bet_still_explains_the_gap():
    why = picks.explain_pick(_pick(sources="market"), {})
    assert "hasn't caught up with the sharp books" in why


def test_injuries_for_lists_both_teams_and_skips_ir_on_totals():
    inj = {"home": [{"name": "A", "pos": "WR", "status": "Out", "pts": -0.6},
                    {"name": "B", "pos": "CB", "status": "IR", "pts": -0.4},
                    {"name": "Q", "pos": "QB", "status": "IR", "pts": None,
                     "note": "McKee starting"}],
           "away": [{"name": "C", "pos": "LB", "status": "Doubtful", "pts": -0.5}]}
    ctx = {"Philadelphia Eagles": {"ratings": {"injured": inj}}}
    sp = N(home="Philadelphia Eagles", away="Dallas Cowboys", market="spreads")
    tot = N(home="Philadelphia Eagles", away="Dallas Cowboys", market="totals")
    assert [(x["team"], x["name"]) for x in picks.injuries_for(sp, ctx)] == [
        ("Cowboys", "C"), ("Eagles", "A"), ("Eagles", "B"), ("Eagles", "Q")]
    assert [x["name"] for x in picks.injuries_for(tot, ctx)] == ["C", "A", "Q"]
    assert picks.injuries_for(sp, {}) == []
