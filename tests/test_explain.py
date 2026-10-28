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


def test_underdog_reason_names_price_model_and_football():
    p = _pick(market="h2h", point=None, price=170, win_chance=0.41,
              fair_market=4.2, sources="ratings", ev_market=-0.03, p_push=0.0)
    why = picks.explain_pick(p, {"Dallas Cowboys": {"ratings": M}})
    assert why.startswith("FanDuel pays +170 on the Eagles, which needs a 37%")
    assert "we make it 41%" in why
    assert "the model has the Eagles winning by 2.8" in why
    assert "their passing game (+1.9)" in why
    assert "sharp" not in why


def test_underdog_line_shopping_reason_gives_the_sharp_price():
    p = _pick(market="h2h", point=None, price=170, win_chance=0.38,
              fair_market=4.2, sources="market", ev_market=0.03, p_push=0.0)
    why = picks.explain_pick(p, {})
    assert "sharp books' moneylines make the Eagles worth about +162" in why


def test_underdogs_keeps_the_best_few_short_dogs_only():
    import pandas as pd
    import selector as sel
    rows = []
    for i, (price, ev, fair) in enumerate([(150, 0.05, 3.0), (180, 0.08, 4.0),
                                           (450, 0.20, 10.0), (130, 0.01, 2.0),
                                           (120, 0.04, 2.5), (110, 0.03, 1.5),
                                           (-120, 0.06, -1.0)]):
        rows.append({"event_id": f"e{i}", "home": "H", "away": f"A{i}",
                     "market": "h2h", "pick": f"A{i}", "price": price,
                     "ev": ev, "fair_market": fair, "kelly": 0.2,
                     "book": "draftkings", "point": None})
    rows.append({**rows[0], "pick": "H", "price": 105, "ev": 0.3})  # favourite
    d = picks.underdogs(pd.DataFrame(rows), 1000, sel.Policy())
    assert list(d.pick) == ["A1", "A0", "A4"]
    assert (d.stake == 10.0).all()
