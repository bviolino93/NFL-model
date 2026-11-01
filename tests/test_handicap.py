from types import SimpleNamespace as N

import pandas as pd

import nfl_handicap as H


class Fake:
    def __init__(self, decisions):
        self.beta = self.messages = self
        self.decisions = list(decisions)
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        d = self.decisions.pop(0)
        txt = '```json\n{"decision": "%s", "reason": "r%d"}\n```' % (d, self.calls)
        return N(stop_reason="end_turn", content=[N(type="text", text=txt)],
                 usage=N(input_tokens=1000, output_tokens=200))


def _picks(**kw):
    row = dict(home="Washington Commanders", away="Philadelphia Eagles",
               commence="2026-11-02T01:20:00Z", market="spreads",
               pick="Philadelphia Eagles", point=6.5, price=-112, book="DraftKings",
               stake=5.34, win_chance=0.55, ev=0.04, fair_market=6.75, why="model case",
               event_id="e1")
    row.update(kw)
    return pd.DataFrame([row])


def test_bet_needs_the_matchup_case_and_reviews_are_cached():
    st = {}
    out = H.review(_picks(), {}, st, client=Fake(["pass"]), now="2026-11-01T15:00Z")
    assert out.empty
    # same brief again: cached, no new call, same answer
    fake = Fake([])
    assert H.review(_picks(), {}, st, client=fake, now="2026-11-01T15:00Z").empty
    assert fake.calls == 0 and st["_budget"]["usd"] > 0
    kept = H.review(_picks(point=7.0), {}, st, client=Fake(["bet"]), now="2026-11-01T15:00Z")
    assert len(kept) == 1 and kept.why[0].endswith("Handicapper: r1")


def test_lean_can_take_the_other_side():
    p = _picks(opp_pick="Washington Commanders", opp_point=-6.5, opp_price=-108,
               opp_book="FanDuel", opp_ev=0.01, opp_win_chance=0.5)
    out = H.review(p, {}, {}, client=Fake(["other"]), leans=True, now="2026-11-01T15:00Z")
    assert out.pick[0] == "Washington Commanders" and out.point[0] == -6.5
    assert out.why[0].startswith("Handicapper's pick: r1 (Against the model")
    assert H.review(p, {}, {}, client=Fake(["pass"]), leans=True,
                    now="2026-11-01T15:00Z").empty


def test_no_key_or_budget_means_bets_unreviewed_and_no_leans(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert len(H.review(_picks(), {}, {}, now="2026-11-01T15:00Z")) == 1
    assert H.review(_picks(), {}, {}, leans=True, now="2026-11-01T15:00Z").empty
    st = {"_budget": {"month": "2026-11", "usd": H.MONTHLY_BUDGET}}
    assert len(H.review(_picks(), {}, st, client=Fake([]), now="2026-11-01T15:00Z")) == 1


def test_form_reads_closing_lines():
    g = pd.DataFrame([dict(season=2026, week=1, gameday="2026-09-14", home_team="WAS",
                           away_team="PHI", home_score=20, away_score=24,
                           spread_line=3.0, total_line=40.5)])
    t = H.form_text(g, "PHI", "Eagles", "2026-11-02T01:20:00Z")
    assert "1-0, scoring 24.0" in t and "covered by 7" in t and "over by 3.5" in t
