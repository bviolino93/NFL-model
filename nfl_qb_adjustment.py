"""
How much is a quarterback worth? Measure it, then apply it.

Why this is separate from the model
-----------------------------------
Sunday Edge's rating is a ridge on final margins. It has no idea who is
playing. When a starting quarterback is out, the rating still reflects every
game he played, the market moved on the news within minutes, and the model
did not move at all.

That does not merely make the model wrong — it makes it wrong LOUDLY. A QB
swing is 5-7 points against a 4-point betting threshold, so an injury is one
of the most likely reasons the model ends up far enough off the line to
generate a pick. Injuries are currently manufacturing picks rather than
being adjusted out of them.

What this measures
------------------
For every game since 2009, whether each team's quarterback was its usual
starter, and what the actual margin did relative to the CLOSING LINE. The
closing line is the control: it already contains everything the market knew,
including the injury. So the regression asks a narrow, answerable question:

    actual_margin - closing_line  ~  b * (home_qb_out - away_qb_out)

If b is near zero, the market prices QB absence correctly and there is
nothing to add — which is the expected result, and still worth knowing.
If b is materially non-zero, the market systematically over- or
under-reacts, and b is the correction.

SEPARATELY, and more usefully, it measures the RAW penalty against team
rating rather than against the line. That is the number the app needs: not
"can I beat the market on QB news" but "how far should my own rating move so
it stops disagreeing with the market for a reason it cannot see".

Output is a small JSON the app reads at runtime. Nothing is hand-set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class QbAdjustment:
    """Everything the app needs, plus the evidence for it."""
    penalty_points: float        # rating points to subtract when QB is out
    penalty_se: float
    penalty_t: float
    n_games_qb_out: int
    n_games_total: int
    vs_market_coef: float        # does the market misprice it?
    vs_market_t: float
    seasons: str
    note: str = ""

    def to_json(self, path: str):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path


def _ols(X, y):
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    resid = y - Xd @ beta
    dof = max(n - Xd.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * (float(resid @ resid) / dof), 0.0))
    return beta, se


def usual_starter_by_team_season(starters: pd.DataFrame) -> dict:
    """
    The quarterback who took the most starts for each team that season.

    Deliberately season-level rather than rolling: a rolling definition makes
    a backup into "the usual starter" after three weeks, which is exactly the
    case the adjustment exists to catch.
    """
    if starters.empty:
        return {}
    g = (starters.groupby(["season", "posteam", "passer_player_id"])
                 .size().rename("n").reset_index()
                 .sort_values("n", ascending=False)
                 .drop_duplicates(["season", "posteam"], keep="first"))
    return {(int(r.season), str(r.posteam)): str(r.passer_player_id)
            for r in g.itertuples()}


def build_qb_frame(starters: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per game: who started, was it the usual guy, margin, line."""
    usual = usual_starter_by_team_season(starters)
    if not usual:
        return pd.DataFrame()

    st = starters.set_index(["game_id", "posteam"])["passer_player_id"].to_dict()
    rows = []
    for g in sched.dropna(subset=["home_score", "away_score"]).itertuples():
        season = int(getattr(g, "season", 0))
        home, away = str(g.home_team), str(g.away_team)
        hq = st.get((g.game_id, home))
        aq = st.get((g.game_id, away))
        if hq is None or aq is None:
            continue
        hu = usual.get((season, home))
        au = usual.get((season, away))
        if hu is None or au is None:
            continue
        spread = pd.to_numeric(pd.Series([getattr(g, "spread_line", np.nan)]),
                               errors="coerce").iloc[0]
        rows.append({
            "game_id": g.game_id, "season": season,
            "week": int(getattr(g, "week", 0)),
            "home_qb_out": int(str(hq) != str(hu)),
            "away_qb_out": int(str(aq) != str(au)),
            "margin": float(g.home_score) - float(g.away_score),
            "spread_line": float(spread) if np.isfinite(spread) else np.nan,
        })
    return pd.DataFrame(rows)


def measure(starters: pd.DataFrame, sched: pd.DataFrame) -> QbAdjustment:
    """
    Two regressions.

      1. margin ~ (home_qb_out - away_qb_out), WITHOUT the line. This is the
         raw cost of losing your starter, and it is what the app should
         subtract from its own rating.

      2. (margin - spread_line) ~ the same term. The line already knows, so
         this asks whether the market gets the size right. Near zero means it
         does, and there is no edge here — only error prevention.
    """
    fr = build_qb_frame(starters, sched)
    if fr.empty or len(fr) < 500:
        raise ValueError("not enough graded games with identified starters")

    swing = (fr["home_qb_out"] - fr["away_qb_out"]).to_numpy(dtype=float)
    n_out = int((fr["home_qb_out"] | fr["away_qb_out"]).sum())

    b1, se1 = _ols(swing.reshape(-1, 1), fr["margin"].to_numpy(dtype=float))
    raw = -float(b1[1])          # positive = points LOST when your QB is out
    raw_se = float(se1[1])
    # Note on attenuation: "usual starter" is the season's most-used QB, so a
    # team whose backup played most of the year has its backup counted as the
    # starter. In simulation with a true 6-point effect this recovered 4.7 —
    # the measurement is a floor on the real penalty, not an unbiased estimate.

    have = fr["spread_line"].notna()
    if have.sum() > 200:
        resid = (fr.loc[have, "margin"] - fr.loc[have, "spread_line"]).to_numpy(float)
        b2, se2 = _ols(swing[have.to_numpy()].reshape(-1, 1), resid)
        vs_mkt = -float(b2[1])
        # Negate the t as well. Reporting a positive coefficient beside a
        # negative t is the kind of inconsistency that gets read as a bug in
        # the data rather than in the printout.
        vs_t = float(vs_mkt / se2[1]) if se2[1] > 0 else float("nan")
    else:
        vs_mkt, vs_t = float("nan"), float("nan")

    seasons = f"{int(fr.season.min())}-{int(fr.season.max())}"
    if abs(vs_t) < 2:
        note = ("The market prices QB absence about right, so this is not an "
                "edge. Apply it so the rating stops disagreeing with the line "
                "for a reason it cannot see.")
    else:
        note = ("The market appears to misprice QB absence. Unusual — re-run "
                "on held-out seasons before believing it.")

    return QbAdjustment(
        penalty_points=round(raw, 3),
        penalty_se=round(raw_se, 3),
        penalty_t=round(raw / raw_se, 2) if raw_se > 0 else float("nan"),
        n_games_qb_out=n_out,
        n_games_total=int(len(fr)),
        vs_market_coef=round(vs_mkt, 3) if np.isfinite(vs_mkt) else float("nan"),
        vs_market_t=round(vs_t, 2) if np.isfinite(vs_t) else float("nan"),
        seasons=seasons,
        note=note,
    )


def report(adj: QbAdjustment) -> str:
    return "\n".join([
        f"Seasons:                 {adj.seasons}",
        f"Games:                   {adj.n_games_total:,} "
        f"({adj.n_games_qb_out:,} with a non-usual starter)",
        "",
        f"Cost of losing your QB:  {adj.penalty_points:+.2f} points "
        f"(se {adj.penalty_se:.2f}, t = {adj.penalty_t:+.2f})",
        f"Versus the closing line: {adj.vs_market_coef:+.2f} points "
        f"(t = {adj.vs_market_t:+.2f})",
        "",
        adj.note,
    ])


def run(seasons, out_path: str = "qb_adjustment.json") -> QbAdjustment:
    import nfl_data_py as nfl
    from nfl_epa_ratings import fetch_pbp, prepare_pbp, starters_from_pbp

    sched = nfl.import_schedules(list(seasons))
    pbp = prepare_pbp(fetch_pbp(seasons))
    starters = starters_from_pbp(pbp)
    adj = measure(starters, sched)
    print(report(adj))
    adj.to_json(out_path)
    print(f"\nwrote {out_path}")
    return adj
