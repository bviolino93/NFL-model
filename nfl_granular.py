"""
Unit-level NFL ratings.

The idea
--------
A single team rating says "the Chiefs are 6 points better than the Bills".
That is one number doing several jobs. Football is not played that way: a
strong passing offense against a weak passing defense is a different game
from the same two teams if the edge were on the ground, and a team rating
averages those into mush.

This fits FOUR units per team — pass offense, rush offense, pass defense,
rush defense — plus a quarterback term, and then projects a specific matchup
by pairing each offense against the defense it actually faces.

    team A expected EPA/play
        = pass_rate * (A.pass_off - B.pass_def + A.qb)
        + rush_rate * (A.rush_off - B.rush_def)

The pairing is the point. It is strictly more information than the team
model, drawn from exactly the same plays.

What is deliberately NOT here
-----------------------------
Third down and red zone splits. Both are famously unstable year to year —
they look like skill and behave like noise — and adding them would mean more
parameters fitted on fewer plays. If they belong, the evaluation should say
so; they are exposed as a config flag rather than baked in.

Every knob is on GranularConfig so the evaluation can sweep it. Nothing here
is hand-set on the grounds that it sounds right. That is the specific
mistake this whole exercise exists to avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

PLAYS_PER_GAME = 63.0
WP_LOW, WP_HIGH = 0.05, 0.95
PASS_TYPES = {"pass"}
RUSH_TYPES = {"run"}


@dataclass
class GranularConfig:
    """Everything tunable, in one place, so the evaluation can sweep it."""
    # Swept over five simulated seasons, scoring RMSE of projected margin
    # against true expected margin on held-out weeks:
    #     team-level model        4.076   <- what Sunday Edge has now
    #     granular alpha=300      3.734   <- best
    #     granular alpha=800      3.752
    #     granular alpha=2000     4.229
    #     granular early-downs    4.329   <- worse, so it stays off
    alpha_pass: float = 300.0
    alpha_rush: float = 300.0
    alpha_qb: float = 300.0
    min_qb_plays: int = 100
    drop_garbage: bool = True
    # Early-downs-only is a common filter and it HURT here: the plays it
    # removes still carry signal, and throwing away a third of the sample
    # costs more than the noise it avoids. Left available, defaulted off.
    early_downs_only: bool = False
    neutral_script_only: bool = False  # exclude obvious pass/run situations
    # League play mix used when projecting. Team-specific pass rate is
    # endogenous — teams that are winning run more — so using it would leak
    # the outcome back into the projection.
    pass_rate: float = 0.58
    # NOTE: removed. Shrinking each unit toward the team's pass-rate-weighted
    # mean, then projecting with that same pass rate, is algebraically a
    # no-op — the weighted combination is unchanged, so every projection came
    # out identical at any shrink level. Caught by two configs producing
    # byte-identical RMSE. If unit-level overfitting turns out to be a real
    # problem, the fix is a higher alpha on the unit fits, not this.


@dataclass
class GranularRatings:
    pass_off: pd.Series
    rush_off: pd.Series
    pass_def: pd.Series     # higher = better defense
    rush_def: pd.Series
    qb: pd.Series
    home_field: float
    cfg: GranularConfig
    n_pass: int = 0
    n_rush: int = 0
    teams: list = field(default_factory=list)

    # -- helpers ------------------------------------------------------
    def _g(self, s, k, default=0.0):
        try:
            v = float(s.get(k, default))
            return v if math.isfinite(v) else default
        except Exception:
            return default

    def team_epa(self, off: str, deff: str, qb=None) -> float:
        """Expected EPA per play for `off` against `deff`, in EPA units."""
        pr = self.cfg.pass_rate
        p = (self._g(self.pass_off, off) - self._g(self.pass_def, deff)
             + self._g(self.qb, qb))
        r = self._g(self.rush_off, off) - self._g(self.rush_def, deff)
        return pr * p + (1.0 - pr) * r

    def projected_margin(self, home: str, away: str, neutral: bool = False,
                         home_qb=None, away_qb=None) -> float:
        edge = (self.team_epa(home, away, home_qb)
                - self.team_epa(away, home, away_qb)) * PLAYS_PER_GAME
        return edge + (0.0 if neutral else self.home_field)

    def team_summary(self) -> pd.DataFrame:
        """One row per team, in points per game, for display."""
        idx = sorted(set(self.pass_off.index) | set(self.rush_off.index))
        pr = self.cfg.pass_rate
        rows = []
        for t in idx:
            po = self._g(self.pass_off, t) * PLAYS_PER_GAME
            ro = self._g(self.rush_off, t) * PLAYS_PER_GAME
            pd_ = self._g(self.pass_def, t) * PLAYS_PER_GAME
            rd = self._g(self.rush_def, t) * PLAYS_PER_GAME
            rows.append({
                "team": t,
                "pass_off": po, "rush_off": ro,
                "pass_def": pd_, "rush_def": rd,
                "offense": pr * po + (1 - pr) * ro,
                "defense": pr * pd_ + (1 - pr) * rd,
            })
        out = pd.DataFrame(rows)
        out["overall"] = out["offense"] + out["defense"]
        return out.sort_values("overall", ascending=False).reset_index(drop=True)


def _filter(df: pd.DataFrame, cfg: GranularConfig) -> pd.DataFrame:
    x = df
    if cfg.drop_garbage and "wp" in x.columns:
        wp = pd.to_numeric(x["wp"], errors="coerce")
        x = x[wp.between(WP_LOW, WP_HIGH) | wp.isna()]
    if cfg.early_downs_only and "down" in x.columns:
        d = pd.to_numeric(x["down"], errors="coerce")
        x = x[d.isin([1, 2])]
    if cfg.neutral_script_only and {"down", "ydstogo"} <= set(x.columns):
        d = pd.to_numeric(x["down"], errors="coerce")
        y = pd.to_numeric(x["ydstogo"], errors="coerce")
        # Drop 3rd-and-long (obvious pass) and 3rd-and-1 (obvious run):
        # situations where play choice is dictated, not chosen.
        x = x[~((d == 3) & ((y >= 7) | (y <= 1)))]
    return x


def _fit_unit(df: pd.DataFrame, teams, alpha: float,
              qb_col: str | None = None, min_qb_plays: int = 100,
              alpha_qb: float | None = None):
    """
    Ridge of EPA on offense identity, defense identity, optional QB, and
    home field. Returns (off, def, qb, hfa) as Series in EPA-per-play.

    Home field is scaled up 100x before fitting so the shared penalty barely
    touches it — it is one parameter identified by every play, unlike a team
    effect which appears in a slice. The QB block is scaled so its effective
    penalty is alpha_qb rather than alpha.
    """
    if df.empty:
        z = pd.Series(dtype=float)
        return z, z, z, 0.0, 0
    t_idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    n = len(df)
    ar = np.arange(n)
    rows, cols, vals = [ar, ar], [
        df["posteam"].map(t_idx).to_numpy(),
        df["defteam"].map(t_idx).to_numpy() + k,
    ], [np.ones(n), np.ones(n)]
    width = 2 * k

    qbs, qb_scale = [], 1.0
    if qb_col and qb_col in df.columns:
        raw = df[qb_col].astype(str)
        counts = raw.value_counts()
        qbs = sorted(q for q, c in counts.items()
                     if c >= min_qb_plays and q not in ("nan", "None", ""))
        if qbs:
            q_idx = {q: i for i, q in enumerate(qbs)}
            qi = raw.map(q_idx).to_numpy(dtype=float)
            m = ~np.isnan(qi)
            qb_scale = math.sqrt(alpha / float(alpha_qb or alpha))
            rows.append(ar[m])
            cols.append(qi[m].astype(np.int64) + width)
            vals.append(np.full(int(m.sum()), qb_scale))
            width += len(qbs)

    # Home field is NOT estimated here, at all.
    #
    # Measured on simulated data with a true effect of 0.022 EPA/play
    # (1.39 points/game): a joint ridge fit recovered 0.010 and stayed there
    # as alpha went to zero, and a direct home-minus-away difference gave
    # -0.007 on pass plays and +0.020 on rush. Those look like different
    # bugs. They are the same thing — noise. The per-team home-minus-away
    # difference has a standard deviation of 0.095 EPA/play across 32 teams,
    # a standard error near 0.017, which is more than half the effect being
    # measured.
    #
    # Play-level EPA is simply a poor instrument for home field: thousands of
    # noisy plays estimating one small constant. Final margins are a better
    # one — fewer observations, but each is a direct measurement in points.
    # So home field is passed in from the schedule (see hfa_points in
    # fit_granular), exactly as Sunday Edge's margin model already computes
    # it. Team and QB effects, which play-level data estimates far better
    # than margins do, stay here.
    y = df["epa"].to_numpy(dtype=np.float64)
    hfa = 0.0

    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, width))
    mdl = Ridge(alpha=float(alpha), fit_intercept=True,
                solver="sparse_cg").fit(X, y)
    c = mdl.coef_

    off = pd.Series(c[:k], index=teams)
    dfn = pd.Series(c[k:2 * k], index=teams)
    qb = (pd.Series(c[2 * k:2 * k + len(qbs)], index=qbs) * qb_scale
          if qbs else pd.Series(dtype=float))

    off = off - off.mean()
    dfn = -(dfn - dfn.mean())          # higher = better defense
    if len(qb):
        qb = qb - qb.mean()
    return off, dfn, qb, hfa, n


def home_field_from_schedule(sched: pd.DataFrame,
                             default: float = 1.8) -> float:
    """
    League home-field advantage in points, from completed games.

    A paired within-game difference, so team quality cancels over a schedule
    where everyone hosts about half their games. This is the estimator
    Sunday Edge's margin model already uses, and it is much better than
    anything play-level EPA can give for this particular constant.

    `default` is the fallback when there are too few games — modern NFL home
    field has been drifting down and sits near 1.5-2 points, well below the
    3 points folklore still quotes.
    """
    if sched is None or sched.empty:
        return float(default)
    g = sched.dropna(subset=["home_score", "away_score"])
    if len(g) < 100:
        return float(default)
    margin = pd.to_numeric(g["home_score"], errors="coerce") - \
        pd.to_numeric(g["away_score"], errors="coerce")
    m = float(margin.mean())
    return m if math.isfinite(m) else float(default)


def fit_granular(pbp: pd.DataFrame,
                 cfg: GranularConfig | None = None,
                 hfa_points: float | None = None) -> GranularRatings:
    """
    Fit pass and rush units separately, then optionally shrink them toward
    the team average.

    `hfa_points` comes from home_field_from_schedule(). Omit it and the
    model projects neutral-field margins, which is honest — better than
    inventing a home edge from an instrument that cannot measure it.
    """
    cfg = cfg or GranularConfig()
    if pbp is None or pbp.empty:
        raise ValueError("no plays to fit")
    df = _filter(pbp, cfg)
    if df.empty:
        raise ValueError("every play was filtered out")

    teams = sorted(set(df["posteam"]) | set(df["defteam"]))
    is_pass = df["play_type"].astype(str).isin(PASS_TYPES)
    dp, dr = df[is_pass], df[df["play_type"].astype(str).isin(RUSH_TYPES)]

    po, pdf, qb, hfa_p, n_p = _fit_unit(
        dp, teams, cfg.alpha_pass, qb_col="passer_player_id",
        min_qb_plays=cfg.min_qb_plays, alpha_qb=cfg.alpha_qb)
    ro, rdf, _, hfa_r, n_r = _fit_unit(dr, teams, cfg.alpha_rush)

    # Home field comes from the schedule, in points, or is left at zero.
    hfa_pts = float(hfa_points) if hfa_points is not None else 0.0

    return GranularRatings(
        pass_off=po, rush_off=ro, pass_def=pdf, rush_def=rdf,
        qb=qb, home_field=hfa_pts, cfg=cfg,
        n_pass=n_p, n_rush=n_r, teams=teams,
    )
