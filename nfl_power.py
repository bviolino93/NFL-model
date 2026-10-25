"""
Recency-weighted, opponent-adjusted EPA ratings with a quarterback term.

What is different from nfl_epa_ratings / nfl_granular
-----------------------------------------------------
Those fit every prior play with equal weight, so in week 10 of 2024 a 2016
snap counted as much as last Sunday's. Team quality drifts within a season
and turns over between seasons; a model that cannot forget is mostly
measuring rosters that no longer exist. Here every team-game is weighted

    plays * 0.5 ** (age_in_weeks / half_life)

where the offseason counts as `offseason_weeks` of age. Nothing else
regresses ratings toward average between seasons: as last season's games
decay, the ridge penalty pulls harder, which is the Bayesian thing to do.

Quarterbacks
------------
Dropback EPA gets a quarterback term shrunk toward `qb_prior` (in EPA per
dropback) rather than toward zero. That matters: a ridge centred on zero
treats an unknown backup as an average starter, which is exactly the case a
QB term exists to catch. Centring on a below-average prior means a QB with
no history projects as a typical backup, and a QB with a long history
projects as himself. Team offense is then measured net of its quarterback.

Units
-----
Ratings are EPA per play. projected_margin() converts to points with
PLAYS_PER_GAME and adds home field in points, measured from final scores of
completed games (play-level EPA is a poor instrument for that one constant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PLAYS_PER_GAME = 62.0
LEAGUE_PASS_RATE = 0.60
# Weeks per season on the age clock (18 regular + 4 playoff + a bye).
SEASON_WEEKS = 23


@dataclass
class PowerConfig:
    # Swept walk-forward on 2014-2021 (spread MAE vs actual margin):
    #   half-life  4 -> 10.78   10 -> 10.59   16 -> 10.45   24 -> 10.46*
    #   half-life 40 -> 10.53   80 -> 10.62
    # (*best on all seasons). Offseason 8 beat 16 and 30; alpha 300-1200
    # and the QB term each moved MAE by under 0.05.
    half_life: float = 20.0          # weeks
    offseason_weeks: float = 8.0     # age added across each offseason
    alpha: float = 600.0             # ridge penalty on team effects, in plays
    alpha_qb: float = 600.0          # ridge penalty on QB effects, in dropbacks
    qb_prior: float = -0.06          # EPA/dropback for a QB with no history
    use_qb: bool = True
    garbage_filter: bool = True
    pass_rate: float = LEAGUE_PASS_RATE
    min_weight: float = 1e-3         # drop team-games decayed below this


def age_clock(season, week, cfg: PowerConfig):
    """Monotone time in weeks, with offseasons stretched to offseason_weeks."""
    s = np.asarray(season, dtype=float)
    w = np.asarray(week, dtype=float)
    return s * (SEASON_WEEKS + cfg.offseason_weeks) + w


def _ridge(X, y, w, alpha_vec, unpenalized_intercept=True):
    """Weighted ridge via normal equations. Columns are small (<300)."""
    if unpenalized_intercept:
        X = np.column_stack([np.ones(len(y)), X])
        alpha_vec = np.concatenate([[0.0], alpha_vec])
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(alpha_vec)
    b = Xw.T @ y
    coef = np.linalg.solve(A, b)
    return coef[0] if unpenalized_intercept else 0.0, coef[1:] if \
        unpenalized_intercept else coef


@dataclass
class PowerRatings:
    teams: list
    db_off: pd.Series
    db_def: pd.Series        # EPA allowed above average; LOWER is better
    ru_off: pd.Series
    ru_def: pd.Series
    qb: pd.Series            # EPA/dropback relative to the team baseline
    qb_prior: float
    team_qb: dict            # most recent starter per team in the window
    home_field: float        # points
    cfg: PowerConfig
    extra: dict = field(default_factory=dict)

    def qb_value(self, qb_id, team=None) -> float:
        if qb_id is None or (isinstance(qb_id, float) and np.isnan(qb_id)):
            qb_id = self.team_qb.get(team)
        if qb_id is None:
            return 0.0
        return float(self.qb.get(qb_id, self.qb_prior))

    def off_epa(self, team, opp, qb_id=None) -> float:
        """Expected EPA per play for `team` with the ball against `opp`."""
        p = self.cfg.pass_rate
        db = (self.db_off.get(team, 0.0) + self.db_def.get(opp, 0.0)
              + (self.qb_value(qb_id, team) if self.cfg.use_qb else 0.0))
        ru = self.ru_off.get(team, 0.0) + self.ru_def.get(opp, 0.0)
        return p * db + (1 - p) * ru

    def projected_margin(self, home, away, neutral=False, home_qb=None,
                         away_qb=None) -> float:
        edge = (self.off_epa(home, away, home_qb)
                - self.off_epa(away, home, away_qb)) * PLAYS_PER_GAME
        return edge + (0.0 if neutral else self.home_field)

    def projected_epa_sum(self, home, away, home_qb=None, away_qb=None):
        """Combined offensive EPA/play of both teams: the totals signal."""
        return self.off_epa(home, away, home_qb) + \
            self.off_epa(away, home, away_qb)

    def table(self) -> pd.DataFrame:
        p = self.cfg.pass_rate
        rows = []
        for t in self.teams:
            qb = self.qb_value(None, t) if self.cfg.use_qb else 0.0
            off = p * (self.db_off.get(t, 0) + qb) + (1 - p) * self.ru_off.get(t, 0)
            dfn = p * self.db_def.get(t, 0) + (1 - p) * self.ru_def.get(t, 0)
            rows.append({"team": t, "qb": self.team_qb.get(t),
                         "off_pts": off * PLAYS_PER_GAME,
                         "def_pts": -dfn * PLAYS_PER_GAME,
                         "net_pts": (off - dfn) * PLAYS_PER_GAME})
        return (pd.DataFrame(rows).sort_values("net_pts", ascending=False)
                .reset_index(drop=True))


def _unit_fit(tg, teams, y_col, n_col, w_decay, cfg, qb=False):
    n = tg[n_col].to_numpy(float)
    ok = n > 0
    d = tg[ok]
    n = n[ok]
    y = d[y_col].to_numpy(float) / n
    w = n * w_decay[ok]
    k = len(teams)
    t_idx = {t: i for i, t in enumerate(teams)}
    X_parts = []
    off = np.zeros((len(d), k))
    off[np.arange(len(d)), d["team"].map(t_idx).to_numpy()] = 1.0
    dfn = np.zeros((len(d), k))
    dfn[np.arange(len(d)), d["opp"].map(t_idx).to_numpy()] = 1.0
    X_parts += [off, dfn]
    alpha = [np.full(2 * k, cfg.alpha)]
    qbs = []
    y_adj = y
    if qb:
        q = d["qb_id"].astype(str)
        qbs = sorted(set(q) - {"nan", "None"})
        q_idx = {v: i for i, v in enumerate(qbs)}
        Q = np.zeros((len(d), len(qbs)))
        m = q.map(q_idx)
        has = m.notna().to_numpy()
        Q[np.flatnonzero(has), m[has].astype(int).to_numpy()] = 1.0
        # Shrink toward the prior, not zero: b = prior + b', fit b'.
        y_adj = y - Q.sum(axis=1) * cfg.qb_prior
        X_parts.append(Q)
        alpha.append(np.full(len(qbs), cfg.alpha_qb))
    X = np.hstack(X_parts)
    b0, coef = _ridge(X, y_adj, w, np.concatenate(alpha))
    o = pd.Series(coef[:k], index=teams)
    df_ = pd.Series(coef[k:2 * k], index=teams)
    qser = pd.Series(coef[2 * k:] + cfg.qb_prior, index=qbs) if qb else \
        pd.Series(dtype=float)
    # Centre team effects; the intercept carries the league level.
    return o - o.mean(), df_ - df_.mean(), qser


def fit_power(tg: pd.DataFrame, now_season: int, now_week: int,
              home_field: float, cfg: PowerConfig | None = None
              ) -> PowerRatings:
    """Ratings from team-games strictly before (now_season, now_week)."""
    cfg = cfg or PowerConfig()
    prior = tg[(tg["season"] < now_season) |
               ((tg["season"] == now_season) & (tg["week"] < now_week))]
    age = age_clock(now_season, now_week, cfg) - \
        age_clock(prior["season"], prior["week"], cfg)
    w = 0.5 ** (age / cfg.half_life)
    keep = w >= cfg.min_weight
    prior, w = prior[keep], w[keep]
    if len(prior) < 64:
        raise ValueError("not enough history")
    teams = sorted(set(prior["team"]) | set(prior["opp"]))
    sfx = "_live" if cfg.garbage_filter else ""
    db_o, db_d, qb = _unit_fit(prior, teams, "epa_db" + sfx, "n_db" + sfx,
                               w, cfg, qb=cfg.use_qb)
    ru_o, ru_d, _ = _unit_fit(prior, teams, "epa_ru" + sfx, "n_ru" + sfx,
                              w, cfg, qb=False)
    last = prior.sort_values(["season", "week"]).drop_duplicates(
        "team", keep="last")
    team_qb = dict(zip(last["team"], last["qb_id"]))
    return PowerRatings(teams=teams, db_off=db_o, db_def=db_d, ru_off=ru_o,
                        ru_def=ru_d, qb=qb, qb_prior=cfg.qb_prior,
                        team_qb=team_qb, home_field=float(home_field),
                        cfg=cfg)
