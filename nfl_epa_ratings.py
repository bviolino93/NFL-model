"""
Opponent-adjusted EPA ratings for NFL, with a quarterback term.

Why replace the margin model
----------------------------
The current Sunday Edge rating is a ridge on final margins. That is one
observation per game with a standard deviation of 13.19 points — the noisiest
possible summary of sixty minutes of football. The backtest says so: over
4,254 games the model coefficient was +0.099 with t = 1.19, and at NFL volume
(~285 games a season) reaching t = 2 on that effect size would take roughly
42 seasons. The game-level model cannot be validated in a human lifetime.

The same games contain ~150 plays each, and nflverse publishes EPA for every
one of them free, back to 1999. Two orders of magnitude more information from
the identical set of games.

Why the quarterback term
------------------------
An NFL starting quarterback is worth roughly 5-7 points of spread. A
team-level rating averaged over a rolling window cannot know who is playing.
When a backup starts, the rating is stale by more than the app's entire
4-point betting threshold — and the market has already moved. So the model
disagrees loudest exactly where it is most wrong.

Fitting a QB effect alongside the team effects, then projecting with the
quarterback who is ACTUALLY STARTING, removes that failure mode. It is the
single largest structural gap in the margin model.

Method
------
    epa ~ offense_team + defense_team + offense_qb + home_field

fitted by ridge on plays with garbage time removed. Team and QB coefficients
share the design matrix, so a quarterback's effect is measured net of the
teammates around him and the defenses he faced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

# Plays per team per game, to convert EPA/play into points/game.
PLAYS_PER_GAME = 63.0

# Garbage time. nflverse ships a win-probability column, which is a better
# filter than score-and-clock rules: it already accounts for time remaining,
# timeouts and possession. Outside this band the football stops being
# representative — prevent defense, running clock, backups.
WP_LOW, WP_HIGH = 0.05, 0.95

# Only scrimmage plays carry offensive signal.
KEEP_PLAY_TYPES = {"pass", "run", "qb_kneel", "qb_spike"}

# A quarterback needs a real sample before his own coefficient means
# anything; below this his plays still inform the TEAM effect, but he gets no
# personal term.
#
# MEASURED, NOT ASSUMED. Swept across four simulated seasons on a grid of
# alpha x qb_alpha_mult x min_qb_plays, scoring RMSE of projected margin
# against the true expected margin on held-out weeks:
#
#     alpha  300 -> 4.58    qb_mult 1.0 beat 2.0 and 4.0 at every alpha
#     alpha  800 -> 4.37    <- best, and the grid was extended past its
#     alpha 1500 -> 4.45       edge to confirm it is a minimum, not a wall
#     alpha 3000 -> 4.91
#
# The heavier QB penalty I assumed would be needed made things worse: the
# shared alpha at 800 already shrinks the QB block enough.
DEFAULT_ALPHA = 800.0
DEFAULT_QB_ALPHA_MULT = 1.0
MIN_QB_PLAYS = 100


def fetch_pbp(seasons, cache_dir: str = ".cache_nflpbp") -> pd.DataFrame:
    """
    Play-by-play from nflverse, cached to disk as parquet.

    A season is ~48k plays and the download is slow; re-pulling it on every
    experiment is the difference between a fast iteration loop and a
    ten-minute one.
    """
    import os
    import nfl_data_py as nfl

    os.makedirs(cache_dir, exist_ok=True)
    cols = ["game_id", "season", "week", "posteam", "defteam", "home_team",
            "away_team", "epa", "wp", "play_type", "passer_player_id",
            "passer_player_name", "season_type"]
    frames = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"pbp_{yr}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
            continue
        df = nfl.import_pbp_data([int(yr)], downcast=True, cache=False)
        df = df[[c for c in cols if c in df.columns]].copy()
        df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare_pbp(raw: pd.DataFrame, regular_only: bool = True) -> pd.DataFrame:
    """Drop non-scrimmage plays, bad rows and anything unusable."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    need = {"posteam", "defteam", "epa"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"play data missing columns: {sorted(missing)}")

    if regular_only and "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper() == "REG"]
    if "play_type" in df.columns:
        df = df[df["play_type"].astype(str).isin(KEEP_PLAY_TYPES)]

    df["epa"] = pd.to_numeric(df["epa"], errors="coerce")
    df = df.dropna(subset=["posteam", "defteam", "epa"])
    # A handful of rows carry absurd EPA from scoring errors upstream.
    df = df[df["epa"].abs() <= 10.0]
    return df.reset_index(drop=True)


def drop_garbage_time(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only plays run while the game was still in doubt."""
    if df.empty or "wp" not in df.columns:
        return df
    wp = pd.to_numeric(df["wp"], errors="coerce")
    keep = wp.between(WP_LOW, WP_HIGH) | wp.isna()
    return df[keep].reset_index(drop=True)


@dataclass
class NflEpaRatings:
    offense: pd.Series          # points/game above average
    defense: pd.Series          # points/game above average (higher = better D)
    overall: pd.Series
    qb: pd.Series               # points/game above a replacement-ish baseline
    home_field: float
    alpha: float
    n_plays: int
    teams: list = field(default_factory=list)

    def rating(self, team: str) -> float:
        return float(self.overall.get(team, 0.0))

    def qb_value(self, qb_id) -> float:
        if qb_id is None:
            return 0.0
        return float(self.qb.get(qb_id, 0.0))

    def projected_margin(self, home: str, away: str, neutral: bool = False,
                         home_qb=None, away_qb=None) -> float:
        """
        Home margin in points, positive means home favoured.

        Passing the starting quarterbacks is the point of this model. Omit
        them and it degrades to a team-only rating, which is what the old
        margin model already was.
        """
        edge = (self.rating(home) + self.qb_value(home_qb)) - \
               (self.rating(away) + self.qb_value(away_qb))
        return edge + (0.0 if neutral else self.home_field)


def fit_nfl_epa(df: pd.DataFrame, alpha: float = DEFAULT_ALPHA,
                qb_alpha_mult: float = DEFAULT_QB_ALPHA_MULT,
                min_qb_plays: int = MIN_QB_PLAYS) -> NflEpaRatings:
    """
    Ridge on offense, defense, quarterback and home field.

    Three implementation points that matter:

    1. Home field is scaled up before fitting so the ridge barely penalises
       it. Team effects SHOULD be shrunk — each team appears in a fraction of
       the plays — but home field is one parameter identified by every play in
       the sample, and penalising it biases the estimate toward zero.

    2. Quarterbacks get a HEAVIER penalty than teams (qb_alpha_mult). There
       are more of them, many with small samples, and an unshrunk QB term
       will happily explain a hot three-game stretch as talent.

    3. Quarterbacks below min_qb_plays get no personal coefficient. Their
       plays still inform the team effect; they simply do not get to move the
       projection on the strength of forty snaps.
    """
    if df is None or df.empty:
        raise ValueError("no plays to fit")

    teams = sorted(set(df["posteam"]) | set(df["defteam"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)

    has_qb = "passer_player_id" in df.columns
    if has_qb:
        qb_raw = df["passer_player_id"].astype(str)
        counts = qb_raw.value_counts()
        eligible = {q for q, c in counts.items()
                    if c >= min_qb_plays and q not in ("nan", "None", "")}
        qbs = sorted(eligible)
    else:
        qb_raw, qbs = None, []
    q_idx = {q: i for i, q in enumerate(qbs)}
    kq = len(qbs)

    n = len(df)
    rows, cols, vals = [], [], []
    off_i = df["posteam"].map(t_idx).to_numpy()
    def_i = df["defteam"].map(t_idx).to_numpy()
    ar = np.arange(n)
    rows.append(ar); cols.append(off_i); vals.append(np.ones(n))
    rows.append(ar); cols.append(def_i + k); vals.append(np.ones(n))

    if kq:
        qi = qb_raw.map(q_idx).to_numpy(dtype=float)
        m = ~np.isnan(qi)
        # Scale the QB block DOWN so the shared ridge penalty hits it harder,
        # then scale the coefficients back up after fitting.
        qb_scale = 1.0 / math.sqrt(max(qb_alpha_mult, 1e-9))
        rows.append(ar[m]); cols.append(qi[m].astype(np.int64) + 2 * k)
        vals.append(np.full(int(m.sum()), qb_scale))

    HFA_SCALE = 100.0
    width = 2 * k + kq
    if "home_team" in df.columns:
        is_home = (df["posteam"].astype(str) ==
                   df["home_team"].astype(str)).to_numpy(dtype=float)
        rows.append(ar); cols.append(np.full(n, width))
        vals.append(is_home * HFA_SCALE)
        width += 1
        has_home = True
    else:
        has_home = False

    X = sparse.csr_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, width),
    )
    y = df["epa"].to_numpy(dtype=np.float64)

    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="sparse_cg")
    model.fit(X, y)
    c = model.coef_

    off = pd.Series(c[:k], index=teams)
    dfn = pd.Series(c[k:2 * k], index=teams)
    qb = (pd.Series(c[2 * k:2 * k + kq], index=qbs) * qb_scale
          if kq else pd.Series(dtype=float))
    hfa = float(c[width - 1]) * HFA_SCALE if has_home else 0.0

    off = (off - off.mean()) * PLAYS_PER_GAME
    dfn = -(dfn - dfn.mean()) * PLAYS_PER_GAME
    if len(qb):
        qb = (qb - qb.mean()) * PLAYS_PER_GAME

    return NflEpaRatings(
        offense=off.sort_values(ascending=False),
        defense=dfn.sort_values(ascending=False),
        overall=(off + dfn).sort_values(ascending=False),
        qb=qb.sort_values(ascending=False) if len(qb) else qb,
        home_field=hfa * PLAYS_PER_GAME,
        alpha=float(alpha),
        n_plays=n,
        teams=teams,
    )


def starters_from_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Who actually took the snaps, per team per game.

    Used two ways: to know which quarterback to project with, and to detect
    that last week's starter is not this week's.
    """
    if df.empty or "passer_player_id" not in df.columns:
        return pd.DataFrame()
    g = (df.dropna(subset=["passer_player_id"])
           .groupby(["game_id", "season", "week", "posteam",
                     "passer_player_id"], dropna=False)
           .size().rename("plays").reset_index())
    g = g.sort_values("plays", ascending=False)
    return g.drop_duplicates(["game_id", "posteam"], keep="first")


def build_ratings(seasons, through=None, alpha: float = DEFAULT_ALPHA,
                  drop_garbage: bool = True) -> NflEpaRatings:
    """End to end. `through` is (season, week); plays from that week and
    later are excluded, so the ratings are what was knowable beforehand."""
    raw = fetch_pbp(seasons)
    plays = prepare_pbp(raw)
    if through is not None:
        s, w = through
        plays = plays[(plays["season"] < s) |
                      ((plays["season"] == s) & (plays["week"] < w))]
    if drop_garbage:
        plays = drop_garbage_time(plays)
    return fit_nfl_epa(plays, alpha=alpha)
