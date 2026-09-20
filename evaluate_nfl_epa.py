"""
Does the EPA+QB model beat the closing line, and does it beat 0.099?

Sunday Edge's current margin model measured +0.099 with t = 1.19 over 4,254
games. That is the number to beat, and this runs the identical test so the
two are directly comparable:

    actual_home_margin ~ b0 + b1*market_margin + b2*model_margin

b2 is the fraction of the model's disagreement with the market that showed up
in results. Read it against 0.099, not against zero.

Leakage is the whole danger. Ratings for week N are fit only on plays from
before week N, and the quarterback used to project a game is the one who
actually started it — which is knowable pregame from the injury report, but
must be taken from that game's data here rather than from a season aggregate
that would smuggle in the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_epa_ratings import (
    DEFAULT_ALPHA, drop_garbage_time, fetch_pbp, fit_nfl_epa, prepare_pbp,
    starters_from_pbp,
)
from nfl_granular import (
    GranularConfig, fit_granular, home_field_from_schedule,
)


def _fit(prior, kind, alpha, hfa_points):
    """
    One of the candidate models, fitted on plays strictly before the target
    week. Both expose projected_margin(home, away, home_qb=, away_qb=), so
    the rest of the harness does not care which is which.

      "team"     — one rating per team, plus a QB term
      "granular" — pass/rush offense and defense paired by matchup, plus QB

    Home field is supplied from the schedule for the granular model, because
    play-level EPA cannot measure it (see nfl_granular for the numbers).
    """
    if kind == "granular":
        cfg = GranularConfig(alpha_pass=alpha, alpha_rush=alpha,
                             alpha_qb=alpha)
        return fit_granular(prior, cfg, hfa_points=hfa_points)
    r = fit_nfl_epa(prior, alpha=alpha)
    if hfa_points is not None:
        # Same treatment for both, so the comparison is about the ratings
        # rather than about who got the better home-field estimate.
        r.home_field = float(hfa_points)
    return r


def _ols(X: np.ndarray, y: np.ndarray):
    """OLS with standard errors. numpy only — statsmodels is a heavy
    dependency for one regression."""
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    k = Xd.shape[1]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    resid = y - Xd @ beta
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2


def build_eval_frame(pbp: pd.DataFrame, sched: pd.DataFrame,
                     alpha: float = DEFAULT_ALPHA, min_train_plays: int = 20000,
                     use_qb: bool = True, drop_garbage: bool = True,
                     model: str = "team") -> pd.DataFrame:
    """
    One row per game: market margin, model margin, actual margin.

    `sched` needs season, week, game_id, home_team, away_team, home_score,
    away_score, spread_line (nflverse states it from the home side).
    """
    starters = starters_from_pbp(pbp)
    sched = sched.dropna(subset=["home_score", "away_score"]).copy()
    keys = sorted({(int(s), int(w)) for s, w in
                   zip(sched["season"], sched["week"])})
    out = []
    for season, week in keys:
        prior = pbp[(pbp["season"] < season) |
                    ((pbp["season"] == season) & (pbp["week"] < week))]
        if drop_garbage:
            prior = drop_garbage_time(prior)
        if len(prior) < min_train_plays:
            continue
        # Home field from completed games only — never from the week being
        # projected, which would be leakage.
        past_games = sched[(sched["season"] < season) |
                           ((sched["season"] == season) &
                            (sched["week"] < week))]
        hfa = home_field_from_schedule(past_games)
        try:
            rat = _fit(prior, model, alpha, hfa)
        except Exception:
            continue

        wk = sched[(sched["season"] == season) & (sched["week"] == week)]
        for _, g in wk.iterrows():
            home, away = str(g["home_team"]), str(g["away_team"])
            # Both rating objects expose `teams`; only the team-level one has
            # `.overall`. Use the common attribute so either can be scored.
            known = set(getattr(rat, "teams", []) or [])
            if home not in known or away not in known:
                continue
            spread = pd.to_numeric(pd.Series([g.get("spread_line")]),
                                   errors="coerce").iloc[0]
            if not np.isfinite(spread):
                continue
            hq = aq = None
            if use_qb and not starters.empty:
                st_g = starters[starters["game_id"] == g["game_id"]]
                for _, s in st_g.iterrows():
                    if str(s["posteam"]) == home:
                        hq = s["passer_player_id"]
                    elif str(s["posteam"]) == away:
                        aq = s["passer_player_id"]
            out.append({
                "season": season, "week": week, "game_id": g["game_id"],
                # nflverse spread_line is the home line: +3 means home
                # favoured by 3. Verified against outcomes, never assumed.
                "market_margin": float(spread),
                "model_margin": rat.projected_margin(home, away,
                                                     home_qb=hq, away_qb=aq),
                "actual_margin": float(g["home_score"]) - float(g["away_score"]),
            })
    return pd.DataFrame(out)


def score(frame: pd.DataFrame, benchmark: float = 0.099) -> dict:
    if frame is None or len(frame) < 50:
        raise ValueError("not enough graded games to say anything")
    b, se, r2 = _ols(
        frame[["market_margin", "model_margin"]].to_numpy(dtype=float),
        frame["actual_margin"].to_numpy(dtype=float),
    )
    b_model, b_market = float(b[2]), float(b[1])
    t_model = float(b[2] / se[2]) if se[2] > 0 else float("nan")

    disagree = frame["model_margin"] - frame["market_margin"]
    live = frame[disagree.abs() >= 1.0]
    if len(live):
        side = np.sign(disagree.loc[live.index]) * (
            live["actual_margin"] - live["market_margin"])
        cover = float((side > 0).mean())
    else:
        cover = float("nan")

    return {
        "n_games": int(len(frame)),
        "model_coef": b_model,
        "model_t": t_model,
        "model_se": float(se[2]),
        "market_coef": b_market,
        "r2": r2,
        "model_side_cover": cover,
        "n_disagreements": int(len(live)),
        "mae_model": float((frame["model_margin"] - frame["actual_margin"]).abs().mean()),
        "mae_market": float((frame["market_margin"] - frame["actual_margin"]).abs().mean()),
        "benchmark": benchmark,
        "beats_benchmark": bool(b_model > benchmark),
    }


def report(res: dict) -> str:
    lo = res["model_coef"] - 1.96 * res["model_se"]
    hi = res["model_coef"] + 1.96 * res["model_se"]
    lines = [
        f"Games graded:        {res['n_games']:,}",
        f"Model coefficient:   {res['model_coef']:+.4f}  "
        f"(t = {res['model_t']:+.2f}, 95% CI {lo:+.3f} to {hi:+.3f})",
        f"Current model:       {res['benchmark']:+.4f}   <- the number to beat",
        f"Market coefficient:  {res['market_coef']:+.4f}",
        f"Model side cover:    {res['model_side_cover']:.1%} on "
        f"{res['n_disagreements']:,} disagreements  (52.4% breaks even)",
        f"MAE model / market:  {res['mae_model']:.2f} / {res['mae_market']:.2f} points",
        "",
    ]
    b = res["model_coef"]
    if b <= 0.02:
        lines.append("VERDICT: adds nothing to the closing line.")
    elif b <= res["benchmark"]:
        lines.append(
            "VERDICT: no better than the margin model already in the app. "
            "Play-level data did not help here.")
    elif lo > 0:
        lines.append(
            f"VERDICT: beats the current model AND clears zero. Use {b:.3f} as "
            "the blend weight. Re-run on held-out seasons before trusting it.")
    else:
        lines.append(
            f"VERDICT: better than {res['benchmark']:.3f}, but the interval "
            "still contains zero. Promising, not proven.")
    return "\n".join(lines)


def run(seasons, sched: pd.DataFrame, alpha: float = DEFAULT_ALPHA,
        use_qb: bool = True, model: str = "team") -> dict:
    raw = fetch_pbp(seasons)
    pbp = prepare_pbp(raw)
    frame = build_eval_frame(pbp, sched, alpha=alpha, use_qb=use_qb,
                             model=model)
    res = score(frame)
    print(f"[model: {model}, alpha {alpha}, QB {use_qb}]")
    print(report(res))
    return res


def compare(seasons, sched: pd.DataFrame, alpha: float = DEFAULT_ALPHA):
    """
    Score every candidate on the SAME games, so the numbers are comparable.

    A model that quietly drops games it cannot rate will look better than one
    that rates everything, so the frames are intersected on game_id before
    scoring rather than each being scored on whatever it managed.
    """
    raw = fetch_pbp(seasons)
    pbp = prepare_pbp(raw)
    frames = {}
    for kind, qb in (("team", False), ("team", True), ("granular", True)):
        label = f"{kind}{'+QB' if qb else ''}"
        frames[label] = build_eval_frame(pbp, sched, alpha=alpha,
                                         use_qb=qb, model=kind)
    common = None
    for f in frames.values():
        ids = set(f["game_id"])
        common = ids if common is None else (common & ids)
    out = {}
    print(f"Scored on {len(common):,} games common to every model.\n")
    for label, f in frames.items():
        res = score(f[f["game_id"].isin(common)])
        out[label] = res
        print(f"{label:12s} coef {res['model_coef']:+.4f}  "
              f"t {res['model_t']:+5.2f}  "
              f"cover {res['model_side_cover']:.1%}  "
              f"MAE {res['mae_model']:.2f}")
    print(f"\n{'current margin model':12s} coef +0.0990  t +1.19  "
          f"(4,254 games) <- the number to beat")
    return out
