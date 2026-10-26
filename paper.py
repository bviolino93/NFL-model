"""
No-money test of model spread bets at early lines.

opener_research.py found that betting the EPA ratings' side of the opening
spread, when they disagree by 3+ points, has run about break-even (52-54%
where 52.4% breaks even at -110). Too thin to bet, too close to drop. This
logs those bets live, the first time each game's line is seen (usually the
Monday run, when next week's lines are fresh), using the ratings plus each
side's injuries from the latest report and players on IR, and grades them
two ways:

  * against the final score, at -110 on the logged line
  * against the closing line (CLV, in points): the honest early read,
    since a few hundred bets of results are mostly luck

Nothing here is staked or shown as a bet. After a season, if the logged
bets beat the close on average, spreads can move into the real picks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest import calibrate, home_field_before, project
from nfl_data import fix_starters, load_games, load_rosters, load_team_games
from nfl_power import PLAYS_PER_GAME, PowerConfig, fit_power

GAP = 3.0
# Only lines seen this far ahead of kickoff count as early; anything later
# is close to the closing line, which the ratings are known not to beat.
MIN_LEAD_DAYS = 3.0
# Points per unit of snap-weighted injury load (nfl_combined.injury_loads),
# measured against the ratings' margin over 2013-2026 (t -2.3 and -2.8).
# Against the closing line only about a third survives, so the market
# prices most of it; early lines may not have it yet.
INJ_OFF_PTS = -0.78
INJ_DEF_PTS = -0.94


def load(path: str) -> list[dict]:
    try:
        with open(path) as fh:
            return json.load(fh).get("bets", [])
    except (OSError, ValueError):
        return []


def _key(home: str, away: str, commence: str) -> str:
    return f"{away}|{home}|{commence[:10]}"


def model_margins(fair: dict, abbr: dict[str, str]) -> dict[str, dict]:
    """The test's own home margin for each fair-line game: calibrated
    ratings (results through the previous week, with each side's projected
    starting QB) plus each side's injuries from the latest report out."""
    games = fix_starters(load_games())
    games = games[games.game_type == "REG"].copy()
    games["neutral"] = games["location"].astype(str).str.lower() == "neutral"
    games["margin"] = games["home_score"] - games["away_score"]
    games["points"] = games["home_score"] + games["away_score"]
    # Unplayed games carry nflverse's projected starter, which is known
    # before kickoff; without it the ratings assume last game's QB, which
    # is wrong whenever a starter returns from injury.
    want = {}
    for k, f in fair.items():
        kick = pd.Timestamp(f.commence).tz_convert("America/New_York")
        day = kick.tz_localize(None).normalize()
        g = games[(games.home_team == abbr.get(f.home_team))
                  & (games.away_team == abbr.get(f.away_team))
                  & ((pd.to_datetime(games.gameday) - day).abs()
                     <= pd.Timedelta(days=1))]
        if len(g):
            want[g.game_id.iloc[0]] = k
    if not want:
        return {}
    season = int(games.loc[games.game_id.isin(want), "season"].max())
    seasons = list(range(season - 4, season + 1))
    past = games[games.season.isin(seasons)
                 & (games.game_id.isin(want) | games.margin.notna())]
    # Lines only feed project()'s output columns; next week's are often
    # missing from nflverse but the game still needs a rating.
    past = past.assign(spread_line=past.spread_line.fillna(0),
                       total_line=past.total_line.fillna(44))
    tg = load_team_games(range(season - 5, season + 1))
    fr = calibrate(project(tg, past, seasons, PowerConfig()))
    fr = fr[fr.game_id.isin(want)]
    try:
        from nfl_combined import injury_loads
        L = injury_loads([season], include_ir=True).sort_values("week")
        P = injury_loads([season], include_ir=True, players=True)
    except Exception as exc:
        print(f"injuries unavailable: {exc}")
        L = pd.DataFrame(columns=["season", "week", "team", "off", "def"])
        P = pd.DataFrame(columns=["week", "team", "name", "pos", "status",
                                  "side", "load"])

    def load_for(team, week):
        r = L[(L.team == team) & (L.week <= week)]
        return (0.0, 0.0) if r.empty else (float(r.off.iloc[-1]),
                                           float(r["def"].iloc[-1]))

    def players_for(team, week):
        """The players behind a team's injury load, biggest first, each with
        the points the model takes off for him."""
        r = P[(P.team == team) & (P.week <= week)]
        if r.empty:
            return []
        r = r[r.week == r.week.max()]
        pts = np.where(r.side == "off", INJ_OFF_PTS, INJ_DEF_PTS) * r.load
        r = r.assign(pts=pts)
        r = r[r.pts.abs() >= 0.1].sort_values("pts")
        return [{"name": x.name, "pos": x.pos, "status": x.status,
                 "pts": round(float(x.pts), 1)} for x in r.head(6).itertuples()]

    try:
        ro = load_rosters(season).sort_values("week")
        status = ro.drop_duplicates("full_name", keep="last").set_index(
            "full_name")["status"].to_dict()
    except Exception:
        status = {}

    cfg = PowerConfig()
    fits = {}
    out = {}
    for r in fr.itertuples():
        ho, hd = load_for(r.home, r.week)
        ao, ad = load_for(r.away, r.week)
        inj = INJ_OFF_PTS * (ho - ao) + INJ_DEF_PTS * (hd - ad)
        row = {"ratings": float(r.model_margin), "injuries": float(inj),
               "margin": float(r.model_margin + inj),
               "injured": {"home": players_for(r.home, r.week),
                           "away": players_for(r.away, r.week)}}
        try:
            if (r.season, r.week) not in fits:
                fits[r.season, r.week] = fit_power(
                    tg, int(r.season), int(r.week),
                    home_field_before(games, r.season, r.week), cfg)
            g = games[games.game_id == r.game_id].iloc[0]
            row.update(breakdown(fits[r.season, r.week], g, r, cfg))
        except Exception as exc:
            print(f"no breakdown for {r.game_id}: {exc}")
        g = games[games.game_id == r.game_id].iloc[0]
        for side in ("home", "away"):
            swap = qb_swap(games, status, r.season, getattr(r, side),
                           g[f"{side}_qb_name"])
            if swap:
                row["injured"][side].insert(0, swap)
        out[want[r.game_id]] = row
    return out


def qb_swap(games, status: dict, season, team, starter) -> dict | None:
    """The team's regular QB (most starts this season) when someone else is
    projected to start, as an injury-list entry. The QB change itself is
    priced through the ratings' QB term, so it carries no points here."""
    played = games[(games.season == season) & games.margin.notna()]
    names = pd.concat([played.loc[played.home_team == team, "home_qb_name"],
                       played.loc[played.away_team == team, "away_qb_name"]]
                      ).dropna()
    if names.empty or not isinstance(starter, str):
        return None
    regular = names.value_counts().index[0]
    if regular == starter:
        return None
    st = status.get(regular)
    return {"name": regular, "pos": "QB",
            "status": "IR" if st == "RES" else "Out" if st != "ACT" else "Benched",
            "pts": None, "note": f"{starter.split()[-1]} starting"}


def breakdown(pr, g, r, cfg: PowerConfig) -> dict:
    """Where the ratings' home margin comes from, in calibrated points:
    each offense against the other's defense (passing and rushing, with
    the starting QB split out), plus home field. The parts add up to
    `ratings`."""
    h, a = r.home, r.away
    p = cfg.pass_rate

    def qb(team, qid):
        qid = qid if isinstance(qid, str) else None
        return pr.qb_value(qid, team) if cfg.use_qb else 0.0

    k = (r.model_margin - r.hfa) / r.model_edge if abs(r.model_edge) > 1e-6 \
        else 0.0
    pts = PLAYS_PER_GAME * k
    return {
        "hfa": float(r.hfa),
        # home team's edge from the offenses and from the defenses
        "pass_off": pts * p * (pr.db_off.get(h, 0) - pr.db_off.get(a, 0)),
        "run_off": pts * (1 - p) * (pr.ru_off.get(h, 0) - pr.ru_off.get(a, 0)),
        "pass_def": pts * p * (pr.db_def.get(a, 0) - pr.db_def.get(h, 0)),
        "run_def": pts * (1 - p) * (pr.ru_def.get(a, 0) - pr.ru_def.get(h, 0)),
        "qb": pts * p * (qb(h, g.home_qb_id) - qb(a, g.away_qb_id)),
        "home_qb": g.home_qb_name if isinstance(g.home_qb_name, str) else None,
        "away_qb": g.away_qb_name if isinstance(g.away_qb_name, str) else None,
    }


def log(history: list[dict], fair: dict, margins: dict, run: str) -> list:
    """Add a test bet for each game seen for the first time where the
    ratings and the market disagree by GAP points or more."""
    def lead(commence, at):
        return (pd.Timestamp(commence) - pd.Timestamp(at)) / pd.Timedelta(days=1)

    history = [b for b in history
               if lead(b["commence"], b["logged"]) >= MIN_LEAD_DAYS]
    seen = {b["key"] for b in history}
    for k, f in fair.items():
        key = _key(f.home_team, f.away_team, str(f.commence))
        if key in seen or k not in margins or np.isnan(f.market_margin):
            continue
        if lead(f.commence, run) < MIN_LEAD_DAYS:
            continue
        seen.add(key)
        line = round(f.market_margin * 2) / 2     # home margin, bookable
        mine = margins[k]
        gap = mine["margin"] - line
        if abs(gap) < GAP:
            continue
        home = gap > 0
        history.append({
            "key": key, "away": f.away_team, "home": f.home_team,
            "commence": str(f.commence), "logged": run,
            "side": f.home_team if home else f.away_team,
            # The side's own line, e.g. -3 for a 3-point favourite
            "point": -line if home else line,
            "model_margin": round(mine["margin"], 2),
            "ratings": round(mine["ratings"], 2),
            "injuries": round(mine["injuries"], 2), "market_margin": line,
            "gap": round(gap, 2), "close": None, "clv": None,
            "result": None, "final": None})
    return history


def grade(history: list[dict], games: pd.DataFrame,
          abbr: dict[str, str]) -> list[dict]:
    g = games.assign(day=pd.to_datetime(games["gameday"]).dt.normalize())
    for b in history:
        if b["result"]:
            continue
        day = (pd.Timestamp(b["commence"]).tz_convert("America/New_York")
               .tz_localize(None).normalize())
        m = g[(g.home_team == abbr.get(b["home"]))
              & (g.away_team == abbr.get(b["away"]))
              & ((g.day - day).abs() <= pd.Timedelta(days=1))]
        if m.empty:
            continue
        m = m.iloc[0]
        sign = 1 if b["side"] == b["home"] else -1
        if pd.notna(m.spread_line):
            b["close"] = float(m.spread_line)
            # Points the line moved our way: home bet at 3, closed 4 = +1
            b["clv"] = round(sign * (m.spread_line - b["market_margin"]), 1)
        if pd.notna(m.home_score) and pd.notna(m.away_score):
            diff = sign * (m.home_score - m.away_score - b["market_margin"])
            b["result"] = "win" if diff > 0 else "loss" if diff < 0 \
                else "push"
            b["final"] = f"{int(m.away_score)}-{int(m.home_score)}"
    return history


def summary(history: list[dict]) -> dict:
    done = [b for b in history if b["result"]]
    w = sum(b["result"] == "win" for b in done)
    lo = sum(b["result"] == "loss" for b in done)
    clv = [b["clv"] for b in done if b["clv"] is not None]
    return {"bets": len(history), "wins": w, "losses": lo,
            "pushes": len(done) - w - lo,
            "win_rate": round(w / (w + lo), 4) if w + lo else None,
            "avg_clv": round(float(np.mean(clv)), 2) if clv else None,
            "beat_close": round(float(np.mean([c > 0 for c in clv])), 4)
            if clv else None}


def save(path: str, history: list[dict]) -> None:
    with open(path, "w") as fh:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(),
                   "gap": GAP, "summary": summary(history),
                   "bets": history}, fh, indent=2, default=str)
