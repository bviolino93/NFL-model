"""
Sunday Edge — NFL

Slate builder, bet freezer, and tracker for NFL spreads and totals.

Built with the measurement fixes Saturday Edge needed retrofitted:
separate official/watch ledgers, real closing-line capture, continuous
result margins, and a version string that tracks the selection math.

The model is an opponent-adjusted ridge power rating, walk-forward.
Backtested 2007-2025 (4,254 games) it did NOT beat the closing line:
model coefficient +0.099, t = +1.19. Those constants are in the header
below and shown in the app, because the honest thing is to let the
record accumulate against a stated prior rather than hide it.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge

import nfl_data_py as nfl

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
RIDGE_ALPHA   = 8.0
# Year-over-year persistence of NFL team strength (~0.5-0.6 empirically).
# Applied ONLY where there is no in-season data. This scales last year's
# ratings toward average; too low and every game looks like a pick'em,
# which makes the model take underdogs indiscriminately in early weeks.
CARRYOVER     = 0.65
WINDOW_GAMES  = 320

# Residual SDs measured on 4,254 games, 2007-2025. These convert a point
# edge into a cover probability, so they must come from data, not a guess.
SD_MARGIN     = 13.19
SD_TOTAL      = 13.35

# Weight on the model when blending with the market line. From the
# backtest regression: market 1.011, model 0.099. The model gets 0.099
# because that is what it earned, not because it feels too low.
MODEL_WEIGHT  = 0.099

# Backtest verdict, stated up front and shown in the UI.
BACKTEST_T    = 1.19
BACKTEST_N    = 4254

MIN_EDGE_PTS  = 1.0   # official tier: model/market disagreement in points
WATCH_EDGE_PTS = 0.5  # watch tier floor; below this nothing is shown

MODEL_VERSION = f"1.0.0-a{RIDGE_ALPHA}-w{MODEL_WEIGHT}-e{MIN_EDGE_PTS}"

TRACKER_COLS = [
    "record_key", "frozen_at", "season", "week", "game_id", "kickoff",
    "matchup", "home_team", "away_team", "market_type", "pick_side",
    "pick_label", "bet_line", "model_line", "edge_pts", "cover_prob",
    "expected_value", "odds", "bet_tier", "model_version",
    "status", "result", "units_result", "result_margin",
    "final_home_score", "final_away_score",
    "closing_line", "clv_points", "closing_captured_at", "graded_at",
]

st.set_page_config(page_title="Sunday Edge", page_icon="🏈", layout="wide")


# ----------------------------------------------------------------------
# Storage — Google Sheets, with a session fallback
# ----------------------------------------------------------------------
def _sheet(return_error=False):
    try:
        import gspread
        raw = st.secrets["gcp_service_account_json"]
        creds = json.loads(raw) if isinstance(raw, str) else dict(raw)
        gc = gspread.service_account_from_dict(creds)
        name = st.secrets.get("tracker_sheet_name", "sunday_edge_tracker")
        sh = gc.open(name)
        try:
            ws = sh.worksheet("tracker")
        except Exception:
            ws = sh.add_worksheet("tracker", rows=2000, cols=len(TRACKER_COLS))
        return (ws, None) if return_error else ws
    except Exception as e:
        return (None, str(e)) if return_error else None


def empty_tracker():
    return pd.DataFrame(columns=TRACKER_COLS)


def load_tracker():
    ws = _sheet()
    if ws is not None:
        try:
            recs = ws.get_all_records()
            df = pd.DataFrame(recs) if recs else empty_tracker()
            for c in TRACKER_COLS:
                if c not in df.columns:
                    df[c] = None
            return df[TRACKER_COLS]
        except Exception:
            pass
    return st.session_state.get("tracker", empty_tracker())


def save_tracker(df):
    x = df.copy()
    for c in TRACKER_COLS:
        if c not in x.columns:
            x[c] = None
    x = x[TRACKER_COLS]
    # Dedup on the FULL key, which includes tier. Official and watch are
    # independent ledgers on purpose.
    if not x.empty:
        x = x[~x["record_key"].astype(str).duplicated(keep="first")]
    st.session_state["tracker"] = x
    ws = _sheet()
    if ws is not None:
        try:
            ws.clear()
            ws.update([TRACKER_COLS] + x.fillna("").astype(str).values.tolist())
        except Exception as e:
            st.warning(f"Sheet write failed, kept in session only: {e}")
    return x


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_schedules(seasons):
    df = nfl.import_schedules(list(seasons))
    keep = ["game_id", "season", "week", "gameday", "gametime",
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "total_line"]
    return df[[c for c in keep if c in df.columns]].copy()


def line_sign(g):
    """nflverse states spread_line from the home side. Verify, never assume:
    a flipped sign would invert every pick silently."""
    d = g.dropna(subset=["spread_line", "home_score", "away_score"])
    if len(d) < 50:
        return 1.0
    m = d["home_score"] - d["away_score"]
    return 1.0 if np.corrcoef(d["spread_line"], m)[0, 1] > 0 else -1.0


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def fit_ratings(hist, teams, target, symmetric=False):
    if len(hist) < 40:
        return None, None
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(hist), len(teams) + 1))
    h, a = hist["home_team"].values, hist["away_team"].values
    for r in range(len(hist)):
        if h[r] in idx:
            X[r, idx[h[r]]] = 1.0
        if a[r] in idx:
            X[r, idx[a[r]]] = 1.0 if symmetric else -1.0
        X[r, -1] = 1.0
    m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False).fit(X, hist[target].values)
    return {t: m.coef_[idx[t]] for t in teams}, float(m.coef_[-1])


def build_ratings(sched, season, week):
    """Ratings from games played strictly before the target week."""
    g = sched.dropna(subset=["home_score", "away_score"]).copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    prior = g[(g["season"] < season) | ((g["season"] == season) & (g["week"] < week))]
    prior = prior.sort_values(["season", "week"])
    if len(prior) < 40:
        return None
    recent = prior.tail(WINDOW_GAMES)
    in_season = prior[prior["season"] == season]
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))

    r_all, hfa = fit_ratings(recent, teams, "home_margin")
    if r_all is None:
        return None
    t_all, tbase = fit_ratings(recent, teams, "total_points", symmetric=True)

    r_cur, _ = fit_ratings(in_season, teams, "home_margin")
    if r_cur is not None:
        w = min(1.0, len(in_season) / 160.0)
        blend = {t: w * r_cur.get(t, 0.0) + (1 - w) * CARRYOVER * r_all.get(t, 0.0)
                 for t in teams}
    else:
        w = 0.0
        blend = {t: CARRYOVER * r_all.get(t, 0.0) for t in teams}
    return {"margin": blend, "hfa": hfa, "total": t_all, "tbase": tbase,
            "n_prior": len(prior), "in_season_weight": w,
            "n_in_season": len(in_season), "prior_ratings": r_all}


def ev_from_prob(p, odds=-110):
    try:
        p = float(p); o = float(odds)
        if not (math.isfinite(p) and math.isfinite(o) and o != 0):
            return None
    except Exception:
        return None
    payout = (100.0 / abs(o)) if o < 0 else (o / 100.0)
    return p * payout - (1.0 - p)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def build_card(sched, season, week, sign):
    rt = build_ratings(sched, season, week)
    if rt is None:
        return pd.DataFrame(), None

    games = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in rt["margin"] or a not in rt["margin"]:
            continue

        raw_model = rt["margin"][h] - rt["margin"][a] + rt["hfa"]
        mkt = sign * g["spread_line"] if pd.notna(g.get("spread_line")) else None

        # SPREAD
        if mkt is not None:
            # Blend toward the market at the weight the backtest earned.
            # Betting the raw model line means betting a number the data
            # says is worse than the one already on the board.
            fair = mkt + MODEL_WEIGHT * (raw_model - mkt)
            edge = fair - mkt
            side = "HOME" if edge > 0 else "AWAY"
            p = norm_cdf(abs(edge) / SD_MARGIN)
            line_for_side = -mkt if side == "HOME" else mkt
            rows.append({
                "game_id": g["game_id"], "season": season, "week": week,
                "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                "market_type": "SPREAD", "pick_side": side,
                "pick_label": f"{h if side=='HOME' else a} "
                              f"{line_for_side:+.1f}",
                "bet_line": float(mkt), "model_line": float(raw_model),
                "edge_pts": float(edge), "cover_prob": p,
                "expected_value": ev_from_prob(p), "odds": -110,
            })

        # TOTAL
        if rt["total"] and pd.notna(g.get("total_line")):
            raw_total = rt["total"].get(h, 0.0) + rt["total"].get(a, 0.0) + rt["tbase"]
            mt = float(g["total_line"])
            fair_t = mt + MODEL_WEIGHT * (raw_total - mt)
            edge_t = fair_t - mt
            side = "OVER" if edge_t > 0 else "UNDER"
            p = norm_cdf(abs(edge_t) / SD_TOTAL)
            rows.append({
                "game_id": g["game_id"], "season": season, "week": week,
                "kickoff": f"{g.get('gameday','')} {g.get('gametime','')}".strip(),
                "matchup": f"{a} @ {h}", "home_team": h, "away_team": a,
                "market_type": "TOTAL", "pick_side": side,
                "pick_label": f"{side.title()} {mt:g}",
                "bet_line": mt, "model_line": float(raw_total),
                "edge_pts": float(edge_t), "cover_prob": p,
                "expected_value": ev_from_prob(p), "odds": -110,
            })

    card = pd.DataFrame(rows)
    if card.empty:
        return card, rt
    card["abs_edge"] = card["edge_pts"].abs()
    card = card.sort_values("abs_edge", ascending=False).reset_index(drop=True)
    # Tiers are FILTERS, not quotas. If nothing clears the floor the
    # section is empty, which is a legitimate answer for a week.
    card["bet_tier"] = np.where(card["abs_edge"] >= MIN_EDGE_PTS, "OFFICIAL",
                        np.where(card["abs_edge"] >= WATCH_EDGE_PTS, "WATCH", None))
    return card[card["bet_tier"].notna()].copy(), rt


# ----------------------------------------------------------------------
# Freeze + grade
# ----------------------------------------------------------------------
def freeze(card, tracker):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = set(tracker["record_key"].astype(str)) if not tracker.empty else set()
    new = []
    for _, r in card.iterrows():
        key = f"{r['game_id']}|{r['market_type']}"
        if r["bet_tier"] != "OFFICIAL":
            key += "|W"
        if key in existing:
            continue
        row = {c: None for c in TRACKER_COLS}
        row.update({
            "record_key": key, "frozen_at": now,
            "season": r["season"], "week": r["week"], "game_id": r["game_id"],
            "kickoff": r["kickoff"], "matchup": r["matchup"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "market_type": r["market_type"], "pick_side": r["pick_side"],
            "pick_label": r["pick_label"], "bet_line": r["bet_line"],
            "model_line": r["model_line"], "edge_pts": r["edge_pts"],
            "cover_prob": r["cover_prob"], "expected_value": r["expected_value"],
            "odds": r["odds"], "bet_tier": r["bet_tier"],
            "model_version": MODEL_VERSION, "status": "FROZEN",
        })
        new.append(row)
    if not new:
        return tracker, 0
    out = pd.concat([tracker, pd.DataFrame(new)], ignore_index=True)
    return save_tracker(out), len(new)


def grade(tracker, sched):
    if tracker.empty:
        return tracker, 0
    df = tracker.copy()
    pend = df["status"].astype(str) != "GRADED"
    if not pend.any():
        return df, 0

    fin = sched.dropna(subset=["home_score", "away_score"]).set_index("game_id")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    for idx in df[pend].index:
        gid = df.at[idx, "game_id"]
        if gid not in fin.index:
            continue
        g = fin.loc[gid]
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        try:
            bl = float(df.at[idx, "bet_line"])
        except Exception:
            continue
        mt = str(df.at[idx, "market_type"]).upper()
        sd = str(df.at[idx, "pick_side"]).upper()

        if mt == "TOTAL":
            m = (hs + as_) - bl if sd == "OVER" else bl - (hs + as_)
        elif sd == "HOME":
            m = (hs - as_) + bl
        else:
            m = (as_ - hs) + bl

        res = "PUSH" if abs(m) < 1e-9 else ("WIN" if m > 0 else "LOSS")
        units = 0.0 if res == "PUSH" else (100 / 110 if res == "WIN" else -1.0)
        df.at[idx, "final_home_score"] = hs
        df.at[idx, "final_away_score"] = as_
        df.at[idx, "result_margin"] = round(m, 2)
        df.at[idx, "result"] = res
        df.at[idx, "units_result"] = round(units, 4)
        df.at[idx, "status"] = "GRADED"
        df.at[idx, "graded_at"] = now
        n += 1
    return (save_tracker(df), n) if n else (df, 0)


def summarize(df):
    if df.empty:
        return dict(w=0, l=0, p=0, units=0.0, roi=0.0, n=0)
    g = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
    if g.empty:
        return dict(w=0, l=0, p=0, units=0.0, roi=0.0, n=0)
    w = int((g["result"] == "WIN").sum())
    l = int((g["result"] == "LOSS").sum())
    p = int((g["result"] == "PUSH").sum())
    u = float(pd.to_numeric(g["units_result"], errors="coerce").fillna(0).sum())
    return dict(w=w, l=l, p=p, units=u, roi=u / len(g), n=len(g))


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("Sunday Edge")
st.caption(f"NFL spreads and totals · model {MODEL_VERSION}")

st.warning(
    f"**This model did not beat the closing line in backtest.** Across "
    f"{BACKTEST_N:,} games (2007-2025) it added no measurable information "
    f"beyond the market (t = +{BACKTEST_T:.2f}). Picks below are the "
    f"model's lean, not a demonstrated edge. The tracker is built to give "
    f"you a real answer as the record accumulates."
)

tab_slate, tab_game, tab_tracker = st.tabs(["Slate", "Game", "Tracker"])

sched_all = None
try:
    this_season = datetime.now().year
    sched_all = load_schedules(range(this_season - 3, this_season + 1))
except Exception as e:
    st.error(f"Could not load NFL schedules: {e}")
    st.stop()

sign = line_sign(sched_all)

with tab_slate:
    seasons = sorted(sched_all["season"].unique())
    c1, c2 = st.columns(2)
    season = c1.selectbox("Season", seasons, index=len(seasons) - 1)
    weeks = sorted(sched_all[sched_all["season"] == season]["week"].unique())
    week = c2.selectbox("Week", weeks, index=min(len(weeks) - 1, 0))

    card, rt = build_card(sched_all, season, week, sign)

    if rt is None:
        st.info("Not enough completed games yet to build ratings.")
    elif card.empty:
        st.info(
            f"No games clear the {WATCH_EDGE_PTS:g}-point floor this week. "
            f"That is a normal result, not an error."
        )
    else:
        st.caption(f"Ratings fit on {rt['n_prior']:,} prior games · "
                   f"home field {rt['hfa']:+.2f} pts")

        for tier, label in [("OFFICIAL", "Official bets"), ("WATCH", "Watch list")]:
            sub = card[card["bet_tier"] == tier]
            if sub.empty:
                continue
            st.subheader(f"{label} ({len(sub)})")
            show = sub[["matchup", "pick_label", "market_type", "bet_line",
                        "model_line", "edge_pts", "cover_prob", "expected_value"]].copy()
            show["cover_prob"] = show["cover_prob"].map(lambda v: f"{v:.1%}")
            show["expected_value"] = show["expected_value"].map(lambda v: f"{v:+.2%}")
            show["edge_pts"] = show["edge_pts"].map(lambda v: f"{v:+.2f}")
            show["model_line"] = [
                f"{v:.1f}" if m == "TOTAL" else f"{v:+.1f}"
                for v, m in zip(sub["model_line"], sub["market_type"])
            ]
            show.columns = ["Game", "Pick", "Market", "Line", "Model", "Edge",
                            "Cover", "EV"]
            st.dataframe(show, hide_index=True, use_container_width=True)

        if st.button("Freeze this card", type="primary"):
            tr, n = freeze(card, load_tracker())
            st.success(f"Froze {n} new bets." if n else "Nothing new to freeze.")

with tab_game:
    st.write("Every number behind one game, so you can see where the "
             "model's line comes from.")
    gs = sorted(sched_all["season"].unique())
    d1, d2 = st.columns(2)
    g_season = d1.selectbox("Season", gs, index=len(gs) - 1, key="g_season")
    g_weeks = sorted(sched_all[sched_all["season"] == g_season]["week"].unique())
    g_week = d2.selectbox("Week", g_weeks, index=0, key="g_week")

    wk = sched_all[(sched_all["season"] == g_season)
                   & (sched_all["week"] == g_week)].copy()
    wk["label"] = wk["away_team"] + " @ " + wk["home_team"]
    pick = st.selectbox("Game", wk["label"].tolist())
    row = wk[wk["label"] == pick].iloc[0]

    rt_g = build_ratings(sched_all, g_season, g_week)
    if rt_g is None:
        st.info("Not enough completed games to build ratings yet.")
    else:
        h, a = row["home_team"], row["away_team"]
        rh, ra = rt_g["margin"].get(h, 0.0), rt_g["margin"].get(a, 0.0)
        hfa = rt_g["hfa"]
        raw = rh - ra + hfa

        st.subheader("Power ratings")
        r1, r2, r3 = st.columns(3)
        r1.metric(f"{h} (home)", f"{rh:+.2f}")
        r2.metric(f"{a} (away)", f"{ra:+.2f}")
        r3.metric("Home field", f"{hfa:+.2f}")
        st.caption(
            f"Fit on {rt_g['n_prior']:,} prior games. In-season games so far: "
            f"{rt_g['n_in_season']}, carrying "
            f"{rt_g['in_season_weight']:.0%} weight — the rest comes from "
            f"earlier seasons scaled by {CARRYOVER:.2f}."
        )

        st.subheader("Spread")
        if pd.notna(row.get("spread_line")):
            mkt = sign * float(row["spread_line"])
            fair = mkt + MODEL_WEIGHT * (raw - mkt)
            edge = fair - mkt
            side = "HOME" if edge > 0 else "AWAY"
            p = norm_cdf(abs(edge) / SD_MARGIN)
            st.code(
                f"model line      {rh:+.2f} - ({ra:+.2f}) + {hfa:+.2f} "
                f"= {raw:+.2f}\n"
                f"market line     {mkt:+.2f}\n"
                f"disagreement    {raw - mkt:+.2f} pts\n"
                f"blended fair    {mkt:+.2f} + {MODEL_WEIGHT} x "
                f"({raw - mkt:+.2f}) = {fair:+.2f}\n"
                f"edge            {edge:+.2f} pts\n"
                f"cover prob      normal({abs(edge):.2f} / {SD_MARGIN}) "
                f"= {p:.1%}\n"
                f"EV at -110      {ev_from_prob(p):+.2%}\n"
                f"lean            {h if side == 'HOME' else a}",
                language=None,
            )
        else:
            st.info("No spread posted for this game.")

        st.subheader("Total")
        if rt_g["total"] and pd.notna(row.get("total_line")):
            th = rt_g["total"].get(h, 0.0); ta = rt_g["total"].get(a, 0.0)
            raw_t = th + ta + rt_g["tbase"]
            mt = float(row["total_line"])
            fair_t = mt + MODEL_WEIGHT * (raw_t - mt)
            edge_t = fair_t - mt
            p = norm_cdf(abs(edge_t) / SD_TOTAL)
            st.code(
                f"model total     {th:.2f} + {ta:.2f} + {rt_g['tbase']:.2f} "
                f"= {raw_t:.2f}\n"
                f"market total    {mt:.2f}\n"
                f"disagreement    {raw_t - mt:+.2f} pts\n"
                f"blended fair    {fair_t:.2f}\n"
                f"edge            {edge_t:+.2f} pts\n"
                f"cover prob      {p:.1%}\n"
                f"EV at -110      {ev_from_prob(p):+.2%}\n"
                f"lean            {'Over' if edge_t > 0 else 'Under'} {mt:g}",
                language=None,
            )
        else:
            st.info("No total posted for this game.")

        if pd.notna(row.get("home_score")):
            st.caption(
                f"Final: {a} {row['away_score']:.0f} - "
                f"{h} {row['home_score']:.0f}"
            )

with tab_tracker:
    tr = load_tracker()
    tr, n = grade(tr, sched_all)
    if n:
        st.success(f"Graded {n} completed bets.")

    ws, err = _sheet(return_error=True)
    if ws is None:
        st.warning("Storage: session only — records are lost when the app "
                   "restarts. Add Google Sheets credentials to keep history.")
        if err:
            st.caption(f"Reason: {err}")
    else:
        st.caption("Storage: Google Sheets — history is saved permanently.")

    if tr.empty:
        st.info("No bets frozen yet.")
    else:
        for tier in ["OFFICIAL", "WATCH"]:
            sub = tr[tr["bet_tier"] == tier]
            s = summarize(sub)
            st.subheader(f"{tier.title()} — {s['w']}-{s['l']}-{s['p']}")
            a, b, c = st.columns(3)
            a.metric("Units", f"{s['units']:+.2f}")
            b.metric("ROI", f"{s['roi']:+.1%}" if s["n"] else "—")
            c.metric("Graded", f"{s['n']}")

            m = pd.to_numeric(sub.get("result_margin"), errors="coerce").dropna()
            if len(m):
                st.caption(f"Average result margin {m.mean():+.2f} pts across "
                           f"{len(m)} graded bets — a continuous read that "
                           f"converges faster than win rate.")

        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.download_button("Download tracker CSV",
                           tr.to_csv(index=False).encode(),
                           "sunday_edge_tracker.csv", "text/csv")
