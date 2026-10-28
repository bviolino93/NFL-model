"""
Underdog moneylines: is there anything to bet?

Every test uses nflverse's CLOSING moneylines and spreads, 2010 on, and
bets $1 on the underdog's moneyline when the rule says so. Win chances come
from the same empirical margin distribution the live selector uses
(selector.OutcomeModel), rebuilt each season from earlier seasons only.

  1. all dogs       every underdog, by price band (the baseline: the
                    favourite-longshot bias says long dogs lose most)
  2. spread vs ML   the dog's win chance implied by the closing SPREAD
                    against its moneyline price; bet when that says +EV.
                    This is what the live selector does with "market" alone.
  3. model nudge    the live spread adjustment (a fifth of the ratings +
                    injuries gap to the line, capped at a point) moves the
                    margin first; bet the dog when the nudged margin makes
                    its price +EV. The ratings use the starting QB, which a
                    live run only knows from the projected starter, so this
                    is a little optimistic.
  4. model full     the ratings + injuries margin taken at face value.

    python moneyline_research.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import selector as sel
from nfl_data import load_games

INJ_OFF_PTS, INJ_DEF_PTS = -0.78, -0.94     # paper.py
NUDGE_WEIGHT, NUDGE_CAP = 0.2, 1.0          # picks.py RATINGS_SPREAD_*
BANDS = [(100, 150), (150, 250), (250, 400), (400, 2000)]


def dog_frame(games: pd.DataFrame, first: int = 2010) -> pd.DataFrame:
    """One row per graded game with a clear underdog at the close."""
    g = games[(games.season >= first) & games.home_moneyline.notna()
              & games.away_moneyline.notna() & games.home_score.notna()
              & games.spread_line.notna()].copy()
    g["margin"] = g.home_score - g.away_score
    home_dog = g.home_moneyline > g.away_moneyline
    g["dog_home"] = home_dog
    g["dog_price"] = np.where(home_dog, g.home_moneyline, g.away_moneyline)
    g["fav_price"] = np.where(home_dog, g.away_moneyline, g.home_moneyline)
    g = g[g.dog_price >= 100].copy()
    # +1 dog won, 0 tie (push), -1 lost
    g["dog_result"] = np.sign(g.margin) * np.where(g.dog_home, 1, -1)
    return g


def profit(price, result) -> np.ndarray:
    price = np.asarray(price, float)
    result = np.asarray(result, float)
    return np.where(result > 0, price / 100.0, np.where(result < 0, -1.0, 0.0))


def add_probs(g: pd.DataFrame, centre_col: str, out: str) -> pd.DataFrame:
    """Dog's P(win), P(tie) when the home margin is centred on centre_col,
    from games in earlier seasons only."""
    pw = np.full(len(g), np.nan)
    pt = np.full(len(g), np.nan)
    allg = load_games()
    allg = allg[allg.home_score.notna() & allg.spread_line.notna()]
    for s in sorted(g.season.unique()):
        past = allg[(allg.season < s) & (allg.season >= 1999)]
        mm = sel.OutcomeModel(past.spread_line.to_numpy(float),
                              (past.home_score - past.away_score).to_numpy(float))
        idx = np.where(g.season.to_numpy() == s)[0]
        for i in idx:
            r = g.iloc[i]
            ph, ptie = mm.probs(float(r[centre_col]), 0.0)
            pw[i] = (1 - ph - ptie) if not r.dog_home else ph
            pt[i] = ptie
    g[out] = pw
    g[out + "_tie"] = pt
    return g


def ev(g: pd.DataFrame, p: str) -> np.ndarray:
    b = g.dog_price.to_numpy(float) / 100.0
    pw, pt = g[p].to_numpy(float), g[p + "_tie"].to_numpy(float)
    return pw * b - (1 - pw - pt)


def record(g: pd.DataFrame, mask, label: str) -> dict:
    x = g[mask]
    pr = profit(x.dog_price, x.dog_result)
    n = len(x)
    return {"rule": label, "bets": n,
            "won": int((x.dog_result > 0).sum()),
            "win%": (x.dog_result > 0).mean() if n else np.nan,
            "avg price": x.dog_price.mean() if n else np.nan,
            "roi": pr.mean() if n else np.nan,
            "roi se": pr.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan}


def eras(g: pd.DataFrame, mask, label: str) -> list[dict]:
    out = []
    for lo, hi in ((2010, 2015), (2016, 2021), (2022, 2026)):
        m = mask & (g.season >= lo) & (g.season <= hi)
        out.append(record(g, m, f"{label} {lo}-{str(hi)[2:]}"))
    out.append(record(g, mask, f"{label} all"))
    return out


def main(frame_path: str | None = None):
    pd.set_option("display.width", 200)
    g = dog_frame(load_games())
    rows = []
    for lo, hi in BANDS:
        m = (g.dog_price >= lo) & (g.dog_price < hi)
        rows += eras(g, m, f"all dogs +{lo}..+{hi}")
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    # The closing spread as the fair centre (the live consensus does the
    # same from every book's de-vigged spread price).
    g = add_probs(g, "spread_line", "p_spread")
    g["ev_spread"] = ev(g, "p_spread")
    rows = []
    for cut in (0.0, 0.02, 0.05):
        rows += eras(g, g.ev_spread >= cut, f"spread>ML ev>={cut:.2f}")
    print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

    if frame_path:
        fr = pd.read_parquet(frame_path)
        fr["mine"] = (fr.model_margin + INJ_OFF_PTS * fr.inj_off
                      + INJ_DEF_PTS * fr.inj_def)
        g = g.merge(fr[["game_id", "mine"]], on="game_id", how="inner")
        g["nudged"] = g.spread_line + np.clip(
            NUDGE_WEIGHT * (g.mine - g.spread_line), -NUDGE_CAP, NUDGE_CAP)
        g = add_probs(g, "nudged", "p_nudge")
        g = add_probs(g, "mine", "p_model")
        g["ev_nudge"] = ev(g, "p_nudge")
        g["ev_model"] = ev(g, "p_model")
        # Does the model lean to the dog at all?
        lean = np.where(g.dog_home, 1, -1) * (g.mine - g.spread_line)
        rows = []
        for cut in (0.02, 0.05):
            rows += eras(g, g.ev_nudge >= cut, f"nudge ev>={cut:.2f}")
        for cut in (0.02, 0.10, 0.20):
            rows += eras(g, g.ev_model >= cut, f"model ev>={cut:.2f}")
        for cut in (2, 3, 4):
            rows += eras(g, lean >= cut, f"model likes dog by {cut}+")
            rows += eras(g, (lean >= cut) & (g.dog_price < 250),
                         f"  ...and price<+250")
        print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

        # 5. The live design: the moneyline market's own de-vigged price
        # is the fair chance, and the model nudge only moves it by the
        # change in win chance the nudge makes on the margin distribution.
        # One closing book, so the only edge left is the nudge itself.
        pd_ = sel.american_to_prob
        q_d = g.dog_price.map(pd_)
        q_f = g.fav_price.map(pd_)
        g["p_ml"] = q_d / (q_d + q_f)
        g["p_live"] = g.p_ml + (g.p_nudge - g.p_spread)
        g["p_live_tie"] = g.p_nudge_tie
        g["ev_live"] = ev(g, "p_live")
        rows = []
        for cut in (0.0, 0.02):
            rows += eras(g, g.ev_live >= cut, f"ML fair+nudge ev>={cut:.2f}")
        print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

        # Calibration: what each method says dogs win vs what they did.
        g["won"] = (g.dog_result > 0).astype(float)
        g["band"] = pd.cut(g.dog_price, [99, 150, 250, 400, 2000])
        print(g.groupby("band", observed=True).agg(
            games=("won", "size"), won=("won", "mean"),
            p_ml=("p_ml", "mean"), p_spread=("p_spread", "mean"),
            p_model=("p_model", "mean")).round(3).to_string())
        g.to_parquet(frame_path.replace(".parquet", "_dogs.parquet"))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
