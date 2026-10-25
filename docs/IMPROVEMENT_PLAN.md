# NFL model: where the edge can come from

_Written 2026-10-25. Scope: spreads and totals first, with moneylines and props later if a model can be shown to work._

## What the first backtest says

`backtest.py` runs the new recency-weighted EPA + QB ratings (`nfl_power.py`) week by week from 2013 through week 7 of 2026. Each projection uses only data from before that week and the quarterback listed as the game's starter. Everything is scored against the **closing** line.

| Spreads, 2022-2026 held out (1,233 games) | Value |
|---|---|
| Slope on (model − close) | −0.14 (t −1.31) |
| ATS, edge ≥ 3 | 189-188 (50.1%) |
| MAE: model / close | 10.00 / 9.51 |

| Totals, 2022-2026 held out | Value |
|---|---|
| Slope on (model − close) | +0.04 (t +0.33) |
| O/U, edge ≥ 3 | 212-223 (48.7%) |
| MAE: model / close | 10.67 / 10.21 |

Across all 72 swept configurations and every season split, no setting gets close to beating the closing line. This matches the app's existing margin model (+0.099, t 1.19). **Ratings built from public play-by-play are already priced into the close.** That is the expected result, not a bug. The close is the sharpest number in sports betting, and nflverse EPA is the most widely used public dataset there is.

What the sweep did teach:

- Recency matters, though less than I expected. A 20-week half-life with an 8-week offseason gap beats both a 4-week half-life and no decay.
- The QB term helps a little (spread MAE about 0.06 better on held-out seasons).
- Weeks 17 and 18 have the largest model-vs-market gaps because of resting starters. The model should never bet them without availability data.

## Where money is actually made

1. **Bet before the market finishes moving, and measure CLV.** A model doesn't need to beat the close. It needs to agree with the close before the close exists. The real test is whether the model's picks beat the line you got compared with the closing line, and Sunday Edge already captures closing lines. **Next step:** backtest against early-week lines. nflverse only has closing numbers, so this needs historical odds. The Odds API sells historical snapshots back to mid-2020 on paid plans. Alternatively, the app can start logging Tuesday and Wednesday lines from now on, which is free but takes a season to build up.
2. **Line shopping.** At -110 you need 52.4% to break even. Taking the best of 4 or 5 books routinely gains half a point on spreads, which is worth more than any rating tweak. The app's `best_offer` already does part of this.
3. **Information the close sees but the ratings don't.** That means injuries beyond QB (OL and secondary clusters), rest and travel, and wind for totals. The app has injury and weather modules. They should be scored in `backtest.py` on the same closing-line test before they're trusted.
4. **Softer markets.** Player props, alt lines and team totals are lower-limit and less efficient. Props are where public-data models have the best real track record (yardage and reception projections from snap and target share). This is the most promising "other market" once the spread/total harness is settled.

## Bankroll sizing

- Size with fractional Kelly on the **shrunk** edge: market probability plus b times (model − market), where b comes from the backtest. b is currently ~0, so the honest stake against the close is zero.
- Until about 300 live bets show positive CLV, use flat 0.5–1% units, and treat the tracker's CLV as the go/no-go signal rather than win rate.
- Cap any one game at 2% and any one week at 10%. NFL outcomes are correlated within a slate (weather, officiating trends, market-wide moves).

## Proposed order

1. ✅ Fast walk-forward harness vs closing lines (`nfl_data.py`, `nfl_power.py`, `backtest.py`, CI workflow).
2. Decide on historical opening/early-week lines (paid Odds API history, or start logging now), then score the model on CLV instead of closing ATS.
3. Port the app's injury, rest and wind adjustments into the harness and keep only what survives.
4. Wire `nfl_power` into Sunday Edge as the rating, with fractional Kelly on the shrunk edge.
5. Props model (receiving/rushing yards) using the same walk-forward discipline.

## Update: weather, injuries and props (same day)

**Weather, rest and injuries** (`market_tests.py`, residual vs close, split into 2006-2015 and 2016-2026):

- Wind is the only consistent signal: about −0.15 total points per mph beyond the closing total, with t ≈ −2.5 and −2.9. nflverse wind is *observed* at kickoff, though, so this overstates what forecasts can deliver. It needs a re-test on archived forecasts (open-meteo is blocked in the cloud sandbox).
- Injury loads move margins about 1 point per missing starter, and the close already prices about 85% of that. What's left flips sign between halves.
- Rest, byes, division games, QB changes and weeks 17-18 show nothing consistent.

**Props** (`nfl_props.py`) projects receiving yards, receptions and rushing yards from target/carry share × team volume × shrunk efficiency, then adjusts for opponent and the market's implied team total, walk-forward. On held-out 2022-2026 it beats the naive baselines:

| Market | MAE model | Season avg | Last 4 |
|---|---|---|---|
| Receiving yards | 25.1 | 26.9 | 27.3 |
| Receptions | 1.78 | 1.90 | 1.92 |
| Rushing yards | 25.9 | 27.1 | 27.4 |

Beating naive averages is necessary but not enough, because books beat them too. The real test is against prop lines, and there's no free history of those. `log_lines.py` plus `.github/workflows/log_lines.yml` snapshot game lines and props from The Odds API every Tuesday and Sunday into `data/lines/`. That builds the props test set and an opener-vs-close history for game-line CLV at the same time, at roughly 400 credits a month.

`python nfl_props.py --week N` prints this week's projections, with mean and median, for unplayed games. It does not yet know about inactives, so check the injury report.
