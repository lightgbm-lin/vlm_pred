# Volume factor research log

## Current model
- Features: `vlm_1_ratio` + `h001_vlm_5_ratio`, `h001_vlm_21_ratio` + `h002_opex`, `h002_russell`,
  `h002_half_day`, `h002_holiday_adj`, `h002_month_end`, `h002_dow`, `h002_prev_special`
  + `h004_y_q_tight`, `h004_y_q_wide`, `h004_y_yr_tight`, `h004_y_yr_wide`
  + `h005_absret_z`, `h005_range_z`, `h005_gap_z` + `h006_idio_vlm_1_ratio`, `h006_idio_vlm_5_ratio`
- Selected candidates (see candidates/selected.txt): h001_vlm_term_structure, h002_calendar_schedule,
  h004_earnings_seasonality, h005_price_shock, h006_idio_volume_ratio
- Walk-forward R² (2002–2014, ins only): 0.27353  (t = 736.1)
- Baseline history: `vlm_1_ratio` only = 0.23088 (t = 657.3)

## Data notes
- `price_close` is split-adjusted (the ratio `price_adj / price_close` moves smoothly; only ~19 jumps
  above 20%, and these look like spin-offs such as KSU in 2000-07 and FMC in 2002-01). Volume is probably
  split-adjusted too, but this hasn't been checked directly. If so, splits aren't a mechanical volume effect, but
  spin-off dates are.
- Unconditional median y by calendar bucket: half day −0.65, pre-/post-holiday −0.30/−0.24, quad
  witch +0.20, opex +0.01, Russell recon +0.05, month end −0.07 (vs −0.12 overall), Monday −0.17, Aug/Dec ≈ −0.20.

## Tested hypotheses

### H001 — Volume term structure (5d / 21d halflife ratios) — ACCEPTED
- File: `candidates/h001_vlm_term_structure.py` · columns: `h001_vlm_5_ratio`, `h001_vlm_21_ratio`
- Hypothesis / mechanism: volume has a slow regime component (weeks) on top of transient spikes;
  weekly and monthly EWMA ratios vs the 60d EWMA separate a persistent level shift from a 1-day spike.
- Result: R² 0.23088 → 0.23257 (Δ +0.00169), years improved 12/13, gain share 0.026 + 0.026
- Notes: gains are small but steady. The only losing year is 2002, which had just one year of training data.
  Halflives were picked from the mechanism (week and month), with no sweep.

### H002 — Scheduled calendar events — ACCEPTED
- File: `candidates/h002_calendar_schedule.py` · columns: `h002_opex` (0/1/2 = quad witch),
  `h002_russell`, `h002_half_day`, `h002_holiday_adj`, `h002_month_end`, `h002_dow`,
  `h002_prev_special` (T-1 was an opex, Russell, or half-day session)
- Hypothesis / mechanism: on some days, volume is set by the schedule (derivative expiry, index
  rebalances, short sessions, holidays, weekday), not by information flow. The EWMA features can't see this.
  When T-1 was such a day, `vlm_1_ratio` is distorted in a way that doesn't persist.
- Result: R² 0.23257 → 0.26034 (Δ +0.02777), years improved 13/13, gain share: half_day 0.068,
  dow 0.038, opex 0.020, holiday_adj 0.011, prev_special 0.010, month_end 0.005, russell 0.002
- Notes: by far the largest gain so far, and it is stable across years (+0.012 to +0.048, growing after
  2011). At 7 columns it's above the ~5-column guideline, but it is all one idea (the calendar). Ablate
  later. `opex`, `holiday_adj`, and `month_end` need the next trading date and are declared
  `KNOWN_IN_ADVANCE`. All columns are functions of the date only.

### H003 — Market-wide volume shock at T-1 and idiosyncratic split — REJECTED
- File: `candidates/h003_market_volume_shock.py` · columns: `h003_mkt_y_1` (sp_weight-weighted
  market y at T-1), `h003_mkt_vlm_1_ratio`, `h003_idio_vlm_1_ratio` (stock minus market)
- Hypothesis / mechanism: common and idiosyncratic volume shocks decay at different rates, and
  `vlm_1_ratio` mixes them.
- Result: R² 0.26034 → 0.26129 (Δ +0.00095), years improved 6/13, gain share 0.049 + 0.035 + 0.011
- Notes: the features have high gain share but inconsistent OOS gains (range −0.0064 to +0.0079 by year).
  Date-level features take only ~250 distinct values per year, so the tree likely fits date-specific
  noise (for example, one market-shock regime in the training data). Rank-corr with y is 0.25, but most of it is
  already in `vlm_1_ratio`. The idiosyncratic column got little gain. Follow-up: H006.

### H004 — Earnings seasonality (spike ~1 quarter / ~1 year ago) — ACCEPTED
- File: `candidates/h004_earnings_seasonality.py` · columns: `h004_y_q_tight` (max y over lags
  62–64), `h004_y_q_wide` (57–69), `h004_y_yr_tight` (251–253), `h004_y_yr_wide` (245–259)
- Hypothesis / mechanism: earnings are the largest recurring stock-specific volume events and come ~63
  trading days apart. Firms report in nearly the same week each year, so the 252-day echo is the most
  precise. EWMA and calendar features can't anticipate stock-specific scheduled events.
- Result: R² 0.26034 → 0.26583 (Δ +0.00549), years improved 12/13, gain share 0.017 / 0.011 /
  0.014 / 0.009 (tight / wide, quarter / year)
- Notes: raw signal is modest (mean y −0.03 → +0.2 across tight-window buckets; median barely moves)
  because 63 trading days is only an approximation of the reporting cycle. Tight windows beat wide ones,
  consistent with scheduled events. The only losing year is 2003 (−0.0007). Follow-ups: an explicit
  "days since the stock's largest spike in the last ~70 days" feature, the 126/189 lags, or a
  spike-alignment score across several past quarters (more robust to one-off spikes).

### H005 — T-1 price shock relative to own volatility — ACCEPTED
- File: `candidates/h005_price_shock.py` · columns: `h005_absret_z` (|ret_{t-1}| / 21d-halflife
  EWMA std through t-2), `h005_range_z` (log(high/low)_{t-1} / its 21d EWMA through t-2),
  `h005_gap_z` (|log open_{t-1}/close_{t-2}| / the same vol)
- Hypothesis / mechanism: mixture of distributions. Volume that comes with a big price move signals
  information that keeps being traded, while volume without a move is flow that reverts.
- Result: R² 0.26583 → 0.27220 (Δ +0.00637), years improved 13/13, gain share 0.018 / 0.017 / 0.006
- Notes: confirmed before the run. Given `vlm_1_ratio` > 1, median y_t rises from 0.51 to 0.98 across
  |ret| z-score quartiles. The gain is positive every year (+0.0015 to +0.0106). The gap column gets the least
  gain. Some extreme values (gap_z up to 102) suggest occasional bad open prices, which trees tolerate.

### H006 — Idiosyncratic volume ratio (stock minus market), no market-level column — ACCEPTED
- File: `candidates/h006_idio_volume_ratio.py` · columns: `h006_idio_vlm_1_ratio`,
  `h006_idio_vlm_5_ratio` (stock ratio minus the lagged-sp_weight-weighted market mean on the same date)
- Hypothesis / mechanism: follow-up to H003. Idiosyncratic shocks revert faster than common ones.
  Dropping the date-level columns avoids fitting date-specific noise.
- Result: R² 0.27220 → 0.27353 (Δ +0.00134), years improved 10/13, gain share 0.012 + 0.005
- Notes: passes the rule, with small gains spread across years. Losing years 2003/2007/2012 are all ≥ −0.0013.
  Evidence is weaker than for H004/H005, and this is a second try at the market-shock idea (selection-bias
  risk), so it's a prime candidate for a later ablation. The market level itself still isn't in the model.

## Backlog
- **Ablation pass**: drop H006 and individual H002/H004 columns (russell, month_end, wide windows) and
  check whether R² holds. Several columns have gain share < 0.006.
- **Days since last spike / decay after spikes**: shape of post-event decay vs the EWMA. Also serves as
  an earnings-cycle phase feature (follow-up to H004).
- **Earnings alignment across quarters**: e.g. mean or min of the tight-window max y at lags 63, 126,
  189, 252. A spike that recurs in every quarter is almost surely earnings (H004 follow-up).
- **Return sign asymmetry**: signed ret_{t-1} z-score, or 5-day cumulative return / vol (disposition
  effect, attention to losers). Not a monotone transform of `h005_absret_z`, because the sign is new.
- **Volatility regime**: 5d vs 60d realized-vol ratio (the price analogue of H001).
- **Calendar ablation / extensions**: S&P quarterly rebalance vs opex, Dec/Aug seasonality (`month`),
  the week between Christmas and New Year, and the day after Thanksgiving vs other half days. Also check
  whether `h002_russell` (gain 0.002) adds anything.
- **Index flows**: change in lagged `sp_weight`, and first days in the universe (short history).
- **Heterogeneity**: lagged `sp_weight` (size) as a conditioning variable for persistence.
