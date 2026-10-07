# Volume factor research log

## Current model
- Features: baseline features (`vlm_pred/feature.py`, 40 columns). H007, H009 and H010 were moved into the
  baseline (2026-10-05) as `enrich_earnings_schedule` (`eday_prob_13w`, `eday_prob_sum`, `eday_prob_gap_sum`),
  `enrich_price_path` (`ret_20d`, `ret_60d`, `pos_52w`) and `enrich_overnight_gap` (`gap_z_1`, `intra_z_1`).
  Outputs identical to the candidates; only the `hNNN_` prefixes were dropped.
- Selected candidates (see candidates/selected.txt): none
- Walk-forward R² (2002–2009, train only): 0.30331  (t = 609.3)
  - by year: 2002 .2911 · 2003 .2234 · 2004 .2539 · 2005 .2564 · 2006 .2509 · 2007 .3286 · 2008 .4065 · 2009 .3812
- History: baseline only 0.28774 (t = 586.9; top gains y_1 .494, vlm_1_ratio .226, dow .039) → +H003 0.29481
  → H003 replaced by H005 0.29690 → H005 replaced by H007 0.29854 → +H009 0.30025 → +H010 0.30331

## Tested hypotheses

### H001 — market-wide volume surprise — REJECTED
- File: `candidates/h001_mkt_vlm_surprise.py` · columns: `h001_mkt_y_1`, `h001_mkt_y_ewm5`, `h001_idio_y_1`
- Hypothesis / mechanism: split y_1 into a persistent common component (sp_weight-weighted cross-sectional
  mean of y at t-1, plus its 5d EWMA) and a fast-decaying idiosyncratic residual.
- Result: R² 0.28774 → 0.28507 (Δ −0.00267), years improved 2/8, gain share mkt_y_1 .039, mkt_y_ewm5 .025, idio .007
- Notes: high gain share yet clearly worse OOS in 6/8 years → overfitting. Date-constant features take a
  distinct value per date, so trees (min_child_samples=20, ~450 rows/date) can isolate individual training
  dates and memorize date fixed effects. Lesson: be wary of continuous date-level features; prefer
  stock-relative versions (e.g. stock vs market) or coarse regimes.

### H002 — intraday high-low range vs own norm — REJECTED
- File: `candidates/h002_intraday_range.py` · columns: `h002_range_1_rel`, `h002_range_5_rel`
- Hypothesis / mechanism: log(high/low) at t-1 (and 5d EWMA) / own 60d EWMA range − 1. Range is an
  efficient vol estimator and flags intraday disagreement that |ret_1| misses (mixture-of-distributions).
- Result: R² 0.28774 → 0.28806 (Δ +0.00032), years improved 5/8, gain share range_1 .009, range_5 .005
- Notes: small, mostly positive 2002–2006, negative 2007/2009. Spearman 0.27 with y but 0.45 with y_1 — the
  information is largely already in y_1/vlm_1_ratio. Below both thresholds.

### H003 — quarterly earnings schedule (13/26/39-week spike lags) — ACCEPTED, later REPLACED by H005
- Replaced: H005 refines the spike definition and beats H003 head-to-head (see H005); removed from selected.txt.
- File: `candidates/h003_quarterly_earnings_schedule.py` · columns: `h003_eday_prob_13w`, `h003_eday_prob_sum`
- Hypothesis / mechanism: firms report on a ~13-week cycle; last quarter's max-y spike is a timelier, first-year-
  available predictor of the next earnings day than the baseline's 52-week lag, and agreement across lags adds
  confidence. Same triangular ±2 bday pmf as baseline; `_sum` = 13w + 26w + 39w + baseline 52w probabilities.
  Window dates on/before the spike's quarter end are dropped (quarter max not yet known → look-ahead).
- Motivation: days with y > 1.5 are 2.6% of rows but ~58% of weighted y variance; baseline flags 12% of them,
  the consensus flags 32%. Recurrence hit rate (big day within ±2 bdays): 13w .167, 26w .158, 39w .152, 52w .188.
- Result: R² 0.28774 → 0.29481 (Δ +0.00707), years improved 8/8, gain share prob_sum .018, prob_13w .003
- Notes: largest gains 2004/2005 (+.014), 2007 (+.010), 2009 (+.010). Mean y by prob_sum: 0 → .014,
  (.4,.7] → .36, (.7,1] → .89, >1 → 1.77. Earnings-day prediction is the richest vein so far.
- Look-ahead verification (added later): targeted truncation check on the last 3 trading days of every quarter
  2001–2009 (108 dates, 45k rows), where a quarter-max leak would land: 0 mismatches. Control with the
  quarter-end filter removed: 13 dates / 15 rows mismatch, so the check has power and the filter is necessary.
  The trailing-max alternative needing no filter was tested as H004 and scored lower.

### H004 — earnings schedule with trailing-max spikes (H003 variant) — REJECTED
- File: `candidates/h004_trailing_max_earnings_schedule.py` · columns: `h004_eday_prob_13w`, `h004_eday_prob_sum`
- Hypothesis / mechanism: as H003, but a spike is a day with y > 1 that is the max of the trailing 63 trading
  days, which is known on the spike day itself. That removes H003's quarter-end filter, and spikes are no
  longer tied to calendar quarters. Same lags, pmf, and consensus construction, so only the spike definition differs.
- Spike sets: quarter-max 14.1k, trailing-max 15.6k, 10.8k in both; big-day coverage 32.5% vs H003 31.5%;
  consensus sums correlate 0.84. Extended truncation check (205 dates incl. quarter ends): 0 mismatches.
- Result, as a **replacement** for H003 (h003 temporarily commented out of selected.txt, same baseline-only
  reference as H003): R² 0.28774 → 0.29306 (Δ +0.00533, 7/8 years) vs H003's +0.00707 (8/8).
  Saved as `results/h004_trailing_max_earnings_schedule_vs_baseline.json`.
- Result, **on top of** H003: R² 0.29481 → 0.29529 (Δ +0.00049), years improved 6/8, gain share prob_sum .008
- Notes: H003 wins head-to-head in 6/8 years; H004 is better in 2002 and especially 2006 (+.0091 vs +.0007).
  Incremental gain is just under threshold and the feature is a near-duplicate → reject, keep H003.
  Interpretation: "one biggest spike per calendar quarter" is a cleaner earnings proxy than "every new
  63-day high". The extra trailing-max spikes (secondary events after earnings) add false positives that
  outweigh the windows H003's filter drops.

### H005 — earnings schedule with idiosyncratic spikes + gap-confirmed consensus — ACCEPTED (replaces H003), later REPLACED by H007
- Replaced: H007 adds a one-week shift for projections crossing the annual report and beats H005 head-to-head
  (see H007); removed from selected.txt.
- File: `candidates/h005_idio_gap_earnings_schedule.py` · columns: `h005_eday_prob_13w`, `h005_eday_prob_sum`,
  `h005_eday_prob_gap_sum`
- Hypothesis / mechanism: earnings news is firm-specific and released outside trading hours, so a true earnings
  day has a non-market-wide volume spike and an overnight gap. (1) Each quarter's spike = max of y minus that
  day's cross-sectional median y (still > 1), so market-frenzy days stop winning quarters. (2) Spikes with
  |overnight return| ≥ 2σ (vol_ewm_21; overnight = (1+ret_raw)/(close/open) − 1, split-safe) feed a separate
  consensus column instead of a filter, keeping recall. Lags 13/26/39/52w, all from the new spike definition;
  same pmf and quarter-end filter as H003.
- Pre-test diagnostic on H003's spikes (full train sample, 2000–2009): 13w/52w recurrence 14%/15% for gap < 1σ,
  22–25%/25–29% for gap ≥ 2σ; 18%/20% on normal days vs 11%/9% when market median y ≥ 0.5. Only 42% of spikes
  have gap ≥ 1σ, so a hard filter would cost too much recall. The 2σ cutoff is conventional, but it was chosen after
  seeing this diagnostic, so it is mildly in-sample. No other cutoffs were tried.
- Look-ahead: harness check passed; extended check (205 dates incl. last 3 trading days of each quarter): 0 mismatches.
- Result, as a **replacement** for H003 (vs baseline only): R² 0.28774 → 0.29690 (Δ +0.00916, 8/8 years) vs
  H003's +0.00707. Saved as `results/h005_idio_gap_earnings_schedule_vs_baseline.json`.
  - Ablation without the gap column: Δ +0.00821 (8/8). So the idiosyncratic pick (plus its own 52w lag) is worth
    ≈ +0.0011 over H003, mostly 2009 (+.0202 vs H003 +.0098): 2008 crash days had won many quarters, and H003
    projected them into 2009 as false earnings days. The gap column adds ≈ +0.0010, mostly 2006–2007.
- Result, **on top of** H003: R² 0.29481 → 0.29580 (Δ +0.00099, 5/8 years) — worse than H005 alone, so the two
  are redundant and keeping both adds noise.
- Decision: swap H003 → H005. Current model 0.29481 → 0.29690 (Δ +0.0021), 6/8 years improved (2004 −.0002,
  2008 −.0016; 2007 +.0059, 2009 +.0095). Passes the rule. Caveat: about half the gain vs H003 is in 2009, and
  that gain follows directly from the mechanism (post-crash false spikes), so I'm accepting it, but watch it on VAL.
- Mean y by gap_sum: (.2,.4] → .21 and (.4,.7] → .57, vs .11 / .37 for the unconfirmed consensus at the same
  levels. Gain share prob_sum .015, gap_sum .005, prob_13w .002.

### H006 — second-largest idiosyncratic spike per quarter (low-confidence schedule column) — REJECTED
- File: `candidates/h006_second_spike_schedule.py` · columns: `h006_eday_prob2_sum`
- Hypothesis / mechanism: when a quarter's top idiosyncratic spike wasn't earnings (M&A news, warning, investor
  day), the earnings day is the runner-up. Second spike = max idio y (> 1) more than 5 trading days from the
  quarter's top day, so it isn't the same event's follow-through. Consensus over 13/26/39/52w lags, same pmf and
  quarter-end filter as H005, kept as its own column so the model can weight it lower.
- Pre-test diagnostic: second spikes recur on-cycle (13w/52w) 14%/14% vs 11%/11% at off-cycle placebo lags
  (6w/19w); top spikes 17%/19% vs 7%/9%. So second spikes carry a weak schedule signal (~1.3× placebo vs ~2.4×)
  and mostly flag spike-prone stocks. Design was fixed before this and run unchanged.
- Feature stats: uncorrelated with h005_eday_prob_sum (0.02); raises big-day coverage ~32% → ~45%; where H005 is
  silent, mean y .009 (prob2 = 0) → .054 (.2–.4] → .149 (> .4).
- Result: R² 0.29690 → 0.29693 (Δ +0.00003), years improved 5/8, gain share .004
- Notes: year deltas swing ±.003 (2004 +.0034, 2007 −.0034, 2009 −.0025) with no net gain, which looks like noise.
  The extra coverage is low-precision: the conditional lift is real but small, and spike-proneness is probably already
  in max_y_20 / y lags. Lesson (with H004): adding more, lower-quality spikes doesn't help; precision of the
  spike set matters more than recall. Don't retry top-k variants without a sharper earnings filter.
- Run stopped at the user's request once the result was clear; the extended look-ahead check was not completed.

### H007 — annual-report one-week shift in the earnings schedule (H005 variant) — ACCEPTED (replaces H005)
- File: `candidates/h007_annual_report_shift_schedule.py` · columns: `h007_eday_prob_13w`, `h007_eday_prob_sum`,
  `h007_eday_prob_gap_sum`
- Hypothesis / mechanism: Dec-FY firms report Q4 with/before the audited 10-K in calendar Q1, about a week later
  in the cycle than interim reports. So projections *into* calendar Q1 from another quarter tend to land a week
  late, and projections *out of* Q1 a week early; same-quarter projections (52w) are unaffected. Rule for every lag:
  shift = +7d if the projected centre is in Q1, −7d if the source spike is in Q1, 0 if both or neither. Only some
  firms shift, so crossing projections put 50% on the usual ±2 bday triangle and 50% on the same triangle
  shifted one week (same weekday). Everything else as H005 (idio spikes, gap-confirmed column, quarter-end filter).
- Pre-test diagnostic (H005 spikes, offsets of later spikes from k×91 days, k = 1–4): unaffected pairs peak at 0
  (21%); into-Q1 pairs peak at 0 (10.1%) and +7 (10.0%); out-of-Q1 pairs at 0 (11.9%) and −7 (8.9%); days in
  between stay at ~2–3%. So a split/shift fits, not a wider window (the request was "quarter-dependent
  tolerance"; a wider window would spread mass over the empty days in between). The 7-day shift is the
  mechanism's own unit (same weekday). The 50/50 split was set from the mechanism, with no attempt to tune it.
- Feature stats vs H005: big-day coverage Q1 .27 → .40, Q2 .32 → .39, Q3 .34 → .40, Q4 .33 → .38; mean y by sum
  is sharper at every level ((.2,.4] .112 → .134, (.4,.7] .365 → .432) and lower at 0 (.014 → .009).
- Look-ahead: harness check passed; extended check (205 dates incl. last 3 trading days of each quarter):
  0 mismatches. The −7d shift moves some windows earlier; the quarter-end filter still covers them.
- Result, as a **replacement** for H005 (vs baseline only): R² 0.28774 → 0.29854 (Δ +0.01081, 8/8 years) vs
  H005's +0.00916. Saved as `results/h007_annual_report_shift_schedule_vs_baseline.json`.
- Decision: swap H005 → H007. Current model 0.29690 → 0.29854 (Δ +0.00165), 6/8 years improved (2002 −.0023,
  2005 −.0006; 2007 +.0051, 2009 +.0040, 2006 +.0027, 2004 +.0020). Passes the rule. Gains are spread
  across years, not concentrated in one. The on-top-of-H005 run was skipped: H007 is a strict refinement of the
  same construction, so keeping both would just duplicate it (cf. H003 + H005).

### Residual screen of the H007 model (diagnostic, not a hypothesis)
- Walk-forward predictions of the H007 model, residual = y − WLS fit. Date-level (common across stocks) share of
  weighted residual variance: 6.9%. Worst dates: early closes (see "Ruled out") and crisis days (2008-10-10,
  2007-08-16, 2007-02-27: not predictable from t-1 data).
- Stock-level signals, share of residual variance explained by 20 quantile bins (in-sample, optimistic): 52w-low
  proximity .0015, market y_1 (date-level) .0014, 20d return .0012, overnight gap_1/vol .0011, 60d return .0008,
  52w-high proximity .0007, naive/EWMA250 volume .0004, |market ret_1| .0004, price level .0003, days since
  y > 1 .0002, sp_weight .0002, y_1 − market y_1 .0001, 60d y dispersion .0001. Screening many signals on the
  same data invites selection bias, so each chosen one is still tested once with a mechanism-fixed design.

### H009 — longer-horizon price path (20d/60d returns, 52-week range position) — ACCEPTED
- File: `candidates/h009_price_path.py` · columns: `h009_ret_20d`, `h009_ret_60d`, `h009_pos_52w`
- Hypothesis / mechanism: turnover rises after large past returns (disposition effect, attention trading;
  Statman–Thorley–Vorkink 2006), and stocks near 52-week extremes draw breakout trading, distress selling and
  anchoring. The baseline sees only 5 days of returns; one- to three-month and yearly-range context is new.
  price_adj based; 52w range = rolling 252d max/min through t-1 (min 60 obs).
- Shape (mean y by decile): ret_20d .28 (worst decile) → −.04 (middle) → .08 (best); pos_52w .22 (at the low) →
  ~0 → .10 (at the high). U-shaped, stronger on the downside, as expected.
- Result: R² 0.29854 → 0.30025 (Δ +0.00171), years improved 6/8, gain share ret_20d .014, pos_52w .013, ret_60d .006
- Notes: gains spread across years (2002 +.0039, 2008 +.0039, 2006 +.0023, 2004 +.0019); 2007 −.0012, 2009 −.0002.
  First accepted non-earnings feature. Follow-ups: market-relative versions (idiosyncratic 20d return), or
  the same horizon on volume instead of price (naive vs 250d volume level ranked lower in the screen).

### H010 — overnight gap vs intraday split of ret_1, in vol units — ACCEPTED
- File: `candidates/h010_overnight_gap.py` · columns: `h010_gap_z_1`, `h010_intra_z_1`
- Hypothesis / mechanism: the overnight part of a return reflects news released outside trading hours (a
  discrete information event with lasting volume); intraday moves are more often flow-driven. ret_1 mixes them,
  and price_open was unused. gap = (1+ret_raw)/(close/open) − 1 (split-safe), intra = close/open − 1, both at
  t-1 and divided by vol_ewm_21.
- Pre-test was flat: at fixed |ret_1|/vol, mean y barely varied with the unsigned gap share of the move. It was
  run anyway with the design unchanged.
- Result: R² 0.30025 → 0.30331 (Δ +0.00306), years improved 6/8, gain share intra_z_1 .016, gap_z_1 .005
  (ret_1 fell .017 → .008, so it took over part of ret_1's role)
- Ablation (`ret_1 / vol_ewm_21` alone, no split): Δ +0.00067, 5/8 years. So vol-scaling explains only ~20% of
  the gain, and the split itself adds ~+0.0024. The pre-test was too coarse: the signal is probably in signed
  combinations (e.g. a gap one way reversing intraday), which an unsigned share hides.
- Notes: by year 2002 +.0065, 2005 +.0070, 2007 +.0056, 2006 +.0025; 2003 .0000, 2009 −.0008. Data has some
  extreme z (|z| > 100, probably bad open prices); trees are robust to them, but a cleaned version could help.
  Follow-ups: multi-day gap EWMA (persistent overnight news flow), market-relative gap.

### H011 — overnight news flow: EWMA (hl 5d) of |gap| in vol units — REJECTED
- File: `candidates/h011_gap_ewm.py` · columns: `h011_gap_absz_ewm5`
- Hypothesis / mechanism: a stock that keeps gapping overnight is "in the news" (a running M&A, litigation or
  guidance story); each new item draws traders back. One day's gap (H010) is a noisy measure of that state, so
  use an EWMA with a ~1-week halflife. Day-t |gap| / vol_ewm_21 (vol through t-1), EWMA, shifted to end at t-1.
- Data check: no row has its open outside [low, high]; only 17 rows have |gap| > 20σ (likely real events), so
  no cleaning (the "clean bad opens" idea is moot). Max EWMA 1.5k comes from near-zero vol early in a stock's history.
- Shape: mean y by decile −.03 → .27, monotone; among quiet-yesterday rows (|y_1| < .2) .015 → .061 at the top
  decile. Spearman .40 with vol_ewm_5/vol_ewm_21, .31 with vlm_5_ratio.
- Result: R² 0.30331 → 0.30345 (Δ +0.00015), years improved 5/8; year deltas swing ±.002 (2002 −.0019,
  2008 −.0022, 2004 +.0017) with no net gain.
- Notes: the persistent "in the news" state is already measured better by volume itself (vlm_5/10_ratio, y lags)
  than by a price proxy for it. The gap's value is in the fresh, signed, one-day split (H010), not in its
  persistence. Don't retry other gap halflives.

### H012 — market-relative overnight gap at t-1 — REJECTED
- File: `candidates/h012_idio_gap.py` · columns: `h012_gap_idio_z_1`
- Hypothesis / mechanism: the market-shared part of an overnight gap is macro news (broad, small per-stock volume
  effect); the remainder is company news, which drives the stock's own volume surprise. Stock gap minus the
  sp_weight-weighted market gap that morning (beta fixed at 1), in vol_ewm_21 units, at t-1. Stock-level column only.
- Pre-test (univariate) looked strong: at fixed raw |gap z| 2–4, mean y_t .78 / .61 / .48 from mostly-idiosyncratic
  to mostly-market gaps (≥ 4σ: 1.71 → .97).
- Result: R² 0.30331 → 0.30328 (Δ −0.00002), years improved 4/8; year deltas ±.002, no pattern.
- Why it failed: same-day volume already shows it. Idiosyncratic gaps come with much higher y_1 (2.70 vs 1.07
  for market-shared gaps at |gap z| 2–4), and within y_1 terciles the effect disappears or reverses (low y_1:
  .04 idio vs .15 market; mid: .39 vs .58). Market-shared shocks persist slightly *more*, given y_1.
- Lesson: univariate pre-tests mislead when the input moves with y_1 or vlm_1_ratio; condition the pre-test on
  y_1 terciles before running (cf. H010, where a flat pre-test still gained). Gap direction: the value is in H010's
  signed one-day split; the persistence (H011) and market-relative (H012) variants add nothing.

### H013 — correlation-peer spillovers (peers' y_1, peers' earnings prob today) — REJECTED
- File: `candidates/h013_peer_spillover.py` · columns: `h013_peer_y_1`, `h013_peer_eday_prob_0`
- Hypothesis / mechanism: stocks that move together share news (sector stories, competitor events, pair
  trading), and a peer's earnings re-prices the industry (intra-industry information transfer). Peers = top 10
  by daily-return correlation over the prior 252 trading days (≥ 120 obs), recomputed at each quarter start
  from returns before the quarter. Features: mean peer y on t-1; mean peer h007_eday_prob_sum at t. Stock-level
  (each stock has its own peer set), not date-level. Coverage 94.5% (first year has no peers).
- Pre-test: survives the y_1 control (top vs bottom quintile residual +.01 to +.04); expected ≲ +.001.
- Result: R² 0.30331 → 0.30406 (Δ +0.00075), years improved 3/8, gain share peer_y_1 .013, peer_eday small
- Notes: the gain comes from 2004 (+.0041) and 2008 (+.0031); the other six years are −.0012 to +.0007. Fails the
  year rule, and the gain is concentrated, so not overruled. peer_y_1 takes real gain share (.013) but it mostly
  overlaps with y_1 / vlm_1_ratio (Spearman .42 with y_1). Possible follow-up only if sector labels become
  available (correlation peers are noisy); otherwise closed.

### H014 — gap reversal at t-1 (signed geometric mean of gap and intraday move) — REJECTED
- File: `candidates/h014_gap_reversal.py` · columns: `h014_gap_intra_geo`
- Hypothesis / mechanism: a gap one way that trades back the other way intraday shows unresolved disagreement
  about the news, so trading continues next day; a gap that continues is news absorbed. H010's columns let trees
  find the reversal quadrant but not its size. sign(g·i)·sqrt(|g·i|) in vol_ewm_21 units at t-1 (negative = reversal).
- Pre-test (current-model residual within y_1 terciles): no consistent pattern. Big reversals +.015 at low y_1 but
  −.037 at high y_1; continuations similar. Days with an exactly-zero gap or intraday move show +.02–.035
  (probably stale-price data, not this mechanism).
- Result: R² 0.30331 → 0.30278 (Δ −0.00052), years improved 2/8, gain share below .004
- Notes: the gap/intraday information is fully used by H010's two signed columns; the interaction only adds
  noise. Gap direction closed: H010 accepted; H011 (EWMA), H012 (market-relative), H014 (reversal) gave nothing.
  Unexplained side finding: rows with an exactly-zero gap or intraday move at t-1 have positive residual. Worth a
  data-quality look (stale or filled prices?) before treating it as a signal.

### Search for large gains (diagnostics on the H010 model, 0.30331; no candidates run)
- Where the residual is: y > 3 rows are 0.7% of rows but 45% of residual variance (model predicts ~.85 there);
  y in (1.5, 3] adds 15%. ~60% of remaining error is on 2.5% of rows. Of big-day residual variance, 26% is on
  rows the earnings schedule flags (predicted too small) and 34% on rows with no signal (unscheduled news).
- Spike magnitude (stock's mean idio size of its last 4 quarterly spikes): matters only at prob_sum ≥ .6 (3.5k
  rows; top-quintile resid +.31); at prob_sum .3–.6, mean y is ~.24 whatever the magnitude. The bottleneck is timing.
- Timing estimators (share of spikes hit within ±2 bdays, 12k spikes): H007-style 13w projection .163; median of
  the last 4 same-type report lags .180; same snapped to the stock's modal weekday .175. No better
  estimator available without true earnings labels; timing looks saturated with this data.
- Linear stacking (does the GBM miss smooth structure?): an OOS-by-year linear fit of the residual on all 40 model
  features removes −0.5% of residual SS (i.e. hurts). The GBM isn't underfitting linear structure.
- Correlation peers (top-10 by trailing 252d return corr, recomputed quarterly, point-in-time;
  scratch code `peers.py`): residual by quintile within y_1 terciles, top vs bottom quintile: peer y_1 +.01 to
  +.03, peer y_1 − market y_1 +.02 to +.03, peer earnings prob today +.01 to +.04, yesterday +.01 to +.04.
  Real (survives the y_1 control) but small; expected ΔR² ≲ .001. Not run yet; parked in the backlog.
- Conclusion: no feature-only idea found with a large expected gain. The remaining large error is unscheduled
  news on spike days, which daily price/volume history can't anticipate.

### H015 — stock's typical earnings spike size (alone and × H007 schedule) — REJECTED
- File: `candidates/h015_earnings_spike_size.py` · columns: `h015_spike_mean_4q`, `h015_eday_x_spike`
- Hypothesis / mechanism: H007 says *when* earnings are likely, not *how big* the reaction is, and reaction size is
  a stable trait of a stock (coverage, retail interest, guidance practice). Size = each calendar quarter's max idio
  y (H007's spike day; every quarter counts, including sub-threshold ones, since a muted reaction is also
  information), mean over the last 4 completed quarters (min 2). A date in Q uses quarters ≤ Q−1. Plus the product
  with `h007_eday_prob_sum`, because the earlier diagnostic showed size matters only at high prob.
- User-requested (follows the "spike magnitude" diagnostic above).
- Shape (mean y by size quintile within prob_sum bins): prob 0: .020 → −.002; (0,.3]: .084 → .059; (.3,.6]:
  .21 → .25; > .6: .70 → 1.22 (but only ~750 rows per cell). Spearman with y: size −.09, product +.06.
- Result: R² 0.30331 → 0.30374 (Δ +0.00044), years improved 4/8, gain share eday_x_spike .008, spike_mean .003
- Notes: by year 2007 +.0051, 2006 +.0018, 2003 +.0012, 2008 +.0009; 2009 −.0024, 2005 −.0012, 2002/2004 −.0010.
  Fails both rules, and the gain is mostly 2007. This confirms the diagnostic: magnitude matters only on the
  few rows where the schedule is confident (prob_sum > .6 is < 0.4% of rows), so the effect is real but too small
  to move pooled R², and at low prob the column mostly marks spike-prone stocks (already in max_y_*, y lags).
  The bottleneck is timing, not size. Not worth retrying with other windows (8q, median) unless timing improves.

### H016 — earnings schedule centred by last year's same-quarter gap (replaces H007's Q1 rule) — REJECTED
- File: `candidates/h016_last_year_gap_schedule.py` · columns: `h016_eday_prob_13w`, `h016_eday_prob_sum`,
  `h016_eday_prob_gap_sum`
- Hypothesis / mechanism: H007's rule assumes the late annual report falls in calendar Q1, true only for Dec/Jan
  fiscal year-ends. Instead centre a 13k-week projection from quarter q at spike + last year's gap between the
  stock's q−4 and q−4+k spikes, rounded to whole weeks, all mass on that centre; used only within ±14d of 13k
  weeks, else (or if last year's pair is missing) the fixed lag. 52w lags unchanged. Same spikes, gap-confirmed
  column and quarter-end filter as H007. User-requested (option 2 of a fiscal-year-end discussion).
- Context: per-stock latest-reporting quarter (from 13w-pair offsets, 81 stocks with ≥ 3 pairs per quarter): Q1
  47%, Q2 17%, Q3 20%, Q4 16%. Q1 is clearly over-represented; no other quarter clusters, but per-stock
  estimates are noisy.
- Pre-test (4,300 13/26/39w pairs with both years' offsets within 14d): last year's week offset predicts this
  year's only weakly (last year +7 → this year +7 30%, 0 35%; last year 0 → 0 52%). Mean pmf mass on the
  realised date: fixed lag .097, H007 .096, H016 .084, last-year 50/50 with fixed .091. Corr with y (sum):
  H016 .090 vs H007 .102.
- Result on top of the baseline: R² 0.30331 → 0.30335 (Δ +0.00004), years improved 4/8, gain share h016 sum .003.
- Result as a **replacement** for `eday_prob_*`: R² 0.30331 → 0.30103 (Δ −0.00228), years improved 2/8 (2004
  −.0111, 2006 −.0044, 2007 −.0038; 2008 +.0045). Saved as `results/h016_last_year_gap_schedule_replacing_eday.json`.
- Decision: reject. Year-to-year spike timing is too noisy for one past pair to set the centre: committing all
  mass to last year's offset moves many projections off a date that the fixed lag would have hit. The pooled
  Q1 rule's 50/50 hedge works better than a per-stock point estimate. Lesson: per-stock timing needs several
  years of evidence (or a hedge), not one pair.

### Baseline construction checks (2026-10-06, review of `vlm_pred/feature.py`; both REVERTED)
- **z-score vol excludes the move** (`enrich_overnight_gap`): `gap_z_1`/`intra_z_1` divide t−1's move by vol
  through t−1, which includes that move, so |z| is compressed towards 1/sqrt(EWM weight) ≈ 5.5 (99.99th pct 6.0;
  10.3 with vol through t−2). Scaling by vol through t−2: R² 0.30331 → 0.30280 (Δ −0.00051), 2/8 years. The
  compressed z is ≈ a monotone transform of the uncompressed one, so trees lose nothing; the difference is
  noise. Kept the original and noted in the docstring not to use |z| with fixed cutoffs.
- **RMS vol instead of demeaned EWM std, min_periods 10** (`enrich_vol_ewm`): with `_daily_vol` (and so
  `gap_z_*` and the gap-confirmed spikes) also switched: R² 0.30271 (Δ −0.00060), 3/8 years. With only the
  `vol_ewm_*` columns switched: 0.30316 (Δ −0.00015), 3/8 years. Demeaning, even at halflife 1, does not hurt
  in practice. Not tried: dropping `vol_ewm_1` / adding a short/long vol ratio (new information, would be its own
  hypothesis).
- **Log returns for vol** (`enrich_vol_ewm`, and so `_daily_vol`): EWM std of log1p(ret_raw), otherwise unchanged.
  R² 0.30328 (Δ −0.00003), 5/8 years: noise. log1p ≈ r − r²/2 only differs on large moves. (Log returns in the
  `ret_*` columns would be a monotone transform, so identical for trees; not run.)
- **Log-volume ratios** (`enrich_vlm_ratio`): EWM of log volume (zero-volume days skipped) minus log
  `vlm_pred_naive`, replacing EWM(volume) / naive − 1 at the same halflives (a geometric mean, so single spike days
  don't dominate). R² 0.30350 (Δ +0.00019), 4/8 years (2002 +.0029, 2009 +.0019; 2004 −.0030): fails the rule.
  Not tried: log ratios *alongside* the arithmetic ones (their gap measures how spiky recent volume was).
- **Log returns in `ret_1..ret_5`** (`enrich_lagged_ret`): log1p(ret_raw). R² 0.3033073933793766 vs
  0.30330739337937673: identical, every year. Confirms that a monotone transform of a single feature cannot change
  the trees (LightGBM bins by value order); don't retry for `ret_20d`/`ret_60d` or other single columns.
- **Vol-scaled lagged returns** (`enrich_lagged_ret`): kept signed `ret_1`, replaced `ret_2..ret_5` with
  |ret| / vol through the day before each return (`ret_absz_2..5`). R² 0.30314 (Δ −0.00017), 4/8 years (2002
  +.0029, 2004 +.0024; 2007 −.0031, 2008 −.0027). Gain shares fell (ret_2 .0070 → ret_absz_2 .0032): the sign of
  earlier returns, or raw size together with the vol columns, carries what the model uses. Rejected.

## Ruled out (not testable under the research rules)
- Early-close sessions (Dec 24, Jul 3, day after Thanksgiving) and holiday-eve flags (e.g. day before
  Thanksgiving). Ruled out by the user as contextual information (exchange-schedule knowledge, not data), even
  when computable from T's date. Context: a residual screen of the H007 model found the 8 worst dates of
  2002–2009 were Dec 24 / Jul 3 (mean residual −0.49 / −0.43; ceiling ≈ +0.003 R²). Day after Thanksgiving and
  Dec 26–31 are already learned from the baseline calendar features.

## Backlog
- Further earnings-spike refinements — spike size (idio y of the source spike) as a confidence weight in the
  consensus. (Top-2 spikes per quarter: H006, no gain. Annual-report shift: H007, accepted. Stock-level average
  spike size, alone and × schedule: H015, Δ +.00044, rejected; size matters only where timing is confident.)
- Per-stock shift choice (follow-up to H007) — use the stock's own last-year Q3→Q1 offset (known point-in-time)
  to put H007's mass on the shifted or unshifted triangle instead of 50/50. The diagnostic's two sharp peaks
  suggest firms are consistently one type or the other. Caution from H016: a single last-year offset persists
  only weakly (+7 → +7 30% vs 0 35%), so weight by several years' evidence rather than switching on one pair.
- Stock-level schedule reliability — fraction of a stock's past spikes that recurred on-cycle (point-in-time).
  Some firms report on a fixed calendar and others drift; H006's placebo test shows on-cycle vs off-cycle
  recurrence can be measured per spike set, and so per stock.
- Earnings timing relative to peers — earnings season clustering: fraction of stocks with a spike in the last
  few days (cross-sectional, but stock-relative / sector-level to avoid H001's date-memorization problem).
- Post-event decay: days since last quarterly spike, and size of that spike — volume after earnings decays over
  ~1–2 weeks with a shape lags y_1..y_5 only partly capture.
- Market volume regime done safely — coarse buckets of market y, or stock y_1 minus market y_1 only (H001 showed
  continuous date-level features overfit).
- Gap direction closed: EWMA (H011), market-relative (H012), and reversal (H014) all gave no gain beyond H010.
- Idiosyncratic longer-horizon return (after H009) — 20d return minus market 20d return; market-wide selloffs vs
  stock-specific declines may have different volume persistence.
- Index weight change (sp_weight lagged vs its own history) — index rebalancing / additions drive one-off volume.
