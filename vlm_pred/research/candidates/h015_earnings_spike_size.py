"""H015: stock's typical earnings volume spike (mean of its last 4 quarterly spikes), alone and times H007's schedule.

Mechanism: the earnings schedule (H007) says *when* a report is likely, but not *how big* the volume reaction is.
That size is a fairly stable trait of the stock: analyst coverage, retail interest, how much news each report
carries, guidance practice and options activity all differ across firms, so the same earnings probability means a
2x day for one stock and a 6x day for another. Each calendar quarter's earnings day is proxied by H007's spike day
(max idiosyncratic y, i.e. y minus that day's cross-sectional median), and its size is that idiosyncratic y. Every
quarter counts, including ones whose max stays below H007's spike threshold: a muted reaction is information too.
A quarter's max is known only at its end, so a date in quarter Q uses quarters up to Q-1 (4 of them, min 2).
Expected: on its own, mildly positive (spike-prone stocks); the product with eday_prob_sum should be strongly
positive and convex, i.e. bigger predicted y on predicted earnings days of stocks that react strongly.
"""
import numpy as np
import pandas as pd

N_QUARTERS = 4
MIN_QUARTERS = 2


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values('date')
    uspn = df.index.get_level_values('uspn')
    q = dates.to_period('Q')

    idio = df['y'] - df['y'].groupby(dates).transform('median')
    # Size of each (stock, quarter)'s spike: its quarter max of idiosyncratic y
    spike = idio.groupby([uspn, q]).max().dropna().rename_axis(['uspn', 'q'])
    mean_4q = (spike.groupby(level='uspn', group_keys=False)
               .apply(lambda s: s.rolling(N_QUARTERS, min_periods=MIN_QUARTERS).mean()))
    # The mean through quarter Q is known from the start of quarter Q+1
    mean_4q.index = pd.MultiIndex.from_arrays([mean_4q.index.get_level_values('uspn'),
                                               mean_4q.index.get_level_values('q') + 1])
    size = mean_4q.reindex(pd.MultiIndex.from_arrays([uspn, q])).to_numpy()

    return pd.DataFrame({
        'h015_spike_mean_4q': size,
        'h015_eday_x_spike': df['eday_prob_sum'].to_numpy() * size,
    }, index=df.index)
