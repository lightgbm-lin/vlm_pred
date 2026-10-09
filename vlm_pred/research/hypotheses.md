# Volume factor research log

## Current model
- Features: baseline features (`FEATURE_FUNCS` in `vlm_pred/main.py`, 39 columns)
- Selected candidates (see candidates/selected.txt): none
- Walk-forward R² (2002–2009, train only): 0.30300  (t = 608.9)
  - by year: 2002 .2918 · 2003 .2237 · 2004 .2519 · 2005 .2549 · 2006 .2495 · 2007 .3288 · 2008 .4071 · 2009 .3821

## Tested hypotheses

## Ruled out (not testable under the research rules)
- Early-close sessions (Dec 24, Jul 3, day after Thanksgiving) and holiday-eve flags (e.g. day before
  Thanksgiving). Ruled out by the user as contextual information (exchange-schedule knowledge, not data), even
  when computable from T's date.

## Backlog
