from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILENAME = ROOT / 'data' / 'sp500.h5'
VAL_CUTOFF = datetime.strptime('2010-01-01', '%Y-%m-%d')
OOS_CUTOFF = datetime.strptime('2015-01-01', '%Y-%m-%d')
