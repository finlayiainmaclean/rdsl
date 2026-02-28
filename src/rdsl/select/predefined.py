from pathlib import Path

import pandas as pd

PREDEFINED_SMARTS = {}

# Load alerts from data/alerts.csv
_ALERTS_PATH = Path(__file__).parent.parent / "data" / "alerts.csv"
_alerts_df = pd.read_csv(_ALERTS_PATH)
_unique_smarts = _alerts_df["smarts"].dropna().unique().tolist()
PREDEFINED_SMARTS["alerts"] = (_unique_smarts, "smarts_list")
