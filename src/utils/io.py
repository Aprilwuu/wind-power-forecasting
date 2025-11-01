import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["TIMESTAMP"])
        return df
    except Exception:
        #in case of no "TIMESTAMP" column, just read normally
        return pd.read_csv

