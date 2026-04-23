import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data. If TIMESTAMP column exists, parse it as datetime.
    """
    df = pd.read_csv(path)

    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

    return df