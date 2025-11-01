import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # unify column names
    df = df.rename(columns={
        "ZONEID": "zone_id",
        "TIMESTAMP": "datetime",
        "TARGETVAR": "target"
    })

    # convert to datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df
