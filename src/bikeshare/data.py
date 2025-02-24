"""Load the Seoul Bike Sharing data with correct dates and tidy column names."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "SeoulBikeData.csv"

COLUMNS = {
    "Date": "date",
    "Rented Bike Count": "rentals",
    "Hour": "hour",
    "Temperature(C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Visibility (10m)": "visibility",
    "Dew point temperature(C)": "dew_point",
    "Solar Radiation (MJ/m2)": "solar_radiation",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "season",
    "Holiday": "holiday",
    "Functioning Day": "functioning",
}
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]


def load(path=DATA_PATH) -> pd.DataFrame:
    """Return one row per hour (8,760 rows, Dec 2017 - Nov 2018) with a `timestamp` column.

    Dates in the file are day-first with a two-digit year (01/12/17 is 1 Dec 2017).
    Parsing them without a format reads them month-first, which was the v1 bug,
    so the format is always given explicitly.
    """
    df = pd.read_csv(path).rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y")
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df["holiday"] = df["holiday"].eq("Holiday")
    df["functioning"] = df["functioning"].eq("Yes")
    df["season"] = pd.Categorical(df["season"], categories=SEASONS)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if not df["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("expected exactly one row per consecutive hour")
    return df
