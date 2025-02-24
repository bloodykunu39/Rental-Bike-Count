"""Features for a day-ahead forecast.

The forecast for day D+1 is made at the end of day D, so a feature may use demand
up to the end of D plus the weather for D+1. The dataset has no weather forecasts,
so observed weather stands in for them; a live system would be somewhat less accurate.
"""
import numpy as np
import pandas as pd

CALENDAR = ["hour", "day_of_week", "is_weekend", "is_holiday", "season_code"]
WEATHER = ["temperature", "humidity", "wind_speed", "visibility", "solar_radiation", "rainfall", "snowfall"]
RECENT_RAIN = ["rain_prev_3h"]
LAGS = ["lag_24h", "lag_168h", "same_hour_7d_mean", "prev_day_mean"]
FEATURES = CALENDAR + WEATHER + RECENT_RAIN + LAGS

# v1 used weather + calendar only, with month instead of day of week
V1_FEATURES = ["hour", *WEATHER, "season_code", "is_holiday", "is_weekend", "month"]

LABELS = {
    "hour": "hour of day", "day_of_week": "day of week", "is_weekend": "weekend", "is_holiday": "holiday",
    "season_code": "season", "temperature": "temperature", "humidity": "humidity", "wind_speed": "wind speed",
    "visibility": "visibility", "solar_radiation": "solar radiation", "rainfall": "rain this hour",
    "snowfall": "snowfall", "rain_prev_3h": "rain, previous 3 h", "lag_24h": "same hour yesterday",
    "lag_168h": "same hour last week", "same_hour_7d_mean": "same hour, 7-day mean",
    "prev_day_mean": "yesterday's hourly mean",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features to the full hourly series from `data.load` (no gaps allowed)."""
    out = df.copy()
    out["day_of_week"] = out["date"].dt.weekday
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = out["holiday"].astype(int)
    out["season_code"] = out["season"].cat.codes
    out["month"] = out["date"].dt.month
    out["hour_weekend"] = out["hour"] + 24 * out["is_weekend"]  # used by the linear baseline
    # Roads stay wet after rain, so recent rain matters as well as rain in the hour itself
    out["rain_prev_3h"] = out["rainfall"].shift(1).rolling(3, min_periods=1).sum()

    # Closed hours had no rentals possible, so treat them as unknown rather than zero
    rentals = out["rentals"].where(out["functioning"])
    out["lag_24h"] = rentals.shift(24)
    out["lag_168h"] = rentals.shift(168)
    out["same_hour_7d_mean"] = pd.concat([rentals.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)
    out["prev_day_mean"] = rentals.groupby(out["date"]).transform("mean").shift(24)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Features for every open hour; closed hours are dropped because no rentals were possible."""
    out = add_features(df)
    return out[out["functioning"]]


def time_band(frame: pd.DataFrame) -> pd.Series:
    """Day type x part of day, e.g. 'weekday morning peak'; prediction intervals are calibrated per band."""
    part = pd.cut(frame["hour"], bins=[-1, 6, 9, 16, 20, 23],
                  labels=["night", "morning peak", "daytime", "evening peak", "late evening"]).astype(str)
    day = np.where((frame["is_weekend"] == 1) | (frame["is_holiday"] == 1), "weekend/holiday", "weekday")
    return pd.Series(day, index=frame.index) + " " + part
