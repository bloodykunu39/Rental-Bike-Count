import pandas as pd

from bikeshare import data


def test_dates_are_read_day_first():
    df = data.load()
    days = df["date"].drop_duplicates().reset_index(drop=True)
    # 01/12/17 must be 1 Dec 2017 (v1 read it as 12 Jan 2017)
    assert days.iloc[0] == pd.Timestamp("2017-12-01")
    assert days.iloc[12] == pd.Timestamp("2017-12-13")
    assert days.iloc[-1] == pd.Timestamp("2018-11-30")
    assert days.diff().iloc[1:].eq(pd.Timedelta(days=1)).all()


def test_one_row_per_hour_with_no_gaps():
    df = data.load()
    assert len(df) == 365 * 24
    assert df["timestamp"].is_unique
    assert df.isna().sum().sum() == 0
