import pandas as pd
import pytest

from bikeshare import data, features


@pytest.fixture(scope="module")
def hourly():
    return data.load()


def test_a_days_features_use_only_earlier_days(hourly):
    """Changing demand from day D onwards must not change any feature of day D (no leakage)."""
    day_start = 24 * 200
    changed = hourly.copy()
    changed.loc[day_start:, "rentals"] += 1000

    original = features.add_features(hourly)
    perturbed = features.add_features(changed)
    same_day = slice(day_start, day_start + 23)
    pd.testing.assert_frame_equal(original.loc[:day_start + 23, features.FEATURES],
                                  perturbed.loc[:day_start + 23, features.FEATURES])
    # ...while the next day does see the change, so the check above is not vacuous
    next_day = slice(day_start + 24, day_start + 47)
    assert not original.loc[next_day, "lag_24h"].equals(perturbed.loc[next_day, "lag_24h"])
    assert original.loc[same_day, "prev_day_mean"].notna().all()


def test_closed_hours_count_as_missing_not_zero(hourly):
    feats = features.add_features(hourly)
    closed = hourly.index[~hourly["functioning"]]
    day_after = closed[closed + 24 < len(hourly)] + 24
    assert feats.loc[day_after, "lag_24h"].isna().all()


def test_build_keeps_only_open_hours(hourly):
    frame = features.build(hourly)
    assert frame["functioning"].all()
    assert len(frame) == hourly["functioning"].sum()
