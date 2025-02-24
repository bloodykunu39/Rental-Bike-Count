"""Time-ordered splits: a model is never scored on hours earlier than the ones it trained on."""
import pandas as pd

TEST_START = pd.Timestamp("2018-10-01")  # test on Oct-Nov 2018, the last two months
CV_MONTHS = ["2018-06", "2018-07", "2018-08", "2018-09"]


def train_test_split_by_time(frame: pd.DataFrame, test_start=TEST_START):
    """Index labels of the training rows (before `test_start`) and the test rows."""
    is_test = frame["timestamp"] >= test_start
    return frame.index[~is_test], frame.index[is_test]


def rolling_origin_folds(frame: pd.DataFrame, months=CV_MONTHS):
    """Expanding-window folds: validate on each month, train on everything before it."""
    folds = []
    for month in months:
        start = pd.Timestamp(month)
        end = start + pd.offsets.MonthBegin(1)
        train = frame.index[frame["timestamp"] < start]
        val = frame.index[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
        folds.append((train, val))
    return folds
