import pytest

from bikeshare import data, features, validation


@pytest.fixture(scope="module")
def frame():
    return features.build(data.load())


def test_test_period_comes_strictly_after_training(frame):
    train, test = validation.train_test_split_by_time(frame)
    assert frame.loc[train, "timestamp"].max() < frame.loc[test, "timestamp"].min()
    assert len(train) + len(test) == len(frame)


def test_folds_expand_and_stay_before_the_test_period(frame):
    folds = validation.rolling_origin_folds(frame)
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes)
    for train, val in folds:
        assert len(val) > 0
        assert frame.loc[train, "timestamp"].max() < frame.loc[val, "timestamp"].min()
        assert frame.loc[val, "timestamp"].max() < validation.TEST_START
