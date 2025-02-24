import numpy as np

from bikeshare import evaluate, stats


def test_games_howell_separates_different_groups_only():
    rng = np.random.default_rng(0)
    groups = {"a": rng.normal(0, 1, 200), "b": rng.normal(0, 3, 200), "c": rng.normal(2, 1, 200)}
    result = stats.games_howell(groups).set_index(["group_a", "group_b"])["p_adj"]
    assert result[("a", "b")] > 0.05
    assert result[("a", "c")] < 0.001


def test_total_variation_distance_bounds():
    p = np.array([0.5, 0.5, 0.0])
    assert stats.total_variation_distance(p, p) == 0
    assert stats.total_variation_distance(p, np.array([0.0, 0.0, 1.0])) == 1


def test_permutation_test_finds_no_difference_between_identical_groups():
    rng = np.random.default_rng(1)
    profiles = rng.dirichlet(np.ones(24) * 50, size=120)
    labels = np.repeat(["x", "y"], 60)
    _, p_value, _ = stats.permutation_test_profiles(profiles, labels, n_perm=2000)
    assert p_value > 0.05


def test_per_group_margins_fix_coverage_for_a_noisier_group():
    rng = np.random.default_rng(3)
    groups = np.repeat(["calm", "busy"], 3000)
    truth = np.where(groups == "busy", rng.poisson(400, 6000) * rng.uniform(0.6, 1.4, 6000), rng.poisson(400, 6000))
    predicted = np.full(6000, 400.0)
    calib, check = np.arange(6000) % 2 == 0, np.arange(6000) % 2 == 1
    one = evaluate.conformal_quantile(truth[calib], predicted[calib])
    per_group = evaluate.conformal_quantiles(truth[calib], predicted[calib], groups[calib])
    busy = check & (groups == "busy")
    q_rows = np.vectorize(per_group.get)(groups[busy])
    assert evaluate.coverage(truth[busy], *evaluate.prediction_interval(predicted[busy], one)) < 0.85
    assert 0.87 < evaluate.coverage(truth[busy], *evaluate.prediction_interval(predicted[busy], q_rows)) < 0.93


def test_conformal_interval_reaches_target_coverage():
    rng = np.random.default_rng(2)
    truth = rng.poisson(400, size=4000).astype(float)
    predicted = np.full_like(truth, 400.0)
    q = evaluate.conformal_quantile(truth[:2000], predicted[:2000], coverage=0.9)
    lower, upper = evaluate.prediction_interval(predicted[2000:], q)
    assert 0.87 < evaluate.coverage(truth[2000:], lower, upper) < 0.93
