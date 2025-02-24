"""Metrics in bikes per hour, and split-conformal prediction intervals."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual, predicted) -> dict:
    """MAE, RMSE and R2 in bikes, plus R2 on the square-root scale that v1 reported."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2_score(actual, predicted),
        "R2_sqrt": r2_score(np.sqrt(actual), np.sqrt(predicted)),
    }


def conformal_quantile(actual, predicted, coverage=0.9) -> float:
    """Residual quantile on the square-root scale, from out-of-sample predictions.

    Working on the square-root scale makes the interval widen as demand grows,
    which is how the noise in count data behaves.
    """
    scores = np.abs(np.sqrt(np.asarray(actual, dtype=float)) - np.sqrt(np.clip(predicted, 0, None)))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_quantiles(actual, predicted, groups, coverage=0.9) -> dict:
    """`conformal_quantile` for each group separately (Mondrian conformal prediction).

    One margin for all hours can be right on average yet too narrow at busy hours;
    calibrating per group targets the coverage within each group.
    """
    actual, predicted, groups = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float), np.asarray(groups)
    return {str(g): conformal_quantile(actual[groups == g], predicted[groups == g], coverage)
            for g in np.unique(groups)}


def prediction_interval(predicted, q):
    """Lower and upper bounds in bikes for a residual quantile `q` (one value, or one per row)."""
    root = np.sqrt(np.clip(np.asarray(predicted, dtype=float), 0, None))
    q = np.asarray(q, dtype=float)
    return np.clip(root - q, 0, None) ** 2, (root + q) ** 2


def coverage(actual, lower, upper) -> float:
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))
