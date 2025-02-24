"""Model factories, baselines, cross-validation and Optuna tuning on time-ordered folds."""
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .evaluate import metrics

SEED = 42


def _square_positive(z):
    return np.square(np.clip(z, 0, None))


def sqrt_target(model):
    """Train on sqrt(rentals) to tame the right skew; predictions come back in bikes."""
    return TransformedTargetRegressor(regressor=model, func=np.sqrt, inverse_func=_square_positive,
                                      check_inverse=False)


def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


def xgb(**params):
    return XGBRegressor(random_state=SEED, n_jobs=8, **params)


def lgbm(**params):
    # subsample_freq=1 turns on row subsampling whenever a tuned `subsample` is below 1
    return LGBMRegressor(random_state=SEED, n_jobs=8, subsample_freq=1, deterministic=True, force_col_wise=True,
                         verbose=-1, **params)


def ridge(numeric):
    """Linear baseline: one-hot hour-of-week profile and season, scaled numeric features."""
    prep = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["hour_weekend", "day_of_week", "season_code"]),
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
    ])
    return make_pipeline(prep, Ridge(alpha=1.0))


def naive(frame, column):
    """Copy past demand as the forecast, falling back to the 7-day same-hour mean after closures."""
    return frame[column].fillna(frame["same_hour_7d_mean"])


def cross_validate(make_model, X, y, folds):
    """Fit on each expanding window and score the following month.

    Returns per-fold metrics and the out-of-fold predictions (used to calibrate intervals).
    """
    rows, oof = [], pd.Series(np.nan, index=y.index)
    for train_idx, val_idx in folds:
        model = make_model().fit(X.loc[train_idx], y.loc[train_idx])
        oof.loc[val_idx] = model.predict(X.loc[val_idx])
        rows.append(metrics(y.loc[val_idx], oof.loc[val_idx]))
    validated = np.concatenate([val_idx for _, val_idx in folds])
    return pd.DataFrame(rows), oof.loc[validated]


def xgb_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def lgbm_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def tune(build, space, X, y, folds, n_trials=100, seed=SEED):
    """Optuna search that minimises mean RMSE (bikes) over the rolling folds.

    Only training-period data is used, so the test months stay unseen until the end.
    """
    def objective(trial):
        params = space(trial)
        rmse = []
        for step, (train_idx, val_idx) in enumerate(folds):
            model = build(**params).fit(X.loc[train_idx], y.loc[train_idx])
            rmse.append(metrics(y.loc[val_idx], model.predict(X.loc[val_idx]))["RMSE"])
            trial.report(float(np.mean(rmse)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmse))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials)
    return study
