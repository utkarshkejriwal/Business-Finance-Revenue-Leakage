"""Dispute prediction: compare 3 classifiers, then score September.

September shipments have no dispute labels yet, so the models are:
  1. validated out-of-time: train on 1 Jul-15 Aug, evaluate on 16-31 Aug;
  2. refit on July + August and used to predict September.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, build

OUT = Path(__file__).resolve().parent / "outputs"
VALID_START, TEST_START = "2026-08-16", "2026-09-01"
FEATURES = NUMERIC + CATEGORICAL
SEED = 42
# Up-weight disputed shipments (~15% of rows): "balanced" weights each class by
# n_samples / (2 * n_class), i.e. roughly 3.3x for disputes vs 0.6x for the rest.
# This makes predicted probabilities higher than real dispute rates, so treat them as risk scores.
CLASS_WEIGHT = "balanced"

# Hyperparameter grids searched by tune.py (keys address the "clf" step of each pipeline).
PARAM_GRIDS = {
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 0.5, 1, 10],
    },
    "logistic_elastic_net": {
        "clf__C": [0.01, 0.1, 0.5, 1, 10],
        "clf__l1_ratio": [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9],
    },
    "random_forest": {
        "clf__max_depth": [5, 10, 15, None],
        "clf__min_samples_leaf": [2, 5, 10, 20],
    },
    "gradient_boosting": {
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_iter": [100, 150, 200, 300],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__min_samples_leaf": [5, 10, 20, 40],
    },
}


def make_models():
    def prep(scale):
        num = StandardScaler() if scale else "passthrough"
        return ColumnTransformer([
            ("num", num, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])

    # Every tunable hyperparameter is listed explicitly; the comment gives the
    # scikit-learn default. Lines marked "changed" differ from the default.
    return {
        "logistic_regression": Pipeline([
            ("prep", prep(scale=True)),
            ("clf", LogisticRegression(
                C=0.5,                      # changed; default 1.0 - inverse regularisation strength (lower = stronger)
                l1_ratio=0.0,               # default 0.0 - 0 = pure L2 (ridge) penalty
                solver="lbfgs",             # default "lbfgs" - lbfgs/newton-cg/sag support L2 only
                fit_intercept=True,         # default True
                intercept_scaling=1,        # default 1 - only used by solver="liblinear"
                class_weight=CLASS_WEIGHT,  # changed; default None
                max_iter=2000,              # changed; default 100 - solver iteration cap
                tol=1e-4,                   # default 1e-4 - convergence tolerance
                dual=False,                 # default False - liblinear only
            )),
        ]),
        # Elastic net = mix of L1 (drops weak features to zero) and L2 (shrinks correlated ones).
        "logistic_elastic_net": Pipeline([
            ("prep", prep(scale=True)),
            ("clf", LogisticRegression(
                C=0.5,                      # changed; default 1.0 - inverse regularisation strength (lower = stronger)
                l1_ratio=0.2,               # changed; default 0.0 - 0 = pure L2, 1 = pure L1 (lasso)
                solver="saga",              # changed; default "lbfgs" - saga is required for L1/elastic net
                fit_intercept=True,         # default True
                class_weight=CLASS_WEIGHT,  # changed; default None
                max_iter=5000,              # changed; default 100 - saga converges slowly
                tol=1e-4,                   # default 1e-4 - convergence tolerance
                random_state=SEED,          # default None
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", prep(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=400,               # changed; default 100 - number of trees
                criterion="gini",               # default "gini" - also "entropy", "log_loss"
                max_depth=None,                 # default None - grow until leaves are pure / min_samples_*
                min_samples_split=2,            # default 2 - min rows needed to split a node
                min_samples_leaf=5,             # changed; default 1 - min rows in each leaf
                min_weight_fraction_leaf=0.0,   # default 0.0
                max_features="sqrt",            # default "sqrt" - features considered at each split
                max_leaf_nodes=None,            # default None - cap on leaves per tree
                min_impurity_decrease=0.0,      # default 0.0 - min gain required to split
                bootstrap=True,                 # default True - sample rows with replacement per tree
                max_samples=None,               # default None - rows drawn per tree (None = all)
                ccp_alpha=0.0,                  # default 0.0 - cost-complexity pruning strength
                class_weight=CLASS_WEIGHT,      # changed; default None
                n_jobs=-1,                      # changed; default None - use all CPU cores
                random_state=SEED,              # default None
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", prep(scale=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,             # changed; default 0.1 - shrinkage per boosting step
                max_iter=300,                   # changed; default 100 - number of boosting rounds (trees)
                max_depth=4,                    # changed; default None - max depth of each tree
                max_leaf_nodes=31,              # default 31 - max leaves per tree
                min_samples_leaf=20,            # default 20 - min rows in each leaf
                l2_regularization=1.0,          # changed; default 0.0 - L2 penalty on leaf values
                max_features=1.0,               # default 1.0 - share of features sampled per split
                max_bins=255,                   # default 255 - histogram bins per feature
                early_stopping="auto",          # default "auto" - only switches on above 10,000 rows, so off here
                validation_fraction=0.1,        # default 0.1 - held-out share when early stopping is on
                n_iter_no_change=10,            # default 10 - patience for early stopping
                tol=1e-7,                       # default 1e-7 - early-stopping tolerance
                class_weight=CLASS_WEIGHT,      # changed; default None
                random_state=SEED,              # default None
            )),
        ]),
    }


def best_threshold(y, p):
    """Threshold maximising F1 on the given (training-period) data."""
    grid = [t / 100 for t in range(5, 96)]
    return max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0))


def evaluate(y, p, threshold):
    pred = p >= threshold
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "predicted_rate": pred.mean(),
    }


def run_validation():
    df = build()
    train = df[df.ship_date < VALID_START]
    valid = df[(df.ship_date >= VALID_START) & (df.ship_date < TEST_START)]
    ytr, yva = train.dispute_raised.astype(int), valid.dispute_raised.astype(int)

    rows, fitted = [], {}
    for name, model in make_models().items():
        model.fit(train[FEATURES], ytr)
        thr = best_threshold(ytr, model.predict_proba(train[FEATURES])[:, 1])
        p = model.predict_proba(valid[FEATURES])[:, 1]
        rows.append({"model": name, **evaluate(yva, p, thr)})
        fitted[name] = (model, thr)
    rows.append({"model": "baseline_is_overbilled", **evaluate(yva, valid.is_overbilled.values, 0.5)})
    return pd.DataFrame(rows).set_index("model"), fitted, valid


def feature_importance(model, X, y):
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def predict_september(best_name):
    df = build()
    train = df[df.ship_date < TEST_START]
    test = df[df.ship_date >= TEST_START].copy()
    y = train.dispute_raised.astype(int)

    preds = test[["awb", "ship_date", "merchant_id", "billed_total", "exp_total", "overbill_amt", "zone_mismatch"]].copy()
    for name, model in make_models().items():
        model.fit(train[FEATURES], y)
        thr = best_threshold(y, model.predict_proba(train[FEATURES])[:, 1])
        preds[f"p_{name}"] = model.predict_proba(test[FEATURES])[:, 1].round(4)
        preds[f"pred_{name}"] = (preds[f"p_{name}"] >= thr).astype(int)
    preds["dispute_probability"] = preds[f"p_{best_name}"]
    preds["predicted_dispute"] = preds[f"pred_{best_name}"]
    return preds.sort_values("dispute_probability", ascending=False)


def main():
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 140)

    metrics, fitted, valid = run_validation()
    metrics.round(4).to_csv(OUT / "validation_metrics.csv")
    print("Validation (train: 1 Jul-15 Aug, test: 16-31 Aug)\n", metrics.round(3), "\n")

    best = metrics.drop("baseline_is_overbilled").pr_auc.idxmax()
    print(f"Best model by PR-AUC: {best}\n")

    imp = feature_importance(fitted[best][0], valid[FEATURES], valid.dispute_raised.astype(int))
    imp.round(4).to_csv(OUT / f"feature_importance_{best}.csv", header=["pr_auc_drop"])
    print("Top features (permutation importance on 16-31 Aug):\n", imp.head(12).round(4), "\n")

    preds = predict_september(best)
    preds.to_csv(OUT / "september_predictions.csv", index=False)
    summary = preds.groupby("merchant_id").agg(
        shipments=("awb", "size"),
        flagged=("predicted_dispute", "sum"),
        mean_risk_score=("dispute_probability", "mean"),
    )
    summary["flagged_rate"] = summary.flagged / summary.shipments
    summary.round(3).to_csv(OUT / "september_merchant_summary.csv")
    print(f"September (train: July+Aug) - {len(preds)} shipments, "
          f"{preds.predicted_dispute.sum()} flagged ({preds.predicted_dispute.mean():.1%})")
    print(summary.round(3))


if __name__ == "__main__":
    main()
