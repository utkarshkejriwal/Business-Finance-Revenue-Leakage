"""Grid search over PARAM_GRIDS (defined in train.py).

To avoid choosing settings on the same data used to report them:
  - tune:      fit each combination on July
  - validate:  score it on 1-15 Aug                 -> picks the best combination (PR-AUC)
  - evaluate:  refit on 1 Jul-15 Aug, score on 16-31 Aug -> same holdout as train.py
Every combination is scored on both, and all results are saved.
"""
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

from features import build
from train import FEATURES, OUT, PARAM_GRIDS, TEST_START, VALID_START, best_threshold, evaluate, make_models

TUNE_START = "2026-08-01"
TUNE_OUT = OUT / "tuning"


def splits():
    df = build()
    y = df.dispute_raised
    sel_tr = df.ship_date < TUNE_START
    sel_va = (df.ship_date >= TUNE_START) & (df.ship_date < VALID_START)
    rep_tr = df.ship_date < VALID_START
    rep_va = (df.ship_date >= VALID_START) & (df.ship_date < TEST_START)
    X = df[FEATURES]
    return (X[sel_tr], y[sel_tr], X[sel_va], y[sel_va]), (X[rep_tr], y[rep_tr], X[rep_va], y[rep_va])


def score_one(base, params, select, report):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)

        Xtr, ytr, Xva, yva = select
        m = clone(base).set_params(**params).fit(Xtr, ytr.astype(int))
        p = m.predict_proba(Xva)[:, 1]
        row = {"val_pr_auc": average_precision_score(yva, p), "val_roc_auc": roc_auc_score(yva, p)}

        Xtr, ytr, Xva, yva = report
        m = clone(base).set_params(**params).fit(Xtr, ytr.astype(int))
        thr = best_threshold(ytr.astype(int), m.predict_proba(Xtr)[:, 1])
        metrics = evaluate(yva.astype(int), m.predict_proba(Xva)[:, 1], thr)
        row.update({f"test_{k}": v for k, v in metrics.items()})

    row["converged"] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {**{k.removeprefix("clf__"): v for k, v in params.items()}, **row}


def main():
    TUNE_OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    select, report = splits()
    models = make_models()

    summary = []
    for name, grid in PARAM_GRIDS.items():
        combos = list(ParameterGrid(grid))
        # Random forest already uses all cores per fit; parallelise the others across combinations.
        jobs = 1 if name == "random_forest" else -1
        rows = Parallel(n_jobs=jobs)(delayed(score_one)(models[name], p, select, report) for p in combos)

        res = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
        res.round(4).to_csv(TUNE_OUT / f"{name}_grid.csv", index=False)

        current = {k: models[name].get_params()[k] for k in grid}
        cur = score_one(models[name], current, select, report)
        best = res.iloc[0]
        params = [k.removeprefix("clf__") for k in grid]
        summary.append({
            "model": name,
            "combinations": len(combos),
            "best_params": {k: best[k].item() if hasattr(best[k], "item") else best[k] for k in params},
            "current_params": {k.removeprefix("clf__"): v for k, v in current.items()},
            "best_val_pr_auc": best.val_pr_auc,
            "current_val_pr_auc": cur["val_pr_auc"],
            "best_test_pr_auc": best.test_pr_auc,
            "current_test_pr_auc": cur["test_pr_auc"],
            "best_test_roc_auc": best.test_roc_auc,
            "best_test_f1": best.test_f1,
            "max_test_pr_auc_in_grid": res.test_pr_auc.max(),
            "not_converged": int((~res.converged).sum()),
        })
        print(f"\n{name}: {len(combos)} combinations (top 5 by validation PR-AUC)")
        print(res.head(5).round(4).to_string())

    summary = pd.DataFrame(summary).set_index("model")
    summary.round(4).to_csv(TUNE_OUT / "tuning_summary.csv")
    print("\nSummary\n", summary.round(4).to_string())


if __name__ == "__main__":
    main()
