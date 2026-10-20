"""Score September shipments with the tuned elastic-net logistic regression.

Model: logistic regression, elastic-net penalty, C=0.01, l1_ratio=0.1 (chosen in tune.py),
trained on every labelled shipment from July and August.
"""
import numpy as np
import pandas as pd

from features import build
from train import FEATURES, OUT, TEST_START, best_threshold, make_models

PARAMS = {"clf__C": 0.01, "clf__l1_ratio": 0.1}


def main():
    df = build()
    train = df[df.ship_date < TEST_START]
    sept = df[df.ship_date >= TEST_START].copy()
    y = train.dispute_raised.astype(int)

    model = make_models()["logistic_elastic_net"].set_params(**PARAMS)
    model.fit(train[FEATURES], y)

    # The model is trained with class_weight="balanced", which inflates its scores.
    # Undo that on the odds scale to get probabilities that match real dispute rates.
    weight_ratio = (y == 1).sum() / (y == 0).sum()

    def calibrate(p):
        odds = p / (1 - p) * weight_ratio
        return odds / (1 + odds)

    p_train = model.predict_proba(train[FEATURES])[:, 1]
    threshold = best_threshold(y, p_train)

    sept["risk_score"] = model.predict_proba(sept[FEATURES])[:, 1]
    sept["dispute_probability"] = calibrate(sept.risk_score)
    sept["predicted_dispute"] = (sept.risk_score >= threshold).astype(int)

    out = sept.sort_values("dispute_probability", ascending=False).round(4)
    out[["awb", "dispute_probability"]].to_csv(OUT / "september_dispute_probabilities_elastic_net.csv", index=False)

    summary = out.groupby("merchant_id").agg(
        shipments=("awb", "size"),
        expected_disputes=("dispute_probability", "sum"),
        mean_probability=("dispute_probability", "mean"),
        flagged=("predicted_dispute", "sum"),
    ).round(3)
    summary.to_csv(OUT / "september_merchant_summary_elastic_net.csv")

    print(f"Trained on {len(train)} shipments (Jul-Aug), actual dispute rate {y.mean():.3f}")
    print(f"Mean calibrated probability on training data: {calibrate(p_train).mean():.3f}")
    print(f"Decision threshold on risk score (max F1 on training): {threshold}")
    print(f"September: {len(out)} shipments, expected disputes {out.dispute_probability.sum():.0f} "
          f"({out.dispute_probability.mean():.1%}), flagged {out.predicted_dispute.sum()}")
    print(summary.to_string())
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=model.named_steps["prep"].get_feature_names_out())
    print(f"\nNon-zero coefficients: {int((coef != 0).sum())} of {len(coef)}")
    print(coef[coef != 0].sort_values(key=np.abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
