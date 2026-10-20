"""Feature engineering for the dispute model.

Recomputes the contract-correct charge for every shipment (Schedule A + merchant
overrides) so billing errors can be used as features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load():
    ship = pd.read_csv(DATA / "shipments.csv", parse_dates=["ship_date"])
    contracts = pd.read_csv(
        DATA / "merchant_contracts.csv",
        parse_dates=["discount_amendment_effective_from", "cod_waiver_start", "cod_waiver_end"],
    )
    rates = pd.read_csv(DATA / "schedule_a_freight_rates.csv").set_index("zone")
    regions = pd.read_csv(DATA / "schedule_a_region_lists.csv")
    return ship, contracts, rates, regions


def expected_zone(df, regions):
    special = set(regions.loc[regions.category == "Special Region", "name"])
    metro = set(regions.loc[regions.category == "Metro", "name"])
    return np.select(
        [
            df.origin_city == df.dest_city,
            df.dest_state.isin(special),
            df.pickup_state == df.dest_state,
            df.origin_city.isin(metro) & df.dest_city.isin(metro),
        ],
        ["A", "E", "B", "C"],
        default="D",
    )


def add_expected_charges(df, rates, regions):
    df["exp_zone"] = expected_zone(df, regions)

    vol_wt = df.length_cm * df.width_cm * df.height_cm / df.volumetric_divisor
    raw = np.maximum(df.dead_wt_kg, vol_wt)
    df["vol_wt_kg"] = vol_wt
    df["exp_chargeable_kg"] = np.maximum(np.ceil(raw / 0.5) * 0.5, df.min_chargeable_weight_kg)

    first = df.exp_zone.map(rates.first_slab_rate_inr)
    extra = df.exp_zone.map(rates.additional_slab_rate_inr)
    gross = first + (df.exp_chargeable_kg / 0.5 - 1) * extra
    amended = df.discount_amendment_effective_from.notna() & (df.ship_date >= df.discount_amendment_effective_from)
    disc = np.where(amended, df.discount_amended_pct, df.freight_discount_pct)
    df["exp_freight"] = (gross * (1 - disc / 100)).round(2)

    cod = np.maximum(df.cod_min_fee_inr, df.cod_rate_pct / 100 * df.cod_value)
    waived = df.ship_date.between(df.cod_waiver_start, df.cod_waiver_end)
    df["exp_cod_fee"] = np.where((df.payment_mode == "COD") & ~waived, cod, 0.0).round(2)
    df["exp_total"] = df.exp_freight + df.exp_cod_fee
    return df


def add_billing_gap_features(df):
    df["overbill_amt"] = df.billed_total - df.exp_total
    df["overbill_pct"] = df.overbill_amt / df.exp_total
    df["freight_gap"] = df.billed_freight - df.exp_freight
    df["cod_fee_gap"] = df.billed_cod_fee - df.exp_cod_fee
    df["weight_gap_kg"] = df.billed_chargeable_kg - df.exp_chargeable_kg
    df["zone_mismatch"] = (df.billed_zone != df.exp_zone).astype(int)
    df["is_overbilled"] = (df.overbill_amt > 0.5).astype(int)
    df["is_cod"] = (df.payment_mode == "COD").astype(int)
    return df


# Left out because they carry the same information as features below:
#   billed_total (= billed_freight + billed_cod_fee), overbill_amt (= freight_gap + cod_fee_gap),
#   exp_total (= billed parts minus gaps), billed/dead vs declared weight (differences of weights here),
#   payment_mode (= is_cod), and contract terms that are constant per merchant (= merchant_id).
NUMERIC = [
    "declared_wt_kg", "dead_wt_kg", "vol_wt_kg", "cod_value", "is_cod",
    "billed_chargeable_kg", "billed_freight", "billed_cod_fee",
    "freight_gap", "cod_fee_gap", "weight_gap_kg", "overbill_pct", "zone_mismatch", "is_overbilled",
]
CATEGORICAL = ["merchant_id", "billed_zone"]


def build():
    ship, contracts, rates, regions = load()
    df = ship.merge(contracts, on="merchant_id", how="left")
    df = add_expected_charges(df, rates, regions)
    return add_billing_gap_features(df)
