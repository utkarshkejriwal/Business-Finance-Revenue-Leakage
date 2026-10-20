
import math
import sys

import pandas as pd

# The ₹ symbol cannot be printed on a default Windows (cp1252) console.
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)

# Variances at or below this are treated as rounding, not leakage.
ROUNDING_TOLERANCE = 0.01

# Read source files

shipments = pd.read_csv("./data/shipments.csv")
contracts = pd.read_csv("./data/merchant_contracts.csv")
freight_rates = pd.read_csv("./data/schedule_a_freight_rates.csv")
charging_rules = pd.read_csv("./data/schedule_a_charging_rules.csv")
regions = pd.read_csv("./data/schedule_a_region_lists.csv")
zone_rules = pd.read_csv("./data/schedule_a_zone_rules.csv")

shipments["ship_date"] = pd.to_datetime(shipments["ship_date"])

date_columns = [
    "effective_date",
    "discount_amendment_effective_from",
    "cod_waiver_start",
    "cod_waiver_end",
]
for col in date_columns:
    contracts[col] = pd.to_datetime(contracts[col], errors="coerce")

# Fail early if any shipment has no matching contract.
contract_by_id = contracts.set_index("merchant_id")
missing = set(shipments["merchant_id"]) - set(contract_by_id.index)
if missing:
    raise ValueError(f"Missing contracts for merchant IDs: {sorted(missing)}")

rate_by_zone = freight_rates.set_index("zone")
defaults = dict(zip(charging_rules["parameter"],
                    charging_rules["default_value"]))

def default_number(parameter):
    return float(defaults[parameter])

default_divisor = default_number("volumetric_divisor")
rounding_slab = default_number("weight_rounding_slab")
default_min_weight = default_number("min_chargeable_weight")
default_cod_min = default_number("cod_min_fee")
default_cod_pct = default_number("cod_rate_pct") / 100
prepaid_cod_fee = default_number("cod_fee_on_prepaid")

special_states = set(
    regions.loc[
        (regions["category"] == "Special Region")
        & (regions["type"] == "state"), "name"
    ].astype(str)
)
metro_cities = set(
    regions.loc[
        (regions["category"] == "Metro")
        & (regions["type"] == "city"), "name"
    ].astype(str)
)

zone_rules = zone_rules.sort_values("priority")

# Contract terms per merchant, falling back to Schedule A defaults
def contract_value(contract, column, default):
    return float(contract[column]) if pd.notna(contract[column]) else default

terms_by_merchant = {}
for merchant_id, contract in contract_by_id.iterrows():
    custom_cod = str(contract["cod_fee_basis"]).strip().lower() != "schedule a"
    terms_by_merchant[merchant_id] = {
        "divisor": contract_value(contract, "volumetric_divisor",
                                  default_divisor),
        "slab_size": contract_value(contract, "weight_slab_kg",
                                    rounding_slab),
        "min_weight": contract_value(contract, "min_chargeable_weight_kg",
                                     default_min_weight),
        "custom_cod": custom_cod,
        "cod_min": (float(contract["cod_min_fee_inr"])
                    if custom_cod else default_cod_min),
        "cod_pct": (float(contract["cod_rate_pct"]) / 100
                    if custom_cod else default_cod_pct),
    }

# retrieve zone based on Schedule A
def get_zone(row):
    contract = contract_by_id.loc[row["merchant_id"]]

    origin_city = str(row["origin_city"]).strip()
    dest_city = str(row["dest_city"]).strip()
    origin_state = str(contract["pickup_state"]).strip()
    dest_state = str(row["dest_state"]).strip()

    for _, rule in zone_rules.iterrows():
        condition = str(rule["condition"]).lower()
        zone = rule["zone"]

        if "same city" in condition:
            if origin_city.casefold() == dest_city.casefold():
                return zone

        elif "special region" in condition:
            if dest_state in special_states:
                return zone

        elif "same state" in condition:
            if origin_state.casefold() == dest_state.casefold():
                return zone

        elif "both metro" in condition:
            if origin_city in metro_cities and dest_city in metro_cities:
                return zone

        elif "all other" in condition:
            return zone

    raise ValueError(f"No zone rule matched AWB {row['awb']}")

def chargeable_weight(row, divisor, min_weight, slab_size):
    volumetric_weight = (
        row["length_cm"] * row["width_cm"] * row["height_cm"]
    ) / divisor
    raw_weight = max(
        float(row["dead_wt_kg"]),
        volumetric_weight,
        min_weight,
    )
    return round(math.ceil(raw_weight / slab_size - 1e-9) * slab_size, 3)

def gross_freight(zone, weight):
    rate = rate_by_zone.loc[zone]
    number_of_slabs = max(
        1, math.ceil(weight / float(rate["additional_slab_kg"]) - 1e-9)
    )
    return (
        float(rate["first_slab_rate_inr"])
        + (number_of_slabs - 1) * float(rate["additional_slab_rate_inr"])
    )

def discounted_freight(zone, weight, discount_pct):
    return round(gross_freight(zone, weight) * (1 - discount_pct / 100), 2)

def cod_fee(cod_value, cod_min, cod_pct):
    return round(max(cod_min, cod_pct * float(cod_value)), 2)

# Expected charges
def calculate_expected(row):
    contract = contract_by_id.loc[row["merchant_id"]]
    terms = terms_by_merchant[row["merchant_id"]]
    zone = get_zone(row)

    weight = chargeable_weight(
        row, terms["divisor"], terms["min_weight"], terms["slab_size"]
    )

    base_discount_pct = float(contract["freight_discount_pct"])
    discount_pct = base_discount_pct

    if (
        pd.notna(contract["discount_amended_pct"])
        and pd.notna(contract["discount_amendment_effective_from"])
        and row["ship_date"]
        >= contract["discount_amendment_effective_from"]
    ):
        discount_pct = float(contract["discount_amended_pct"])

    expected_freight = discounted_freight(zone, weight, discount_pct)

    is_cod = str(row["payment_mode"]).strip().upper() == "COD"
    waiver_start = contract["cod_waiver_start"]
    waiver_end = contract["cod_waiver_end"]
    waiver_active = (
        is_cod
        and pd.notna(waiver_start)
        and pd.notna(waiver_end)
        and waiver_start <= row["ship_date"] <= waiver_end
    )

    if not is_cod:
        expected_cod_fee = prepaid_cod_fee
    elif waiver_active:
        expected_cod_fee = 0.0
    else:
        expected_cod_fee = cod_fee(
            row["cod_value"], terms["cod_min"], terms["cod_pct"]
        )

    expected_total = round(expected_freight + expected_cod_fee, 2)

    # Standard Schedule A charge with only the base discount, i.e. what the
    # shipment would cost without the merchant's special contract terms.
    standard_weight = chargeable_weight(
        row, default_divisor, default_min_weight, rounding_slab
    )
    standard_cod_fee = (
        cod_fee(row["cod_value"], default_cod_min, default_cod_pct)
        if is_cod else prepaid_cod_fee
    )
    standard_total = round(
        discounted_freight(zone, standard_weight, base_discount_pct)
        + standard_cod_fee, 2
    )

    terms_applied = []
    if (
        terms["divisor"] != default_divisor
        and chargeable_weight(row, default_divisor, terms["min_weight"],
                              terms["slab_size"]) != weight
    ):
        terms_applied.append(f"Volumetric divisor {terms['divisor']:g}")
    if (
        terms["min_weight"] != default_min_weight
        and chargeable_weight(row, terms["divisor"], default_min_weight,
                              terms["slab_size"]) != weight
    ):
        terms_applied.append(
            f"Minimum chargeable weight {terms['min_weight']:g} kg"
        )
    if discount_pct != base_discount_pct:
        terms_applied.append(f"Discount amended to {discount_pct:g}%")
    if waiver_active:
        terms_applied.append("COD fee waiver")
    elif is_cod and expected_cod_fee != standard_cod_fee:
        terms_applied.append("Custom COD fee")

    return pd.Series({
        "expected_zone": zone,
        "expected_weight": weight,
        "expected_discount_pct": discount_pct,
        "expected_freight": expected_freight,
        "expected_cod_fee": round(expected_cod_fee, 2),
        "expected_total": expected_total,
        "standard_total": standard_total,
        "contract_terms_applied": "; ".join(terms_applied),
    })

# calculating the leakage audit
calculated = shipments.apply(calculate_expected, axis=1)
results = pd.concat([shipments, calculated], axis=1)

results["variance"] = (
    results["expected_total"] - results["billed_total"]
).round(2)

results["audit_status"] = "No variance"
results.loc[results["variance"] > ROUNDING_TOLERANCE,
            "audit_status"] = "Underbilled"
results.loc[results["variance"] < -ROUNDING_TOLERANCE,
            "audit_status"] = "Overbilled"
results.loc[(results["variance"] != 0)
            & (results["variance"].abs() <= ROUNDING_TOLERANCE),
            "audit_status"] = "Rounding"

underbilled = results[results["audit_status"] == "Underbilled"]
overbilled = results[results["audit_status"] == "Overbilled"]

print(f"Shipments audited: {len(results):,}")
print(f"Underbilled shipments: {len(underbilled):,}")
print(f"Underbilling amount: ₹{underbilled['variance'].sum():,.2f}")
print(f"Overbilled shipments: {len(overbilled):,}")
print(f"Potential overbilling amount: ₹{-overbilled['variance'].sum():,.2f}")
print(f"No-variance shipments: {(results['variance'] == 0).sum():,}")

print("\nTop 3 merchants by underbilling:")
print(
    underbilled.groupby("merchant_id")["variance"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
    .to_string()
)

# What is wrong in billing: split each shipment's variance into causes.
# Freight is walked from the billed inputs to the expected inputs one at a
# time (zone, then weight, then discount), so the impacts add up exactly.
def weight_issue(row, terms, billed_weight, expected_weight):
    if (
        terms["divisor"] != default_divisor
        and billed_weight == chargeable_weight(
            row, default_divisor, terms["min_weight"], terms["slab_size"])
    ):
        return (f"Volumetric divisor {default_divisor:g} used instead of "
                f"{terms['divisor']:g}")
    if (
        terms["min_weight"] != default_min_weight
        and billed_weight == chargeable_weight(
            row, terms["divisor"], default_min_weight, terms["slab_size"])
    ):
        return (f"Minimum chargeable weight {terms['min_weight']:g} kg "
                f"not applied")
    if billed_weight < expected_weight:
        return "Chargeable weight billed below measured weight"
    return "Chargeable weight billed above measured weight"

def discount_issue(row, billed_discount_pct):
    base_discount_pct = float(
        contract_by_id.loc[row["merchant_id"], "freight_discount_pct"]
    )
    expected_discount_pct = row["expected_discount_pct"]
    if (
        expected_discount_pct != base_discount_pct
        and abs(billed_discount_pct - base_discount_pct) < 0.05
    ):
        return (f"Discount amendment to {expected_discount_pct:g}% not "
                f"applied (billed {billed_discount_pct:g}%)")
    return (f"Discount billed at {billed_discount_pct:g}% instead of "
            f"{expected_discount_pct:g}%")

def cod_issue(row, terms, billed_cod_fee, expected_cod_fee):
    if str(row["payment_mode"]).strip().upper() != "COD":
        return "COD fee charged on prepaid shipment"
    if expected_cod_fee == 0:
        return "COD fee charged during waiver period"
    if billed_cod_fee == 0:
        return "COD fee not charged"
    if terms["custom_cod"]:
        if billed_cod_fee == cod_fee(row["cod_value"], default_cod_min,
                                     default_cod_pct):
            return "Schedule A COD fee used instead of contract COD fee"
        return (f"Contract COD fee (higher of ₹{terms['cod_min']:g} or "
                f"{terms['cod_pct'] * 100:g}%) not applied")
    if billed_cod_fee < expected_cod_fee:
        return "COD fee billed below Schedule A"
    return "COD fee billed above Schedule A"

def billing_issues(row):
    terms = terms_by_merchant[row["merchant_id"]]
    billed_zone = row["billed_zone"]
    expected_zone = row["expected_zone"]
    billed_weight = round(float(row["billed_chargeable_kg"]), 3)
    expected_weight = row["expected_weight"]
    billed_discount_pct = round(
        (1 - row["billed_freight"]
         / gross_freight(billed_zone, billed_weight)) * 100, 1
    )

    freight_billed = discounted_freight(
        billed_zone, billed_weight, billed_discount_pct)
    freight_zone_fixed = discounted_freight(
        expected_zone, billed_weight, billed_discount_pct)
    freight_weight_fixed = discounted_freight(
        expected_zone, expected_weight, billed_discount_pct)

    issues = []

    def add(error_type, issue, impact):
        impact = round(impact, 2)
        if impact != 0:
            issues.append((error_type, issue, impact))

    if billed_zone != expected_zone:
        add("Zone", f"Zone {billed_zone} billed instead of {expected_zone}",
            freight_zone_fixed - freight_billed)
    if billed_weight != expected_weight:
        add("Weight",
            weight_issue(row, terms, billed_weight, expected_weight),
            freight_weight_fixed - freight_zone_fixed)
    if abs(billed_discount_pct - row["expected_discount_pct"]) >= 0.05:
        add("Discount", discount_issue(row, billed_discount_pct),
            row["expected_freight"] - freight_weight_fixed)

    explained_freight = sum(impact for _, _, impact in issues)
    add("Other", "Unexplained freight difference",
        row["expected_freight"] - row["billed_freight"] - explained_freight)

    billed_cod_fee = round(float(row["billed_cod_fee"]), 2)
    expected_cod_fee = row["expected_cod_fee"]
    if billed_cod_fee != expected_cod_fee:
        add("COD", cod_issue(row, terms, billed_cod_fee, expected_cod_fee),
            expected_cod_fee - billed_cod_fee)

    return issues

issue_rows = []
for _, row in results.iterrows():
    for error_type, issue, impact in billing_issues(row):
        issue_rows.append({
            "awb": row["awb"],
            "merchant_id": row["merchant_id"],
            "ship_date": row["ship_date"],
            "error_type": error_type,
            "issue": issue,
            "impact": impact,
        })
issue_details = pd.DataFrame(issue_rows)

# The per-cause impacts must add back to each shipment's variance.
reconciled = (
    issue_details.groupby("awb")["impact"].sum()
    .reindex(results["awb"], fill_value=0)
    .values
)
unreconciled = (reconciled - results["variance"].values).round(2) != 0
if unreconciled.any():
    raise ValueError(
        f"{unreconciled.sum()} shipments do not reconcile to their variance"
    )

def summarise_issues(details, keys):
    return (
        details.groupby(keys)
        .agg(
            merchants=("merchant_id",
                       lambda ids: ", ".join(sorted(ids.unique()))),
            first_ship_date=("ship_date", "min"),
            last_ship_date=("ship_date", "max"),
            shipments=("awb", "nunique"),
            underbilled_inr=("impact", lambda x: x[x > 0].sum()),
            overbilled_inr=("impact", lambda x: abs(x[x < 0].sum())),
            net_impact_inr=("impact", "sum"),
        )
        .round(2)
        .reset_index()
        .sort_values("net_impact_inr", key=abs, ascending=False)
    )

issue_summary = summarise_issues(issue_details, ["error_type", "issue"])
issue_by_merchant = (
    summarise_issues(issue_details, ["error_type", "issue", "merchant_id"])
    .drop(columns="merchants")
    .sort_values(["error_type", "issue", "net_impact_inr"],
                 ascending=[True, True, False])
)
for frame in (issue_summary, issue_by_merchant):
    for col in ("first_ship_date", "last_ship_date"):
        frame[col] = frame[col].dt.date

print("\nWhat is wrong in billing "
      "(net impact: + underbilled, - overbilled; a shipment can have "
      "more than one issue):")
print(issue_summary.to_string(index=False))

print("\nBilling issues by merchant:")
print(issue_by_merchant.to_string(index=False))

# False leakages: shipments that look underbilled against the standard
# Schedule A charge, but are correctly billed under the merchant's special
# contract terms, plus variances that are only rounding.
results["standard_variance"] = (
    results["standard_total"] - results["billed_total"]
).round(2)
correctly_billed = results["variance"].abs() <= ROUNDING_TOLERANCE
contract_false = correctly_billed & (
    results["standard_variance"] > ROUNDING_TOLERANCE
)
results["false_leakage_inr"] = 0.0
results["false_leakage_reason"] = ""
results.loc[contract_false, "false_leakage_inr"] = results.loc[
    contract_false, "standard_variance"]
results.loc[contract_false, "false_leakage_reason"] = results.loc[
    contract_false, "contract_terms_applied"]

rounding = (results["audit_status"] == "Rounding") & ~contract_false
results.loc[rounding, "false_leakage_inr"] = results.loc[
    rounding, "variance"].abs()
results.loc[rounding, "false_leakage_reason"] = (
    f"Rounding difference (≤ ₹{ROUNDING_TOLERANCE:g})"
)

false_leakages = results[results["false_leakage_inr"] > 0]
false_leakage_summary = (
    false_leakages.groupby(["false_leakage_reason", "merchant_id"])
    .agg(
        first_ship_date=("ship_date", "min"),
        last_ship_date=("ship_date", "max"),
        shipments=("awb", "count"),
        false_leakage_inr=("false_leakage_inr", "sum"),
    )
    .round(2)
    .reset_index()
)
for col in ("first_ship_date", "last_ship_date"):
    false_leakage_summary[col] = false_leakage_summary[col].dt.date

print(f"\nFalse leakages: {len(false_leakages):,} shipments, "
      f"₹{false_leakages['false_leakage_inr'].sum():,.2f} that would look "
      f"underbilled against standard Schedule A but is correctly billed")
print(false_leakage_summary.to_string(index=False))

results.to_csv("leakage_audit_results.csv", index=False)

results[results["variance"] != 0].to_csv(
    "billing_exceptions.csv", index=False
)
issue_details.to_csv("billing_issue_details.csv", index=False)
issue_summary.to_csv("billing_issue_summary.csv", index=False)
issue_by_merchant.to_csv("billing_issue_by_merchant.csv", index=False)
false_leakages.to_csv("false_leakages.csv", index=False)

print("\nSaved leakage_audit_results.csv, billing_exceptions.csv, "
      "billing_issue_details.csv, billing_issue_summary.csv, "
      "billing_issue_by_merchant.csv and false_leakages.csv")
