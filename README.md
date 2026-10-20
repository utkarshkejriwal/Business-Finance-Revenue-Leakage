# Shadowfax Business Finance – Revenue Leakage

Three independent pieces of work over the same data (shipments for 1 Jul – 30 Sep 2026, 12 merchant contracts, and the Schedule A rate card):

| Section | Folder | What it does |
|---|---|---|
| 1. Leakage audit | [leakage_audit/](leakage_audit/) | Re-prices every shipment from the contracts and compares it with what was billed: underbilling, overbilling, root causes, and "false leakages". |
| 2. Dispute model | [dispute_model/](dispute_model/) | Predicts which shipments a merchant will dispute. Trained on July–August labels, scores September. |
| 3. Contract assistant | [contract_assistant/](contract_assistant/) | RAG chatbot that answers questions about the merchant agreements and Schedule A, with clause citations. |

## Repository layout

```
data/                              Inputs shared by all three sections
  shipments.csv                    7,000 shipments with billed values and dispute_raised flag
  merchant_contracts.csv           Contract terms for M01–M12 (discounts, amendments, COD terms, waivers…)
  schedule_a_*.csv                 Rate card: freight rates, zone rules, region lists, charging defaults
  contracts_and_rate_card.md       Full text of the agreements + Schedule A (source for the assistant)

leakage_audit/main.py              Section 1
*.csv (repo root)                  Section 1 outputs (written to the current directory)

dispute_model/
  features.py                      Recomputes expected charges and builds billing-gap features
  train.py                         Compares 4 classifiers + baseline, scores September
  tune.py                          Grid search over the hyperparameters in train.py
  predict_september.py             Final model (tuned elastic net) with calibrated probabilities
  outputs/                         Metrics, tuning grids, feature importance, September scores

contract_assistant/
  chunking.py → vector_store.py → retriever.py → graph.py → chat.py / app.py
  vector_store/                    Pre-built chunks + embeddings (rebuilt automatically if the source changes)
```

## Setup

Python 3.13 was used.

```bash
# Sections 1 and 2
pip install pandas numpy scikit-learn joblib

# Section 3
pip install -r contract_assistant/requirements.txt
```

Section 3 needs at least one LLM key in a `.env` file at the repo root (or in the environment):

```
ANTHROPIC_API_KEY=...      # or
GEMINI_API_KEY=...         # or
OPENROUTER_APIKEY=...      # OPENROUTER_API_KEY also works
```

---

## 1. Leakage audit

### Architecture

[leakage_audit/main.py](leakage_audit/main.py) is a single script:

1. **Load** shipments, contracts and the Schedule A tables from `./data`.
2. **Resolve terms per merchant**: contract values where set, Schedule A defaults otherwise (volumetric divisor, weight slab, minimum chargeable weight, COD fee basis).
3. **Re-price each shipment**
   - Zone from the Schedule A zone rules, applied in priority order (same city → special region → same state → metro–metro → rest).
   - Chargeable weight = max(dead, volumetric, minimum), rounded up to the slab.
   - Freight from the zone rate card, less the contract discount (or the amended discount from its effective date).
   - COD fee = max(minimum, % of COD value); zero inside a waiver window; prepaid fee for non-COD.
4. **Classify** each shipment by `variance = expected_total − billed_total` as Underbilled, Overbilled, No variance or Rounding (≤ ₹0.01).
5. **Root-cause** each variance. Freight is walked from billed to expected one input at a time (zone, then weight, then discount), plus a separate COD fee step. Each step's impact is recorded, and the script checks that the impacts add back exactly to the shipment's variance.
6. **False leakages**: shipments that look underbilled against standard Schedule A but are billed correctly under the merchant's special terms, plus rounding-only differences.

### Reproduce

Run from the **repo root**, because the script reads `./data/` and writes its CSVs to the current directory:

```bash
python leakage_audit/main.py
```

Outputs (repo root):

| File | Contents |
|---|---|
| `leakage_audit_results.csv` | Every shipment with expected values, variance and audit status |
| `billing_exceptions.csv` | Shipments with a non-zero variance |
| `billing_issue_details.csv` | One row per shipment × cause, with its ₹ impact |
| `billing_issue_summary.csv` | Impact by cause |
| `billing_issue_by_merchant.csv` | Impact by cause and merchant |
| `false_leakages.csv` | Shipments that are correctly billed but would look underbilled under standard Schedule A |

Expected headline: 7,000 shipments audited; 1,427 underbilled (₹32,899.52); 544 overbilled (₹6,585.00); 420 false leakages (₹3,754.13).

> The CSVs inside `leakage_audit/` come from an earlier version of the script (no root-cause or false-leakage columns). The current outputs are the ones in the repo root.

---

## 2. Dispute model

### Architecture

- **Features** ([features.py](dispute_model/features.py)): joins shipments to contracts, recomputes the contract-correct charge (vectorised, same rules as Section 1), and derives billing-gap features: `overbill_pct`, `freight_gap`, `cod_fee_gap`, `weight_gap_kg`, `zone_mismatch`, `is_overbilled`, plus weights, COD value, billed amounts, `merchant_id` and `billed_zone`. Features that duplicate information already in the set are left out on purpose; the reasons are in comments in the file.
- **Models** ([train.py](dispute_model/train.py)): logistic regression (L2), logistic elastic net, random forest and histogram gradient boosting, each in a scikit-learn pipeline (scaling + one-hot encoding). All use `class_weight="balanced"`. A rule baseline (`is_overbilled`) is included for comparison.
- **Out-of-time validation** (September has no labels yet):

  | Step | Train | Evaluate |
  |---|---|---|
  | Tuning (`tune.py`) | 1–31 Jul | 1–15 Aug (selects the hyperparameters) |
  | Reported holdout | 1 Jul – 15 Aug | 16–31 Aug |
  | September scoring | 1 Jul – 31 Aug | 1–30 Sep (unlabelled) |

  The decision threshold is the one that maximises F1 on the training period. The primary metric is PR-AUC.
- **Final model** ([predict_september.py](dispute_model/predict_september.py)): elastic net with `C=0.01, l1_ratio=0.1`, chosen by the tuning step. Its scores are re-calibrated on the odds scale to undo the class weighting, so `dispute_probability` matches real dispute rates. The uncalibrated `risk_score` is what gets thresholded.

### Reproduce

Run from inside `dispute_model/`, because the scripts import `features` and `train` as local modules:

```bash
cd dispute_model
python train.py              # model comparison, feature importance, September scores for every model
python tune.py               # grid search (312 combinations; takes a few minutes)
python predict_september.py  # final tuned elastic-net scores
```

To capture the console logs that are stored in `outputs/`: `python train.py > outputs/train_log.txt` and `python tune.py > outputs/tune_log.txt`.

Outputs (`dispute_model/outputs/`):

| Script | Files |
|---|---|
| `train.py` | `validation_metrics.csv`, `feature_importance_<best>.csv`, `september_predictions.csv`, `september_merchant_summary.csv` |
| `tune.py` | `tuning/<model>_grid.csv`, `tuning/tuning_summary.csv` |
| `predict_september.py` | `september_dispute_probabilities_elastic_net.csv`, `september_merchant_summary_elastic_net.csv` |

Expected holdout results (16–31 Aug, default hyperparameters): PR-AUC ranges from 0.50 to 0.52 across the models, against 0.39 for the `is_overbilled` baseline. Gradient boosting is best by PR-AUC, and `overbill_pct` and `freight_gap` are the top features. After tuning, every model scores PR-AUC 0.51–0.53. `predict_september.py` uses the tuned elastic net (`C=0.01, l1_ratio=0.1`). All models are seeded (`SEED = 42`), so re-runs give the same numbers.

---

## 3. Contract assistant

### Architecture

```
contracts_and_rate_card.md
   │  chunking.py      one chunk per "##" section; metadata: parent (M01…M12 / Schedule A),
   │                   section id, merchant, cross-references
   ▼
vector_store.py        BAAI/bge-small-en-v1.5 embeddings (fastembed), saved as
                       vector_store/{chunks.json, embeddings.npy, manifest.json}
   ▼
retriever.py           dense + BM25 → reciprocal rank fusion → merchant filter →
                       cross-encoder rerank (ms-marco-MiniLM-L-12-v2) → cross-reference expansion
   ▼
graph.py (LangGraph)   input_guard → rewrite_query → retrieve → generate → output_guard → finalize
                       (blocked / no match → finalize; failed output check → regenerate once)
   ▼
chat.py (CLI)  /  app.py (Streamlit)
```

- **Guardrails** ([guardrails.py](contract_assistant/guardrails.py)): regex prompt-injection and length checks, then an LLM scope classifier (in scope / small talk / out of scope / injection). On the output side, the answer is checked for valid clause citations, and its percentages and day counts must appear in the retrieved text.
- **LLM providers** ([llm.py](contract_assistant/llm.py)): Anthropic, Gemini or OpenRouter behind one interface. Each provider has a fallback list of models. Choose one with `LLM_PROVIDER=anthropic|gemini|openrouter` or `--provider`; otherwise the first provider with a key set is used.
- **Memory**: LangGraph `MemorySaver` keyed by thread id, so follow-up questions ("what about M03?") are rewritten into standalone questions.

### Reproduce

Run from inside `contract_assistant/`:

```bash
cd contract_assistant
python vector_store.py                       # optional: rebuild the store (also happens automatically if the source .md changes)
python chat.py                               # interactive CLI
python chat.py --sources "What is Kesari Foods' COD fee in September?"
python chat.py --provider gemini "Compare the liability caps across merchants"
streamlit run app.py                         # web UI
```

The first run downloads the embedding and reranker models into the fastembed cache. Retrieval is deterministic, but answer wording depends on the LLM provider and model, so answers can differ from run to run.
