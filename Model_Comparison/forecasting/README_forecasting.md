# Retrospective Forecasting Module — SPB Winter 2022

## 1. What this module does

This module implements a **retrospective forecasting experiment** using the real
confirmed-case incidence curve from Saint Petersburg during the winter 2022
COVID-19 wave.

The experiment evaluates whether the SBM-SIR mechanistic model and/or its LSTM
surrogate can correctly predict the short-term evolution of the epidemic when
only a partial initial segment of the curve is available.

> **Key distinction:**
> The previous validation plot shows that the full SBM-SIR/surrogate framework
> can reproduce the observed SPB 2022 wave when the complete curve is available.
> In contrast, this forecasting module hides part of the observed curve and
> evaluates whether the model can predict the next 14 days from partial
> information.

---

## 2. Files required

The script must be run from inside `Model_Comparison/` (or equivalently with
the path `python forecasting/forecast_spb_2022.py` called from that directory).
It expects the following files to be present relative to `Model_Comparison/`:

| File | Purpose |
|---|---|
| `stopkoronavirus_clean_wave_winter.csv` | Real SPB 2022 incidence data |
| `LSTM_SBM.py` | Surrogate model architecture and SBM simulator wrapper |
| `test_simulation.py` | Model configuration templates |
| `simple_sbm_generator.py` | SBM-SIR network generator |
| `output/ai_sbm/dataset_normalized.npz` | Training dataset (scalers) |
| `output/ai_sbm/surrogate_model_normalized.pth` | Trained surrogate weights |

> If the `.npz` or `.pth` files are missing, run `LSTM_SBM.py` first.

---

## 3. How to run

```bash
# From Model_Comparison/ directory:
python forecasting/forecast_spb_2022.py
```

To switch between surrogate and mechanistic mode, edit the top of
`forecast_spb_2022.py`:

```python
MODE = "surrogate"   # fast  — uses LSTM surrogate
MODE = "sbm"         # slow  — uses the mechanistic SBM-SIR directly
```

To switch peak detection method:

```python
PEAK_MODE = "raw"     # uses argmax of the observed series (recommended)
PEAK_MODE = "manual"  # uses MANUAL_PEAK_DATE = "2022-02-08"
```

---

## 4. Outputs generated

All outputs are written to `forecasting/output/`:

| File | Description |
|---|---|
| `forecast_tpeak_minus_14.png` | Forecast figure: cut at t_peak − 14 |
| `forecast_tpeak_minus_7.png` | Forecast figure: cut at t_peak − 7 |
| `forecast_tpeak_plus_7.png` | Forecast figure: cut at t_peak + 7 |
| `forecast_all_scenarios.png` | Three-panel combined figure |
| `forecast_metrics.csv` | Evaluation metrics per scenario |
| `forecast_parameters.csv` | Top-K calibrated parameters per scenario |
| `observed_hidden_splits.csv` | Known/hidden data split per scenario |
| `forecast_summary.md` | Auto-generated markdown summary |

---

## 5. Scenarios explained

| Scenario | Cut date | Interpretation |
|---|---|---|
| `tpeak_minus_14` | Peak − 14 days | Early forecast; model sees only growth phase |
| `tpeak_minus_7` | Peak − 7 days | Near-peak forecast; more wave shape is visible |
| `tpeak_plus_7` | Peak + 7 days | Post-peak; validates model decline behavior |

For each scenario:
- **Known segment**: all data up to and including the cut date.
- **Hidden segment**: the 14 days immediately after the cut date.
- The model is calibrated **exclusively** on the known segment.
- The hidden segment is used **only** for evaluation.

---

## 6. Full-curve validation vs. retrospective forecasting

| Aspect | Full-curve validation | Retrospective forecasting |
|---|---|---|
| Data used for fitting | Complete observed curve | Only data before cut date |
| Data used for evaluation | Same curve (in-sample) | Hidden future data (out-of-sample) |
| Purpose | Confirm model can reproduce the wave | Test predictive ability from partial info |
| Risk of overfitting | High | Low (strict data separation) |

---

## 7. Thesis note

> "After validating the SBM-SIR model against the complete SPB 2022 epidemic
> wave, a retrospective forecasting experiment was performed. The real incidence
> curve was divided into observed and hidden segments using three forecast
> origins: 14 days before the observed peak, 7 days before the observed peak,
> and 7 days after the observed peak. For each forecast origin, the model was
> calibrated only on the available historical data and then used to generate a
> 14-day forecast. The hidden real data were used exclusively for evaluation."
