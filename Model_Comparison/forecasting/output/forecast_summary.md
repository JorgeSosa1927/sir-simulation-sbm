# Retrospective Forecasting Experiment — SPB Winter 2022

## Objective
This experiment evaluates whether the SBM-SIR model and/or its surrogate LSTM approximation can forecast the short-term evolution of the SPB winter 2022 epidemic wave using only partial observed data.

## Method
The observed incidence curve is split into known and hidden segments using three forecast origins:
- 14 days before the epidemic peak.
- 7 days before the epidemic peak.
- 7 days after the epidemic peak.

For each scenario, the model is calibrated only on the known segment. The following 14 days are then forecasted and compared against the hidden real data.

## Peak Definition
- **Raw peak date (argmax):** 2022-02-11
- **Selected peak mode:** `raw`
- **Peak date used:** 2022-02-11

The observed SPB winter 2022 wave shows a high-incidence peak window around February 7–11, 2022. For reproducibility, the main forecasting experiment uses the raw maximum of the observed series as the official peak date.

## Metrics

| Scenario | Cut Date | RMSE | MAE | R² | Peak Δt (d) | Peak Height Err | Coverage |
|---|---|---|---|---|---|---|---|
| tpeak_minus_14 | 2022-01-28 | 5977.1 | 5247.0 | -1.658 | -6 | -0.365 | 42.86% |
| tpeak_minus_7 | 2022-02-04 | 4777.8 | 4466.1 | -0.040 | -6 | -0.217 | 85.71% |
| tpeak_plus_7 | 2022-02-18 | 2481.2 | 1865.7 | 0.675 | 8 | -0.315 | 100.00% |

## Interpretation

**t_peak − 14:** Early forecast. Higher uncertainty is expected because the model sees only the initial growth phase.

**t_peak − 7:** Near-peak forecast. The model has more information about the wave growth, so better peak timing and magnitude are expected.

**t_peak + 7:** Post-peak forecast. Less useful for early warning, but useful to validate whether the model captures the epidemic decline.

## Methodological Note

This is a retrospective forecasting experiment using real SPB 2022 incidence data. The model is calibrated only on the observed segment before each forecast origin, while the hidden future segment is used exclusively for evaluation.
