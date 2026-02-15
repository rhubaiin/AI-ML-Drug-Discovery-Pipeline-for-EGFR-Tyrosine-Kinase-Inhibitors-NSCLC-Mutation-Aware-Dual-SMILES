# Experiment Plan — Models 1, 2, 3

## Objective

Cross-validate models 1, 2, and 3 by training each on both available datasets. This produces two output directories per model (6 runs total), allowing comparison of model performance across different training data.

---

## Models

| # | Name | Type | Inference available |
|---|------|------|---------------------|
| 1 | `adv_physchem5f2` | Advanced Feed-Forward NN (hierarchical + RNN) | Yes |
| 2 | `kan_bspline` | KAN + B-Spline | No |
| 3 | `kan_navier_stokes` | KAN + Fourier/RBF | Yes |

## Datasets

| Label | File | Rows | Notes |
|-------|------|------|-------|
| **Small** | `data/df_3_shuffled.csv` | ~751 | Cleaned and shuffled subset |
| **Large** | `data/manual_egfr3_mini_dock_fixed.csv` | ~2,677 | Includes docking scores |

---

## Run Matrix

| Run | Model | Training dataset | Inference | Output directory |
|-----|-------|-----------------|-----------|-----------------|
| A1 | 1 — Advanced FF | `df_3_shuffled.csv` | Yes | `experiments/model_1_adv_physchem5f2/df_3_shuffled/` |
| A2 | 2 — KAN B-Spline | `df_3_shuffled.csv` | No | `experiments/model_2_kan_bspline/df_3_shuffled/` |
| A3 | 3 — KAN Fourier | `df_3_shuffled.csv` | Yes | `experiments/model_3_kan_navier_stokes/df_3_shuffled/` |
| B1 | 1 — Advanced FF | `manual_egfr3_mini_dock_fixed.csv` | Yes | `experiments/model_1_adv_physchem5f2/manual_egfr3_mini_dock_fixed/` |
| B2 | 2 — KAN B-Spline | `manual_egfr3_mini_dock_fixed.csv` | No | `experiments/model_2_kan_bspline/manual_egfr3_mini_dock_fixed/` |
| B3 | 3 — KAN Fourier | `manual_egfr3_mini_dock_fixed.csv` | Yes | `experiments/model_3_kan_navier_stokes/manual_egfr3_mini_dock_fixed/` |

---

## Commands

All commands run from the `arshath/` directory.

### Run A — Train on small dataset (default)

```bash
python run_experiment.py --model 1 2 3 --predict_input data/test_egfr3.csv
```

### Run B — Train on large dataset (swapped)

```bash
python run_experiment.py --model 1 2 3 \
  --train_data data/manual_egfr3_mini_dock_fixed.csv \
  --predict_input data/test_egfr3.csv
```

> Model 2 has no inference script. The orchestrator will skip inference for it automatically and print a note.

---

## Expected Output Structure

```
experiments/
├── model_1_adv_physchem5f2/
│   ├── df_3_shuffled/
│   │   ├── hierarchical_model_full.h5
│   │   ├── hierarchical_model_hinge_p_loop.h5
│   │   ├── hierarchical_model_c_helix.h5
│   │   ├── hierarchical_model_a_loop_dfg.h5
│   │   ├── hierarchical_model_hrd_cat.h5
│   │   ├── rnn_sequential_model.h5
│   │   ├── feature_scalers.pkl
│   │   ├── y_scalers.pkl
│   │   ├── mutation_profiles.csv
│   │   ├── control_predictions_rnn.csv
│   │   ├── drug_predictions_rnn.csv
│   │   ├── rnn_training_history.png
│   │   └── predictions/
│   │       ├── predictions_adv_physchem5f2.csv
│   │       ├── metrics/
│   │       └── plots/
│   └── manual_egfr3_mini_dock_fixed/
│       ├── (same files as above)
│       └── predictions/
│
├── model_2_kan_bspline/
│   ├── df_3_shuffled/
│   │   ├── hierarchical_model_*.h5 (x5)
│   │   ├── rnn_sequential_model.h5
│   │   ├── feature_scalers.pkl
│   │   ├── y_scalers.pkl
│   │   ├── mutation_profiles.csv
│   │   ├── control_predictions_rnn.csv
│   │   ├── drug_predictions_rnn.csv
│   │   └── kan_training_history.png
│   └── manual_egfr3_mini_dock_fixed/
│       └── (same files as above)
│
└── model_3_kan_navier_stokes/
    ├── df_3_shuffled/
    │   ├── hierarchical_model_*.h5 (x5)
    │   ├── rnn_sequential_model.h5
    │   ├── feature_scalers.pkl
    │   ├── y_scalers.pkl
    │   ├── mutation_profiles.csv
    │   ├── control_predictions_rnn.csv
    │   ├── drug_predictions_rnn.csv
    │   ├── kan_training_history.png
    │   └── predictions/
    │       ├── predictions_rnn_lstm_kan.csv
    │       ├── metrics/
    │       └── plots/
    └── manual_egfr3_mini_dock_fixed/
        ├── (same files as above)
        └── predictions/
```

---

## What to Compare

After all 6 runs complete, compare across the two dataset columns for each model:

| Metric | Where to find it |
|--------|-----------------|
| Training loss curves | `*_training_history.png` in each run directory |
| Control compound predictions | `control_predictions_rnn.csv` — check if known drugs (erlotinib, osimertinib, etc.) rank correctly |
| Drug candidate predictions | `drug_predictions_rnn.csv` — compare predicted activity/docking across datasets |
| Inference metrics (models 1, 3) | `predictions/metrics/` — per-mutation and overall error metrics |
| Inference plots (models 1, 3) | `predictions/plots/` — predicted vs actual scatter plots |

### Key questions to answer

1. **Dataset sensitivity**: Do models trained on the small dataset (751 rows) perform comparably to the large dataset (2,677 rows)?
2. **Model comparison**: Which architecture (Advanced FF vs KAN B-Spline vs KAN Fourier) produces the best activity/docking predictions?
3. **Control validation**: Do all models correctly identify known EGFR TKIs as active compounds?
4. **Generalization**: Do models trained on one dataset produce reasonable predictions on the test set (`test_egfr3.csv`)?
