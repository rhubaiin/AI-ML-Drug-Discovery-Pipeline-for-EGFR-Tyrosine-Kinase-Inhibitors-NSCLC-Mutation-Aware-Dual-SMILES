# Experiment Comparison: New Run (Feb 28) vs Backup Run (Feb 18)

## Overview

This document compares cross-CSV prediction metrics between two independent training runs:
- **New Run** (`experiments/`) - Feb 28, 2026
- **Backup Run** (`experiments_backup/`) - Feb 18, 2026

Both runs used the same code, datasets, and hyperparameters. Differences arise from random initialization, stochastic training, and non-deterministic GPU operations.

---

## 1. Overall Pearson R Comparison

### Direction A: Trained on df_3_shuffled -> Predict on manual_egfr3_mini_dock_fixed (N=2676)

| Model | Activity R (New) | Activity R (Backup) | Delta | Docking R (New) | Docking R (Backup) | Delta |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| M0 - Dummy PhysChem | -0.005 | 0.033 | -0.038 | **0.415** | **0.460** | -0.045 |
| M1 - Adv PhysChem | -0.026 | 0.073 | -0.099 | 0.190 | 0.132 | +0.058 |
| M2 - KAN B-Spline | 0.036 | 0.067 | -0.031 | -0.044 | 0.067 | **-0.111** |
| M3 - KAN Navier-Stokes | **0.161** | **0.170** | -0.009 | **0.316** | 0.206 | **+0.110** |
| M4 - ChemBERTa | -0.047 | -0.107 | +0.060 | 0.122 | **0.241** | -0.119 |
| M5 - GNN | **0.257** | 0.023 | **+0.234** | -0.009 | 0.017 | -0.026 |

### Direction B: Trained on manual_egfr3_mini_dock_fixed -> Predict on df_3_shuffled (N=748-750)

| Model | Activity R (New) | Activity R (Backup) | Delta | Docking R (New) | Docking R (Backup) | Delta |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| M0 - Dummy PhysChem | **0.087** | 0.012 | +0.075 | **0.465** | **0.490** | -0.025 |
| M1 - Adv PhysChem | -0.032 | -0.024 | -0.008 | 0.161 | 0.430 | **-0.269** |
| M2 - KAN B-Spline | -0.042 | -0.029 | -0.013 | **-0.315** | 0.065 | **-0.380** |
| M3 - KAN Navier-Stokes | 0.025 | 0.019 | +0.006 | **0.534** | 0.363 | **+0.171** |
| M4 - ChemBERTa | -0.018 | 0.003 | -0.021 | **0.543** | 0.498 | +0.045 |
| M5 - GNN | 0.006 | -0.024 | +0.030 | 0.510 | 0.506 | +0.004 |

---

## 2. Average Across Both Directions

| Model | Avg Activity R (New) | Avg Activity R (Backup) | Delta | Avg Docking R (New) | Avg Docking R (Backup) | Delta |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| M0 - Dummy PhysChem | 0.041 | 0.023 | +0.018 | **0.440** | **0.475** | -0.035 |
| M1 - Adv PhysChem | -0.029 | 0.024 | -0.053 | 0.176 | 0.281 | -0.105 |
| M2 - KAN B-Spline | -0.003 | 0.019 | -0.022 | -0.180 | 0.066 | **-0.246** |
| M3 - KAN Navier-Stokes | **0.093** | **0.095** | -0.002 | **0.425** | 0.285 | **+0.140** |
| M4 - ChemBERTa | -0.033 | -0.052 | +0.019 | 0.332 | 0.369 | -0.037 |
| M5 - GNN | 0.131 | -0.000 | **+0.131** | 0.250 | 0.262 | -0.012 |

---

## 3. Stability Analysis: Which Models Are Reproducible?

Measuring absolute delta between runs (smaller = more stable):

| Model | |Delta Activity R| (Dir A) | |Delta Activity R| (Dir B) | |Delta Docking R| (Dir A) | |Delta Docking R| (Dir B) | Avg |Delta| |
|-------|:---:|:---:|:---:|:---:|:---:|
| M0 - Dummy PhysChem | 0.038 | 0.075 | 0.045 | 0.025 | 0.046 |
| M1 - Adv PhysChem | 0.099 | 0.008 | 0.058 | 0.269 | 0.109 |
| M2 - KAN B-Spline | 0.031 | 0.013 | 0.111 | 0.380 | **0.134** |
| M3 - KAN Navier-Stokes | 0.009 | 0.006 | 0.110 | 0.171 | 0.074 |
| M4 - ChemBERTa | 0.060 | 0.021 | 0.119 | 0.045 | 0.061 |
| M5 - GNN | 0.234 | 0.030 | 0.026 | 0.004 | 0.074 |

**Stability ranking** (most to least stable across runs):
1. **M0 - Dummy PhysChem** (avg |delta| = 0.046) - Most reproducible
2. **M4 - ChemBERTa** (avg |delta| = 0.061)
3. **M3 - KAN Navier-Stokes** (avg |delta| = 0.074)
4. **M5 - GNN** (avg |delta| = 0.074)
5. **M1 - Adv PhysChem** (avg |delta| = 0.109)
6. **M2 - KAN B-Spline** (avg |delta| = 0.134) - Least reproducible

---

## 4. Per-Mutation Comparison (Selected Key Mutations)

### 4a. `del` mutation (Exon 19 deletion) - Activity Pearson R

| Model | Dir A New | Dir A Backup | Dir B New | Dir B Backup |
|-------|:---:|:---:|:---:|:---:|
| M0 | -0.021 | -0.029 | **0.970** | 0.219 |
| M1 | -0.042 | 0.100 | 0.147 | **0.462** |
| M2 | -0.032 | 0.055 | 0.062 | -0.042 |
| M3 | **0.354** | **0.185** | **0.904** | 0.019 |
| M4 | -0.055 | -0.073 | 0.333 | **0.710** |
| M5 | 0.162 | 0.021 | **0.863** | **0.570** |

### 4b. `l858r/t790m double` mutation - Activity Pearson R

| Model | Dir A New | Dir A Backup | Dir B New | Dir B Backup |
|-------|:---:|:---:|:---:|:---:|
| M0 | -0.014 | 0.074 | **0.514** | **0.325** |
| M1 | -0.062 | 0.067 | -0.378 | -0.307 |
| M2 | -0.017 | 0.109 | -0.445 | -0.555 |
| M3 | **0.122** | **0.269** | **0.585** | **0.500** |
| M4 | -0.012 | -0.088 | 0.194 | -0.067 |
| M5 | **0.247** | 0.017 | -0.089 | -0.465 |

### 4c. `del` mutation - Docking Pearson R

| Model | Dir A New | Dir A Backup | Dir B New | Dir B Backup |
|-------|:---:|:---:|:---:|:---:|
| M0 | **0.594** | **0.545** | **0.647** | **0.625** |
| M1 | **0.502** | **0.468** | 0.489 | **0.544** |
| M2 | 0.087 | -0.119 | -0.076 | **0.641** |
| M3 | **0.322** | 0.140 | **0.873** | **0.755** |
| M4 | **0.324** | **0.625** | **0.835** | **0.532** |
| M5 | 0.172 | 0.084 | **0.723** | **0.958** |

---

## 5. MAE Comparison (Overall)

### Direction A: df_3_shuffled -> manual_egfr3_mini_dock_fixed

| Model | Activity MAE (New) | Activity MAE (Backup) | Docking MAE (New) | Docking MAE (Backup) |
|-------|:---:|:---:|:---:|:---:|
| M0 | 4.06e+12 | 1.01e+08 | 0.657 | **0.689** |
| M1 | 3214 | 2935 | 0.797 | 0.876 |
| M2 | 2900 | 2566 | 0.715 | 0.867 |
| M3 | **2759** | **2891** | **0.627** | **0.692** |
| M4 | 2910 | 2714 | 0.716 | 0.682 |
| M5 | **2561** | **2563** | 0.700 | 1.049 |

### Direction B: manual_egfr3_mini_dock_fixed -> df_3_shuffled

| Model | Activity MAE (New) | Activity MAE (Backup) | Docking MAE (New) | Docking MAE (Backup) |
|-------|:---:|:---:|:---:|:---:|
| M0 | **2455** | **2539** | **0.397** | **0.461** |
| M1 | 2564 | 2582 | 0.503 | 0.521 |
| M2 | 2568 | 2547 | 0.611 | 2.240 |
| M3 | 2522 | 2529 | **0.386** | **0.488** |
| M4 | 2862 | 2549 | 0.488 | 0.460 |
| M5 | 2552 | 2547 | **0.358** | 0.630 |

---

## 6. Key Findings

### Models that improved (New > Backup):
1. **M5 - GNN**: Major activity improvement in Direction A (R: 0.023 -> 0.257). This is the largest single improvement across all comparisons, suggesting GNN training has high variance but can find good solutions.
2. **M3 - KAN Navier-Stokes**: Docking improved substantially in both directions (Dir A: 0.206 -> 0.316, Dir B: 0.363 -> 0.534). Activity remained stable. Most consistent improver.
3. **M0 - Dummy PhysChem**: Activity improved in Direction B (0.012 -> 0.087).
4. **M4 - ChemBERTa**: Docking improved in Direction B (0.498 -> 0.543).

### Models that degraded (New < Backup):
1. **M2 - KAN B-Spline**: Docking collapsed in both directions (Dir A: 0.067 -> -0.044, Dir B: 0.065 -> -0.315). This model shows the most instability between runs.
2. **M1 - Adv PhysChem**: Docking degraded heavily in Direction B (0.430 -> 0.161). Activity also dropped in Direction A (0.073 -> -0.026).
3. **M4 - ChemBERTa**: Docking degraded in Direction A (0.241 -> 0.122).

### Consistently stable:
1. **M3 - KAN Navier-Stokes**: Activity correlation is nearly identical across runs (Dir A: 0.170 vs 0.161, Dir B: 0.019 vs 0.025). Best activity predictor in Direction A in both runs.
2. **M0 - Dummy PhysChem**: Docking performance is stable (Dir A: 0.460 vs 0.415, Dir B: 0.490 vs 0.465). Consistently the best or near-best docking generalizer.

---

## 7. Reproducibility Concerns

### High-variance models (unreliable across runs):

| Model | Metric | Direction | New | Backup | |Delta| |
|-------|--------|-----------|:---:|:---:|:---:|
| M2 - KAN B-Spline | Docking R | B | -0.315 | 0.065 | 0.380 |
| M1 - Adv PhysChem | Docking R | B | 0.161 | 0.430 | 0.269 |
| M5 - GNN | Activity R | A | 0.257 | 0.023 | 0.234 |
| M3 - KAN Navier-Stokes | Docking R | B | 0.534 | 0.363 | 0.171 |

The M2 docking flip from positive to negative correlation is particularly concerning - it means the model's docking predictions are directionally wrong in the new run.

### Low-variance models (reliable across runs):

| Model | Metric | Direction | New | Backup | |Delta| |
|-------|--------|-----------|:---:|:---:|:---:|
| M5 - GNN | Docking R | B | 0.510 | 0.506 | 0.004 |
| M3 - KAN Navier-Stokes | Activity R | B | 0.025 | 0.019 | 0.006 |
| M1 - Adv PhysChem | Activity R | B | -0.032 | -0.024 | 0.008 |
| M3 - KAN Navier-Stokes | Activity R | A | 0.161 | 0.170 | 0.009 |

---

## 8. Generalizability Rankings (Averaged Across Both Runs)

Taking the mean of New and Backup overall Pearson R for each model:

### Activity (cross-run average):

| Rank | Model | Dir A Avg | Dir B Avg | Grand Avg |
|:---:|-------|:---:|:---:|:---:|
| 1 | M3 - KAN Navier-Stokes | 0.165 | 0.022 | **0.094** |
| 2 | M5 - GNN | 0.140 | -0.009 | 0.066 |
| 3 | M0 - Dummy PhysChem | 0.014 | 0.050 | 0.032 |
| 4 | M2 - KAN B-Spline | 0.052 | -0.036 | 0.008 |
| 5 | M1 - Adv PhysChem | 0.023 | -0.028 | -0.003 |
| 6 | M4 - ChemBERTa | -0.077 | -0.008 | -0.042 |

### Docking (cross-run average):

| Rank | Model | Dir A Avg | Dir B Avg | Grand Avg |
|:---:|-------|:---:|:---:|:---:|
| 1 | M0 - Dummy PhysChem | 0.437 | 0.478 | **0.458** |
| 2 | M3 - KAN Navier-Stokes | 0.261 | 0.449 | **0.355** |
| 3 | M4 - ChemBERTa | 0.181 | 0.520 | **0.351** |
| 4 | M1 - Adv PhysChem | 0.161 | 0.296 | 0.229 |
| 5 | M5 - GNN | 0.004 | 0.508 | 0.256 |
| 6 | M2 - KAN B-Spline | 0.011 | -0.125 | **-0.057** |

---

## 9. Conclusions

1. **M3 (KAN Navier-Stokes) is the most reliable activity generalizer** across both runs, with stable Direction A performance (~0.165) and improved docking.

2. **M0 (Dummy PhysChem) is the most reliable docking generalizer**, with the smallest variance and consistently strong docking R (~0.44-0.48).

3. **M5 (GNN) shows high potential but high variance** - it achieved the best single-run activity R (0.257 in new run Dir A) but was near zero in the backup run. Its docking in Direction B is stable (~0.51).

4. **M2 (KAN B-Spline) is unreliable** - docking predictions flipped sign between runs. Not recommended for deployment without stability improvements (e.g., ensemble or fixed seeds).

5. **M1 (Adv PhysChem) has moderate instability** - docking dropped from 0.430 to 0.161 in Direction B between runs.

6. **Activity prediction remains the harder task** - most models have near-zero or negative overall activity R, with only M3 and M5 showing consistent positive signal.

7. **Docking prediction is more learnable** - most models achieve R > 0.3 in at least one direction, with M0 and M3 being the most consistent.

8. **For deployment, ensemble M0 + M3 + M5** - this combines M0's stable docking, M3's balanced performance, and M5's activity potential while hedging against individual model variance.

9. **Random seed control would help** - the variance observed between runs suggests setting fixed random seeds (PyTorch, NumPy, CUDA) would improve reproducibility.
