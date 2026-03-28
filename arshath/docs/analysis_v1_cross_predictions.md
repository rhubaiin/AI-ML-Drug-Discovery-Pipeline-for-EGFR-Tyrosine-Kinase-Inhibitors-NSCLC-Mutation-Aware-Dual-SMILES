# Cross-Dataset Prediction Analysis (v2 - Re-run Feb 28, 2026)

Evaluation of model generalizability by training on one dataset and predicting on the other. This tests whether learned structure-activity/docking relationships transfer across dataset boundaries.

All models were retrained from scratch and all predictions were regenerated on Feb 28, 2026 with fresh feature computation (no cached features).

## Experimental Design

| Direction | Trained On | Predicted On | Train Samples | Prediction Samples |
|-----------|-----------|-------------|:---:|:---:|
| A | df_3_shuffled | manual_egfr3_mini_dock_fixed | 750 | 2676 |
| B | manual_egfr3_mini_dock_fixed | df_3_shuffled | 2676 | 748-750 |

Results are stored in `predictions_cross_csv/` within each model's experiment directory.

---

## 1. Overall Cross-Prediction Performance

### Direction A: df_3_shuffled trained --> predict on manual (n=2676)

| Model | Activity R | Activity p | Activity MAE | Docking R | Docking p | Docking MAE |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | -0.005 | 0.813 | 4,058,843,366,000 | **0.415** | <0.001 | 0.657 |
| 1 - Adv PhysChem 5F2 | -0.026 | 0.177 | 3,214 | **0.190** | <0.001 | 0.797 |
| 2 - KAN B-Spline | 0.036 | 0.063 | 2,900 | -0.044 | 0.023 | 0.715 |
| 3 - KAN Navier-Stokes | **0.161** | <0.001 | 2,759 | **0.316** | <0.001 | 0.627 |
| 4 - ChemBERTa CrossAttn | -0.047 | 0.014 | 2,910 | **0.122** | <0.001 | 0.716 |
| 5 - GNN/MolCLR | **0.257** | <0.001 | 2,561 | -0.009 | 0.646 | 0.700 |

### Direction B: manual trained --> predict on df_3_shuffled (n=748-750)

| Model | Activity R | Activity p | Activity MAE | Docking R | Docking p | Docking MAE |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 0.087 | 0.017 | 2,455 | **0.465** | <0.001 | 0.397 |
| 1 - Adv PhysChem 5F2 | -0.032 | 0.387 | 2,564 | **0.161** | <0.001 | 0.503 |
| 2 - KAN B-Spline | -0.042 | 0.246 | 2,568 | -0.315 | <0.001 | 0.611 |
| 3 - KAN Navier-Stokes | 0.025 | 0.492 | 2,522 | **0.534** | <0.001 | 0.386 |
| 4 - ChemBERTa CrossAttn | -0.018 | 0.616 | 2,862 | **0.543** | <0.001 | 0.488 |
| 5 - GNN/MolCLR | 0.006 | 0.876 | 2,552 | **0.510** | <0.001 | 0.358 |

---

## 2. Performance Comparison: Direction A vs Direction B

### Activity Pearson R

| Model | Direction A (df3->manual) | Direction B (manual->df3) | Average |
|-------|:---:|:---:|:---:|
| 0 - Dummy PhysChem | -0.005 | 0.087 | 0.041 |
| 1 - Adv PhysChem 5F2 | -0.026 | -0.032 | -0.029 |
| 2 - KAN B-Spline | 0.036 | -0.042 | -0.003 |
| 3 - KAN Navier-Stokes | **0.161** | 0.025 | **0.093** |
| 4 - ChemBERTa CrossAttn | -0.047 | -0.018 | -0.033 |
| 5 - GNN/MolCLR | **0.257** | 0.006 | **0.131** |

### Docking Pearson R

| Model | Direction A (df3->manual) | Direction B (manual->df3) | Average |
|-------|:---:|:---:|:---:|
| 0 - Dummy PhysChem | **0.415** | **0.465** | **0.440** |
| 1 - Adv PhysChem 5F2 | **0.190** | **0.161** | **0.176** |
| 2 - KAN B-Spline | -0.044 | -0.315 | -0.180 |
| 3 - KAN Navier-Stokes | **0.316** | **0.534** | **0.425** |
| 4 - ChemBERTa CrossAttn | **0.122** | **0.543** | **0.333** |
| 5 - GNN/MolCLR | -0.009 | **0.510** | **0.250** |

---

## 3. Per-Mutation Cross-Prediction Breakdown

### 3.1 Direction A: df_3_shuffled trained --> predict on manual

#### Activity Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 230 | -0.021 | -0.042 | -0.032 | **0.354** | -0.055 | 0.162 |
| del/t790m double | 181 | -0.027 | -0.135 | 0.014 | **0.550** | **-0.260** | **0.300** |
| del/t790m/c797s triple | 105 | -0.023 | 0.022 | 0.079 | -0.117 | 0.039 | -0.007 |
| ins 20 | 166 | 0.027 | -0.080 | 0.079 | **0.205** | -0.104 | -0.152 |
| l858r/t790m double | 499 | -0.014 | -0.062 | -0.017 | 0.122 | -0.012 | **0.247** |
| l858r/t790m/c797s triple | 498 | -0.015 | 0.007 | 0.079 | 0.084 | 0.009 | **0.214** |
| subs l858r | 499 | -0.011 | -0.032 | -0.108 | **0.278** | -0.128 | **0.291** |
| wild adeno lung | 498 | -0.018 | -0.010 | 0.118 | **0.200** | -0.002 | **0.271** |

#### Docking Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 230 | **0.594** | **0.502** | 0.087 | **0.322** | **0.324** | 0.172 |
| del/t790m double | 181 | 0.171 | 0.136 | -0.091 | 0.070 | **0.230** | 0.006 |
| del/t790m/c797s triple | 105 | 0.072 | -0.027 | -0.020 | -0.003 | -0.009 | 0.127 |
| ins 20 | 166 | -0.063 | -0.048 | 0.022 | -0.090 | -0.097 | **-0.200** |
| l858r/t790m double | 499 | **0.486** | **0.369** | -0.024 | **0.486** | 0.111 | 0.006 |
| l858r/t790m/c797s triple | 498 | **0.437** | **0.214** | **-0.253** | **0.326** | 0.014 | -0.091 |
| subs l858r | 499 | **0.537** | **0.408** | 0.013 | **0.498** | 0.153 | 0.035 |
| wild adeno lung | 498 | **0.531** | **0.399** | 0.076 | **0.392** | **0.236** | -0.014 |

### 3.2 Direction B: manual trained --> predict on df_3_shuffled

#### Activity Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 117 | **0.970** | 0.147 | 0.062 | **0.904** | **0.333** | **0.863** |
| del/t790m double | 41 | **0.712** | **-0.561** | **-0.713** | 0.008 | -0.185 | -0.147 |
| del/t790m/c797s triple | 32 | **-0.469** | **-0.434** | **-0.658** | 0.172 | 0.244 | 0.120 |
| ins 20 | 51 | 0.148 | -0.249 | **-0.408** | **0.566** | -0.136 | 0.079 |
| l858r/t790m double | 251 | **0.514** | **-0.378** | **-0.445** | **0.585** | 0.194 | -0.089 |
| l858r/t790m/c797s triple | 51 | 0.034 | 0.003 | -0.011 | -0.051 | 0.040 | 0.031 |
| subs l858r | 105 | **0.865** | 0.057 | 0.049 | **0.490** | 0.014 | **0.278** |
| wild adeno lung | 100 | 0.044 | 0.016 | 0.043 | -0.022 | -0.040 | 0.034 |

#### Docking Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 117 | **0.647** | **0.489** | -0.076 | **0.873** | **0.835** | **0.723** |
| del/t790m double | 41 | **0.992** | **0.476** | **-0.677** | **0.641** | **0.978** | **0.883** |
| del/t790m/c797s triple | 32 | -0.196 | **-0.661** | **-0.794** | **0.745** | **0.983** | -0.145 |
| ins 20 | 51 | **0.883** | **0.723** | -0.284 | **0.922** | **0.805** | **0.994** |
| l858r/t790m double | 251 | **0.766** | -0.026 | -0.128 | **0.980** | **0.688** | **0.759** |
| l858r/t790m/c797s triple | 51 | **0.894** | -0.154 | **-0.566** | **0.517** | **0.895** | **0.420** |
| subs l858r | 105 | **0.922** | **0.571** | **-0.727** | **0.576** | **0.933** | **0.848** |
| wild adeno lung | 100 | **0.875** | 0.214 | **-0.468** | **0.794** | **0.888** | **0.774** |

---

## 4. Activity MAE / RMSE (Cross-Dataset)

### Direction A: df_3_shuffled trained --> predict on manual (n=2676)

| Model | Activity MAE | Activity RMSE | Docking MAE | Docking RMSE |
|-------|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 4,058,843,366,000 | 209,937,482,471,000 | 0.657 | 1.026 |
| 1 - Adv PhysChem 5F2 | 3,214 | 10,942 | 0.797 | 1.183 |
| 2 - KAN B-Spline | 2,900 | 10,960 | 0.715 | 1.061 |
| 3 - KAN Navier-Stokes | 2,759 | 10,917 | 0.627 | 0.995 |
| 4 - ChemBERTa CrossAttn | 2,910 | 11,013 | 0.716 | 1.076 |
| 5 - GNN/MolCLR | 2,561 | 11,097 | 0.700 | 1.056 |

### Direction B: manual trained --> predict on df_3_shuffled (n=748-750)

| Model | Activity MAE | Activity RMSE | Docking MAE | Docking RMSE |
|-------|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 2,455 | 18,592 | 0.397 | 0.527 |
| 1 - Adv PhysChem 5F2 | 2,564 | 18,630 | 0.503 | 0.605 |
| 2 - KAN B-Spline | 2,568 | 18,630 | 0.611 | 0.748 |
| 3 - KAN Navier-Stokes | 2,522 | 18,612 | 0.386 | 0.473 |
| 4 - ChemBERTa CrossAttn | 2,862 | 18,602 | 0.488 | 0.558 |
| 5 - GNN/MolCLR | 2,552 | 18,629 | 0.358 | 0.494 |

---

## 5. Key Findings

### 5.1 Activity Predictions Do Not Transfer (Overall)

Cross-dataset activity prediction remains near zero for most models at the overall level:
- **Direction A** (df3 -> manual): Best overall R = 0.257 (Model 5 GNN), followed by 0.161 (Model 3)
- **Direction B** (manual -> df3): Best overall R = 0.087 (Model 0), all others below 0.03

This confirms fundamental differences between the two datasets in how activity values are distributed. However, per-mutation activity transfer can be strong (see Section 5.4).

### 5.2 Docking Predictions Partially Transfer

Docking score predictions show moderate cross-dataset generalizability:
- **Direction B** (manual -> df3) is notably better, with 4 of 6 models achieving Docking R > 0.46 (Models 0, 3, 4, 5)
- **Direction A** (df3 -> manual) is weaker, with Model 0 (R=0.415) and Model 3 (R=0.316) leading

This asymmetry is consistent: models trained on more data (manual, 2676 samples) learn more robust docking relationships that transfer better to the smaller dataset.

### 5.3 Per-Mutation Docking Transfer is Remarkably Strong

Despite variable overall cross-prediction, per-mutation docking Pearson R values in Direction B are exceptionally strong:
- `del/t790m double`: R = 0.99 (M0), 0.98 (M4), 0.88 (M5)
- `ins 20`: R = 0.99 (M5), 0.92 (M3), 0.88 (M0)
- `l858r/t790m double`: R = 0.98 (M3), 0.77 (M0), 0.76 (M5)
- `del`: R = 0.87 (M3), 0.84 (M4), 0.72 (M5)
- `subs l858r`: R = 0.93 (M4), 0.92 (M0), 0.85 (M5)
- `wild adeno lung`: R = 0.89 (M4), 0.88 (M0), 0.79 (M3)
- `l858r/t790m/c797s triple`: R = 0.89 (M4), 0.89 (M0)
- `del/t790m/c797s triple`: R = 0.98 (M4), 0.74 (M3)

This demonstrates that within a mutation class, the structural relationship between compound features and docking scores is highly conserved across datasets.

### 5.4 Model 0 (Dummy PhysChem) Shows Exceptional Per-Mutation Activity Transfer

In Direction B, Model 0 achieves remarkable activity transfer for specific mutations:
- `del`: R = 0.970 (near-perfect)
- `subs l858r`: R = 0.865
- `del/t790m double`: R = 0.712
- `l858r/t790m double`: R = 0.514

The simple feedforward architecture on physicochemical descriptors produces the most transferable activity representations for many mutation types.

### 5.5 Model 3 (KAN Navier-Stokes) Shows Best Balanced Transfer

Model 3 demonstrates the most balanced cross-dataset performance:
- **Docking**: R = 0.316 (Direction A) and R = 0.534 (Direction B), average = 0.425
- **Activity Direction B per-mutation**: Strong R for del (0.90), ins 20 (0.57), l858r/t790m double (0.59), subs l858r (0.49)
- **Activity Direction A**: Best overall R = 0.161 among models with consistent docking transfer

The Fourier KAN layers capture more generalizable frequency-domain patterns in the chemical features.

### 5.6 Model 5 (GNN) Shows Strong Asymmetric Activity Transfer

Model 5 GNN is the strongest activity predictor in Direction A:
- **Direction A overall activity**: R = 0.257 (best of all models, p<0.001)
- Per-mutation: del/t790m double (0.30), l858r/t790m double (0.25), subs l858r (0.29), wild adeno lung (0.27)

However, in Direction B, GNN activity transfer drops to near zero overall (R=0.006), though del mutation retains R=0.86. GNN docking transfer follows the opposite pattern: poor in Direction A (R=-0.009) but strong in Direction B (R=0.510).

### 5.7 Model 4 (ChemBERTa CrossAttn) Best Docking Transfer in Direction B

Model 4 achieves the highest overall docking R in Direction B (0.543), driven by outstanding per-mutation correlations:
- `del/t790m/c797s triple`: R = 0.983
- `del/t790m double`: R = 0.978
- `subs l858r`: R = 0.933
- `l858r/t790m/c797s triple`: R = 0.895
- `wild adeno lung`: R = 0.888
- `del`: R = 0.835
- `ins 20`: R = 0.805

This is the most consistent per-mutation docking performer across all mutations in Direction B.

### 5.8 Model 2 (KAN B-Spline) Fails to Generalize

Model 2 shows near-zero or negative cross-dataset performance in both directions:
- Direction A: Activity R = 0.036, Docking R = -0.044
- Direction B: Activity R = -0.042, Docking R = -0.315 (strong negative correlation)

The negative docking R in Direction B is notable -- the model's predictions are systematically inverted relative to actual values. The B-spline KAN architecture consistently underperforms.

---

## 6. Generalizability Ranking

### Docking (Cross-Dataset)

Best to worst for cross-dataset docking generalization:

| Rank | Model | Direction A (R) | Direction B (R) | Average |
|:---:|-------|:---:|:---:|:---:|
| 1 | 0 - Dummy PhysChem | 0.415 | 0.465 | **0.440** |
| 2 | 3 - KAN Navier-Stokes | 0.316 | 0.534 | **0.425** |
| 3 | 4 - ChemBERTa CrossAttn | 0.122 | 0.543 | **0.333** |
| 4 | 5 - GNN/MolCLR | -0.009 | 0.510 | **0.250** |
| 5 | 1 - Adv PhysChem 5F2 | 0.190 | 0.161 | **0.176** |
| 6 | 2 - KAN B-Spline | -0.044 | -0.315 | **-0.180** |

### Activity (Cross-Dataset)

No model achieves strong cross-dataset activity generalization at the overall level. All overall R values are below 0.26.

| Rank | Model | Direction A (R) | Direction B (R) | Average |
|:---:|-------|:---:|:---:|:---:|
| 1 | 5 - GNN/MolCLR | 0.257 | 0.006 | **0.131** |
| 2 | 3 - KAN Navier-Stokes | 0.161 | 0.025 | **0.093** |
| 3 | 0 - Dummy PhysChem | -0.005 | 0.087 | **0.041** |
| 4 | 2 - KAN B-Spline | 0.036 | -0.042 | **-0.003** |
| 5 | 1 - Adv PhysChem 5F2 | -0.026 | -0.032 | **-0.029** |
| 6 | 4 - ChemBERTa CrossAttn | -0.047 | -0.018 | **-0.033** |

---

## 7. Comparison with Previous Run (Feb 18 vs Feb 28)

Key differences from the previous experiment run:

| Metric | Previous (Feb 18) | Current (Feb 28) | Change |
|--------|:---:|:---:|:---:|
| **Best Docking (Dir A)** | M0: 0.460 | M0: 0.415 | -0.045 |
| **Best Docking (Dir B)** | M5: 0.506 | M4: 0.543 | +0.037 |
| **Best Activity (Dir A)** | M3: 0.170 | M5: 0.257 | +0.087 |
| **Best Activity (Dir B)** | M3: 0.019 | M0: 0.087 | +0.068 |
| **M2 Docking (Dir B)** | 0.065 | -0.315 | -0.380 |

Notable shifts:
- **Model 5 (GNN)** gained significant activity transfer in Direction A (0.023 -> 0.257), likely due to different random initialization
- **Model 4 (ChemBERTa)** improved Direction B docking from 0.498 to 0.543
- **Model 2 (KAN B-Spline)** worsened dramatically in Direction B docking (0.065 -> -0.315), confirming instability
- **Model 0** remains the most consistent docking generalizer across both runs
- **Model 3** retains its balanced transfer profile

These differences reflect the stochastic nature of neural network training. The overall conclusions remain consistent: docking transfers better than activity, simpler models are more stable, and per-mutation correlations are much stronger than overall correlations.

---

## 8. Conclusions

1. **Activity prediction is dataset-specific.** The IC50/potency measurements in the two datasets are not on comparable scales or reflect different experimental conditions. Models cannot reliably transfer activity predictions across datasets at the overall level.

2. **Docking prediction partially generalizes**, especially at the per-mutation level. The physics of protein-ligand binding is more consistent across datasets than activity measurements.

3. **Simpler models generalize more consistently for docking.** Model 0 (physicochemical descriptors) has the most stable cross-dataset docking performance across both runs, while complex models show more variance between training runs.

4. **Training on more data helps transfer.** Direction B (manual -> df3) consistently outperforms Direction A (df3 -> manual) for docking, confirming that models trained on 2676 samples learn more robust patterns than those trained on 750.

5. **Model 3 (KAN Navier-Stokes) provides the best balanced generalization** for both activity and docking across directions. Its Fourier-basis KAN layers may capture more transferable chemical patterns.

6. **Per-mutation docking transfer is exceptional.** Multiple model-mutation combinations achieve R > 0.90 in Direction B, demonstrating that within a mutation class, docking structure-activity relationships are highly conserved.

7. **Model 2 (KAN B-Spline) should be deprioritized** -- it fails to generalize and shows instability between training runs, with negative docking correlations in Direction B.

8. **Model 4 (ChemBERTa CrossAttn) is the top docking predictor in Direction B**, with the most consistently high per-mutation R values across all 8 mutations.

9. **For deployment**, if cross-dataset generalization matters, consider:
   - Combining both datasets for training
   - Normalizing activity values to a common scale before training
   - Using domain adaptation techniques
   - Focusing on docking predictions (which transfer better) and using them as a proxy for activity ranking
   - Using an ensemble of Model 0 (stable docking) + Model 3 (balanced) + Model 4 (Direction B docking) for maximum generalizability
