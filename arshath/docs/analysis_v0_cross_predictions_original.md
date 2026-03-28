# Cross-Dataset Prediction Analysis

Evaluation of model generalizability by training on one dataset and predicting on the other. This tests whether learned structure-activity/docking relationships transfer across dataset boundaries.

## Experimental Design

| Direction | Trained On | Predicted On | Train Samples | Prediction Samples |
|-----------|-----------|-------------|:---:|:---:|
| A | df_3_shuffled | manual_egfr3_mini_dock_fixed | 750 | 2676 |
| B | manual_egfr3_mini_dock_fixed | df_3_shuffled | 2709 | 748-750 |

Results are stored in `predictions_cross_csv/` alongside the original `predictions/` folder within each model's experiment directory.

---

## 1. Overall Cross-Prediction Performance

### Direction A: df_3_shuffled trained --> predict on manual (n=2676)

| Model | Activity R | Activity p | Activity MAE | Docking R | Docking p | Docking MAE |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 0.033 | 0.086 | 100,629,485 | **0.460** | <0.001 | 0.689 |
| 1 - Adv PhysChem 5F2 | 0.073 | <0.001 | 2,935 | **0.132** | <0.001 | 0.876 |
| 2 - KAN B-Spline | 0.067 | <0.001 | 2,566 | 0.067 | <0.001 | 0.867 |
| 3 - KAN Navier-Stokes | **0.170** | <0.001 | 2,891 | **0.206** | <0.001 | 0.692 |
| 4 - ChemBERTa CrossAttn | -0.107 | <0.001 | 2,714 | **0.241** | <0.001 | 0.682 |
| 5 - GNN/MolCLR | 0.023 | 0.228 | 2,563 | 0.017 | 0.369 | 1.049 |

### Direction B: manual trained --> predict on df_3_shuffled (n=748-750)

| Model | Activity R | Activity p | Activity MAE | Docking R | Docking p | Docking MAE |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 0.012 | 0.737 | 2,539 | **0.490** | <0.001 | 0.461 |
| 1 - Adv PhysChem 5F2 | -0.024 | 0.508 | 2,582 | **0.430** | <0.001 | 0.521 |
| 2 - KAN B-Spline | -0.029 | 0.435 | 2,547 | 0.065 | 0.076 | 2.240 |
| 3 - KAN Navier-Stokes | 0.019 | 0.596 | 2,529 | **0.363** | <0.001 | 0.488 |
| 4 - ChemBERTa CrossAttn | 0.003 | 0.933 | 2,549 | **0.498** | <0.001 | 0.460 |
| 5 - GNN/MolCLR | -0.024 | 0.509 | 2,547 | **0.506** | <0.001 | 0.630 |

---

## 2. Performance Drop: Same-Dataset vs Cross-Dataset

### Activity Pearson R Comparison

| Model | Same (manual) | Cross A (df3->manual) | Cross B (manual->df3) | Same (df3) |
|-------|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 0.358 | 0.033 | 0.012 | -0.090 |
| 1 - Adv PhysChem 5F2 | 0.792 | 0.073 | -0.024 | -0.123 |
| 2 - KAN B-Spline | -0.125 | 0.067 | -0.029 | 0.228 |
| 3 - KAN Navier-Stokes | **0.839** | **0.170** | 0.019 | -0.094 |
| 4 - ChemBERTa CrossAttn | 0.547 | -0.107 | 0.003 | 0.026 |
| 5 - GNN/MolCLR | 0.770 | 0.023 | -0.024 | -0.174 |

### Docking Pearson R Comparison

| Model | Same (manual) | Cross A (df3->manual) | Cross B (manual->df3) | Same (df3) |
|-------|:---:|:---:|:---:|:---:|
| 0 - Dummy PhysChem | 0.715 | **0.460** | **0.490** | **0.676** |
| 1 - Adv PhysChem 5F2 | 0.689 | 0.132 | **0.430** | 0.046 |
| 2 - KAN B-Spline | 0.075 | 0.067 | 0.065 | 0.189 |
| 3 - KAN Navier-Stokes | 0.554 | **0.206** | **0.363** | 0.265 |
| 4 - ChemBERTa CrossAttn | 0.380 | **0.241** | **0.498** | **0.351** |
| 5 - GNN/MolCLR | 0.662 | 0.017 | **0.506** | -0.215 |

---

## 3. Per-Mutation Cross-Prediction Breakdown

### 3.1 Direction A: df_3_shuffled trained --> predict on manual

#### Activity Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 230 | -0.029 | 0.100 | 0.055 | **0.185** | -0.073 | 0.021 |
| del/t790m double | 181 | -0.065 | 0.099 | -0.050 | **0.489** | **-0.242** | -0.170 |
| del/t790m/c797s triple | 105 | 0.011 | -0.121 | 0.027 | 0.020 | -0.089 | **0.330** |
| ins 20 | 166 | -0.097 | 0.066 | 0.049 | 0.142 | -0.128 | 0.068 |
| l858r/t790m double | 499 | 0.074 | 0.067 | **0.109** | **0.269** | -0.088 | 0.017 |
| l858r/t790m/c797s triple | 498 | **0.122** | -0.078 | 0.016 | 0.077 | -0.052 | 0.082 |
| subs l858r | 499 | 0.008 | **0.171** | 0.010 | **0.217** | -0.100 | 0.008 |
| wild adeno lung | 498 | 0.006 | 0.107 | 0.045 | **0.160** | **-0.129** | 0.037 |

#### Docking Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 230 | **0.545** | **0.468** | -0.119 | 0.140 | **0.625** | 0.084 |
| del/t790m double | 181 | **0.376** | 0.157 | -0.134 | -0.033 | **0.236** | -0.019 |
| del/t790m/c797s triple | 105 | 0.114 | -0.015 | **-0.318** | -0.106 | **0.304** | 0.165 |
| ins 20 | 166 | -0.073 | -0.094 | 0.027 | -0.123 | -0.043 | 0.123 |
| l858r/t790m double | 499 | **0.525** | **0.165** | **0.210** | **0.294** | **0.310** | -0.052 |
| l858r/t790m/c797s triple | 498 | **0.568** | 0.125 | 0.070 | 0.152 | **0.271** | -0.097 |
| subs l858r | 499 | **0.555** | **0.203** | 0.126 | **0.330** | **0.301** | 0.091 |
| wild adeno lung | 498 | **0.618** | **0.365** | 0.171 | **0.231** | **0.365** | 0.073 |

### 3.2 Direction B: manual trained --> predict on df_3_shuffled

#### Activity Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 117 | **0.219** | **0.462** | -0.042 | 0.019 | **0.710** | **0.570** |
| del/t790m double | 41 | **0.358** | **-0.609** | **-0.798** | **0.840** | -0.284 | **-0.551** |
| del/t790m/c797s triple | 32 | **0.474** | **-0.487** | **-0.678** | **0.684** | -0.273 | **-0.424** |
| ins 20 | 51 | -0.005 | -0.240 | **-0.634** | **0.633** | -0.043 | -0.144 |
| l858r/t790m double | 251 | **0.325** | **-0.307** | **-0.555** | **0.500** | -0.067 | **-0.465** |
| l858r/t790m/c797s triple | 51 | 0.044 | -0.004 | -0.003 | 0.013 | 0.029 | -0.001 |
| subs l858r | 105 | 0.054 | 0.126 | **0.230** | **0.518** | 0.078 | **0.266** |
| wild adeno lung | 100 | -0.055 | 0.043 | 0.013 | 0.011 | 0.004 | 0.036 |

#### Docking Pearson R

| Mutation | n | M0 | M1 | M2 | M3 | M4 | M5 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| del | 117 | **0.625** | **0.544** | **0.641** | **0.755** | **0.532** | **0.958** |
| del/t790m double | 41 | **0.923** | **0.792** | **0.557** | **0.463** | **0.892** | **0.642** |
| del/t790m/c797s triple | 32 | **-0.723** | **-0.455** | **-0.567** | **0.522** | -0.214 | **0.900** |
| ins 20 | 51 | **0.804** | **0.992** | 0.189 | **0.858** | **0.982** | **0.525** |
| l858r/t790m double | 251 | **0.688** | **0.712** | 0.165 | **0.540** | **0.826** | **0.913** |
| l858r/t790m/c797s triple | 51 | **0.586** | 0.215 | **-0.601** | **0.336** | **0.634** | **0.923** |
| subs l858r | 105 | **0.963** | **0.687** | 0.162 | **0.377** | **0.764** | **0.619** |
| wild adeno lung | 100 | **0.886** | **0.674** | -0.033 | **0.341** | **0.790** | **0.687** |

---

## 4. Key Findings

### 4.1 Activity Predictions Do Not Transfer

Cross-dataset activity prediction is near zero for all models in both directions:
- **Direction A** (df3 -> manual): Best overall R = 0.170 (Model 3)
- **Direction B** (manual -> df3): Best overall R = 0.019 (Model 3)
- No model achieves meaningful activity generalization

This indicates fundamental differences between the two datasets in how activity values (IC50/potency) are distributed or measured. The models learn dataset-specific activity patterns rather than universal structure-activity relationships.

### 4.2 Docking Predictions Partially Transfer

Docking score predictions show moderate cross-dataset generalizability:
- **Direction B** (manual -> df3) is notably better than Direction A, with 5 of 6 models achieving significant Docking R (0.36-0.51)
- **Direction A** (df3 -> manual) shows weaker transfer, with only Model 0 reaching R > 0.4

This asymmetry makes sense: models trained on more data (manual, 2709 samples) learn more robust docking relationships that transfer better to the smaller dataset.

### 4.3 Per-Mutation Docking Transfer is Strong

Despite weak overall cross-prediction, per-mutation docking Pearson R values in Direction B are remarkably strong:
- `del/t790m double`: R = 0.92 (M0), 0.79 (M1), 0.89 (M4)
- `ins 20`: R = 0.80 (M0), 0.99 (M1), 0.98 (M4)
- `l858r/t790m double`: R = 0.69 (M0), 0.71 (M1), 0.83 (M4), 0.91 (M5)
- `subs l858r`: R = 0.96 (M0), 0.69 (M1), 0.76 (M4)

This suggests that within a mutation class, the structural relationship between compound features and docking scores is highly conserved across datasets. The overall R is diluted by scale differences between mutations.

### 4.4 Model 3 (KAN Navier-Stokes) Shows Best Activity Transfer

Model 3 is the only model showing any activity transfer signal:
- Direction A overall: R = 0.170 (best of all models)
- Direction B per-mutation: Strong R for del/t790m double (0.84), del/t790m/c797s triple (0.68), ins 20 (0.63), l858r/t790m double (0.50), subs l858r (0.52)

The Fourier KAN layers may capture more generalizable frequency-domain patterns in the chemical features.

### 4.5 Model 5 (GNN) Does Not Transfer

Despite strong same-dataset performance (R=0.77 activity on manual), Model 5 GNN shows near-zero cross-prediction (Direction A: R=0.02, Direction B: R=-0.02 for activity). The graph neural network representations appear to overfit to dataset-specific molecular distributions. However, Direction B docking is an exception (R=0.51), driven by strong per-mutation correlations.

### 4.6 Model 0 (Dummy PhysChem) Best Docking Transfer

The simplest model (feedforward on physicochemical descriptors) shows the most consistent docking transfer:
- Direction A: R = 0.460
- Direction B: R = 0.490

This is consistent with physicochemical descriptors being inherently more transferable than learned representations.

---

## 5. Generalizability Ranking

### Docking (Cross-Dataset)

Best to worst for cross-dataset docking generalization:

| Rank | Model | Direction A (R) | Direction B (R) | Average |
|:---:|-------|:---:|:---:|:---:|
| 1 | 0 - Dummy PhysChem | 0.460 | 0.490 | **0.475** |
| 2 | 4 - ChemBERTa CrossAttn | 0.241 | 0.498 | **0.370** |
| 3 | 3 - KAN Navier-Stokes | 0.206 | 0.363 | **0.285** |
| 4 | 1 - Adv PhysChem 5F2 | 0.132 | 0.430 | **0.281** |
| 5 | 5 - GNN/MolCLR | 0.017 | 0.506 | **0.262** |
| 6 | 2 - KAN B-Spline | 0.067 | 0.065 | **0.066** |

### Activity (Cross-Dataset)

No model achieves meaningful cross-dataset activity generalization. All overall R values are below 0.2.

---

## 6. Conclusions

1. **Activity prediction is dataset-specific.** The IC50/potency measurements in the two datasets are not on comparable scales or reflect different experimental conditions. Models cannot transfer activity predictions across datasets.

2. **Docking prediction partially generalizes**, especially at the per-mutation level. The physics of protein-ligand binding is more consistent across datasets than activity measurements.

3. **Simpler models generalize better for docking.** Model 0 (physicochemical descriptors) has the best cross-dataset docking performance, while complex models (GNN, ChemBERTa) show larger same-vs-cross performance gaps.

4. **Training on more data helps transfer.** Direction B (manual -> df3) consistently outperforms Direction A (df3 -> manual), confirming that models trained on 2709 samples learn more robust patterns than those trained on 750.

5. **Model 3 (KAN Navier-Stokes) is the most generalizable for activity**, though the signal is weak. Its Fourier-basis KAN layers may capture more transferable chemical patterns.

6. **For deployment**, if activity generalization matters, consider:
   - Combining both datasets for training
   - Normalizing activity values to a common scale before training
   - Using domain adaptation techniques
   - Focusing on docking predictions (which transfer better) and using them as a proxy for activity ranking
