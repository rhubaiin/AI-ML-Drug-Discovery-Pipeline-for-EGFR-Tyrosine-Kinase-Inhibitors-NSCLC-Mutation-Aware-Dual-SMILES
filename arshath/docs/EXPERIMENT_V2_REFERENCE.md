# Experiment V2 Reference Guide

## Origin: Rhubain's Corrected Scripts (March 2026)

### Messages from Rhubain

> "This is the NON-optimised corrected scripts, there are subtle changes (minor bugs)":
> https://github.com/rhubaiin/AI-ML-Drug-Discovery-Pipeline-for-EGFR-Tyrosine-Kinase-Inhibitors-in-NSCLC-Using-Dual-SMILES/tree/main/physicochem_activity_main_pseudocode
>
> "Last time you managed to Optimise them by saving the cache to disk so it trains each model quicker"
>
> "This is the dataset trimmed version (no outliers)":
> https://github.com/rhubaiin/AI-ML-Drug-Discovery-Pipeline-for-EGFR-Tyrosine-Kinase-Inhibitors-in-NSCLC-Using-Dual-SMILES/tree/main/dataset
>
> "Please do Smoke Test first, worried the corrected small bugs might crash"
>
> "Train on small set, predict on big, then reverse"

---

## What We Did (Step by Step)

### Phase 1: Organize & Preserve Existing Work

1. **Renamed experiment directories** for clear three-version tracking:
   - `experiments_backup/` -> `experiments_v0/` (Feb 18, 2026 run - original scripts)
   - `experiments/` -> `experiments_v1/` (Feb 28, 2026 run - optimized scripts)
   - `experiments_v2/` (new - corrected + optimized scripts)

2. **Updated `.gitignore`** to cover all experiment dirs, checkpoints, caches:
   ```
   */experiments_v*/
   */experiments_backup/
   */checkpoints/
   .DS_Store
   .feature_cache/
   ```

3. **Copied analysis markdowns to `docs/`** so they're tracked in git:
   - `docs/analysis_v0_cross_predictions.md`
   - `docs/analysis_v1_cross_predictions.md`
   - `docs/comparison_v0_vs_v1.md`

4. **Committed and pushed** to `arshath/pipeline-optimizations` branch.

### Phase 2: Download & Diff Corrected Scripts

5. **Downloaded all 12 corrected scripts** from Rhubain's GitHub repo:
   - 6 training scripts + 6 inference scripts
   - Saved to `corrected_scripts/` staging directory

6. **Downloaded trimmed dataset**: `valid_drug_tki_trimmed.csv`
   - 718 rows (vs `df_3_shuffled.csv` with 750 rows)
   - Same columns — this is the outlier-cleaned version
   - Maps to `df_3_shuffled.csv` (32 outlier rows removed)
   - `manual_egfr3_mini_dock_fixed.csv` has no replacement (different schema, 2676 rows)

7. **Diffed all 12 corrected scripts against our current optimized versions**:

   | Script | File | Bug Fixes Found |
   |--------|------|----------------|
   | Training 0 | `dummy_physchem_5f2.py` | **None** |
   | Training 1 | `adv_physchem5f2.py` | **None** |
   | Training 2 | `adv_physchem_KAN3_b_spline1a.py` | **None** |
   | Training 3 | `adv_physchem_KAN_navier_stokes_sinusoid.py` | **None** |
   | Training 4 | `adv_physchem_chemerta_crossattention.py` | **6 bugs** |
   | Training 5 | `adv_physchem_gnn.py` | **2 bugs** |
   | Inference 0 | `predict_dummy_physchem_5f2.py` | **None** |
   | Inference 1 | `predict_adv_physchem5f2.py` | **None** |
   | Inference 2 | `predict_adv_physchem_KAN_bspline.py` | **None** |
   | Inference 3 | `predict_adv_physchem_KAN_navier_stokes.py` | **None** |
   | Inference 4 | `predict_adv_physchem_chembert_crossattention.py` | **5 bugs** |
   | Inference 5 | `predict_adv_physchem_gnn.py` | **4 bugs** |

   **Scripts 0-3: No bug fixes needed.** All differences were our previously added optimizations (caching, argparse, encoding).

### Phase 3: Bug Fix Details

#### Script 4 — ChemBERTa Cross-Attention (MOST IMPACTED)

| # | Bug | Severity | Details |
|---|-----|----------|---------|
| 1 | **Electrostatic formula wrong array** | HIGH | `lig_intra[14]` (sp3 carbon count) was used instead of `lig_inter[14]` (MolWt). The Coulomb-like normalization should be charge/size, not charge/hybridization. |
| 2 | **SeqLen=1 collapsed attention** | HIGH | With `Reshape((1, 768))`, softmax over a single token is always `[1.0]` — attention learns nothing. Fix: reshape to `(8, 96)` giving 8 pseudo-tokens with real attention competition. |
| 3 | **Missing FFN sub-layer** | MEDIUM | Standard transformer block has 2 sub-layers: MHA + FFN. Only MHA was implemented. Added: `Dense(TOKEN_DIM*4) -> ReLU -> Dense(TOKEN_DIM) -> Dropout -> Residual -> LayerNorm` |
| 4 | **192x compression destroyed ChemBERTa signal** | HIGH | `768 -> 32 -> 4` = 192x compression stripped all semantic info. Fix: gradual `768 -> 768(adapt) -> 512 -> 128 -> 32` = 24x compression. |
| 5 | **ChemBERTa was minority signal** | MEDIUM | Old: 4 of 16 dims in priority1 (25%). New: 32 of 44 dims (73%). ChemBERTa now dominates Priority 1 as intended. |
| 6 | **Inference embeddings not filtered** | CRITICAL | If any compound fails RDKit parsing, physico arrays have fewer rows than embedding arrays -> shape mismatch crash. Fix: pre-compute all embeddings, then slice to valid indices. |

#### Script 5 — GNN/MolCLR

| # | Bug | Severity | Details |
|---|-----|----------|---------|
| 1 | **Same electrostatic formula bug** | HIGH | `lig_intra[14]` -> `lig_inter[14]` |
| 2 | **Duplicate feature append** | LOW | Electrostatic features were computed and appended twice |
| 3 | **Missing `max(0,...)` clamp** | MEDIUM | `np.sqrt(1 - cosine_sim**2)` can produce NaN if cosine_sim > 1.0 due to floating point. Fix: `np.sqrt(max(0, ...))` |
| 4 | **Site ordering mismatch** | HIGH | Inference had FULL_SMILES last instead of first. This misaligns scalers with sites and changes RNN input sequence. |

### Phase 4: Apply Bug Fixes + Re-Optimize

8. **Created transformation script** (`apply_optimizations.py`) that:
   - Starts from Rhubain's corrected scripts (bug fixes preserved)
   - Adds disk-backed feature cache (`_generate_lig_features()` with `.feature_cache/`)
   - Adds `MPLBACKEND=Agg` and `CUDA_VISIBLE_DEVICES=''`
   - Adds `encoding='latin-1'` to CSV reads
   - Replaces paired `generate_lig_inter_features()` + `generate_lig_intra_features()` calls with single cached `_generate_lig_features()` call
   - Only scripts 4 and 5 were modified; scripts 0-3 untouched

9. **Added argparse parameterization** to training scripts 4 and 5:
   - `--output_dir`: where to save models/outputs
   - `--train_data`: which CSV to train on
   - Resolves all paths to absolute before `os.chdir(output_dir)`
   - Compatible with `run_experiment.py` orchestrator

10. **Fixed GNN pretrained weights path** to use script-relative path instead of CWD-relative.

### Phase 5: Environment Fixes

11. **Fixed `torch_geometric` compatibility** with PyTorch 2.11.0:
    - `torch_geometric` 2.7.0, `torch_scatter` 2.1.2, `torch_sparse` 0.6.18 crashed at C++ level
    - Reinstalled from PyG wheels for PyTorch 2.11.0:
      ```
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
      pip install torch-geometric -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
      ```

### Phase 6: Smoke Test

12. **Model 4 (ChemBERTa)**: PASS — loaded data (750 samples), started ChemBERTa embeddings, generated features across all 6 mutation sites.

13. **Model 5 (GNN)**: PASS — loaded data, initialized GNN, saved mutation profiles, started embedding extraction. Needed `torch_geometric` reinstall (Phase 5) and pretrained weights path fix.

### Phase 7: Full Experiment V2

14. **Updated `run_all_experiments.sh`** to output to `experiments_v2/`:
    - Same structure: 6 models x 2 datasets x 2 directions = 24 runs
    - Same datasets as v0/v1 to isolate bug fix impact
    - Clears feature cache for fresh start

15. **Running all 24 experiments** (12 training + 12 cross-CSV inference).

---

## Git Commits on `arshath/pipeline-optimizations`

| Commit | Description |
|--------|-------------|
| `e5d3d9b` | Add experiment orchestration script and analysis docs for v0/v1 runs |
| `d5578f5` | Apply bug fixes from corrected scripts to models 4 (ChemBERTa) and 5 (GNN) |
| `10d551f` | Fix path resolution in corrected training scripts 4 and 5 |

---

## File Mapping: Corrected -> Our Scripts

### Training Scripts

| # | Rhubain's Corrected Name | Our Filename | Changed? |
|---|--------------------------|-------------|----------|
| 0 | `dummy_physchem_5f2.py` | `training_scripts/0_dummy_physchem_5f2.py` | No |
| 1 | `adv_physchem5f2_hierachichal_ltsm_gru_custom.py` | `training_scripts/1_adv_physchem5f2.py` | No |
| 2 | `adv_physchem_KAN3_b_spline1a.py` | `training_scripts/2_adv_physchem_KAN3_b_spline1a.py` | No |
| 3 | `adv_physchem_KAN_base2_navier_stokes_sinusoid.py` | `training_scripts/3_adv_physchem_KAN_navier_stokes_sinusoid.py` | No |
| 4 | `adv_physchem_chemberta_crossattention2_corrected.py` | `training_scripts/4_adv_physchem_chemerta_crossattention.py` | **YES** |
| 5 | `adv_physchem_gnn_base1b.py` | `training_scripts/5_adv_physchem_gnn.py` | **YES** |

### Inference Scripts

| # | Rhubain's Corrected Name | Our Filename | Changed? |
|---|--------------------------|-------------|----------|
| 0 | `predict_dummy_physchem_5f2_updated.py` | `inference_scripts/0_predict_dummy_physchem_5f2.py` | No |
| 1 | `predict_adv_physchem5f2_hierachical_ltsm_gru_custom.py` | `inference_scripts/1_predict_adv_physchem5f2.py` | No |
| 2 | `predict_adv_physchem_KAN3_b_spline1a.py` | `inference_scripts/2_predict_adv_physchem_KAN_bspline.py` | No |
| 3 | `predict_adv_physchem_KAN_base2_navier_stokes.py` | `inference_scripts/3_predict_adv_physchem_KAN_navier_stokes.py` | No |
| 4 | `predict_adv_physchem_chemberta_crossattention2_corrected.py` | `inference_scripts/4_predict_adv_physchem_chembert_crossattention.py` | **YES** |
| 5 | `predict_adv_physchem_gnn_base1b.py` | `inference_scripts/5_predict_adv_physchem_gnn.py` | **YES** |

---

## Dataset Mapping

| Our Dataset | Rows | Rhubain's Equivalent | Rows | Notes |
|-------------|------|---------------------|------|-------|
| `df_3_shuffled.csv` | 750 | `valid_drug_tki_trimmed.csv` | 718 | Trimmed = outliers removed |
| `manual_egfr3_mini_dock_fixed.csv` | 2676 | (no direct equivalent) | — | Larger, different schema |

For experiment_v2, we used the SAME datasets as v0/v1 (`df_3_shuffled.csv` + `manual_egfr3_mini_dock_fixed.csv`) to isolate the impact of bug fixes only.

---

## Experiment Structure

```
arshath/
├── experiments_v0/          # Feb 18 run (original scripts, before optimization)
├── experiments_v1/          # Feb 28 run (optimized scripts with caching)
├── experiments_v2/          # Mar 28 run (corrected + optimized scripts)
│   ├── model_0_dummy_physchem/
│   │   ├── df_3_shuffled/           (trained + predictions_cross_csv/)
│   │   └── manual_egfr3_mini_dock_fixed/  (trained + predictions_cross_csv/)
│   ├── model_1_adv_physchem5f2/
│   │   ├── df_3_shuffled/
│   │   └── manual_egfr3_mini_dock_fixed/
│   ├── model_2_kan_bspline/
│   │   ├── df_3_shuffled/
│   │   └── manual_egfr3_mini_dock_fixed/
│   ├── model_3_kan_navier_stokes/
│   │   ├── df_3_shuffled/
│   │   └── manual_egfr3_mini_dock_fixed/
│   ├── model_4_chembert_crossattention/
│   │   ├── df_3_shuffled/
│   │   └── manual_egfr3_mini_dock_fixed/
│   └── model_5_gnn/
│       ├── df_3_shuffled/
│       └── manual_egfr3_mini_dock_fixed/
├── training_scripts/        # 6 scripts (4 and 5 updated with bug fixes)
├── inference_scripts/       # 6 scripts (4 and 5 updated with bug fixes)
├── corrected_scripts/       # Rhubain's originals (staging area)
├── data/
│   ├── df_3_shuffled.csv
│   ├── manual_egfr3_mini_dock_fixed.csv
│   ├── valid_drug_tki_trimmed.csv    # NEW - trimmed dataset
│   ├── egfr_tki_valid_cleaned.csv    # Control SMILES
│   └── drugs.csv                      # Drug candidates
├── checkpoints/
│   └── molclr_pretrained.pth          # GNN pretrained weights
├── run_experiment.py                  # Orchestrator
├── run_all_experiments.sh             # Master script
└── docs/                              # Analysis & reference docs
```

---

## What Changed Between Experiments

| Comparison | What Changed | What It Shows |
|-----------|-------------|---------------|
| v0 vs v1 | Nothing (same scripts, same data) | Reproducibility/variance |
| v1 vs v2 | Scripts 4 & 5 bug fixes | Impact of corrected physics + architecture |
| v0 vs v2 | Scripts 4 & 5 bug fixes | Overall improvement |

**Expected impact**: Models 0-3 should produce very similar results in v2 (same code). Models 4 and 5 may show significant changes due to:
- Fixed electrostatic features (correct physics normalization)
- Fixed cross-attention architecture (actual attention instead of constant projection)
- Fixed ChemBERTa signal pathway (meaningful representation instead of crushed)
- Fixed site ordering in inference (correct scaler alignment)
