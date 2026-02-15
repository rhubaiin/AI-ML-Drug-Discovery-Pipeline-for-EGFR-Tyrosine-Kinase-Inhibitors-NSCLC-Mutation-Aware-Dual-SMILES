# ML-Driven Pipeline for 4th Generation EGFR TKIs — Experiment Guide

## What This Project Is

This is a **drug discovery ML pipeline** for predicting 4th-generation EGFR inhibitor activity against Non-Small Cell Lung Cancer (NSCLC). It uses molecular data from ChEMBL to train neural networks that predict:

1. **Activity** (IC50/EC50 — how potent a drug is)
2. **Docking score** (how well a drug binds to the EGFR protein)

---

## Directory Structure

```
arshath/
├── run_experiment.py           # Experiment orchestrator
├── data/                       # All datasets
│   ├── manual_egfr3_mini_dock_fixed.csv   # Large training set (2,677 rows)
│   ├── df_3_shuffled.csv                  # Smaller training set (751 rows)
│   ├── egfr_tki_valid_cleaned.csv         # Control compounds (13 known drugs)
│   ├── drugs.csv                          # Reference drug molecules (7 rows)
│   ├── test_egfr3.csv                     # Test set (41 rows)
│   └── test_df_3_shuffled.csv             # Test set (41 rows)
│
├── training_scripts/           # Model training (numbered 0–5)
│   ├── 0_dummy_physchem_5f2.py
│   ├── 1_adv_physchem5f2.py
│   ├── 2_adv_physchem_KAN3_b_spline1a.py
│   ├── 3_adv_physchem_KAN_navier_stokes_sinusoid.py
│   ├── 4_adv_physchem_chemerta_crossattention.py
│   └── 5_adv_physchem_gnn.py
│
├── inference_scripts/          # Prediction / inference (numbered to match)
│   ├── 0_predict_dummy_physchem_5f2.py
│   ├── 1_predict_adv_physchem5f2.py
│   └── 3_predict_adv_physchem_KAN_navier_stokes.py
│
├── experiments/                # Auto-organized experiment outputs
│   ├── model_0_dummy_physchem/
│   │   └── <dataset_name>/     # e.g. manual_egfr3_mini_dock_fixed/
│   ├── model_1_adv_physchem5f2/
│   │   └── <dataset_name>/
│   └── ...
│
└── docs/                       # Documentation
    ├── EXPERIMENT_GUIDE.md     # This file
    └── readme                  # Original quick reference
```

> **Note:** Scripts #2, #4, and #5 do not have inference scripts yet.

---

## The Two Datasets

| Dataset | File | Rows | Purpose |
|---------|------|------|---------|
| **Large dataset** | `data/manual_egfr3_mini_dock_fixed.csv` | ~2,677 | Has docking scores included |
| **Smaller dataset** | `data/df_3_shuffled.csv` | ~751 | Cleaned and shuffled subset |

Both have the same column structure (64 columns) including SMILES strings, physicochemical properties, mutation info, and activity values.

### Supporting Files

| File | Rows | Purpose |
|------|------|---------|
| `data/egfr_tki_valid_cleaned.csv` | 13 | Known EGFR drugs (erlotinib, osimertinib, etc.) — **control compounds** for validation |
| `data/drugs.csv` | 7 | Reference drug molecules |
| `data/test_egfr3.csv` | 41 | Test set for evaluation |
| `data/test_df_3_shuffled.csv` | 41 | Smaller test version |

---

## Script Pairing (Training ↔ Inference)

The numbering prefix matches each training script to its inference counterpart:

| # | Training Script | Inference Script | Model Type |
|---|----------------|-----------------|-----------|
| 0 | `0_dummy_physchem_5f2.py` | `0_predict_dummy_physchem_5f2.py` | Simple Feed-Forward NN (baseline) |
| 1 | `1_adv_physchem5f2.py` | `1_predict_adv_physchem5f2.py` | Advanced Feed-Forward NN |
| 2 | `2_adv_physchem_KAN3_b_spline1a.py` | — (not yet available) | KAN + B-Spline |
| 3 | `3_adv_physchem_KAN_navier_stokes_sinusoid.py` | `3_predict_adv_physchem_KAN_navier_stokes.py` | KAN + Fourier/RBF |
| 4 | `4_adv_physchem_chemerta_crossattention.py` | — (not yet available) | ChemBERTa + Cross-Attention |
| 5 | `5_adv_physchem_gnn.py` | — (not yet available) | Graph Neural Network |

---

## How the Experiment Works

### Step 1: Train on one dataset

Each training script:
- Loads a training CSV from `data/`
- Generates physicochemical features from SMILES (intermolecular, intramolecular, similarity metrics — ~72 features)
- Encodes mutation types via LabelEncoder
- Trains a dual-output model (activity + docking)
- Saves the trained model (`.h5`), scalers (`.pkl`), and encoder (`.pkl`)

### Step 2: Predict on the other dataset

Each inference script:
- Loads the saved model, scalers, and encoder
- Runs predictions on control compounds (`data/egfr_tki_valid_cleaned.csv`) and drug candidates (`data/drugs.csv`)
- Outputs `control_predictions.csv` and `drug_predictions.csv`

### Step 3: Swap datasets and repeat

The core idea is **cross-validation between the two datasets**:

- **Run A**: Train on `manual_egfr3_mini_dock_fixed.csv` → predict on `df_3_shuffled.csv` / test sets
- **Run B**: Train on `df_3_shuffled.csv` → predict on `manual_egfr3_mini_dock_fixed.csv` / test sets

To swap datasets, use the `--train_data` flag:
```bash
python run_experiment.py --model 1 --train_data data/manual_egfr3_mini_dock_fixed.csv
```

Each dataset run gets its own output directory, so results from different datasets never overwrite each other.

---

## Recommended Execution Order

### Option A: Use the orchestrator (recommended)

The `run_experiment.py` orchestrator automates training and inference, organizing all outputs into `experiments/<model_name>/<dataset_name>/`.

```bash
# Run from arshath/ directory

# --- Model 0: Train + Inference ---
python run_experiment.py --model 0 --predict_input data/test_egfr3.csv

# --- Models 0, 1, 3: Train + Inference ---
python run_experiment.py --model 0 1 3 --predict_input data/test_egfr3.csv

# --- All 6 models: Train only ---
python run_experiment.py --model all

# --- Swap dataset: Train model 1 with the larger dataset ---
python run_experiment.py --model 1 --train_data data/manual_egfr3_mini_dock_fixed.csv
```

Outputs land in structured directories:
```
experiments/model_0_dummy_physchem/manual_egfr3_mini_dock_fixed/
experiments/model_1_adv_physchem5f2/df_3_shuffled/
experiments/model_1_adv_physchem5f2/manual_egfr3_mini_dock_fixed/  # different dataset run
```

### Option B: Run scripts directly with CLI args

All training scripts now accept `--output_dir`, `--train_data`, `--control_data`, and `--drug_data`:

```bash
# Run from arshath/ directory

# --- Model 0: Baseline ---
python training_scripts/0_dummy_physchem_5f2.py --output_dir /tmp/model0
python inference_scripts/0_predict_dummy_physchem_5f2.py --input data/test_egfr3.csv --model_dir /tmp/model0

# --- Model 1: Advanced Feed-Forward ---
python training_scripts/1_adv_physchem5f2.py --output_dir /tmp/model1
python inference_scripts/1_predict_adv_physchem5f2.py --input data/test_egfr3.csv --model_dir /tmp/model1

# --- Model 3: KAN Fourier ---
python training_scripts/3_adv_physchem_KAN_navier_stokes_sinusoid.py --output_dir /tmp/model3
python inference_scripts/3_predict_adv_physchem_KAN_navier_stokes.py --input data/test_egfr3.csv --model_dir /tmp/model3
```

When `--output_dir` is omitted from inference scripts, predictions default to `{model_dir}/predictions/`.

### Option C: Run with defaults (legacy behavior)

Scripts still work without any arguments -- they use the same defaults as before (CWD for output, hardcoded data paths).

---

## EGFR Mutation Targets

The pipeline targets these EGFR mutations:

| Mutation | Type |
|----------|------|
| WT | Wild-type (normal EGFR) |
| del19 | Deletion exon 19 (sensitizing) |
| L858R | Point mutation (sensitizing) |
| ins20 | Insertion exon 20 (sensitizing) |
| T790M | Acquired resistance to 1st/2nd gen TKIs |
| L858R/T790M | Double mutation |
| C797S | 3rd gen resistance |
| L858R/T790M/C797S | Triple mutation |

### Protein Regions Captured as SMILES

1. Full EGFR sequence (721–862 ATP pocket)
2. P-loop/hinge (719–724) — ATP binding
3. C-helix (752–760) — Structural stabilization
4. A-loop/DFG (857–859) — Activation loop
5. Catalytic HRD motif (831–839) — Catalytic activity

---

## Model Training Details

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Epochs | 100–150 |
| Validation split | 0.2 (80/20) |
| Early stopping patience | 30–40 epochs |
| Activity loss weight | 1.0 |
| Docking loss weight | 0.6–0.7 |
| Activity transform | log(1+x) |
| Feature dimensions | ~72 (27 inter + 30 intra + 15 similarity) |

---

## Required Dependencies

- `tensorflow` / `keras`
- `rdkit`
- `scikit-learn`
- `pandas`, `numpy`
- `torch-geometric` (for GNN variant #5 only)
- `transformers` (for ChemBERTa variant #4 only)

---

## Output Files

All outputs are organized under `experiments/<model_name>/<dataset_name>/`:

```
experiments/model_1_adv_physchem5f2/df_3_shuffled/
├── hierarchical_model_full.h5          # Per-site models
├── hierarchical_model_hinge_p_loop.h5
├── hierarchical_model_c_helix.h5
├── hierarchical_model_a_loop_dfg.h5
├── hierarchical_model_hrd_cat.h5
├── rnn_sequential_model.h5             # Combined RNN model
├── feature_scalers.pkl                 # Input feature scalers
├── y_scalers.pkl                       # Output target scalers
├── mutation_profiles.csv               # Per-mutation activity profiles
├── control_predictions_rnn.csv         # Predictions for known EGFR TKI drugs
├── drug_predictions_rnn.csv            # Predictions for candidate molecules
├── rnn_training_history.png            # Training loss/metric curves
└── predictions/                        # Inference outputs (from inference scripts)
    ├── predictions_adv_physchem5f2.csv
    ├── metrics/
    └── plots/
```

### Per-model output differences

| Model | Weights | Extra scalers | Predictions |
|-------|---------|---------------|-------------|
| 0 — Dummy | `feedforward_model.h5` | `mutant_encoder.pkl`, `mutant_mapping.pkl` | `control_predictions.csv`, `drug_predictions.csv` |
| 1 — Advanced | `hierarchical_model_{site}.h5` (×5), `rnn_sequential_model.h5` | — | `control_predictions_rnn.csv`, `drug_predictions_rnn.csv` |
| 2 — KAN B-Spline | `hierarchical_model_{site}.h5` (×5), `rnn_sequential_model.h5` | — | `control_predictions_rnn.csv`, `drug_predictions_rnn.csv` |
| 3 — KAN Fourier | `hierarchical_model_{site}.h5` (×5), `rnn_sequential_model.h5` | — | `control_predictions_rnn.csv`, `drug_predictions_rnn.csv` |
| 4 — ChemBERTa | `hierarchical_model_{site}.h5` (×5), `rnn_sequential_model.h5` | `chembert_scalers.pkl` | `control_predictions_rnn.csv`, `drug_predictions_rnn.csv` |
| 5 — GNN | `gnn_hierarchical_{site}.h5` (×5), `gnn_rnn_model.h5` | `gnn_embedding_scalers.pkl` | `gnn_{name}_predictions.csv` |

All models produce `feature_scalers.pkl`, `y_scalers.pkl`, `mutation_profiles.csv`, and a training history plot.

---

## CLI Arguments Reference

### Training scripts (all 6)

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `.` (CWD) | Directory for all saved outputs |
| `--train_data` | Model-specific default CSV | Training data CSV path |
| `--control_data` | `data/egfr_tki_valid_cleaned.csv` | Control compounds CSV |
| `--drug_data` | `data/drugs.csv` | Drug compounds CSV |

### Inference scripts (0, 1, 3)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Input CSV for predictions |
| `--model_dir` | *(required)* | Directory containing trained model files |
| `--output_dir` | `{model_dir}/predictions/` | Directory for prediction outputs |

### Orchestrator (`run_experiment.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model IDs (0-5) or `all` |
| `--predict_input` | *(none)* | Input CSV for inference after training |
| `--train_data` | Model-specific default | Override training CSV |
| `--control_data` | *(none)* | Override control compounds CSV |
| `--drug_data` | *(none)* | Override drug compounds CSV |
| `--experiments_dir` | `experiments/` | Base output directory |
