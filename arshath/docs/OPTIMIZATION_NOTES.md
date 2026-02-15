# Pipeline Optimization Notes

## Overview

Five performance optimizations were applied to all 9 scripts (6 training + 3 inference) in the EGFR TKI pipeline. The primary bottleneck was redundant RDKit feature computation from SMILES strings -- the same molecules were recomputed across mutation sites, across mutation profiles, and across separate runs.

All scripts remain standalone. No shared utility modules were introduced.

---

## Changes

### 1. Disk-backed feature cache

**Scripts:** All 9

Computed RDKit features (25 intermolecular + 30 intramolecular) are saved as pickle files in `data/.feature_cache/`, keyed by MD5 hash of the SMILES string. On subsequent runs, features are loaded from disk instead of recomputed.

- Cache path: `data/.feature_cache/{md5}.pkl`
- Each file stores `{'inter': np.array, 'intra': np.array}`
- Features are deterministic, so no invalidation logic is needed
- To force recomputation: `rm -rf arshath/data/.feature_cache/`
- Hit/miss counters are printed at the end of each run

### 2. Persistent in-memory caches across mutation sites

**Scripts:** Training scripts 1-5

`generate_hierarchical_features` is called 6 times per training run (once per mutation site). Previously, each call created fresh `ligand_cache = {}` and `mutation_cache = {}` dicts that were discarded after each call. Since ligand and mutation features are independent of which site is being processed, these caches are now created once in `main()` and passed into all 6 calls.

The function signature was changed from:
```python
def generate_hierarchical_features(ligand_smiles_series, mutation_smiles_series)
```
to:
```python
def generate_hierarchical_features(ligand_smiles_series, mutation_smiles_series, ligand_cache=None, mutation_cache=None)
```

### 3. Embedding model creation hoisted outside loops

**Scripts:** Training script 1, inference scripts 1 and 3

Previously, `load_model()` and `Model(inputs=..., outputs=...)` were called inside per-sample or per-site loops, redundantly reconstructing the same Keras models. These are now created once before the loops:

- **Training script 1:** `_loaded_embedding_models` dict built once before the prediction loop. Both control and drug prediction sections reference it.
- **Inference script 1:** `embedding_models` dict built once after `load_models_and_scalers`.
- **Inference script 3:** All hierarchical models loaded once with `custom_objects={'KANLayer': KANLayer, 'FourierKANLayer': FourierKANLayer}`, and `_embedding_models` dict built once.

### 4. Batch predictions in inference scripts

**Scripts:** Inference scripts 1 and 3

Replaced per-row iteration (`for idx, row in df_pred.iterrows()`) with group-by-mutation batch processing:

1. Group prediction rows by mutation type
2. Per mutation group: batch-compute ligand features, batch-scale, batch-predict embeddings through hierarchical models, batch-predict through RNN
3. Collect results and reassemble in original order

### 5. Single mol parse per SMILES

**Scripts:** All 9

Added a `_generate_lig_features(smiles)` helper that:
- Checks the disk cache first
- Calls `Chem.MolFromSmiles()` once
- Computes both inter (25) and intra (30) feature arrays from the same mol object
- Computes `MaxPartialCharge`/`MinPartialCharge` once (previously computed 3 times each in `generate_lig_inter_features`)
- Saves to disk cache on completion

The original `generate_lig_inter_features` and `generate_lig_intra_features` functions are preserved unchanged but are no longer called on the hot path.

---

## Files modified

| Script | #1 Disk cache | #2 Persistent cache | #3 Model hoist | #4 Batch predict | #5 Combined parse |
|--------|:---:|:---:|:---:|:---:|:---:|
| `training_scripts/0_dummy_physchem_5f2.py` | x | | | | x |
| `training_scripts/1_adv_physchem5f2.py` | x | x | x | | x |
| `training_scripts/2_adv_physchem_KAN3_b_spline1a.py` | x | x | | | x |
| `training_scripts/3_adv_physchem_KAN_navier_stokes_sinusoid.py` | x | x | | | x |
| `training_scripts/4_adv_physchem_chemerta_crossattention.py` | x | x | | | x |
| `training_scripts/5_adv_physchem_gnn.py` | x | x | | | x |
| `inference_scripts/0_predict_dummy_physchem_5f2.py` | x | | | | x |
| `inference_scripts/1_predict_adv_physchem5f2.py` | x | | x | x | x |
| `inference_scripts/3_predict_adv_physchem_KAN_navier_stokes.py` | x | | x | x | x |

---

## Verification

To verify optimizations produce identical results:

1. Run the original version of a script, save output CSVs
2. Run the optimized version on the same input
3. Diff outputs -- should be numerically identical
4. Check that `data/.feature_cache/` was populated
5. Run again -- confirm cache hits are logged and execution is faster
