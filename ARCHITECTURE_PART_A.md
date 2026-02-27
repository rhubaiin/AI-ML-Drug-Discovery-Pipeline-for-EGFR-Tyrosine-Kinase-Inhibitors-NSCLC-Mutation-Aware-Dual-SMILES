# Architecture of Part A — Activity Prediction ML Pipeline

## Overview

Part A is a **two-stage neural network pipeline** that runs sequentially:

1. **Stage 1 — Priority Hierarchical Model:** A forward neural network trained independently for each of 6 EGFR structural mutation sites. Extracts a 16-dimensional embedding from 8 physicochemical feature inputs, guided by a cascading priority-gating mechanism.

2. **Stage 2 — RNN-LSTM Sequential Model:** Receives the 6 extracted embeddings stacked as a temporal sequence `(batch, 6, 16)`. Runs two parallel recurrent paths (Bidirectional LSTM and Bidirectional GRU) to learn structural–sequential dependencies, then outputs final activity and docking predictions.

---

## Stage 1 — Feature Generation

Before the neural network, 8 feature arrays are computed per ligand–mutation pair using RDKit. These are the exact 8 inputs to the hierarchical model.

```
Ligand SMILES ─┐
               ├─→ generate_lig_inter_features()   ─→  lig_inter   (intermolecular: H-bond, charge, PSA, LogP...)
               ├─→ generate_lig_intra_features()   ─→  lig_intra   (intramolecular: bonds, rings, complexity...)
               └─→ [used in pair-level features]

Mutation SMILES ─┐
                 ├─→ generate_mut_inter_features()  ─→  mut_inter   (same engine as lig_inter)
                 ├─→ generate_mut_intra_features()  ─→  mut_intra   (same engine as lig_intra)
                 └─→ [used in pair-level features]

Pair (Lig + Mut) ─┐
                  ├─→ generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra)
                  │     ├─→  lig_mut_inter  →┐ concatenated into...
                  │     ├─→  lig_mut_intra  →┘
                  │     └─→  lig_mut_mix_inter_intra  (H-path, charge×size, pi-pi ratios)
                  │
                  ├─→ generate_inter_interaction_features(lig_inter, mut_inter)
                  │     └─→  [cosine_sim, sine_dissim] + prepend lig_mut_inter
                  │           = inter_interaction  (cross-similarity on intermolecular vectors)
                  │
                  ├─→ generate_intra_interaction_features(lig_intra, mut_intra)
                  │     └─→  [cosine_sim, sine_dissim] + prepend lig_mut_intra
                  │           = intra_interaction  (cross-similarity on intramolecular vectors)
                  │
                  └─→ generate_final_interaction_features(lig_smi, mut_smi)
                        └─→  [Dice sim, Tanimoto sim] (Morgan FP radius=2, nBits=2048)
                              = final_fp_interaction
```

### The 8 Feature Inputs (in priority order)

| Priority | Input Name | What it Contains |
|:---:|---|---|
| **P1a** | `final_fp_interaction` | Morgan fingerprint Tanimoto + Dice similarity between ligand and mutation SMILES |
| **P1b** | `lig_mut_mix_inter_intra` | Custom features mixing inter+intra forces: H-path, charge×size products, pi-pi ratios, hybridisation differences |
| **P2** | `inter_interaction` | Custom intermolecular ratios (H-bond, charge, TPSA, LogP differences) + cosine/sine similarity of intermolecular vectors |
| **P3** | `intra_interaction` | Custom intramolecular ratios (bond rigidity, hybridisation fractions, Bertz ratio) + cosine/sine similarity of intramolecular vectors |
| **P4** | `mut_inter` | Raw mutation intermolecular descriptors (H-bond, partial charge, PSA, LogP, MolWt, aromaticity, halogens) |
| **P5** | `lig_inter` | Raw ligand intermolecular descriptors (same descriptor set as P4) |
| **P6** | `mut_intra` | Raw mutation intramolecular descriptors (bonds, ring count, hybridisation, ring strain, Bertz, Kappa) |
| **P7** | `lig_intra` | Raw ligand intramolecular descriptors (same descriptor set as P6) |

> `inter_interaction` and `intra_interaction` are **augmented vectors**: custom ratio/difference features are prepended to the cosine/sine similarity values, making them richer than pure similarity metrics.

---

## Stage 1 — Priority Hierarchical Model

This model is trained **6 times** — once per mutation site. For each run, it receives the same 8 inputs and produces a 16-dimensional embedding from the `embedding_output` layer (just before the auxiliary heads).

### Gating Logic

> **Concatenation** preserves higher-priority features as the dominant baseline. **Multiplication** (Sigmoid gate) filters what lower-priority features contribute. Each gate is computed *from all previously accumulated priority embeddings*, so every new layer is conditioned on all higher-priority context.

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │           PRIORITY HIERARCHICAL MODEL                        │
                  │    (trained once per mutation site, 6 sites total)           │
                  └─────────────────────────────────────────────────────────────┘

 INPUTS                     BRANCHES                      EMBEDDING         DIMENSION

 final_fp_interaction   ──→  Dense(32)→LeakyReLU→BN
                              →Dropout(0.1)→Dense(16)
                              →LeakyReLU→Dense(8,tanh)   = final_emb            (8)
                                                                 │
                                                          Concatenate
                                                                 │
 lig_mut_mix_inter_intra ──→  Dense(8)→LeakyReLU
                              →Dense(4,tanh)              = mix_emb             (4)
                                                                 │
                                                  = priority1_combined          (12)
                                                          ┌──────┴──────────────────────────────┐
                                                          │  Gate source for P2                  │
                                                          ▼                                       │
 inter_interaction   ──→  Dense(48)→LeakyReLU→BN                                               │
                                  │                                                              │
                          priority1_combined(12) → Dense(48,sigmoid) = inter_gate               │
                                  │                       │                                      │
                          inter_branch ⊙ inter_gate  →  Dropout(0.1)→Dense(24)                 │
                          →LeakyReLU→Dense(12,tanh)  = inter_emb                (12)            │
                                                                 │                               │
                                              priority1_combined(12) + inter_emb(12)             │
                                            = priority1_2_combined               (24)            │
                                                          ┌──────┴──────────────────────────────┘
                                                          │  Gate source for P3
                                                          ▼
 intra_interaction   ──→  Dense(48)→LeakyReLU→BN
                                  │
                          priority1_2_combined(24) → Dense(48,sigmoid) = intra_gate
                                  │                        │
                          intra_branch ⊙ intra_gate  →  Dropout(0.1)→Dense(24)
                          →LeakyReLU→Dense(12,tanh)  = intra_emb               (12)
                                                                 │
                                           priority1_2_combined(24) + intra_emb(12)
                                         = priority1_2_3_combined               (36)
                                                    ┌───────────┴─────────────────────────────────┐
                                                    │  Gate source for P4 and P5 (independent gates│
                                                    │  from same source)                           │
                                                    ▼                                              │
 mut_inter   ──→  Dense(32)→LeakyReLU→BN                                                         │
                        │                                                                          │
             priority1_2_3_combined(36) → Dense(32,sigmoid) = mut_inter_gate                     │
                        │                        │                                                 │
             mut_inter_branch ⊙ mut_inter_gate → Dropout(0.1)→Dense(16)                          │
             →LeakyReLU→Dense(8,tanh)         = mut_inter_emb               (8)                  │
                                                             │                                     │
                                                      Concatenate                                  │
                                                             │                                     │
 lig_inter   ──→  Dense(32)→LeakyReLU→BN                   │                                     │
                        │                                                                          │
             priority1_2_3_combined(36) → Dense(32,sigmoid) = lig_inter_gate                     │
                        │                        │                                                 │
             lig_inter_branch ⊙ lig_inter_gate → Dropout(0.1)→Dense(16)                          │
             →LeakyReLU→Dense(8,tanh)         = lig_inter_emb               (8)                  │
                                                             │                                     │
                                              = inter_combined               (16)                  │
                                                             │                                     │
                              priority1_2_3_combined(36) + inter_combined(16)                     │
                            = priority1_to_5_combined        (52) [GATING USE ONLY]               │
                                                    ┌─────────────────────────────────────────────┘
                                                    │  Gate source for P6 and P7 (independent gates
                                                    │  from same source)
                                                    ▼
 mut_intra   ──→  Dense(32)→LeakyReLU→BN
                        │
             priority1_to_5_combined(52) → Dense(32,sigmoid) = mut_intra_gate
                        │                         │
             mut_intra_branch ⊙ mut_intra_gate → Dropout(0.25)→Dense(16)
             →LeakyReLU→Dense(8,tanh)          = mut_intra_emb              (8)
                                                             │
                                                      Concatenate
                                                             │
 lig_intra   ──→  Dense(32)→LeakyReLU→BN
                        │
             priority1_to_5_combined(52) → Dense(32,sigmoid) = lig_intra_gate
                        │                         │
             lig_intra_branch ⊙ lig_intra_gate → Dropout(0.25)→Dense(16)
             →LeakyReLU→Dense(8,tanh)          = lig_intra_emb              (8)
                                                             │
                                              = intra_combined               (16)


 ─────────────────────────── FINAL COMBINATION ──────────────────────────────────

 Concatenate[priority1_2_3_combined(36), inter_combined(16), intra_combined(16)]
                                   = all_combined                              (68)

 NOTE: priority1_to_5_combined(52) feeds P6/P7 gates ONLY — it is NOT in all_combined.

 ──────────────────────────── INTEGRATION LAYERS ────────────────────────────────

 all_combined(68)
   → Dense(128) → LeakyReLU → BN → Dropout(0.3)
   → Dense(64)  → LeakyReLU → BN → Dropout(0.2)
   → Dense(32)  → LeakyReLU
   → Dense(16)  → LeakyReLU    ← 'embedding_output'  (16)  ← EXTRACTED FOR RNN

 ──────────────────────────── AUXILIARY HEADS ────────────────────────────────────

 embedding_output(16)
   ├──→ Dense(8)→LeakyReLU→Dropout(0.2)→Dense(1,linear) = activity_output
   └──→ Dense(8)→LeakyReLU→Dropout(0.2)→Dense(1,linear) = docking_output
```

### Gating Summary Table

| Priority Block | Input | Gate Source | Gate Produces | Embedding Output |
|---|---|---|---|:---:|
| **P1a — FP** | `final_fp_interaction` | *(none, no gate)* | — | `final_emb` (8) |
| **P1b — Mix** | `lig_mut_mix_inter_intra` | *(none, no gate)* | — | `mix_emb` (4) |
| P1a + P1b concatenated | — | — | `priority1_combined` | (12) |
| **P2 — Inter-Sim** | `inter_interaction` | `priority1_combined` (12) | `inter_gate` (48) | `inter_emb` (12) |
| P1+P2 concatenated | — | — | `priority1_2_combined` | (24) |
| **P3 — Intra-Sim** | `intra_interaction` | `priority1_2_combined` (24) | `intra_gate` (48) | `intra_emb` (12) |
| P1+P2+P3 concatenated | — | — | `priority1_2_3_combined` | (36) |
| **P4 — Mut Inter** | `mut_inter` | `priority1_2_3_combined` (36) | `mut_inter_gate` (32) | `mut_inter_emb` (8) |
| **P5 — Lig Inter** | `lig_inter` | `priority1_2_3_combined` (36) | `lig_inter_gate` (32) | `lig_inter_emb` (8) |
| P4+P5 concatenated | — | — | `inter_combined` | (16) |
| P1–P5 combined (gating only) | — | — | `priority1_to_5_combined` | (52) |
| **P6 — Mut Intra** | `mut_intra` | `priority1_to_5_combined` (52) | `mut_intra_gate` (32) | `mut_intra_emb` (8) |
| **P7 — Lig Intra** | `lig_intra` | `priority1_to_5_combined` (52) | `lig_intra_gate` (32) | `lig_intra_emb` (8) |
| P6+P7 concatenated | — | — | `intra_combined` | (16) |
| **all_combined** | P1-P3 (36) + P4-P5 (16) + P6-P7 (16) | — | — | **(68)** |

> **P4 and P5 share the same gate *source* but have independent gate *weights* (`mut_inter_gate` vs `lig_inter_gate`). The same is true for P6 and P7.** Each pair processes its respective feature through its own separately learned gate, even though both are conditioned on the same accumulated priority context.

---

## Stage 2 — Embedding Extraction & Sequence Assembly

After Stage 1 training, the `embedding_output` layer (the 16-dim layer before the auxiliary heads) is extracted from each of the 6 trained hierarchical models:

```
Mutation sites processed in order:

  t=0  FULL_SMILES      → Hierarchical Model → embedding (batch, 16)
  t=1  ATP_POCKET       → Hierarchical Model → embedding (batch, 16)
  t=2  P_LOOP_HINGE     → Hierarchical Model → embedding (batch, 16)
  t=3  C_HELIX          → Hierarchical Model → embedding (batch, 16)
  t=4  DFG_A_LOOP       → Hierarchical Model → embedding (batch, 16)
  t=5  HRD_CAT          → Hierarchical Model → embedding (batch, 16)

  np.stack(all_embeddings, axis=1)
  → sequential_embeddings shape: (batch, 6, 16)
```

The sequence order reflects the **mechanistic EGFR signal transduction pathway**, so the forward LSTM/GRU reads it in biological order (overall context → active site → catalytic outcome) and the backward pass reads in reverse.

---

## Stage 2 — RNN Sequential Model

This model receives the stacked embeddings `(batch, 6, 16)` and runs **two parallel recurrent paths** from the same input.

```
 sequence_input  (batch, 6, 16)
       │
       ├─────────────────────────────────────────────────────────────────────────┐
       │                                                                         │
       ▼  LSTM PATH                                                              ▼  GRU PATH
                                                                                    (parallel, same input)
 BiLSTM Layer 1                                                            BiGRU Layer 1
   LSTM(128 units, return_sequences=True)                                    GRU(128 units, return_sequences=True)
   Forward:  t=0→1→2→3→4→5  →  (batch, 6, 128)                             Forward:  t=0→1→2→3→4→5  →  (batch, 6, 128)
   Backward: t=5→4→3→2→1→0  →  (batch, 6, 128)                             Backward: t=5→4→3→2→1→0  →  (batch, 6, 128)
   Concat:   [fwd ║ bwd]     →  (batch, 6, 256)                             Concat:   [fwd ║ bwd]     →  (batch, 6, 256)
   BatchNorm                  →  (batch, 6, 256)                             BatchNorm                  →  (batch, 6, 256)
       │                                                                         │
       ▼                                                                         ▼
 BiLSTM Layer 2                                                            BiGRU Layer 2
   LSTM(64 units, return_sequences=False)                                    GRU(64 units, return_sequences=False)
   Forward:  final hidden state  →  (batch, 64)                             Forward:  final hidden state  →  (batch, 64)
   Backward: final hidden state  →  (batch, 64)                             Backward: final hidden state  →  (batch, 64)
   Concat:   [fwd ║ bwd]         →  (batch, 128)                            Concat:   [fwd ║ bwd]         →  (batch, 128)
   BatchNorm                     →  (batch, 128)                             BatchNorm                     →  (batch, 128)
       │                                                                         │
       └────────────────────────────┬────────────────────────────────────────────┘
                                    │
                         Concatenate[lstm(128) ║ gru(128)]
                                    │
                              (batch, 256)
                                    │
                         ─── DENSE INTEGRATION ───
                                    │
                         Dense(128, relu) → BN → Dropout(0.3)
                                    │
                         Dense(64, relu)  → BN → Dropout(0.2)
                                    │
                         Dense(32, relu)  → Dropout(0.1)
                                    │
                       ┌────────────┴────────────┐
                       │                         │
              ACTIVITY HEAD                DOCKING HEAD
                       │                         │
           Dense(16, relu)              Dense(16, relu)
           Dropout(0.15)                Dropout(0.15)
           Dense(1, linear)             Dense(1, linear)
                       │                         │
            final_activity_output      final_docking_output
             (IC50 prediction)         (Docking score prediction)
```

### Bidirectional Reading — Biological Rationale

| Direction | Timestep Sequence | Biological Meaning |
|---|---|---|
| **Forward** | t=0 → t=5 | Overall protein context → binding pocket → catalytic outcome |
| **Backward** | t=5 → t=0 | Catalytic outcome ← activation loop ← regulatory region ← binding mechanics ← overall context |

LSTM and GRU are run in parallel because they capture complementary patterns: LSTM better preserves **long-range dependencies** across timesteps (via forget gate), while GRU is lighter and tends to capture **shorter sequential patterns** more efficiently. Concatenating both gives the dense integration layers access to both types of temporal information.

---

## Dimension Flow Summary

### Stage 1 — Hierarchical Model

```
8 feature inputs (variable dims)
        ↓
P1a final_emb:             (8)   ─┐
P1b mix_emb:               (4)   ─┴─ priority1_combined         (12)
P2  inter_emb:            (12)      → priority1_2_combined       (24)
P3  intra_emb:            (12)      → priority1_2_3_combined     (36) ─┐
P4  mut_inter_emb:         (8)   ─┐                                    │ + for gating P6/P7
P5  lig_inter_emb:         (8)   ─┴─ inter_combined              (16) ─┴─ priority1_to_5_combined (52)
P6  mut_intra_emb:         (8)   ─┐
P7  lig_intra_emb:         (8)   ─┴─ intra_combined              (16)

all_combined = priority1_2_3_combined(36) + inter_combined(16) + intra_combined(16) = (68)
        ↓
Dense(128) → Dense(64) → Dense(32) → Dense(16) = embedding_output  (16)
        ↓
Auxiliary: activity_output (1),  docking_output (1)
```

### Stage 2 — RNN Model

```
6 × embedding_output(16) stacked
        → sequential_embeddings  (batch, 6, 16)
        → BiLSTM Layer 1         (batch, 6, 256)  [return_sequences=True]
        → BiLSTM Layer 2         (batch, 128)     [return_sequences=False]
        ║
        → BiGRU Layer 1          (batch, 6, 256)  [return_sequences=True]
        → BiGRU Layer 2          (batch, 128)     [return_sequences=False]
        ↓
Concatenate                       (batch, 256)
        ↓
Dense(128) → Dense(64) → Dense(32)
        ↓
final_activity_output (1)    final_docking_output (1)
```

---

## Training Configuration

### Stage 1 — Hierarchical Model
- **Optimizer:** Adam (lr=0.003)
- **Loss:** MSE on both outputs
- **Loss weights:** Activity = 1.0, Docking = 0.6
- **Epochs:** 100 with EarlyStopping (patience=30)
- **Validation split:** 20%
- **Batch size:** 32
- **Saved artifact:** `hierarchical_model_{site_name}.h5`

### Stage 2 — RNN Sequential Model
- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE on both outputs
- **Loss weights:** Activity = 1.0, Docking = 0.7
- **Input:** `(batch, 6, 16)` stacked embeddings from Stage 1
- **Saved artifact:** `rnn_sequential_model.h5`

### Target Transforms
- **Activity (IC50):** `log1p(y)` → StandardScaler; inverse: `expm1(inverse_transform(y))`
- **Docking Score:** StandardScaler only; inverse: `inverse_transform(y)`

---

## Script Versions

| Version | Location | Notes |
|---|---|---|
| **Pseudocode (Set A)** | `physicochem_activity_main_pseudocode/adv_physchem5f2_hierachichal_ltsm_gru_custom.py` | Annotated with detailed logger messages and pseudocode rationale |
| **Optimised (Set B)** | `physicochem_activity_main_optimised/1_adv_physchem5f2_hierachichal_ltsm_gru_custom.py` | Production-ready with disk-backed `.feature_cache/`, `loguru` structured logging |
