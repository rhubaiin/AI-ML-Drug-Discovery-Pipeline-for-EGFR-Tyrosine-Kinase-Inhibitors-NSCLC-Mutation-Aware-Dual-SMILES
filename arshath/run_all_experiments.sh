#!/usr/bin/env bash
# run_all_experiments.sh
# Replicate all EGFR TKI pipeline experiments from scratch:
#   - 12 training runs (6 models x 2 datasets)
#   - 12 cross-CSV inference runs (6 models x 2 directions)
#
# Usage: bash run_all_experiments.sh

set -euo pipefail

# Force non-interactive matplotlib backend to prevent plt.show() from hanging
export MPLBACKEND=Agg

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE=experiments
DATA_A=data/manual_egfr3_mini_dock_fixed.csv
DATA_B=data/df_3_shuffled.csv

echo "======================================"
echo "  EGFR TKI Pipeline - Full Replication"
echo "======================================"
echo ""

# ─────────────────────────────────────────
# Step 1: Back up existing experiments
# ─────────────────────────────────────────
echo "[Step 1/4] Backing up experiments/ → experiments_backup/"
if [ -d "$BASE" ]; then
    if [ -d "experiments_backup" ]; then
        echo "  WARNING: experiments_backup/ already exists. Removing it first."
        rm -rf experiments_backup
    fi
    mv "$BASE" experiments_backup
    echo "  Done."
else
    echo "  No existing experiments/ directory to back up."
fi
mkdir -p "$BASE"
echo ""

# Clear feature cache for a fully fresh start
echo "[Step 1b/4] Clearing feature cache..."
if [ -d "data/.feature_cache" ]; then
    rm -rf data/.feature_cache
    echo "  Removed data/.feature_cache"
else
    echo "  No feature cache found."
fi
echo ""

# ─────────────────────────────────────────
# Step 2: Train all 6 models on default datasets
# ─────────────────────────────────────────
echo "[Step 2/4] Training all 6 models on default datasets..."
echo "  Model 0 → manual_egfr3_mini_dock_fixed.csv"
echo "  Models 1-5 → df_3_shuffled.csv"
echo ""
python run_experiment.py --model all
echo ""
echo "  Default training complete."
echo ""

# ─────────────────────────────────────────
# Step 3: Train all 6 models on swapped datasets
# ─────────────────────────────────────────
echo "[Step 3/4] Training all 6 models on swapped datasets..."

echo "  Model 0 → df_3_shuffled.csv"
python run_experiment.py --model 0 --train_data "$DATA_B"
echo ""

echo "  Models 1-5 → manual_egfr3_mini_dock_fixed.csv"
python run_experiment.py --model 1 2 3 4 5 --train_data "$DATA_A"
echo ""
echo "  Swapped training complete."
echo ""

# ─────────────────────────────────────────
# Step 4: Cross-CSV inference (12 runs)
# ─────────────────────────────────────────
echo "[Step 4/4] Running cross-CSV inference..."

# Direction 1: Models trained on df_3_shuffled → predict on manual_egfr3_mini_dock_fixed
echo ""
echo "  Direction 1: trained on df_3_shuffled → predict on manual_egfr3_mini_dock_fixed"

echo "    Model 0..."
python inference_scripts/0_predict_dummy_physchem_5f2.py \
    --input "$DATA_A" \
    --model_dir "$BASE/model_0_dummy_physchem/df_3_shuffled" \
    --output_dir "$BASE/model_0_dummy_physchem/df_3_shuffled/predictions_cross_csv"

echo "    Model 1..."
python inference_scripts/1_predict_adv_physchem5f2.py \
    --input "$DATA_A" \
    --model_dir "$BASE/model_1_adv_physchem5f2/df_3_shuffled" \
    --output_dir "$BASE/model_1_adv_physchem5f2/df_3_shuffled/predictions_cross_csv"

echo "    Model 2..."
python inference_scripts/2_predict_adv_physchem_KAN_bspline.py \
    --input "$DATA_A" \
    --model_dir "$BASE/model_2_kan_bspline/df_3_shuffled" \
    --output_dir "$BASE/model_2_kan_bspline/df_3_shuffled/predictions_cross_csv"

echo "    Model 3..."
python inference_scripts/3_predict_adv_physchem_KAN_navier_stokes.py \
    --input "$DATA_A" \
    --model_dir "$BASE/model_3_kan_navier_stokes/df_3_shuffled" \
    --output_dir "$BASE/model_3_kan_navier_stokes/df_3_shuffled/predictions_cross_csv"

echo "    Model 4..."
python inference_scripts/4_predict_adv_physchem_chembert_crossattention.py \
    --input "$DATA_A" \
    --model_dir "$BASE/model_4_chembert_crossattention/df_3_shuffled" \
    --output_dir "$BASE/model_4_chembert_crossattention/df_3_shuffled/predictions_cross_csv"

echo "    Model 5..."
python inference_scripts/5_predict_adv_physchem_gnn.py \
    --input "$DATA_A" \
    --model_dir "$BASE/model_5_gnn/df_3_shuffled" \
    --output_dir "$BASE/model_5_gnn/df_3_shuffled/predictions_cross_csv"

# Direction 2: Models trained on manual_egfr3_mini_dock_fixed → predict on df_3_shuffled
echo ""
echo "  Direction 2: trained on manual_egfr3_mini_dock_fixed → predict on df_3_shuffled"

echo "    Model 0..."
python inference_scripts/0_predict_dummy_physchem_5f2.py \
    --input "$DATA_B" \
    --model_dir "$BASE/model_0_dummy_physchem/manual_egfr3_mini_dock_fixed" \
    --output_dir "$BASE/model_0_dummy_physchem/manual_egfr3_mini_dock_fixed/predictions_cross_csv"

echo "    Model 1..."
python inference_scripts/1_predict_adv_physchem5f2.py \
    --input "$DATA_B" \
    --model_dir "$BASE/model_1_adv_physchem5f2/manual_egfr3_mini_dock_fixed" \
    --output_dir "$BASE/model_1_adv_physchem5f2/manual_egfr3_mini_dock_fixed/predictions_cross_csv"

echo "    Model 2..."
python inference_scripts/2_predict_adv_physchem_KAN_bspline.py \
    --input "$DATA_B" \
    --model_dir "$BASE/model_2_kan_bspline/manual_egfr3_mini_dock_fixed" \
    --output_dir "$BASE/model_2_kan_bspline/manual_egfr3_mini_dock_fixed/predictions_cross_csv"

echo "    Model 3..."
python inference_scripts/3_predict_adv_physchem_KAN_navier_stokes.py \
    --input "$DATA_B" \
    --model_dir "$BASE/model_3_kan_navier_stokes/manual_egfr3_mini_dock_fixed" \
    --output_dir "$BASE/model_3_kan_navier_stokes/manual_egfr3_mini_dock_fixed/predictions_cross_csv"

echo "    Model 4..."
python inference_scripts/4_predict_adv_physchem_chembert_crossattention.py \
    --input "$DATA_B" \
    --model_dir "$BASE/model_4_chembert_crossattention/manual_egfr3_mini_dock_fixed" \
    --output_dir "$BASE/model_4_chembert_crossattention/manual_egfr3_mini_dock_fixed/predictions_cross_csv"

echo "    Model 5..."
python inference_scripts/5_predict_adv_physchem_gnn.py \
    --input "$DATA_B" \
    --model_dir "$BASE/model_5_gnn/manual_egfr3_mini_dock_fixed" \
    --output_dir "$BASE/model_5_gnn/manual_egfr3_mini_dock_fixed/predictions_cross_csv"

echo ""
echo "======================================"
echo "  All experiments complete!"
echo "======================================"
echo ""

# ─────────────────────────────────────────
# Verification
# ─────────────────────────────────────────
echo "Verification:"
echo ""

MODELS=("model_0_dummy_physchem" "model_1_adv_physchem5f2" "model_2_kan_bspline" "model_3_kan_navier_stokes" "model_4_chembert_crossattention" "model_5_gnn")
DATASETS=("df_3_shuffled" "manual_egfr3_mini_dock_fixed")

TRAIN_OK=0
INFER_OK=0
TRAIN_FAIL=0
INFER_FAIL=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        dir="$BASE/$model/$dataset"
        if [ -d "$dir" ]; then
            TRAIN_OK=$((TRAIN_OK + 1))
        else
            echo "  MISSING: $dir"
            TRAIN_FAIL=$((TRAIN_FAIL + 1))
        fi

        cross_dir="$dir/predictions_cross_csv"
        if [ -d "$cross_dir" ] && [ "$(ls -A "$cross_dir" 2>/dev/null)" ]; then
            INFER_OK=$((INFER_OK + 1))
        else
            echo "  MISSING: $cross_dir"
            INFER_FAIL=$((INFER_FAIL + 1))
        fi
    done
done

echo ""
echo "  Training dirs:  $TRAIN_OK/12 OK ($TRAIN_FAIL missing)"
echo "  Inference dirs: $INFER_OK/12 OK ($INFER_FAIL missing)"
echo ""

if [ $TRAIN_FAIL -eq 0 ] && [ $INFER_FAIL -eq 0 ]; then
    echo "  All 24 experiment runs verified successfully!"
else
    echo "  WARNING: Some experiment runs are missing. Check output above."
    exit 1
fi
