#!/usr/bin/env python
"""
Experiment orchestrator for ML-Driven EGFR TKI pipeline.

Organizes training and inference outputs into structured directories:
    experiments/<model_name>/<dataset_name>/

Usage examples:
    # Train model 0, then run inference
    python run_experiment.py --model 0 --predict_input data/test_data.csv

    # Train model 1 only (no inference)
    python run_experiment.py --model 1

    # Train with a different dataset
    python run_experiment.py --model 0 --train_data data/my_custom_train.csv

    # Train and predict models 0, 1, 3
    python run_experiment.py --model 0 1 3 --predict_input data/test_data.csv

    # Train all 6 models
    python run_experiment.py --model all
"""

import argparse
import os
import subprocess
import sys

MODELS = {
    0: {
        'name': 'dummy_physchem',
        'train': 'training_scripts/0_dummy_physchem_5f2.py',
        'infer': 'inference_scripts/0_predict_dummy_physchem_5f2.py',
        'default_train_csv': 'manual_egfr3_mini_dock_fixed.csv',
    },
    1: {
        'name': 'adv_physchem5f2',
        'train': 'training_scripts/1_adv_physchem5f2.py',
        'infer': 'inference_scripts/1_predict_adv_physchem5f2.py',
        'default_train_csv': 'df_3_shuffled.csv',
    },
    2: {
        'name': 'kan_bspline',
        'train': 'training_scripts/2_adv_physchem_KAN3_b_spline1a.py',
        'infer': None,
        'default_train_csv': 'df_3_shuffled.csv',
    },
    3: {
        'name': 'kan_navier_stokes',
        'train': 'training_scripts/3_adv_physchem_KAN_navier_stokes_sinusoid.py',
        'infer': 'inference_scripts/3_predict_adv_physchem_KAN_navier_stokes.py',
        'default_train_csv': 'df_3_shuffled.csv',
    },
    4: {
        'name': 'chembert_crossattention',
        'train': 'training_scripts/4_adv_physchem_chemerta_crossattention.py',
        'infer': None,
        'default_train_csv': 'df_3_shuffled.csv',
    },
    5: {
        'name': 'gnn',
        'train': 'training_scripts/5_adv_physchem_gnn.py',
        'infer': None,
        'default_train_csv': 'df_3_shuffled.csv',
    },
}


def resolve_path(path, base_dir):
    """Resolve a path relative to base_dir if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def get_dataset_name(csv_path):
    """Extract dataset name from CSV filename (stem without extension)."""
    return os.path.splitext(os.path.basename(csv_path))[0]


def run_command(cmd, description):
    """Run a subprocess command, streaming output."""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)

    if result.returncode != 0:
        print(f"\nWARNING: {description} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Experiment orchestrator for ML-Driven EGFR TKI pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--model', nargs='+', required=True,
        help='Model IDs to run (0-5) or "all"',
    )
    parser.add_argument(
        '--predict_input', type=str, default=None,
        help='Input CSV for inference (requires --model with inference support)',
    )
    parser.add_argument(
        '--train_data', type=str, default=None,
        help='Training CSV path (overrides model default)',
    )
    parser.add_argument(
        '--control_data', type=str, default=None,
        help='Control compounds CSV path',
    )
    parser.add_argument(
        '--drug_data', type=str, default=None,
        help='Drug compounds CSV path',
    )
    parser.add_argument(
        '--experiments_dir', type=str, default='experiments',
        help='Base directory for experiment outputs (default: experiments/)',
    )

    args = parser.parse_args()

    # Resolve base directory (where this script lives)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse model IDs
    if args.model == ['all']:
        model_ids = sorted(MODELS.keys())
    else:
        try:
            model_ids = [int(m) for m in args.model]
        except ValueError:
            parser.error("--model must be integers (0-5) or 'all'")

        for mid in model_ids:
            if mid not in MODELS:
                parser.error(f"Unknown model ID: {mid}. Valid: {sorted(MODELS.keys())}")

    experiments_dir = resolve_path(args.experiments_dir, base_dir)
    created_dirs = []

    print("="*80)
    print("EXPERIMENT ORCHESTRATOR")
    print("="*80)
    print(f"Models to run: {model_ids}")
    print(f"Experiments dir: {experiments_dir}")
    if args.train_data:
        print(f"Custom train data: {args.train_data}")
    if args.predict_input:
        print(f"Inference input: {args.predict_input}")

    for mid in model_ids:
        model_info = MODELS[mid]
        model_name = model_info['name']

        # Resolve training CSV
        if args.train_data:
            train_csv = resolve_path(args.train_data, base_dir)
        else:
            train_csv = os.path.join(base_dir, 'data', model_info['default_train_csv'])

        dataset_name = get_dataset_name(train_csv)

        # Create output directory
        run_dir = os.path.join(experiments_dir, f'model_{mid}_{model_name}', dataset_name)
        if os.path.exists(run_dir):
            print(f"\nWARNING: Directory already exists, will overwrite: {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        created_dirs.append(run_dir)

        print(f"\n{'#'*80}")
        print(f"# Model {mid}: {model_name}")
        print(f"# Dataset: {dataset_name}")
        print(f"# Output: {run_dir}")
        print(f"{'#'*80}")

        # Build training command
        train_script = resolve_path(model_info['train'], base_dir)
        train_cmd = [
            sys.executable, train_script,
            '--output_dir', run_dir,
            '--train_data', train_csv,
        ]
        if args.control_data:
            train_cmd.extend(['--control_data', resolve_path(args.control_data, base_dir)])
        if args.drug_data:
            train_cmd.extend(['--drug_data', resolve_path(args.drug_data, base_dir)])

        # Run training
        ret = run_command(train_cmd, f"Training model {mid} ({model_name})")
        if ret != 0:
            print(f"Training failed for model {mid}. Skipping inference.")
            continue

        # Run inference if requested and available
        if args.predict_input and model_info['infer']:
            infer_script = resolve_path(model_info['infer'], base_dir)
            predict_input = resolve_path(args.predict_input, base_dir)

            infer_cmd = [
                sys.executable, infer_script,
                '--input', predict_input,
                '--model_dir', run_dir,
            ]
            # output_dir defaults to {model_dir}/predictions/ in the inference scripts

            run_command(infer_cmd, f"Inference model {mid} ({model_name})")
        elif args.predict_input and not model_info['infer']:
            print(f"\nNote: Model {mid} ({model_name}) has no inference script. Skipping inference.")

    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    for d in created_dirs:
        file_count = sum(len(files) for _, _, files in os.walk(d))
        print(f"  {d}/ ({file_count} files)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
