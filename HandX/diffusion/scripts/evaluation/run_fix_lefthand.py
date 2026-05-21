#!/usr/bin/env python3
"""
Fix Left Hand - Generate Right Hand Evaluation Script

This script evaluates spatial masking where the left hand motion is fixed
from ground truth, and the model generates only the right hand motion.

Masking Strategy:
- Spatial mask: Left hand joints (indices 0-20) are fixed
- All 60 frames are masked for the left hand
- Right hand (indices 21-41) is freely generated

This task tests the model's ability to generate coherent bimanual motion
when one hand's motion is given as a constraint.

Usage:
    python run_fix_lefthand.py
"""

import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define all checkpoints to evaluate
CHECKPOINTS = [
    {
        'name': 'your_checkpoint_name',
        'checkpoint_dir': '/path/to/your/checkpoint',
        'model_name': 'model000220000.pt',
        'num_val_samples': 256,
        'data_dir': str(PROJECT_ROOT.parent / 'data/handx'),
        'data_loader': 'src.diffusion.data_loader.handx.HandXDataset',  # Use HandX data loader
        'data_file_name': 'can_pos_all_wotextfeat.npz',  # Data file name for HandX
        'fixed_frames': None,  # [0,1,2,3,4] for temporal-only masking
        'mask_regions': [
            {
                'temporal': list(range(60)),
                'spatial': list(range(21)),     # left hand
                'use_soft_mask': True,          # Enable soft mask (RePaint style)
                'temporal_transition_width': 5,  # Transition width
                'core_mask_value': 0.85,        # Core region mask value
                'edge_mask_value': 0.1,         # Edge/generation region mask value
            }
        ],
        'eval_folder_name': 'generate_fix_left',  # Customize output folder name
        'num_generated': 4,
        # 'mean_std_path': None,  # Optional: Override auto-detection from data_dir/mean_std_{repr}
        'description': 'Your model description here'
    }
]


def run_evaluation(config):
    """Run evaluation for a single checkpoint."""
    print("\n" + "="*80)
    print(f"Evaluating: {config['name']}")
    print(f"Description: {config['description']}")
    print("="*80)

    # Use 'python' from PATH to respect activated conda environment
    cmd = [
        "python",
        str(Path(__file__).parent / "evaluate_val_samples.py"),
        "--checkpoint_dir", config['checkpoint_dir'],
        "--model_name", config['model_name'],
        "--num_val_samples", str(config['num_val_samples']),
    ]

    # Handle old 'fixed_frames' parameter (backward compatibility)
    if config.get('fixed_frames') is not None:
        fixed_frames_str = ','.join(map(str, config['fixed_frames']))
        cmd.extend(["--fixed_frames", fixed_frames_str])

    # Handle new 'mask_regions' parameter (temporal + spatial masking)
    if config.get('mask_regions') is not None:
        mask_regions_str = json.dumps(config['mask_regions'])
        cmd.extend(["--mask_regions", mask_regions_str])

    if config.get('eval_folder_name') is not None:
        cmd.extend(["--eval_folder_name", config['eval_folder_name']])

    if config.get('num_generated') is not None:
        cmd.extend(["--num_generated", str(config['num_generated'])])

    if config.get('mean_std_path') is not None:
        cmd.extend(["--mean_std_path", config['mean_std_path']])

    if config.get('data_dir') is not None:
        cmd.extend(["--data_dir", config['data_dir']])

    if config.get('data_loader') is not None:
        cmd.extend(["--data_loader", config['data_loader']])

    if config.get('data_file_name') is not None:
        cmd.extend(["--data_file_name", config['data_file_name']])

    print(f"Running command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Successfully evaluated: {config['name']}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to evaluate: {config['name']}")
        print(f"Error: {e}")
        return False

    return True


def main():
    print("="*80)
    print(f"Batch Evaluation - {len(CHECKPOINTS)} checkpoints")
    print("="*80)

    results = []
    for i, config in enumerate(CHECKPOINTS, 1):
        print(f"\n[{i}/{len(CHECKPOINTS)}] Processing: {config['name']}")
        success = run_evaluation(config)
        results.append((config['name'], success))

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")

    total_success = sum(1 for _, s in results if s)
    print(f"\nTotal: {total_success}/{len(results)} succeeded")
    print("="*80)


if __name__ == "__main__":
    main()
