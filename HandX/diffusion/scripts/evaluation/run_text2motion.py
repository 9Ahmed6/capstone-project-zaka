#!/usr/bin/env python3
"""
Text-to-Motion Generation Evaluation Script

This script evaluates text-to-motion generation without any masking constraints.
The model freely generates motions based solely on text descriptions.

Features:
- Support for both single-GPU and multi-GPU parallel generation
- Command-line parameter overrides for flexible configuration
- Automatic checkpoint configuration or custom parameters

Masking: None (free generation from text only)

Usage:
    # Use default CHECKPOINTS configuration
    python run_text2motion.py

    # Specify checkpoint and model
    python run_text2motion.py --checkpoint_dir /path/to/checkpoint --model_name model.pt

    # Custom parameters
    python run_text2motion.py --checkpoint_dir /path/to/checkpoint --model_name model.pt \\
        --num_val_samples 1024 --batch_size 128 --gpu_ids "0,1,2,3"
"""

import subprocess
import sys
import json
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define all checkpoints to evaluate
CHECKPOINTS = [
    {
        'name': 'Data100_Layers12',
        'checkpoint_dir': '/your/checkpoint/dir/',
        'model_name': 'model000115000.pt',
        'num_val_samples': 2048,
        'batch_size': 256,  # Batch size for generation (optional, default: 256)
        'data_dir': str(PROJECT_ROOT.parent / 'data/handx'),
        'data_loader': 'src.diffusion.data_loader.handx.HandXDataset',  # Use HandX data loader
        'data_file_name': 'can_pos_all_wotextfeat.npz',  # Data file name for HandX
        'fixed_frames': None,  # No fixed frames for text2motion generation
        'mask_regions': None,  # No masking regions for text2motion generation
        'eval_folder_name': 'generate_text2motion',  # Customize output folder name
        'num_generated': 4,
        'enable_multi_gpu': True,  # Enable multi-GPU parallel generation (True = auto-detect all GPUs)
        'num_gpus': None,  # Number of GPUs to use (None = auto-detect all, or specify 2, 4, etc.)
        'gpu_ids': None,   # Specific GPU IDs (e.g., "0,1,2,3"), overrides num_gpus if set
        'use_fp16': False,  # Use FP16 mixed precision for faster inference (default: False)
        'description': 'Your model description here'
    },
]


def run_evaluation(config):
    """Run evaluation for a single checkpoint (supports both single-GPU and multi-GPU)."""
    print("\n" + "="*80)
    print(f"Evaluating: {config['name']}")
    print(f"Description: {config['description']}")
    print("="*80)

    # Determine if multi-GPU mode should be used
    enable_multi_gpu = config.get('enable_multi_gpu', False)
    use_multi_gpu = enable_multi_gpu or config.get('num_gpus') is not None or config.get('gpu_ids') is not None

    if use_multi_gpu:
        print("Mode: Multi-GPU parallel generation")
        script_name = "multi_gpu_generate.py"

        # Auto-detect GPUs if enabled but not specified
        if enable_multi_gpu and config.get('num_gpus') is None and config.get('gpu_ids') is None:
            import torch
            num_gpus_available = torch.cuda.device_count()
            print(f"Auto-detected {num_gpus_available} GPUs")
    else:
        print("Mode: Single-GPU generation")
        script_name = "evaluate_val_samples.py"

    # Build base command
    cmd = [
        "python",
        str(Path(__file__).parent / script_name),
        "--checkpoint_dir", config['checkpoint_dir'],
        "--model_name", config['model_name'],
        "--num_val_samples", str(config['num_val_samples']),
    ]

    # Add common parameters
    if config.get('num_generated') is not None:
        cmd.extend(["--num_generated", str(config['num_generated'])])

    if config.get('batch_size') is not None:
        cmd.extend(["--batch_size", str(config['batch_size'])])

    if config.get('eval_folder_name') is not None:
        cmd.extend(["--eval_folder_name", config['eval_folder_name']])

    if config.get('data_dir') is not None:
        cmd.extend(["--data_dir", config['data_dir']])

    if config.get('data_loader') is not None:
        cmd.extend(["--data_loader", config['data_loader']])

    if config.get('data_file_name') is not None:
        cmd.extend(["--data_file_name", config['data_file_name']])

    # Handle mask_regions parameter (supported by both single-GPU and multi-GPU modes)
    if config.get('mask_regions'):  # Check for truthy value, not just 'is not None'
        mask_regions_str = json.dumps(config['mask_regions'])
        cmd.extend(["--mask_regions", mask_regions_str])

    # Parameters only for single-GPU mode (evaluate_val_samples.py)
    if not use_multi_gpu:
        # Handle old 'fixed_frames' parameter (backward compatibility)
        if config.get('fixed_frames') is not None:
            fixed_frames_str = ','.join(map(str, config['fixed_frames']))
            cmd.extend(["--fixed_frames", fixed_frames_str])

        if config.get('mean_std_path') is not None:
            cmd.extend(["--mean_std_path", config['mean_std_path']])

    # Multi-GPU specific parameters
    # Note: If num_gpus and gpu_ids are both None, multi_gpu_generate.py will auto-detect all GPUs
    if use_multi_gpu:
        if config.get('num_gpus') is not None:
            cmd.extend(["--num_gpus", str(config['num_gpus'])])
        if config.get('gpu_ids') is not None:
            cmd.extend(["--gpu_ids", config['gpu_ids']])

    # FP16 flag (supported by both modes)
    if config.get('use_fp16'):
        cmd.append("--use_fp16")

    print(f"Running command: {' '.join(cmd)}\n")

    # Set environment variables for torch.compile compatibility
    env = os.environ.copy()
    env['NETWORKX_AUTOMATIC_BACKENDS'] = 'false'
    env['NETWORKX_BACKEND_PRIORITY'] = ''

    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"\n✓ Successfully evaluated: {config['name']}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to evaluate: {config['name']}")
        print(f"Error: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run text-to-motion evaluation on checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default CHECKPOINTS configuration:
  python run_text2motion.py

  # Specify checkpoint_dir and model_name:
  python run_text2motion.py --checkpoint_dir /path/to/checkpoint --model_name model000100000.pt

  # Override multiple parameters:
  python run_text2motion.py --checkpoint_dir /path/to/checkpoint --model_name model.pt --num_val_samples 1024 --batch_size 128
        '''
    )

    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to checkpoint directory (overrides CHECKPOINTS config)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model checkpoint filename (overrides CHECKPOINTS config)')
    parser.add_argument('--num_val_samples', type=int, default=None,
                       help='Number of validation samples to generate (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for generation (default: from config)')
    parser.add_argument('--num_generated', type=int, default=None,
                       help='Number of generations per sample (default: from config)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: from config)')
    parser.add_argument('--eval_folder_name', type=str, default=None,
                       help='Output folder name (default: from config)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Specific GPU IDs to use, e.g., "0,1,2,3" (default: auto-detect all)')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect all)')
    parser.add_argument('--disable_multi_gpu', action='store_true',
                       help='Disable multi-GPU mode (use single GPU)')

    args = parser.parse_args()

    # Determine which checkpoints to process
    checkpoints = CHECKPOINTS.copy()

    # If checkpoint_dir or model_name is specified, override the first checkpoint
    if args.checkpoint_dir or args.model_name:
        if not checkpoints:
            # Create a minimal checkpoint config if CHECKPOINTS is empty
            checkpoints = [{
                'name': 'custom_checkpoint',
                'checkpoint_dir': args.checkpoint_dir or '.',
                'model_name': args.model_name or 'model.pt',
                'num_val_samples': 2048,
                'batch_size': 256,
                'num_generated': 4,
                'enable_multi_gpu': True,
                'description': 'Custom checkpoint from command line'
            }]
        else:
            # Override the first checkpoint in CHECKPOINTS
            checkpoints = [checkpoints[0].copy()]

            if args.checkpoint_dir:
                checkpoints[0]['checkpoint_dir'] = args.checkpoint_dir
                checkpoints[0]['name'] = f"custom_{Path(args.checkpoint_dir).name}"

            if args.model_name:
                checkpoints[0]['model_name'] = args.model_name

    # Apply other command-line overrides to all checkpoints
    for config in checkpoints:
        if args.num_val_samples is not None:
            config['num_val_samples'] = args.num_val_samples
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size
        if args.num_generated is not None:
            config['num_generated'] = args.num_generated
        if args.data_dir is not None:
            config['data_dir'] = args.data_dir
        if args.eval_folder_name is not None:
            config['eval_folder_name'] = args.eval_folder_name
        if args.gpu_ids is not None:
            config['gpu_ids'] = args.gpu_ids
        if args.num_gpus is not None:
            config['num_gpus'] = args.num_gpus
        if args.disable_multi_gpu:
            config['enable_multi_gpu'] = False
            config['num_gpus'] = None
            config['gpu_ids'] = None

    print("="*80)
    print(f"Batch Evaluation - {len(checkpoints)} checkpoints")
    print("="*80)

    results = []
    for i, config in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Processing: {config['name']}")
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
