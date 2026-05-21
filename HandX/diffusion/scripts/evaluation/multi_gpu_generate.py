#!/usr/bin/env python3
"""
Multi-GPU Parallel Generation Wrapper

This module provides multi-GPU acceleration for motion generation by:
1. Splitting the validation samples across available GPUs
2. Running independent generation processes on each GPU
3. Merging results into a single output directory

Features:
- Automatic GPU detection or manual GPU ID specification
- Pre-computation of mean/std to avoid race conditions
- Automatic result merging and cleanup of temporary directories
- Support for all generation modes (text2motion, inpainting, etc.)

Usage:
    # Auto-detect all GPUs
    python multi_gpu_generate.py --checkpoint_dir /path/to/checkpoint --model_name model.pt

    # Specify specific GPUs
    python multi_gpu_generate.py --checkpoint_dir /path/to/checkpoint --model_name model.pt \\
        --gpu_ids "0,1,2,3"

    # With masking
    python multi_gpu_generate.py --checkpoint_dir /path/to/checkpoint --model_name model.pt \\
        --mask_regions '[{"temporal": [0,1,2,3,4], "spatial": [0,21]}]'
"""

import os
import sys
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path


def run_generation_worker(gpu_id, script_path, args, sample_start, sample_end, output_suffix):
    """
    Worker function to run generation on a single GPU.

    Args:
        gpu_id: GPU device ID
        script_path: Path to evaluate_val_samples.py
        args: Namespace with generation arguments
        sample_start: Start index for this worker's samples
        sample_end: End index for this worker's samples
        output_suffix: Suffix to append to output folder name
    """
    # Set environment variable to use only this GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Fix networkx compatibility with torch.compile
    env['NETWORKX_AUTOMATIC_BACKENDS'] = 'false'
    env['NETWORKX_BACKEND_PRIORITY'] = ''

    # Build command
    cmd = [
        'python', script_path,
        '--checkpoint_dir', args.checkpoint_dir,
        '--model_name', args.model_name,
        '--num_val_samples', str(args.num_val_samples),
        '--num_generated', str(args.num_generated),
        '--batch_size', str(args.batch_size),
        '--eval_folder_name', f'{args.eval_folder_name}_{output_suffix}',
        '--seed', str(args.seed),
        '--sample_start', str(sample_start),
        '--sample_end', str(sample_end),
    ]

    # Add optional arguments
    if args.data_dir:
        cmd.extend(['--data_dir', args.data_dir])
    if args.data_loader:
        cmd.extend(['--data_loader', args.data_loader])
    if args.data_file_name:
        cmd.extend(['--data_file_name', args.data_file_name])
    if args.mask_regions:
        cmd.extend(['--mask_regions', args.mask_regions])
    if args.use_fp16:
        cmd.append('--use_fp16')

    print(f"\n{'='*80}")
    print(f"GPU {gpu_id}: Processing samples {sample_start}-{sample_end}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Run the generation process
        subprocess.run(cmd, env=env, check=True)
        print(f"\n✓ GPU {gpu_id} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ GPU {gpu_id} failed: {e}")
        return False


def merge_results(checkpoint_dir, eval_folder_name, checkpoint_name, num_gpus, final_output_dir):
    """
    Merge results from multiple GPU workers into a single directory.

    Args:
        checkpoint_dir: Base checkpoint directory
        eval_folder_name: Evaluation folder name (without gpu suffix)
        checkpoint_name: Name of checkpoint
        num_gpus: Number of GPUs used
        final_output_dir: Final merged output directory
    """
    print(f"\n{'='*80}")
    print("Merging results from all GPUs...")
    print(f"{'='*80}")

    final_output_dir = Path(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PKL files from worker directories
    # Worker directories are: {checkpoint_dir.parent}/{eval_folder_name}_gpu{i}/{checkpoint_name}/
    all_files = []
    checkpoint_parent = Path(checkpoint_dir).parent
    for gpu_id in range(num_gpus):
        worker_dir = checkpoint_parent / f'{eval_folder_name}_gpu{gpu_id}' / checkpoint_name
        if worker_dir.exists():
            # Only include val_sample_*.pkl files (exclude metadata.pkl)
            pkl_files = sorted(worker_dir.glob('val_sample_*.pkl'))
            all_files.extend(pkl_files)
            print(f"  GPU {gpu_id}: {len(pkl_files)} sample files from {worker_dir}")
        else:
            print(f"  GPU {gpu_id}: directory not found: {worker_dir}")

    print(f"\nTotal files to merge: {len(all_files)}")

    # Copy/move files to final directory with renumbering
    import shutil
    for i, src_file in enumerate(sorted(all_files)):
        # Rename to ensure sequential numbering (keep val_sample_ prefix for compatibility)
        dst_file = final_output_dir / f'val_sample_{i:03d}.pkl'
        shutil.copy2(src_file, dst_file)

    print(f"✓ Merged {len(all_files)} files to: {final_output_dir}")

    # Clean up worker directories after merging
    print("\nCleaning up temporary worker directories...")
    for gpu_id in range(num_gpus):
        worker_parent_dir = checkpoint_parent / f'{eval_folder_name}_gpu{gpu_id}'
        if worker_parent_dir.exists():
            shutil.rmtree(worker_parent_dir)
            print(f"  ✓ Removed: {worker_parent_dir}")

    return final_output_dir


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU parallel generation')

    # Generation parameters (pass-through to evaluate_val_samples.py)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--num_val_samples', type=int, default=32)
    parser.add_argument('--num_generated', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_folder_name', type=str, default='auto_eval_text2motion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--data_loader', type=str, default=None)
    parser.add_argument('--data_file_name', type=str, default=None)
    parser.add_argument('--mask_regions', type=str, default=None,
                       help='JSON string of mask regions with temporal and spatial components')

    # Multi-GPU specific parameters
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Comma-separated GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use FP16 (mixed precision) for faster inference (default: False)')

    args = parser.parse_args()

    # Determine GPUs to use
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        import torch
        num_gpus = args.num_gpus if args.num_gpus else torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))

    num_gpus = len(gpu_ids)
    print(f"\n{'='*80}")
    print(f"MULTI-GPU PARALLEL GENERATION")
    print(f"{'='*80}")
    print(f"GPUs: {gpu_ids}")
    print(f"Total samples: {args.num_val_samples}")
    print(f"Samples per GPU: ~{args.num_val_samples // num_gpus}")
    print(f"{'='*80}\n")

    # Calculate sample ranges for each GPU
    samples_per_gpu = args.num_val_samples // num_gpus
    sample_ranges = []
    for i, gpu_id in enumerate(gpu_ids):
        start = i * samples_per_gpu
        end = (i + 1) * samples_per_gpu if i < num_gpus - 1 else args.num_val_samples
        sample_ranges.append((gpu_id, start, end, f'gpu{i}'))

    # Path to evaluate_val_samples.py
    script_dir = Path(__file__).parent
    script_path = str(script_dir / 'evaluate_val_samples.py')

    # Determine paths
    checkpoint_name = Path(args.checkpoint_dir).name
    final_output_dir = Path(args.checkpoint_dir).parent / args.eval_folder_name / checkpoint_name

    # Check if mean/std files already exist (to avoid unnecessary computation)
    from omegaconf import OmegaConf
    config = OmegaConf.load(Path(args.checkpoint_dir) / "config.yaml")

    # Apply overrides to get correct data_dir
    data_dir = args.data_dir if args.data_dir else config.data.data_dir
    repr_type = config.data.repr

    mean_std_path = Path(data_dir) / f'mean_std_{repr_type}'
    mean_file = mean_std_path / "mean.npy"
    std_file = mean_std_path / "std.npy"

    # Check if files exist and are non-empty
    if (mean_file.exists() and std_file.exists() and
        mean_file.stat().st_size > 0 and std_file.stat().st_size > 0):
        print(f"\n{'='*80}")
        print("Mean/std files already exist - skipping pre-computation")
        print(f"{'='*80}")
        print(f"Found existing files at: {mean_std_path}")
        print(f"  mean.npy: {mean_file.stat().st_size} bytes")
        print(f"  std.npy: {std_file.stat().st_size} bytes")
        print("✓ Using existing mean/std files\n")
    else:
        # Pre-compute mean/std by loading train dataset once (to avoid multi-process conflicts)
        print(f"\n{'='*80}")
        print("Pre-computing mean/std from train dataset...")
        print(f"{'='*80}")
        print("Mean/std files not found - computing from training data...")

        # Run a single process to load the train dataset and compute mean/std
        precompute_cmd = [
            'python', '-c',
            f'''
import sys
sys.path.insert(0, "/your/project/path/diffusion")
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Load config
config_path = Path("{args.checkpoint_dir}") / "config.yaml"
config = OmegaConf.load(config_path)

# Override data_dir if specified
if "{args.data_dir}":
    config.data.data_dir = "{args.data_dir}"
if "{args.data_loader}":
    config.data._target_ = "{args.data_loader}"
if "{args.data_file_name}":
    config.data.data_file_name = "{args.data_file_name}"

# Ensure normalize=True to trigger mean/std computation
config.data.normalize = True

# Load train dataset (this will auto-compute mean/std if normalize=True)
print("Loading train dataset to compute mean/std...")
train_dataset = instantiate(config.data, split='train')
print(f"Train dataset loaded: {{len(train_dataset)}} samples")
print(f"Mean/std saved to: {{config.data.data_dir}}/mean_std_{{config.data.repr}}")
del train_dataset
'''
        ]

        # Set environment variables to fix networkx compatibility
        env = os.environ.copy()
        env['NETWORKX_AUTOMATIC_BACKENDS'] = 'false'
        env['NETWORKX_BACKEND_PRIORITY'] = ''

        try:
            subprocess.run(precompute_cmd, check=True, env=env)
            print("✓ Mean/std pre-computation completed\n")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Mean/std pre-computation failed: {e}")
            print("Workers will compute individually (may cause race conditions)\n")

    # Run workers in parallel
    with mp.Pool(num_gpus) as pool:
        results = pool.starmap(
            run_generation_worker,
            [(gpu_id, script_path, args, start, end, suffix)
             for gpu_id, start, end, suffix in sample_ranges]
        )

    # Check if all workers succeeded
    if not all(results):
        print("\n✗ Some GPU workers failed!")
        sys.exit(1)

    # Merge results
    merge_results(
        args.checkpoint_dir,
        args.eval_folder_name,
        checkpoint_name,
        num_gpus,
        final_output_dir
    )

    print(f"\n{'='*80}")
    print("✓ Multi-GPU generation completed successfully")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
