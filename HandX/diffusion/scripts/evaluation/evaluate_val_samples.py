#!/usr/bin/env python3
"""
Unified Evaluation Script for Bimanual Motion Generation Models

This is the core evaluation script that generates motion samples from trained checkpoints.
It supports various generation modes including:
- Normal text-to-motion generation (no masking)
- Temporal masking (fix specific frames from ground truth)
- Spatiotemporal masking (fix specific joints at specific frames)
- Soft masking with RePaint-style blending

Features:
- Batched generation for efficient GPU utilization
- Automatic mean/std computation and caching
- Support for multiple data representations (joint_pos, joint_pos_w_scalar_rot, joint_rot)
- Multi-GPU support via sample range splitting
- Optional FP16 inference for faster generation

Usage:
    # Normal generation (text-to-motion)
    python evaluate_val_samples.py --checkpoint_dir /path/to/checkpoint --model_name model.pt

    # Temporal masking (fix first 5 frames)
    python evaluate_val_samples.py --checkpoint_dir /path/to/checkpoint --model_name model.pt \\
        --fixed_frames "0,1,2,3,4"

    # Spatiotemporal masking with soft mask
    python evaluate_val_samples.py --checkpoint_dir /path/to/checkpoint --model_name model.pt \\
        --mask_regions '[{"temporal": [0,1,2,3,4], "spatial": [0,21], "use_soft_mask": true}]'

Output:
    PKL files containing ground truth and generated motions for each validation sample
"""

import os
import sys
import random
import pickle
import shutil
import argparse
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.model.cls_free_sampler import ClassifierFreeSampleWrapper
from src.diffusion.utils.mics import get_device, fixseed
from src.diffusion.utils.model_utils import create_model_and_diffusion
from hydra.utils import instantiate


def load_checkpoint_config(checkpoint_dir: str):
    """
    Load configuration from checkpoint directory.

    Args:
        checkpoint_dir: Path to the checkpoint directory containing config.yaml

    Returns:
        OmegaConf configuration object
    """
    config_path = pjoin(checkpoint_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    return config


def load_model_and_diffusion(checkpoint_path: str, config):
    """
    Load model and diffusion from checkpoint file.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        config: Model configuration object

    Returns:
        Tuple of (model, diffusion) objects ready for inference
    """
    model, diffusion = create_model_and_diffusion(config.model)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    return model, diffusion


def sample_val_data(dataset, num_samples=100, seed=42, sample_start=None, sample_end=None):
    """
    Sample random items from validation dataset.

    Randomly selects validation samples for evaluation. Supports sample range
    specification for multi-GPU parallel processing.

    Args:
        dataset: Validation dataset object
        num_samples: Total number of samples to select
        seed: Random seed for reproducible sampling
        sample_start: Start index for sample range (for multi-GPU processing)
        sample_end: End index for sample range (for multi-GPU processing)

    Returns:
        List of dicts with keys: 'text', 'motion', 'length', 'index'
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:num_samples]
    selected_indices.sort()
    print(f"Selected validation sample indices: {selected_indices}")

    # Apply sample range if specified (for multi-GPU processing)
    if sample_start is not None and sample_end is not None:
        selected_indices = selected_indices[sample_start:sample_end]
        print(f"Processing samples {sample_start} to {sample_end} (total: {len(selected_indices)})")

    samples = []
    for i, idx in enumerate(selected_indices):
        motion, length, text = dataset[idx]

        if i == 0:
            print("\n" + "="*80)
            print("DEBUG: First sampled item")
            print("="*80)
            print(f"Dataset index: {idx}")
            print(f"Dataset name: {dataset.name_list[idx]}")
            print(f"Motion shape: {motion.shape}")
            print(f"Length: {length}")
            print(f"Text type: {type(text)}")
            if isinstance(text, dict):
                print(f"Text keys: {list(text.keys())}")
                for key in ['left', 'right', 'two_hands_relation']:
                    if key in text:
                        print(f"  {key}: {text[key][:60]}...")
            else:
                print(f"Text value: {str(text)[:100]}...")
            print("="*80 + "\n")

        # Convert motion back to (T, J*F) format
        motion = rearrange(motion, 'j f t -> t (j f)')

        samples.append({
            'index': idx,
            'text': text,
            'motion': motion,
            'length': length
        })

    return samples


def generate_samples_batch(
    model,
    diffusion,
    dataset,
    val_samples_batch,
    num_samples_per_text: int = 4,
    guidance_scale: float = 2.5,
    njoints: int = 42,
    nfeats: int = 3,
    fixed_frames: list = None,
    mask_regions: list = None,
    use_fp16: bool = False,
):
    """
    Generate multiple samples for a batch of text prompts (batched version).

    Args:
        val_samples_batch: List of dicts with keys 'text', 'motion', 'length', 'index'
        num_samples_per_text: Number of samples to generate per text prompt
        fixed_frames: List of frame indices to fix from GT, e.g., [0,1,2,3,4]. None for normal generation. (deprecated)
        mask_regions: List of dicts with 'temporal' and 'spatial' keys for spatiotemporal masking
        use_fp16: Whether to use FP16 mixed precision for inference

    Returns:
        List of tuples (samples_normalized, samples_real_world) for each val_sample
    """
    device = get_device()

    batch_size = len(val_samples_batch)
    motion_lengths = [vs['length'] for vs in val_samples_batch]
    max_length = max(motion_lengths)

    total_samples = batch_size * num_samples_per_text
    shape = (total_samples, njoints, nfeats, max_length)

    text_prompts = [vs['text'] for vs in val_samples_batch]

    # Handle both string and dict text formats
    if isinstance(text_prompts[0], dict):
        text_dict = {'left': [], 'right': [], 'two_hands_relation': []}
        for text_prompt in text_prompts:
            text_dict['left'].extend([text_prompt['left']] * num_samples_per_text)
            text_dict['right'].extend([text_prompt['right']] * num_samples_per_text)
            text_dict['two_hands_relation'].extend([text_prompt['two_hands_relation']] * num_samples_per_text)
    else:
        text_dict = []
        for text_prompt in text_prompts:
            text_dict.extend([text_prompt] * num_samples_per_text)

    lengths_expanded = []
    for length in motion_lengths:
        lengths_expanded.extend([max_length] * num_samples_per_text)

    model_kwargs = dict(
        y=dict(
            lengths=torch.as_tensor(lengths_expanded, device=device),
            text=text_dict
        )
    )

    # Prepare GT motion for inpainting if fixed_frames or mask_regions is specified
    gt_motion_batch = None
    if (fixed_frames is not None and len(fixed_frames) > 0) or mask_regions is not None:
        gt_motions = []
        for vs in val_samples_batch:
            gt_motion = vs['motion']  # (T, J*F)
            gt_motion = rearrange(
                torch.tensor(gt_motion, dtype=torch.float32),
                't (j f) -> j f t',
                j=njoints,
                f=nfeats
            )

            if gt_motion.shape[2] < max_length:
                padding = torch.zeros(njoints, nfeats, max_length - gt_motion.shape[2])
                gt_motion = torch.cat([gt_motion, padding], dim=2)

            gt_motion = gt_motion.unsqueeze(0).repeat(num_samples_per_text, 1, 1, 1)
            gt_motions.append(gt_motion)

        gt_motion_batch = torch.cat(gt_motions, dim=0).to(device)

    # Generate samples with optional FP16
    if use_fp16:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            samples = diffusion.p_sample_loop(
                model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=True,
                gt_motion=gt_motion_batch,
                fixed_frames=fixed_frames,
                mask_regions=mask_regions,
            )
    else:
        samples = diffusion.p_sample_loop(
            model,
            shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
            progress=True,
            gt_motion=gt_motion_batch,
            fixed_frames=fixed_frames,
            mask_regions=mask_regions,
        )

    samples = rearrange(samples, 'b j f t -> b t (j f)').detach().cpu().numpy()

    results = []
    for i, vs in enumerate(val_samples_batch):
        original_length = vs['length']
        start_idx = i * num_samples_per_text
        end_idx = start_idx + num_samples_per_text

        samples_for_text = samples[start_idx:end_idx, :original_length, :]
        samples_real = dataset.inv_transform(samples_for_text)

        results.append((samples_for_text, samples_real))

    return results


def process_gt_motion(gt_motion_raw: np.ndarray, dataset, njoints: int, nfeats: int):
    """Process GT motion to match generated format."""
    gt_normalized = gt_motion_raw
    gt_real_world = dataset.inv_transform(gt_normalized[np.newaxis, ...])[0]
    return gt_normalized, gt_real_world


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on validation set')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--model_name', type=str, default='model000190000.pt', help='Model checkpoint filename')
    parser.add_argument('--num_val_samples', type=int, default=256, help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for generation')
    parser.add_argument('--num_generated', type=int, default=4, help='Number of samples per text')
    parser.add_argument('--fixed_frames', type=str, default=None, help='Comma-separated frame indices to fix, e.g., "0,1,2,3,4" (deprecated by --mask_regions)')
    parser.add_argument('--mask_regions', type=str, default=None, help='JSON string of mask regions with temporal and spatial components')
    parser.add_argument('--eval_folder_name', type=str, default='eval_results', help='Folder name for evaluation results (default: eval_results)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mean_std_path', type=str, default=None, help='Path to pre-computed mean_std directory (default: compute from train set)')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data_dir from checkpoint config (useful if data moved)')
    parser.add_argument('--data_loader', type=str, default=None, help='Override data loader class (e.g., "src.diffusion.data_loader.handx.HandXDataset")')
    parser.add_argument('--data_file_name', type=str, default='can_pos_all_wotextfeat.npz', help='Data file name for HandX dataset')
    parser.add_argument('--sample_start', type=int, default=None, help='Start index for sample range (for multi-GPU processing)')
    parser.add_argument('--sample_end', type=int, default=None, help='End index for sample range (for multi-GPU processing)')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 (mixed precision) for faster inference on A40 (default: False)')

    args = parser.parse_args()

    # Parse fixed_frames (backward compatibility)
    fixed_frames = None
    if args.fixed_frames is not None:
        fixed_frames = [int(x.strip()) for x in args.fixed_frames.split(',')]

    # Parse mask_regions (new parameter)
    mask_regions = None
    if args.mask_regions is not None:
        import json
        mask_regions = json.loads(args.mask_regions)

    # Setup output directory
    checkpoint_name = Path(args.checkpoint_dir).name
    if args.output_dir is None:
        output_dir = pjoin(Path(args.checkpoint_dir).parent, f"{args.eval_folder_name}/{checkpoint_name}")
    else:
        output_dir = args.output_dir

    print("="*80)
    print(f"Evaluating checkpoint: {args.checkpoint_dir}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of val samples: {args.num_val_samples}")
    print(f"Generated samples per text: {args.num_generated}")
    print(f"Batch size: {args.batch_size}")
    if mask_regions is not None:
        print(f"Mask regions (spatiotemporal):")
        for i, region in enumerate(mask_regions):
            print(f"  Region {i+1}: temporal={region['temporal']}, spatial={region['spatial']}")
    elif fixed_frames is not None:
        print(f"Fixed frames (temporal only, all joints): {fixed_frames}")
    else:
        print(f"Fixed frames: None (normal generation)")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    fixseed(args.seed)
    torch.set_float32_matmul_precision("high")

    # Load config
    print("\nLoading configuration...")
    config = load_checkpoint_config(args.checkpoint_dir)

    # Add missing parameters for backward compatibility with old checkpoints
    if not hasattr(config.model.diffusion, 'lambda_repr'):
        print("Warning: Adding missing parameter 'lambda_repr' with default value 0.5")
        config.model.diffusion.lambda_repr = 0.5
    if not hasattr(config.model.diffusion, 'lambda_acce'):
        print("Warning: Adding missing parameter 'lambda_acce' with default value 0.0")
        config.model.diffusion.lambda_acce = 0.0
    if not hasattr(config.model.diffusion, 'lambda_contact'):
        print("Warning: Adding missing parameter 'lambda_contact' with default value 0.0")
        config.model.diffusion.lambda_contact = 0.0
    if not hasattr(config.model.diffusion, 'contact_loss'):
        print("Warning: Adding missing parameter 'contact_loss' with default value 'bce'")
        config.model.diffusion.contact_loss = 'bce'
    if not hasattr(config.model, 'contact_prediction'):
        print("Warning: Adding missing parameter 'contact_prediction' with default value False")
        config.model.contact_prediction = False

    # Override data_dir if specified
    if args.data_dir is not None:
        print(f"Overriding data_dir: {config.data.data_dir} -> {args.data_dir}")
        config.data.data_dir = args.data_dir

    # Override data loader if specified
    if args.data_loader is not None:
        print(f"Overriding data loader: {config.data._target_} -> {args.data_loader}")
        config.data._target_ = args.data_loader
        # Add HandX-specific config
        if 'HandX' in args.data_loader:
            config.data.data_file_name = args.data_file_name
            config.data.contact_label = False
            print(f"  Set data_file_name: {args.data_file_name}")
            print(f"  Set contact_label: False")

    print(f"Model architecture: {config.model.arch}")
    print(f"Data representation: {config.data.repr}")
    if hasattr(config.data, 'text_branch'):
        print(f"Text branch: {config.data.text_branch}")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Data loader: {config.data._target_}")

    # Determine dimensions
    if config.data.repr == 'joint_pos':
        njoints, nfeats = 42, 3
    elif config.data.repr == 'joint_pos_w_scalar_rot':
        njoints, nfeats = 42, 4
    elif config.data.repr == 'joint_rot':
        njoints, nfeats = 34, 6
    else:
        raise ValueError(f"Unknown repr: {config.data.repr}")

    # Load or compute mean/std (following handx.py logic)
    mean_std_loaded = False
    mean_std_from_shared = False

    # Try to load from data_dir/mean_std_{repr} (new method, same as handx.py)
    mean_std_auto_path = Path(config.data.data_dir) / f'mean_std_{config.data.repr}'
    mean_file_auto = mean_std_auto_path / "mean.npy"
    std_file_auto = mean_std_auto_path / "std.npy"

    # Check files exist and are non-empty (avoid loading corrupted/incomplete files)
    if (mean_file_auto.exists() and std_file_auto.exists() and
        mean_file_auto.stat().st_size > 0 and std_file_auto.stat().st_size > 0):
        # Load using the same path convention as handx.py
        print(f"\nLoading pre-computed mean/std from: {mean_std_auto_path}")
        try:
            mean = np.load(mean_file_auto)
            std = np.load(std_file_auto)
            print(f"Loaded mean shape: {mean.shape}, std shape: {std.shape}")
            print("✓ Using existing mean/std, no need to recompute or resave")
            mean_std_loaded = True
            mean_std_from_shared = True
        except Exception as e:
            print(f"Warning: Failed to load mean/std: {e}")
            print("Will compute from train dataset instead...")
            mean_std_loaded = False
    elif args.mean_std_path is not None:
        # Fall back to user-specified path
        mean_std_source = Path(args.mean_std_path)
        mean_file = mean_std_source / "mean.npy"
        std_file = mean_std_source / "std.npy"

        if mean_file.exists() and std_file.exists():
            print(f"\nLoading pre-computed mean/std from: {args.mean_std_path}")
            mean = np.load(mean_file)
            std = np.load(std_file)
            print(f"Loaded mean shape: {mean.shape}, std shape: {std.shape}")
            mean_std_loaded = True
        else:
            print(f"\nWarning: mean/std files not found at {args.mean_std_path}")
            print("Will compute from train dataset instead...")

    if not mean_std_loaded:
        # Compute mean/std from train dataset
        print("\nComputing mean/std from train dataset...")
        config_no_norm = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        config_no_norm.data.normalize = False

        train_dataset_raw = instantiate(config_no_norm.data, split='train')
        print(f"Train dataset loaded: {len(train_dataset_raw)} samples")

        first_motion = train_dataset_raw.data_dict[train_dataset_raw.name_list[0]]['motion']
        feature_shape = first_motion.shape[1:]

        sum_of_data = np.zeros(feature_shape, dtype=np.float64)
        sum_of_squares = np.zeros(feature_shape, dtype=np.float64)
        total_frames = 0

        for data in tqdm(train_dataset_raw.data_dict.values(), desc="Computing mean/std"):
            motion_data = data['motion']
            num_frames = motion_data.shape[0]
            total_frames += num_frames
            sum_of_data += np.sum(motion_data, axis=0)
            sum_of_squares += np.sum(np.square(motion_data), axis=0)

        mean = sum_of_data / total_frames
        variance = (sum_of_squares / total_frames) - np.square(mean)
        variance[variance < 0] = 0
        std = np.sqrt(variance)
        std[std < 1e-4] = 1.0

        del train_dataset_raw

        # Save to shared location (following handx.py convention: data_dir/mean_std_{repr})
        shared_mean_std_path = Path(config.data.data_dir) / f'mean_std_{config.data.repr}'
        shared_mean_std_path.mkdir(parents=True, exist_ok=True)

        np.save(shared_mean_std_path / "mean.npy", mean)
        np.save(shared_mean_std_path / "std.npy", std)
        print(f"Saved mean/std to shared location: {shared_mean_std_path}")
    elif not mean_std_from_shared:
        # Only save to shared location if loaded from custom path (not from shared location)
        shared_mean_std_path = Path(config.data.data_dir) / f'mean_std_{config.data.repr}'
        shared_mean_std_path.mkdir(parents=True, exist_ok=True)

        np.save(shared_mean_std_path / "mean.npy", mean)
        np.save(shared_mean_std_path / "std.npy", std)
        print(f"Saved mean/std to shared location: {shared_mean_std_path}")

    # Load validation dataset
    print("\nLoading validation dataset...")
    dataset = instantiate(config.data, split='val')
    print(f"Validation dataset size: {len(dataset)}")

    # Sample validation data
    print(f"\nSampling {args.num_val_samples} random validation samples...")
    val_samples = sample_val_data(
        dataset,
        num_samples=args.num_val_samples,
        seed=args.seed,
        sample_start=args.sample_start,
        sample_end=args.sample_end
    )

    # Load model
    print("\nLoading model and diffusion...")
    model_path = pjoin(args.checkpoint_dir, args.model_name)
    model, diffusion = load_model_and_diffusion(model_path, config)

    device = get_device()
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    print(f"FP16 enabled: {args.use_fp16}")

    # Wrap in ClassifierFreeSampleWrapper (single GPU per process)
    model_wrapped = ClassifierFreeSampleWrapper(model, scale=2.5)

    # Generate samples
    print(f"\nGenerating {args.num_generated} samples for each text prompt (batch_size={args.batch_size})...")
    print("="*80)

    num_batches = (len(val_samples) + args.batch_size - 1) // args.batch_size
    sample_idx = 0

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(val_samples))
        batch_samples = val_samples[start_idx:end_idx]

        batch_results = generate_samples_batch(
            model=model_wrapped,
            diffusion=diffusion,
            dataset=dataset,
            val_samples_batch=batch_samples,
            num_samples_per_text=args.num_generated,
            guidance_scale=2.5,
            njoints=njoints,
            nfeats=nfeats,
            fixed_frames=fixed_frames,
            mask_regions=mask_regions,
            use_fp16=args.use_fp16,
        )

        for i, (val_sample, (generated_norm, generated_real)) in enumerate(zip(batch_samples, batch_results)):
            text_prompt = val_sample['text']
            gt_motion = val_sample['motion']
            motion_length = val_sample['length']
            val_idx = val_sample['index']

            gt_norm, gt_real = process_gt_motion(gt_motion, dataset, njoints, nfeats)

            save_data = {
                'text_prompt': text_prompt,
                'val_index': val_idx,
                'motion_length': motion_length,
                'gt_motion_normalized': gt_norm,
                'gt_motion_real': gt_real,
                'generated_normalized': generated_norm,
                'generated_real': generated_real,
                'njoints': njoints,
                'nfeats': nfeats,
                'checkpoint_dir': args.checkpoint_dir,
                'model_name': args.model_name,
                'guidance_scale': 2.5,
                'fixed_frames': fixed_frames,
                'mask_regions': mask_regions,
            }

            save_path = pjoin(output_dir, f"val_sample_{sample_idx:03d}_idx{val_idx}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

            sample_idx += 1

    print("\n" + "="*80)
    print("Generation complete!")
    print(f"Total samples processed: {len(val_samples)}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Save metadata
    metadata = {
        'checkpoint_dir': args.checkpoint_dir,
        'model_name': args.model_name,
        'num_val_samples': args.num_val_samples,
        'num_generated_per_text': args.num_generated,
        'batch_size': args.batch_size,
        'guidance_scale': 2.5,
        'seed': args.seed,
        'njoints': njoints,
        'nfeats': nfeats,
        'fixed_frames': fixed_frames,
        'mask_regions': mask_regions,
        'config': OmegaConf.to_container(config, resolve=True),
    }

    metadata_path = pjoin(output_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
