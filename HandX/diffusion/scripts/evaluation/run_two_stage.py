#!/usr/bin/env python3
"""
Two-Stage Generation Batch Evaluation Script

This script performs batch evaluation of two-stage motion generation:
1. Finds consecutive clip pairs from validation set (pairs of clips that are temporally adjacent)
2. For each pair, generates 4 samples with Stage 1 (free generation) and Stage 2 (conditioned)
3. Stage 2 is conditioned on the last 10 frames from Stage 1 using soft masking

Output: PKL files containing GT and generated motions for both stages
"""

import os
import sys
import random
import pickle
import json
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


def load_mean_std(repr: str, mean_std_dir: str = None, data_dir: str = None):
    """
    Load mean/std from specified directory.
    Following evaluate_val_samples.py logic.

    Args:
        repr: Data representation type (e.g., 'joint_pos_w_scalar_rot')
        mean_std_dir: Directory containing mean.npy and std.npy (overrides data_dir)
        data_dir: Data directory (will look for mean_std_{repr} subdirectory)

    Returns:
        mean: (J, F) numpy array
        std: (J, F) numpy array
    """
    # Try mean_std_dir first
    if mean_std_dir is not None:
        mean_std_path = Path(mean_std_dir)
        mean_file = mean_std_path / "mean.npy"
        std_file = mean_std_path / "std.npy"

        if mean_file.exists() and std_file.exists():
            print(f"Loading pre-computed mean/std from: {mean_std_path}")
            mean = np.load(mean_file)
            std = np.load(std_file)
            print(f"Loaded mean shape: {mean.shape}, std shape: {std.shape}")
            return mean, std

    # Try data_dir/mean_std_{repr}
    if data_dir is not None:
        mean_std_path = Path(data_dir) / f'mean_std_{repr}'
        mean_file = mean_std_path / "mean.npy"
        std_file = mean_std_path / "std.npy"

        if mean_file.exists() and std_file.exists():
            print(f"Loading pre-computed mean/std from: {mean_std_path}")
            mean = np.load(mean_file)
            std = np.load(std_file)
            print(f"Loaded mean shape: {mean.shape}, std shape: {std.shape}")
            return mean, std

    # Not found
    raise FileNotFoundError(
        f"Mean/std files not found. Tried:\n"
        f"  1. {mean_std_dir if mean_std_dir else 'None'}\n"
        f"  2. {Path(data_dir) / f'mean_std_{repr}' if data_dir else 'None'}\n"
        f"Please specify --mean_std_dir or ensure mean.npy/std.npy exist."
    )


def normalize_motion(motion: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Normalize motion data.

    Args:
        motion: (T, J*F) or (T, J, F)
        mean: (J, F)
        std: (J, F)

    Returns:
        normalized: same shape as input
    """
    original_shape = motion.shape

    # Reshape to (T, J, F) if needed
    if motion.ndim == 2:
        T = motion.shape[0]
        J, F = mean.shape
        motion = motion.reshape(T, J, F)

    # Normalize
    normalized = (motion - mean[None, :, :]) / std[None, :, :]

    # Reshape back to original
    if len(original_shape) == 2:
        normalized = normalized.reshape(original_shape)

    return normalized


def denormalize_motion(motion: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Denormalize motion data (inverse transform).

    Args:
        motion: (B, T, J*F) or (T, J*F) or (T, J, F)
        mean: (J, F)
        std: (J, F)

    Returns:
        denormalized: same shape as input
    """
    original_shape = motion.shape

    # Handle batch dimension
    if motion.ndim == 3:
        # (B, T, J*F)
        B, T = motion.shape[0], motion.shape[1]
        J, F = mean.shape
        motion = motion.reshape(B, T, J, F)
        denormalized = motion * std[None, None, :, :] + mean[None, None, :, :]
        denormalized = denormalized.reshape(original_shape)
    elif motion.ndim == 2:
        # (T, J*F)
        T = motion.shape[0]
        J, F = mean.shape
        motion = motion.reshape(T, J, F)
        denormalized = motion * std[None, :, :] + mean[None, :, :]
        denormalized = denormalized.reshape(original_shape)
    else:
        raise ValueError(f"Unexpected motion shape: {motion.shape}")

    return denormalized


def find_consecutive_clip_pairs(
    data_dir: str,
    num_samples: int = 256,
    seed: int = 42,
):
    """
    Find consecutive clip pairs from validation set.

    Args:
        data_dir: Path to merged_bihand_data_92k_final
        num_samples: Number of clip pairs to find
        seed: Random seed

    Returns:
        List of dicts with keys:
            - 'seq_name': sequence name
            - 'skeleton_file': path to skeleton npy
            - 'text_file': path to text json
            - 'clip1_start': first clip start frame
            - 'clip1_end': first clip end frame
            - 'clip2_start': second clip start frame
            - 'clip2_end': second clip end frame
            - 'text1': first clip text (dict with 'left', 'right', 'two_hands_relation')
            - 'text2': second clip text
    """
    random.seed(seed)

    print(f"\nSearching for {num_samples} consecutive clip pairs from validation set...")

    # Load data_info to get val sequences
    data_info_path = pjoin(data_dir, "data_info.json")
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)

    val_sequences = [seq_name for seq_name, info in data_info.items() if info['split'] == 'val']
    print(f"Found {len(val_sequences)} validation sequences")

    # Load file_mapping
    file_mapping_path = pjoin(data_dir, "file_mapping.json")
    with open(file_mapping_path, 'r') as f:
        file_mapping = json.load(f)

    # Create mapping from new_name to file info
    file_map_dict = {item['new_name']: item for item in file_mapping}

    # Find ALL consecutive clip pairs first
    all_clip_pairs = []

    for seq_name in tqdm(val_sequences, desc="Scanning validation sequences"):
        if seq_name not in file_map_dict:
            continue

        file_info = file_map_dict[seq_name]
        valid_clips = file_info.get('valid_clips', [])

        # Check if there are consecutive clips
        if len(valid_clips) < 2:
            continue

        # Load text annotations
        text_file = file_info['text_file']
        if not os.path.exists(text_file):
            continue

        with open(text_file, 'r') as f:
            text_data = json.load(f)

        if len(text_data) < 2:
            continue

        # Find ALL consecutive pairs in this sequence
        for i in range(len(text_data) - 1):
            clip1 = text_data[i]
            clip2 = text_data[i + 1]

            # Check if consecutive (clip1 end == clip2 start)
            if clip1['frame_end'] == clip2['frame_start']:
                # Both clips should be 60 frames
                if (clip1['frame_end'] - clip1['frame_start'] == 60 and
                    clip2['frame_end'] - clip2['frame_start'] == 60):

                    # Verify annotation format
                    if (len(clip1.get('annotation', [])) == 0 or
                        len(clip2.get('annotation', [])) == 0):
                        continue

                    # Check if annotations have required keys
                    required_keys = ['left', 'right', 'two_hands_relation']
                    if not all(k in clip1['annotation'][0] for k in required_keys):
                        continue
                    if not all(k in clip2['annotation'][0] for k in required_keys):
                        continue

                    # Store all valid consecutive pairs with ALL annotations
                    all_clip_pairs.append({
                        'seq_name': seq_name,
                        'skeleton_file': file_info['skeleton_file'],
                        'text_file': text_file,
                        'clip1_start': clip1['frame_start'],
                        'clip1_end': clip1['frame_end'],
                        'clip2_start': clip2['frame_start'],
                        'clip2_end': clip2['frame_end'],
                        'clip1_annotations': clip1['annotation'],
                        'clip2_annotations': clip2['annotation'],
                    })

    print(f"\nFound {len(all_clip_pairs)} total consecutive clip pairs from all validation sequences!")

    if len(all_clip_pairs) == 0:
        print("ERROR: No consecutive clip pairs found!")
        return []

    # Randomly sample num_samples pairs
    if len(all_clip_pairs) < num_samples:
        print(f"Warning: Only found {len(all_clip_pairs)} pairs, requested {num_samples}")
        sampled_pairs = all_clip_pairs
    else:
        random.shuffle(all_clip_pairs)
        sampled_pairs = all_clip_pairs[:num_samples]
        print(f"Randomly sampled {num_samples} pairs from {len(all_clip_pairs)} available pairs")

    # Now assign random annotation indices to each sampled pair
    clip_pairs = []
    for pair in sampled_pairs:
        # Randomly select one annotation index (use same index for both clips)
        max_ann_idx = min(len(pair['clip1_annotations']), len(pair['clip2_annotations'])) - 1
        ann_idx = random.randint(0, max_ann_idx)
        ann1 = pair['clip1_annotations'][ann_idx]
        ann2 = pair['clip2_annotations'][ann_idx]

        clip_pairs.append({
            'seq_name': pair['seq_name'],
            'skeleton_file': pair['skeleton_file'],
            'text_file': pair['text_file'],
            'clip1_start': pair['clip1_start'],
            'clip1_end': pair['clip1_end'],
            'clip2_start': pair['clip2_start'],
            'clip2_end': pair['clip2_end'],
            'text1': ann1,
            'text2': ann2,
        })

    return clip_pairs


def generate_stage1(
    model,
    diffusion,
    text_prompt,
    motion_length: int = 60,
    njoints: int = 42,
    nfeats: int = 4,
):
    """
    Stage 1: Free generation (no masking).

    Returns:
        samples: (1, T, J*F) normalized
    """
    device = get_device()
    shape = (1, njoints, nfeats, motion_length)

    text_dict = {
        'left': [text_prompt['left']],
        'right': [text_prompt['right']],
        'two_hands_relation': [text_prompt['two_hands_relation']]
    }

    model_kwargs = dict(
        y=dict(
            lengths=torch.tensor([motion_length], device=device),
            text=text_dict
        )
    )

    samples = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        gt_motion=None,
        fixed_frames=None,
        mask_regions=None,
    )

    samples = rearrange(samples, 'b j f t -> b t (j f)').detach().cpu().numpy()
    return samples


def generate_stage2(
    model,
    diffusion,
    text_prompt,
    stage1_motion,
    motion_length: int = 60,
    njoints: int = 42,
    nfeats: int = 4,
    core_mask_value: float = 0.85,
    edge_mask_value: float = 0.1,
    mask_frames: int = 5,
    transition_frames: int = 5,
):
    """
    Stage 2: Conditioned on Stage 1's last 10 frames.

    Args:
        stage1_motion: (1, T, J*F) normalized

    Returns:
        samples: (1, T, J*F) normalized
    """
    device = get_device()
    shape = (1, njoints, nfeats, motion_length)

    text_dict = {
        'left': [text_prompt['left']],
        'right': [text_prompt['right']],
        'two_hands_relation': [text_prompt['two_hands_relation']]
    }

    model_kwargs = dict(
        y=dict(
            lengths=torch.tensor([motion_length], device=device),
            text=text_dict
        )
    )

    # Extract last 10 frames from stage1
    stage1_sample = stage1_motion[0]  # (T, J*F)
    total_condition_frames = mask_frames + transition_frames
    last_10_frames = stage1_sample[-total_condition_frames:, :]  # (10, J*F)

    # Convert to (J, F, T) format
    condition_motion = rearrange(
        torch.tensor(last_10_frames, dtype=torch.float32),
        't (j f) -> j f t',
        j=njoints,
        f=nfeats
    )

    # Create gt_motion with condition frames at the beginning
    gt_motion = torch.zeros(njoints, nfeats, motion_length)
    gt_motion[:, :, :total_condition_frames] = condition_motion
    gt_motion = gt_motion.unsqueeze(0).to(device)  # (1, J, F, T)

    # Define mask regions
    mask_regions = [
        {
            'temporal': list(range(mask_frames)),  # [0,1,2,3,4]
            'spatial': list(range(njoints)),
            'use_soft_mask': True,
            'core_mask_value': core_mask_value,
            'edge_mask_value': edge_mask_value,
            'temporal_transition_width': transition_frames
        }
    ]

    samples = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        gt_motion=gt_motion,
        fixed_frames=None,
        mask_regions=mask_regions,
    )

    samples = rearrange(samples, 'b j f t -> b t (j f)').detach().cpu().numpy()
    return samples


def generate_stage1_batch(
    model,
    diffusion,
    text_prompts,
    motion_length: int = 60,
    njoints: int = 42,
    nfeats: int = 4,
):
    """
    Stage 1 batch generation: Free generation for multiple text prompts.

    Args:
        text_prompts: List of text dicts

    Returns:
        samples: (B, T, J*F) normalized
    """
    device = get_device()
    batch_size = len(text_prompts)
    shape = (batch_size, njoints, nfeats, motion_length)

    # Prepare text dict
    text_dict = {
        'left': [tp['left'] for tp in text_prompts],
        'right': [tp['right'] for tp in text_prompts],
        'two_hands_relation': [tp['two_hands_relation'] for tp in text_prompts]
    }

    model_kwargs = dict(
        y=dict(
            lengths=torch.tensor([motion_length] * batch_size, device=device),
            text=text_dict
        )
    )

    samples = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        gt_motion=None,
        fixed_frames=None,
        mask_regions=None,
    )

    samples = rearrange(samples, 'b j f t -> b t (j f)').detach().cpu().numpy()
    return samples


def generate_stage2_batch(
    model,
    diffusion,
    text_prompts,
    stage1_motions,
    motion_length: int = 60,
    njoints: int = 42,
    nfeats: int = 4,
    core_mask_value: float = 0.85,
    edge_mask_value: float = 0.1,
    mask_frames: int = 5,
    transition_frames: int = 5,
):
    """
    Stage 2 batch generation: Conditioned on Stage 1's last 10 frames.

    Args:
        text_prompts: List of text dicts
        stage1_motions: (B, T, J*F) normalized

    Returns:
        samples: (B, T, J*F) normalized
    """
    device = get_device()
    batch_size = len(text_prompts)
    shape = (batch_size, njoints, nfeats, motion_length)

    # Prepare text dict
    text_dict = {
        'left': [tp['left'] for tp in text_prompts],
        'right': [tp['right'] for tp in text_prompts],
        'two_hands_relation': [tp['two_hands_relation'] for tp in text_prompts]
    }

    model_kwargs = dict(
        y=dict(
            lengths=torch.tensor([motion_length] * batch_size, device=device),
            text=text_dict
        )
    )

    # Extract last 10 frames from each stage1 motion
    total_condition_frames = mask_frames + transition_frames
    last_10_frames_batch = stage1_motions[:, -total_condition_frames:, :]  # (B, 10, J*F)

    # Convert to (B, J, F, T) format
    gt_motions = []
    for i in range(batch_size):
        last_10_frames = last_10_frames_batch[i]  # (10, J*F)

        condition_motion = rearrange(
            torch.tensor(last_10_frames, dtype=torch.float32),
            't (j f) -> j f t',
            j=njoints,
            f=nfeats
        )

        # Create gt_motion with condition frames at the beginning
        gt_motion = torch.zeros(njoints, nfeats, motion_length)
        gt_motion[:, :, :total_condition_frames] = condition_motion
        gt_motions.append(gt_motion)

    gt_motion_batch = torch.stack(gt_motions, dim=0).to(device)  # (B, J, F, T)

    # Define mask regions (same for all samples in batch)
    mask_regions = [
        {
            'temporal': list(range(mask_frames)),  # [0,1,2,3,4]
            'spatial': list(range(njoints)),
            'use_soft_mask': True,
            'core_mask_value': core_mask_value,
            'edge_mask_value': edge_mask_value,
            'temporal_transition_width': transition_frames
        }
    ]

    samples = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        gt_motion=gt_motion_batch,
        fixed_frames=None,
        mask_regions=mask_regions,
    )

    samples = rearrange(samples, 'b j f t -> b t (j f)').detach().cpu().numpy()
    return samples


def generate_two_stage_batch(
    model,
    diffusion,
    clip_pairs_batch,
    num_samples_per_pair: int = 4,
    njoints: int = 42,
    nfeats: int = 4,
):
    """
    Generate two-stage samples for a batch of clip pairs.
    NOW WITH FULL BATCHING - much faster!

    Returns:
        List of (stage1_results, stage2_results) tuples
        Each result is (num_samples_per_pair, T, J*F)
    """
    batch_size = len(clip_pairs_batch)
    total_samples = batch_size * num_samples_per_pair

    # Prepare text prompts for stage1 (repeat each pair num_samples_per_pair times)
    stage1_texts = []
    stage2_texts = []
    for clip_pair in clip_pairs_batch:
        for _ in range(num_samples_per_pair):
            stage1_texts.append(clip_pair['text1'])
            stage2_texts.append(clip_pair['text2'])

    print(f"  Stage 1: Generating {total_samples} samples in batch...")
    # Stage 1: Generate all samples at once
    stage1_all = generate_stage1_batch(
        model, diffusion, stage1_texts,
        motion_length=60,
        njoints=njoints,
        nfeats=nfeats
    )  # (total_samples, T, J*F)

    print(f"  Stage 2: Generating {total_samples} samples in batch (conditioned on Stage 1)...")
    # Stage 2: Generate all samples at once, conditioned on stage1
    stage2_all = generate_stage2_batch(
        model, diffusion, stage2_texts, stage1_all,
        motion_length=60,
        njoints=njoints,
        nfeats=nfeats
    )  # (total_samples, T, J*F)

    # Reshape results back to per-pair format
    all_results = []
    for i in range(batch_size):
        start_idx = i * num_samples_per_pair
        end_idx = start_idx + num_samples_per_pair

        stage1_samples = stage1_all[start_idx:end_idx]  # (num_samples_per_pair, T, J*F)
        stage2_samples = stage2_all[start_idx:end_idx]

        all_results.append((stage1_samples, stage2_samples))

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Two-stage generation batch evaluation')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--model_name', type=str, default='model000070000.pt', help='Model checkpoint filename')
    parser.add_argument('--num_val_samples', type=int, default=256, help='Number of consecutive clip pairs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--num_generated', type=int, default=4, help='Number of samples per clip pair')
    parser.add_argument('--eval_folder_name', type=str, default='generate_two_stage', help='Folder name for evaluation results')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory (required)')
    parser.add_argument('--mean_std_dir', type=str, default=None, help='Directory containing mean.npy and std.npy (default: look in data_dir/mean_std_{repr})')

    args = parser.parse_args()

    # Setup output directory
    checkpoint_name = Path(args.checkpoint_dir).name
    if args.output_dir is None:
        output_dir = pjoin(Path(args.checkpoint_dir).parent, f"{args.eval_folder_name}/{checkpoint_name}")
    else:
        output_dir = args.output_dir

    print("="*80)
    print(f"Two-Stage Generation Batch Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of consecutive clip pairs: {args.num_val_samples}")
    print(f"Generated samples per pair: {args.num_generated}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data directory: {args.data_dir}")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    fixseed(args.seed)
    torch.set_float32_matmul_precision("high")

    # Load config
    print("\nLoading configuration...")
    config = load_checkpoint_config(args.checkpoint_dir)

    # Add missing parameters for backward compatibility
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

    print(f"Model architecture: {config.model.arch}")
    print(f"Data representation: {config.data.repr}")

    # Determine dimensions
    if config.data.repr == 'joint_pos':
        njoints, nfeats = 42, 3
    elif config.data.repr == 'joint_pos_w_scalar_rot':
        njoints, nfeats = 42, 4
    elif config.data.repr == 'joint_rot':
        njoints, nfeats = 34, 6
    else:
        raise ValueError(f"Unknown repr: {config.data.repr}")

    print(f"Model dimensions: {njoints} joints × {nfeats} features")

    # Load mean/std for normalization
    print("\nLoading mean/std...")
    mean, std = load_mean_std(config.data.repr, mean_std_dir=args.mean_std_dir, data_dir=args.data_dir)

    # Find consecutive clip pairs
    clip_pairs = find_consecutive_clip_pairs(
        args.data_dir,
        num_samples=args.num_val_samples,
        seed=args.seed
    )

    if len(clip_pairs) == 0:
        print("ERROR: No consecutive clip pairs found!")
        return

    # Load model
    print("\nLoading model...")
    checkpoint_path = pjoin(args.checkpoint_dir, args.model_name)
    model, diffusion = load_model_and_diffusion(checkpoint_path, config)

    model_wrapped = ClassifierFreeSampleWrapper(model, scale=2.5)
    device = get_device()
    model_wrapped.to(device)
    model_wrapped.eval()
    print(f"Model loaded on device: {device}")

    # Generate samples in batches
    print(f"\n{'='*80}")
    print(f"Generating two-stage samples...")
    print(f"{'='*80}\n")

    num_batches = (len(clip_pairs) + args.batch_size - 1) // args.batch_size
    sample_idx = 0

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(clip_pairs))
        batch_pairs = clip_pairs[start_idx:end_idx]

        # Generate samples for this batch
        batch_results = generate_two_stage_batch(
            model_wrapped, diffusion, batch_pairs,
            num_samples_per_pair=args.num_generated,
            njoints=njoints,
            nfeats=nfeats
        )

        # Save results for each pair in batch
        for i, (clip_pair, (stage1_norm, stage2_norm)) in enumerate(zip(batch_pairs, batch_results)):
            # Denormalize
            stage1_real = denormalize_motion(stage1_norm, mean, std)
            stage2_real = denormalize_motion(stage2_norm, mean, std)

            # Load GT skeletons
            skeleton_file = clip_pair['skeleton_file']
            skeleton_full = np.load(skeleton_file)  # (T, 2, 21, 3)

            # Extract GT clips
            gt1 = skeleton_full[clip_pair['clip1_start']:clip_pair['clip1_end']]  # (60, 2, 21, 3)
            gt2 = skeleton_full[clip_pair['clip2_start']:clip_pair['clip2_end']]  # (60, 2, 21, 3)

            # Reshape GT to (T, J*F) to match model format
            gt1_flat = gt1.reshape(60, 42, 3)  # (60, 42, 3)
            gt2_flat = gt2.reshape(60, 42, 3)

            # Normalize GT (to match generated format)
            gt1_norm = normalize_motion(gt1_flat, mean[:, :3], std[:, :3])  # (60, 42, 3)
            gt2_norm = normalize_motion(gt2_flat, mean[:, :3], std[:, :3])

            # Flatten to (60, 126)
            gt1_norm = gt1_norm.reshape(60, 42*3)
            gt2_norm = gt2_norm.reshape(60, 42*3)

            save_data = {
                'text1': clip_pair['text1'],
                'text2': clip_pair['text2'],
                'seq_name': clip_pair['seq_name'],
                'clip1_frames': [clip_pair['clip1_start'], clip_pair['clip1_end']],
                'clip2_frames': [clip_pair['clip2_start'], clip_pair['clip2_end']],
                'motion_length': 60,

                # GT motions (normalized and real)
                'gt1_normalized': gt1_norm,  # (T, J*F=126 for xyz)
                'gt1_real': gt1,  # (T, 2, 21, 3)
                'gt2_normalized': gt2_norm,
                'gt2_real': gt2,

                # Generated motions (B, T, J*F)
                'stage1_normalized': stage1_norm,  # (B, T, J*F=168 for xyzr)
                'stage1_real': stage1_real,
                'stage2_normalized': stage2_norm,
                'stage2_real': stage2_real,

                # Model info
                'njoints': njoints,
                'nfeats': nfeats,
                'checkpoint_dir': args.checkpoint_dir,
                'model_name': args.model_name,
                'guidance_scale': 2.5,
                'num_generated': args.num_generated,
            }

            save_path = pjoin(output_dir, f"two_stage_sample_{sample_idx:03d}_{clip_pair['seq_name']}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

            sample_idx += 1

    print(f"\n{'='*80}")
    print("Generation complete!")
    print(f"Total clip pairs processed: {len(clip_pairs)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    # Save metadata
    metadata = {
        'checkpoint_dir': args.checkpoint_dir,
        'model_name': args.model_name,
        'num_val_samples': len(clip_pairs),
        'num_generated_per_pair': args.num_generated,
        'batch_size': args.batch_size,
        'guidance_scale': 2.5,
        'seed': args.seed,
        'njoints': njoints,
        'nfeats': nfeats,
        'data_dir': args.data_dir,
    }

    metadata_path = pjoin(output_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
