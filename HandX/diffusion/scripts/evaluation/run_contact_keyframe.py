#!/usr/bin/env python3
"""
Contact Keyframe Conditioned Generation Evaluation Script

This script evaluates generation conditioned on contact keyframes. It finds
samples with hand-hand contact and uses the contact moment as a conditioning
constraint for generation.

Algorithm:
1. Finds validation samples that contain hand-hand contact
2. For each sample, identifies the contact frame closest to the middle (frame 30)
3. Masks that specific keyframe for the contact joints
4. Generates motion conditioned on the contact constraint

Masking Strategy:
- Temporal mask: Single keyframe where contact occurs
- Spatial mask: Only the contact joints involved in the interaction
- Soft masking with transition for smooth blending

This task tests the model's ability to generate motions that achieve
specific contact interactions at specified moments.

Usage:
    python run_contact_keyframe.py --checkpoint_dir /path/to/checkpoint --model_name model.pt
"""

import os
import sys
import random
import pickle
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
from src.diffusion.metric.interaction import give_contact_label
from src.constant import INTRA_TIP_CONTACT_THRESH, TIP_PALM_CONTACT_THRESH, PALM_PALM_CONTACT_THRESH
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


def detect_contact_frames(motion):
    """
    Detect frames with hand-hand contact.

    Args:
        motion: (T, J*F) array where J=42, F=3 or 4

    Returns:
        contact_frames: List of frame indices with contact
        contact_info: Dictionary with contact details per frame
    """
    # Convert (T, J*F) -> (T, 2, J, 3) format
    if motion.ndim == 2:
        # motion is (T, J*F) where J=42
        T = motion.shape[0]
        # Infer nfeats from shape
        nfeats = motion.shape[1] // 42  # 3 or 4
        # Reshape to (T, 42, nfeats)
        motion = motion.reshape(T, 42, nfeats)
        # Extract only xyz (first 3 features)
        motion = motion[:, :, :3]  # (T, 42, 3)
        # Reshape to (T, 2, 21, 3)
        motion = motion.reshape(T, 2, 21, 3)

    # Get contact labels for all frames
    contact_labels = give_contact_label(
        motion,
        tip_tip_threshold=INTRA_TIP_CONTACT_THRESH,
        tip_palm_threshold=TIP_PALM_CONTACT_THRESH,
        palm_palm_threshold=PALM_PALM_CONTACT_THRESH
    )  # (T, total_pairs)

    # Focus on inter-hand contacts (not intra-hand)
    # contact_labels structure: [left_intra(4), right_intra(4), left_tip_right_palm(5), right_tip_left_palm(5), palm_palm(1)]
    inter_hand_contacts = contact_labels[:, 8:]  # (T, 11) - skip the 8 intra-hand contacts

    # Find frames with any inter-hand contact
    contact_frames = np.where(np.any(inter_hand_contacts, axis=1))[0].tolist()

    # Get detailed info for each frame with contact
    contact_info = {}
    for frame_idx in contact_frames:
        contact_pairs = np.where(inter_hand_contacts[frame_idx])[0].tolist()
        contact_info[int(frame_idx)] = {
            'contact_pairs': contact_pairs,
            'num_contacts': len(contact_pairs)
        }

    return contact_frames, contact_info


def find_closest_contact_frame(contact_frames, target_frame=30):
    """Find the contact frame closest to the target frame."""
    if not contact_frames:
        return None

    contact_frames = np.array(contact_frames)
    distances = np.abs(contact_frames - target_frame)
    closest_idx = np.argmin(distances)
    return int(contact_frames[closest_idx])




def find_contact_samples(dataset, num_samples=256, target_frame=30, seed=42):
    """
    Find samples with most hand-hand contact frames.

    Strategy:
    1. Traverse the entire validation set
    2. Count contact frames for each sample
    3. Select top num_samples samples with most contact frames
    4. For each selected sample, find the contact frame closest to target_frame

    Returns:
        List of dicts with keys: 'text', 'motion', 'length', 'index', 'keyframe', 'joint_to_mask', 'mask_regions'
    """
    random.seed(seed)

    print(f"\nSearching for top {num_samples} samples with most hand-hand contact frames...")
    print(f"Total dataset size: {len(dataset)}")
    print("Step 1: Scanning entire validation set to count contact frames...")

    # Step 1: Scan entire dataset and collect contact frame counts
    all_contact_info = []

    for idx in tqdm(range(len(dataset)), desc="Scanning all samples"):
        motion, length, text = dataset[idx]

        # Convert motion to numpy for contact detection
        # motion is (J, F, T) from dataset
        motion_np = rearrange(motion, 'j f t -> t (j f)')  # (T, J*F)

        # Detect contact frames
        contact_frames, contact_info = detect_contact_frames(motion_np)

        if not contact_frames:
            continue

        # Store basic info for this sample
        all_contact_info.append({
            'index': idx,
            'text': text,
            'motion': motion_np,  # (T, J*F)
            'length': length,
            'dataset_name': dataset.name_list[idx],
            'contact_frames': contact_frames,
            'contact_info': contact_info,
            'num_contact_frames': len(contact_frames),
        })

    print(f"\nFound {len(all_contact_info)} samples with contact in total!")

    if len(all_contact_info) == 0:
        print("ERROR: No contact samples found in the entire dataset!")
        return []

    # Step 2: Sort by number of contact frames (descending) and select top num_samples
    print(f"\nStep 2: Sorting by contact frame count and selecting top {num_samples}...")
    all_contact_info.sort(key=lambda x: x['num_contact_frames'], reverse=True)

    # Select top num_samples
    selected_samples = all_contact_info[:num_samples]

    print(f"Selected {len(selected_samples)} samples")
    print(f"Contact frame range: [{selected_samples[-1]['num_contact_frames']}, {selected_samples[0]['num_contact_frames']}]")

    # Step 3: Process selected samples to find keyframe and create mask_regions
    print(f"\nStep 3: Processing selected samples to find keyframes...")
    contact_samples = []

    for sample_info in tqdm(selected_samples, desc="Processing selected samples"):
        contact_frames = sample_info['contact_frames']
        contact_info = sample_info['contact_info']

        # Find closest contact frame to target_frame
        closest_frame = find_closest_contact_frame(contact_frames, target_frame)

        if closest_frame is None:
            continue

        # Get the contact pairs at this frame
        frame_contact_info = contact_info[closest_frame]
        contact_pairs = frame_contact_info['contact_pairs']

        # Create mask_regions for this sample
        # Mask that one keyframe, all joints
        mask_regions = [
            {
                'temporal': [closest_frame],  # Only the contact keyframe
                'spatial': list(range(42)),   # All joints
                'use_soft_mask': True,
                'temporal_transition_width': 5,
                'core_mask_value': 0.85,
                'edge_mask_value': 0.1,
            }
        ]

        # Store final sample info
        contact_samples.append({
            'index': sample_info['index'],
            'text': sample_info['text'],
            'motion': sample_info['motion'],  # (T, J*F)
            'length': sample_info['length'],
            'keyframe': closest_frame,
            'mask_regions': mask_regions,
            'dataset_name': sample_info['dataset_name'],
            'num_contact_frames': sample_info['num_contact_frames'],
            'distance_to_target': abs(closest_frame - target_frame),
        })

    print(f"\nFinally selected {len(contact_samples)} samples for generation!")

    if len(contact_samples) < num_samples:
        print(f"Warning: Only found {len(contact_samples)} samples, requested {num_samples}")

    # Print statistics
    if contact_samples:
        keyframes = [s['keyframe'] for s in contact_samples]
        distances = [s['distance_to_target'] for s in contact_samples]
        num_contacts = [s['num_contact_frames'] for s in contact_samples]

        print("\n" + "="*80)
        print("Contact Sample Statistics:")
        print("="*80)
        print(f"Number of contact frames per sample:")
        print(f"  Mean: {np.mean(num_contacts):.2f}")
        print(f"  Median: {np.median(num_contacts):.2f}")
        print(f"  Range: [{min(num_contacts)}, {max(num_contacts)}]")
        print(f"\nSelected keyframe positions:")
        print(f"  Average: {np.mean(keyframes):.2f}")
        print(f"  Range: [{min(keyframes)}, {max(keyframes)}]")
        print(f"\nDistance to target frame {target_frame}:")
        print(f"  Average: {np.mean(distances):.2f}")
        print(f"  Max: {max(distances)}")

        # Print first 5 and last 5 samples
        print("\n" + "="*80)
        print("Top 5 samples (most contact frames):")
        print("="*80)
        for i, sample in enumerate(contact_samples[:5]):
            print(f"\nSample {i+1}:")
            print(f"  Dataset name: {sample['dataset_name']}")
            print(f"  Total contact frames: {sample['num_contact_frames']}")
            print(f"  Selected keyframe: {sample['keyframe']} (distance to target: {sample['distance_to_target']})")
            print(f"  Masked: frame {sample['keyframe']}, all 42 joints")

        print("\n" + "="*80)
        print("Bottom 5 samples (least contact frames among selected):")
        print("="*80)
        for i, sample in enumerate(contact_samples[-5:]):
            print(f"\nSample {len(contact_samples)-5+i+1}:")
            print(f"  Dataset name: {sample['dataset_name']}")
            print(f"  Total contact frames: {sample['num_contact_frames']}")
            print(f"  Selected keyframe: {sample['keyframe']} (distance to target: {sample['distance_to_target']})")
            print(f"  Masked: frame {sample['keyframe']}, all 42 joints")

    return contact_samples


def generate_samples_batch(
    model,
    diffusion,
    dataset,
    val_samples_batch,
    num_samples_per_text: int = 4,
    guidance_scale: float = 2.5,
    njoints: int = 42,
    nfeats: int = 3,
):
    """
    Generate samples for a batch of contact samples.
    Each sample has its own mask_regions.
    """
    device = get_device()

    batch_size = len(val_samples_batch)
    motion_lengths = [vs['length'] for vs in val_samples_batch]
    max_length = max(motion_lengths)

    # Process each sample individually since they have different masks
    all_results = []

    for vs in tqdm(val_samples_batch, desc="Generating samples"):
        total_samples = num_samples_per_text
        shape = (total_samples, njoints, nfeats, max_length)

        text_prompt = vs['text']

        # Handle both string and dict text formats
        if isinstance(text_prompt, dict):
            text_dict = {
                'left': [text_prompt['left']] * num_samples_per_text,
                'right': [text_prompt['right']] * num_samples_per_text,
                'two_hands_relation': [text_prompt['two_hands_relation']] * num_samples_per_text
            }
        else:
            text_dict = [text_prompt] * num_samples_per_text

        model_kwargs = dict(
            y=dict(
                lengths=torch.as_tensor([max_length] * num_samples_per_text, device=device),
                text=text_dict
            )
        )

        # Prepare GT motion for inpainting
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

        gt_motion_batch = gt_motion.unsqueeze(0).repeat(num_samples_per_text, 1, 1, 1).to(device)

        # Get mask_regions for this specific sample
        mask_regions = vs['mask_regions']

        # Generate samples
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

        # Process results
        original_length = vs['length']
        samples_for_text = samples[:, :original_length, :]
        samples_real = dataset.inv_transform(samples_for_text)

        all_results.append((samples_for_text, samples_real))

    return all_results


def process_gt_motion(gt_motion_raw: np.ndarray, dataset, njoints: int, nfeats: int):
    """Process GT motion to match generated format."""
    gt_normalized = gt_motion_raw
    gt_real_world = dataset.inv_transform(gt_normalized[np.newaxis, ...])[0]
    return gt_normalized, gt_real_world


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with contact keyframe masking')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--model_name', type=str, default='model000070000.pt', help='Model checkpoint filename')
    parser.add_argument('--num_val_samples', type=int, default=256, help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--num_generated', type=int, default=4, help='Number of samples per text')
    parser.add_argument('--eval_folder_name', type=str, default='generate_contact_keyframe', help='Folder name for evaluation results')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--target_frame', type=int, default=30, help='Target frame to find contact closest to')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory (required)')
    parser.add_argument('--data_loader', type=str, default='src.diffusion.data_loader.handx.HandXDataset', help='Data loader class')
    parser.add_argument('--data_file_name', type=str, default='can_pos_all_wotextfeat.npz', help='Data file name')

    args = parser.parse_args()

    # Setup output directory
    checkpoint_name = Path(args.checkpoint_dir).name
    if args.output_dir is None:
        output_dir = pjoin(Path(args.checkpoint_dir).parent, f"{args.eval_folder_name}/{checkpoint_name}")
    else:
        output_dir = args.output_dir

    print("="*80)
    print(f"Contact Keyframe Masking Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of contact samples to find: {args.num_val_samples}")
    print(f"Generated samples per text: {args.num_generated}")
    print(f"Batch size: {args.batch_size}")
    print(f"Target frame for contact: {args.target_frame}")
    print(f"Masking strategy: temporal=keyframe_only, spatial=all_42_joints")
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

    # Override data settings
    print(f"Overriding data settings...")
    config.data.data_dir = args.data_dir
    config.data._target_ = args.data_loader
    config.data.data_file_name = args.data_file_name
    config.data.contact_label = False

    print(f"Model architecture: {config.model.arch}")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Data loader: {config.data._target_}")

    # Create validation dataset
    print("\nCreating validation dataset...")
    val_dataset = instantiate(config.data, split='val')
    print(f"Validation dataset size: {len(val_dataset)}")

    # Find contact samples
    contact_samples = find_contact_samples(
        val_dataset,
        num_samples=args.num_val_samples,
        target_frame=args.target_frame,
        seed=args.seed
    )

    if len(contact_samples) == 0:
        print("ERROR: No contact samples found!")
        return

    # Load model
    print("\nLoading model...")
    checkpoint_path = pjoin(args.checkpoint_dir, args.model_name)
    model, diffusion = load_model_and_diffusion(checkpoint_path, config)

    # Wrap model with classifier-free guidance
    model_wrapped = ClassifierFreeSampleWrapper(model, scale=2.5)

    device = get_device()
    model_wrapped.to(device)
    model_wrapped.eval()
    print(f"Model loaded on device: {device}")

    # Determine dimensions based on data representation
    if config.data.repr == 'joint_pos':
        njoints, nfeats = 42, 3
    elif config.data.repr == 'joint_pos_w_scalar_rot':
        njoints, nfeats = 42, 4
    elif config.data.repr == 'joint_rot':
        njoints, nfeats = 34, 6
    else:
        raise ValueError(f"Unknown repr: {config.data.repr}")

    print(f"Data representation: {config.data.repr}")
    print(f"Model dimensions: {njoints} joints × {nfeats} features")

    # Generate samples in batches
    print(f"\n{'='*80}")
    print(f"Generating samples...")
    print(f"{'='*80}\n")

    num_batches = (len(contact_samples) + args.batch_size - 1) // args.batch_size
    sample_idx = 0

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(contact_samples))
        batch_samples = contact_samples[start_idx:end_idx]

        # Generate samples for this batch
        batch_results = generate_samples_batch(
            model_wrapped, diffusion, val_dataset, batch_samples,
            num_samples_per_text=args.num_generated,
            njoints=njoints,
            nfeats=nfeats
        )

        # Save results for each sample in batch
        for i, (val_sample, (generated_norm, generated_real)) in enumerate(zip(batch_samples, batch_results)):
            text_prompt = val_sample['text']
            gt_motion = val_sample['motion']
            motion_length = val_sample['length']
            val_idx = val_sample['index']
            keyframe = val_sample['keyframe']
            mask_regions = val_sample['mask_regions']

            gt_norm, gt_real = process_gt_motion(gt_motion, val_dataset, njoints, nfeats)

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
                'fixed_frames': None,  # Keep consistent with other scripts
                'mask_regions': mask_regions,
                # Additional info (does not affect downstream compatibility)
                'keyframe': keyframe,
                'dataset_name': val_sample['dataset_name'],
            }

            save_path = pjoin(output_dir, f"val_sample_{sample_idx:03d}_idx{val_idx}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

            sample_idx += 1

    print(f"\n{'='*80}")
    print("Generation complete!")
    print(f"Total samples processed: {len(contact_samples)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    # Save metadata
    metadata = {
        'checkpoint_dir': args.checkpoint_dir,
        'model_name': args.model_name,
        'num_val_samples': len(contact_samples),
        'num_generated_per_text': args.num_generated,
        'batch_size': args.batch_size,
        'guidance_scale': 2.5,
        'seed': args.seed,
        'njoints': njoints,
        'nfeats': nfeats,
        'target_frame': args.target_frame,
        'mask_strategy': 'keyframe_all_joints',
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
