#!/usr/bin/env python3
"""
Quick Evaluation Script - Using Evaluator to compute FID/Diversity/Multimodality

Features:
- Load trained model
- Evaluate on specified dataset (train/val)
- Output FID, Diversity, Multimodality metrics
- Optionally save generated data to pkl for reproducibility

Usage:
    python scripts/quick_eval.py \
        --model_path /path/to/model.pt \
        --num_samples_on_train 1000 \
        --num_samples_on_val 200 \
        --num_samples_per_condition 10 \
        --split val \
        --save_pkl ./evaluation_results
"""

import argparse
import sys
import pickle
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model.evaluator import Evaluator
from src.diffusion.model.cls_free_sampler import ClassifierFreeSampleWrapper
from src.diffusion.utils.model_utils import create_model_and_diffusion
from src.diffusion.utils.mics import get_device
from src.diffusion.config import DataLoaderConfig


class EvaluatorWithSave(Evaluator):
    """
    Extended Evaluator that can optionally save generated data to pkl
    """
    def __init__(self, *args, save_data=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_data = save_data
        self.saved_gt_motions = None
        self.saved_sample_motions = None
        self.saved_sample_motions_multi = None
        self.saved_masks = None
        self.saved_masks_multi = None

    def evaluate(self, split: str):
        """
        Run evaluation and optionally save generated data
        """
        from src.diffusion.data_loader.get_data import get_dataloader

        # Create subdatasets
        sub_dataset_plain = self.get_subdataset(split)
        dataloader_plain = get_dataloader(sub_dataset_plain, self.dataloader_cfg)

        sub_dataset_for_multimodality = self.get_subdataset(split, for_multimodality=True)
        dataloader_for_multimodality = self.get_dataloader_for_multimodality(sub_dataset_for_multimodality)

        # Collect data (now includes text embeddings)
        gt_motions, sample_motions, masks, text_embeds_list = self.collect_gt_sample_motion_pairs(dataloader_plain, split)
        sample_motions_for_multimodality, masks_for_multimodality = self.collect_sample_motions_for_multimodality(dataloader_for_multimodality, split)

        # Save data if requested
        if self.save_data:
            self.saved_gt_motions = gt_motions
            self.saved_sample_motions = sample_motions
            self.saved_sample_motions_multi = sample_motions_for_multimodality
            self.saved_masks = masks
            self.saved_masks_multi = masks_for_multimodality

        # Get embeddings
        gt_embeddings = self.get_motion_embeddings(gt_motions, masks)
        sample_embeddings = self.get_motion_embeddings(sample_motions, masks)
        text_embeddings = torch.concat(text_embeds_list, dim=0)
        sample_embeddings_for_multimodality = self.get_motion_embeddings(sample_motions_for_multimodality, masks_for_multimodality)

        # Gather across processes (only if distributed is initialized)
        from src.diffusion.dist import gather_tensors, is_main_process, is_dist_avail_and_initialized

        if is_dist_avail_and_initialized():
            gt_embeddings = gather_tensors(gt_embeddings)
            sample_embeddings = gather_tensors(sample_embeddings)
            text_embeddings = gather_tensors(text_embeddings)
            sample_embeddings_for_multimodality = gather_tensors(sample_embeddings_for_multimodality)

            if not is_main_process():
                return None

            # Convert list of tensors to single tensor
            gt_embeddings = torch.concat(gt_embeddings, dim=0)
            sample_embeddings = torch.concat(sample_embeddings, dim=0)
            text_embeddings = torch.concat(text_embeddings, dim=0)
            sample_embeddings_for_multimodality = torch.concat(sample_embeddings_for_multimodality, dim=0)
        else:
            # Single GPU case: embeddings are already single tensors
            pass

        # Convert to numpy
        gt_embeddings = gt_embeddings.detach().cpu().numpy()
        sample_embeddings = sample_embeddings.detach().cpu().numpy()
        text_embeddings = text_embeddings.detach().cpu().numpy()
        sample_embeddings_for_multimodality = sample_embeddings_for_multimodality.detach().cpu().numpy()

        # Calculate metrics
        ret_dict = dict()
        ret_dict[f'{split}_fid'] = self.calc_fid_metric(gt_embeddings, sample_embeddings)

        # Compute FID between text and motion embeddings
        ret_dict[f'{split}_fid_text_gt'] = self.calc_fid_metric(text_embeddings, gt_embeddings)
        ret_dict[f'{split}_fid_text_gen'] = self.calc_fid_metric(text_embeddings, sample_embeddings)

        # Compute cosine similarity between paired text and motion embeddings
        ret_dict[f'{split}_cosine_sim_text_gt'] = self.calc_cosine_similarity(text_embeddings, gt_embeddings)
        ret_dict[f'{split}_cosine_sim_text_gen'] = self.calc_cosine_similarity(text_embeddings, sample_embeddings)

        if sample_embeddings.shape[0] % 2 == 1:
            sample_embeddings = sample_embeddings[:-1]

        sample_embeddings1, sample_embeddings2 = np.split(sample_embeddings, 2, axis=0)
        ret_dict[f'{split}_diversity'] = self.calc_diversity_metric(sample_embeddings1, sample_embeddings2)

        sample_embeddings_for_multimodality = sample_embeddings_for_multimodality.reshape(-1, self.num_samples_per_condition, *sample_embeddings_for_multimodality.shape[1:])

        # If num_samples_per_condition is odd, remove the last sample from each condition
        if self.num_samples_per_condition % 2 == 1:
            sample_embeddings_for_multimodality = sample_embeddings_for_multimodality[:, :-1, :]

        sample_embeddings1_for_multimodality, sample_embeddings2_for_multimodality = np.split(sample_embeddings_for_multimodality, 2, axis=1)
        sample_embeddings1_for_multimodality = sample_embeddings1_for_multimodality.reshape(-1, sample_embeddings1_for_multimodality.shape[-1])
        sample_embeddings2_for_multimodality = sample_embeddings2_for_multimodality.reshape(-1, sample_embeddings2_for_multimodality.shape[-1])
        ret_dict[f'{split}_multimodality'] = self.calc_diversity_metric(sample_embeddings1_for_multimodality, sample_embeddings2_for_multimodality)

        return ret_dict


def main():
    # Start timer
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Quick evaluation using Evaluator")

    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to config file (default: auto-detect from model_path)")
    parser.add_argument("--guidance_param", type=float, default=2.5,
                       help="Classifier-free guidance parameter")

    # Three core Evaluator parameters
    parser.add_argument("--num_samples_on_train", type=int, default=1000,
                       help="Number of samples to evaluate on train set")
    parser.add_argument("--num_samples_on_val", type=int, default=200,
                       help="Number of samples to evaluate on val set")
    parser.add_argument("--num_samples_per_condition", type=int, default=10,
                       help="Number of samples per condition (for Multimodality)")

    # Evaluation parameters
    parser.add_argument("--split", type=str, default='val',
                       choices=['train', 'val'],
                       help="Dataset split to evaluate: 'train' or 'val'")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="DataLoader batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader num_workers")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Dataset root directory (default: use path from config)")

    # Save parameters
    parser.add_argument("--save_pkl", type=str, default=None,
                       help="Directory to save generated data as pkl (optional)")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Quick Evaluation - Using Evaluator")
    print(f"{'='*70}")
    print(f"Model path: {args.model_path}")
    print(f"Evaluation split: {args.split}")
    print(f"Guidance parameter: {args.guidance_param}")
    print(f"Train samples: {args.num_samples_on_train}")
    print(f"Val samples: {args.num_samples_on_val}")
    print(f"Samples per condition: {args.num_samples_per_condition}")
    if args.save_pkl:
        print(f"Save pkl to: {args.save_pkl}")
    print(f"{'='*70}\n")

    # 1. Load configuration
    print("Step 1: Loading model configuration...")
    if args.config_path is None:
        args.config_path = Path(args.model_path).parent / "config.yaml"

    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    cfg = OmegaConf.load(args.config_path)
    print(f"✅ Config file: {args.config_path}")
    print(f"   Data representation: {cfg.data.repr}")

    # Override data_dir if specified
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
        print(f"   Data directory: {args.data_dir}")

    # 2. Load model
    print(f"\nStep 2: Loading Diffusion model...")
    torch.set_float32_matmul_precision("high")

    model, diffusion = create_model_and_diffusion(cfg.model)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print(f"   Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu')

    # Handle state_dict keys (may be wrapped in 'state_dict')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict, strict=False)

    # Wrap as ClassifierFreeSampleWrapper
    sample_model = ClassifierFreeSampleWrapper(model, scale=args.guidance_param)

    device = get_device()
    sample_model.to(device)
    sample_model.eval()
    print(f"✅ Model loaded to {device}")
    print(f"   Joints: {model.njoints}, Feats: {model.nfeats}")

    # 3. Load datasets
    print(f"\nStep 3: Loading datasets...")
    train_dataset = instantiate(cfg.data, split='train', debug=False)
    val_dataset = instantiate(cfg.data, split='val', debug=False)
    print(f"✅ Train set size: {len(train_dataset)}")
    print(f"✅ Val set size: {len(val_dataset)}")

    # 4. Create Evaluator
    print(f"\nStep 4: Creating Evaluator...")

    # Adjust batch_size to be divisible by num_samples_per_condition
    adjusted_batch_size = args.batch_size
    if adjusted_batch_size % args.num_samples_per_condition != 0:
        adjusted_batch_size = (adjusted_batch_size // args.num_samples_per_condition) * args.num_samples_per_condition
        if adjusted_batch_size == 0:
            adjusted_batch_size = args.num_samples_per_condition
        print(f"⚠️  Adjusting batch_size: {args.batch_size} → {adjusted_batch_size} (divisible by {args.num_samples_per_condition})")

    evaluator = EvaluatorWithSave(
        sample_model=sample_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataloader_cfg=DataLoaderConfig(
            batch_size=adjusted_batch_size,
            num_workers=args.num_workers,
            shuffle=False  # No shuffle for evaluation
        ),
        sample_fn=diffusion.p_sample_loop,
        nfeats=model.nfeats,
        njoints=model.njoints,
        sample_length=cfg.data.fixed_length if cfg.data.fixed_length > 0 else cfg.data.max_length,
        num_samples_on_train=args.num_samples_on_train,
        num_samples_on_val=args.num_samples_on_val,
        num_samples_per_condition=args.num_samples_per_condition,
        save_data=args.save_pkl is not None  # Save data if pkl path specified
    )
    print(f"✅ Evaluator created")

    # 5. Run evaluation
    print(f"\n{'='*70}")
    print(f"Step 5: Starting evaluation (on {args.split.upper()} set)...")
    print(f"{'='*70}")

    if args.split == 'train':
        num_samples = args.num_samples_on_train
    else:
        num_samples = args.num_samples_on_val

    num_conditions_for_multi = num_samples // args.num_samples_per_condition

    print(f"\nEvaluation configuration:")
    print(f"  - FID/Diversity: Sample {num_samples} conditions, generate 1 time each")
    print(f"  - Multimodality: Sample {num_conditions_for_multi} conditions, generate {args.num_samples_per_condition} times each")
    print(f"\nStarting generation and computation...\n")

    result = evaluator.evaluate(split=args.split)

    # 6. Save pkl if requested
    if args.save_pkl and result is not None:
        print(f"\nStep 6: Saving generated data to pkl...")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(args.save_pkl)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        # Convert list of tensors to list of numpy arrays
        def tensors_to_numpy(tensor_list):
            """Convert list of tensors to list of numpy arrays"""
            return [t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in tensor_list]

        eval_data = {
            'metadata': {
                'timestamp': timestamp,
                'model_path': str(args.model_path),
                'split': args.split,
                'num_samples': num_samples,
                'num_samples_per_condition': args.num_samples_per_condition,
                'guidance_param': args.guidance_param,
                'data_repr': cfg.data.repr,
            },
            'metrics': result,
            'gt_motions': tensors_to_numpy(evaluator.saved_gt_motions),
            'sample_motions': tensors_to_numpy(evaluator.saved_sample_motions),
            'sample_motions_multi': tensors_to_numpy(evaluator.saved_sample_motions_multi),
            'masks': tensors_to_numpy(evaluator.saved_masks),
            'masks_multi': tensors_to_numpy(evaluator.saved_masks_multi),
        }

        # Save to pkl
        pkl_filename = f"eval_{args.split}_samples{num_samples}_{timestamp}.pkl"
        pkl_path = save_dir / pkl_filename

        with open(pkl_path, 'wb') as f:
            pickle.dump(eval_data, f)

        file_size_mb = pkl_path.stat().st_size / 1024 / 1024
        print(f"✅ Data saved to: {pkl_path}")
        print(f"   File size: {file_size_mb:.2f} MB")

    # 7. Print results
    print(f"\n{'='*70}")
    print(f"Evaluation Results ({args.split.upper()} set)")
    print(f"{'='*70}")

    if result is None:
        print("⚠️  Current process is not main process, no results available")
    else:
        for key, value in sorted(result.items()):
            metric_name = key.replace(f'{args.split}_', '').upper()
            print(f"{metric_name:15s}: {value:8.4f}")

        print(f"{'='*70}")
        print(f"\nMetric interpretation:")
        print(f"  - FID (lower is better):        Distance between generated and real distributions")
        print(f"  - Diversity (higher is better): Diversity of generations for different conditions")
        print(f"  - Multimodality (moderate):     Diversity of multiple generations for same condition")
        print(f"{'='*70}\n")

    # Calculate and display total runtime
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print(f"{'='*70}")
    print(f"Total Runtime")
    print(f"{'='*70}")
    if hours > 0:
        print(f"Time elapsed: {hours}h {minutes}m {seconds:.2f}s ({total_time:.2f}s)")
    elif minutes > 0:
        print(f"Time elapsed: {minutes}m {seconds:.2f}s ({total_time:.2f}s)")
    else:
        print(f"Time elapsed: {seconds:.2f}s")
    print(f"{'='*70}\n")

    print("✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
