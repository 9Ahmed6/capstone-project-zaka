import os
import time
from types import SimpleNamespace
from typing import Dict
from pathlib import Path
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
from hydra.utils import instantiate

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange


# from .. import logger
from ..config import Config, DataLoaderConfig
from ..logger_new import mylogger
from .train_platforms import TrainPlatform
from ..gaussian_diffusion import GaussianDiffusion
from ..model.evaluator import Evaluator
from ..utils.mics import get_device
from ..resample import create_named_schedule_sampler, LossAwareSampler
from ..data_loader.get_data import get_dataloader
from ...utils import debug
from .. import dist as dist_utils
from ...constant import gesture_list
from ..model.cls_free_sampler import ClassifierFreeSampleWrapper
from ...visualize.visualize import MultiMotionVisualizer


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        cfg:Config,
        train_platform:TrainPlatform,
        model:torch.nn.Module,
        diffusion:GaussianDiffusion,
    ):
        self.cfg = cfg
        # self.dataset = cfg.dataset
        self.train_platform = train_platform
        self.diffusion = diffusion
        self.repr = cfg.data.repr
        self.step = 0
        self.resume_step = None
        self.lr = cfg.train.optimizer.lr
        self.log_interval = cfg.train.log_interval
        self.save_interval = cfg.train.save_interval
        self.weight_decay = cfg.train.optimizer.weight_decay
        if cfg.data.fixed_length > 0:
            self.sample_length = cfg.data.fixed_length
        else:
            self.sample_length = cfg.data.max_length
        self.save_dir = cfg.train.save_dir
        self.num_steps = cfg.train.num_steps

        torch.set_float32_matmul_precision("high")

        self.model = model
        self.sample_model = ClassifierFreeSampleWrapper(model, scale=cfg.train.sample.guidance_param)
        self.resume_checkpoint = cfg.train.resume_checkpoint
        self._load_and_sync_parameters()

        self.device = get_device()
        self.model.to(self.device)

        self.set_ddp()

        mylogger.info("Creating dataset ...")
        # self.data = get_dataset_loader('train', cfg, cfg.batch_size, shuffle=True)
        self.train_dataset = instantiate(cfg.data, split='train', debug=False)
        mylogger.info("Creating dataloader ...")
        self.train_dataloader = get_dataloader(self.train_dataset, cfg.train.dataloader)
        self.num_epochs = self.num_steps // len(self.train_dataloader) + 1

        self._add_model_params_to_optimizer()
        if self.resume_step:
            self._load_optimizer_state()

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        # TODO: Evaluation Module
        # if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
        #     mm_num_samples = 0  # mm is super slow hence we won't run it during training
        #     mm_num_repeats = 0  # mm is super slow hence we won't run it during training
        #     gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                     split=args.eval_split,
        #                                     hml_mode='eval')

        #     self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                            split=args.eval_split,
        #                                            hml_mode='gt')
        #     self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
        #     self.eval_data = {
        #         'test': lambda: eval_humanml.get_mdm_loader(
        #             model, diffusion, args.eval_batch_size,
        #             gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
        #             args.eval_num_samples, scale=1.,
        #         )
        #     }

        self.val_during_training = cfg.train.val_during_training
        if self.val_during_training:
            self.val_interval = cfg.train.val_cfg.val_interval
            self.val_dataset = instantiate(cfg.data, split='val', debug=False)
            self.val_dataloader = get_dataloader(self.val_dataset, cfg.train.val_cfg.dataloader)

        self.eval_during_training = cfg.train.eval_during_training
        if self.eval_during_training:
            self.eval_interval = cfg.train.eval_cfg.eval_interval
            self.eval_dataset_on_train = self.train_dataset
            self.eval_dataset_on_val = self.val_dataset if hasattr(self, 'val_dataset') else instantiate(cfg.data, split='val', debug=False)
            self.evaluate_helper = Evaluator(
                sample_model=self.sample_model,
                train_dataset=self.eval_dataset_on_train,
                val_dataset=self.eval_dataset_on_val,
                dataloader_cfg=cfg.train.eval_cfg.dataloader,
                sample_fn=self.diffusion.p_sample_loop,
                nfeats=self.model_without_ddp.nfeats, njoints=self.model_without_ddp.njoints,
                sample_length=self.sample_length,
                num_samples_on_train=cfg.train.eval_cfg.num_samples_on_train,
                num_samples_on_val=cfg.train.eval_cfg.num_samples_on_val,
                num_samples_per_condition=cfg.train.eval_cfg.num_samples_per_condition,
            )

        self.viz_during_training = cfg.train.viz_during_training
        if self.viz_during_training:
            self.viz_dataloader = get_dataloader(self.train_dataset, DataLoaderConfig(batch_size=dist_utils.get_world_size(), num_workers=0, shuffle=True))
            self.iter_viz_data = iter(self.viz_dataloader)
            self.viz_samples_count = cfg.train.viz_cfg.samples_count

    def _add_model_params_to_optimizer(self):
        params_dict = {
            pn: p
            for pn, p in self.model.named_parameters()
            if p.requires_grad
        }

        decay_params = [
            p
            for n, p in params_dict.items()
            if p.dim() >= 2
        ]

        nodecay_params = [
            p
            for n, p in params_dict.items()
            if p.dim() < 2
        ]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        self.opt = AdamW(
            optim_groups,
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True
        )

    def _remove_module_prefix(self, state_dict):
        if any(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v

            return new_state_dict
        return state_dict

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(Path(resume_checkpoint).name)
            self.step = self.resume_step
            mylogger.info(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = torch.load(
                resume_checkpoint, map_location='cpu'
            )
            state_dict = self._remove_module_prefix(state_dict['state_dict'])
            self.model.load_state_dict(state_dict, strict=False)

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if os.path.exists(opt_checkpoint):
            mylogger.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(
                opt_checkpoint, map_location='cpu'
            )
            self.opt.load_state_dict(state_dict)

    def set_ddp(self):
        self.model.to(self.device)
        if dist_utils.is_dist_avail_and_initialized():
            self.model = DDP(self.model, device_ids=[self.device.index], output_device=self.device.index, static_graph=True)
        self.model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            if dist_utils.is_dist_avail_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            for motion, cond in tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1} / {self.num_epochs} RANK {dist_utils.get_rank()}'):

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}


                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for key in mylogger.get_all_metrics().keys():
                        if key.startswith('val_'):
                            continue
                        if key.startswith('train_'):
                            mylogger.debug(f'step [{self.step}]: {key} [{mylogger.get_average(key):0.5f}]')

                        if '_q1' not in key and '_q2' not in key and '_q3' not in key and '_q0' not in key:
                            self.train_platform.report_scalar(
                                name=key,
                                value=mylogger.get_average(key),
                                iteration=self.step,
                                group_name='Train'
                            )
                        mylogger.clear_metric(key)

                    self.train_platform.report_scalar(
                        name='learning_rate',
                        value=self.opt.param_groups[0]['lr'],
                        iteration=self.step,
                        group_name='Train',
                    )

                if self.val_during_training and self.step % self.val_interval == 0:
                    # calculate validation loss
                    with torch.no_grad():
                        for motion, cond in tqdm(self.val_dataloader, desc=f'Validation on val set | RANK {dist_utils.get_rank()}'):
                            motion = motion.to(self.device)
                            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                            self.validation_forward(motion, cond, split='val')

                        for motion, cond in tqdm(self.train_dataloader, desc=f'Validation on train set | RANK {dist_utils.get_rank()}'):
                            motion = motion.to(self.device)
                            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                            self.validation_forward(motion, cond, split='train')

                    for key in mylogger.get_all_metrics().keys():
                        if not key.startswith('val_'):
                            continue

                        mylogger.debug(f'step [{self.step}]: {key} [{mylogger.get_average(key):0.5f}]')
                        if '_q0' not in key and '_q1' not in key and '_q2' not in key and '_q3' not in key:
                            self.train_platform.report_scalar(
                                name=key,
                                value=mylogger.get_average(key),
                                iteration=self.step,
                                group_name='Validation'
                            )
                        mylogger.clear_metric(key)

                if self.eval_during_training and self.step % self.eval_interval == 0:
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                if self.step % self.save_interval == 0:
                    self.save()

                    if self.viz_during_training:
                        for sample_id in range(self.viz_samples_count):
                            samples, gt, lengths, label, mse_losses, vel_losses = self.sample()
                            gt_single = gt[0]
                            length = lengths[0]
                            mse_loss = mse_losses[0]
                            vel_loss = vel_losses[0]

                            # Handle both dict (treble text) and list (normal text) label formats
                            if isinstance(label, dict):
                                # For treble text, create a combined label string
                                label_str = f"step: {self.step}_{dist_utils.get_rank() * self.viz_samples_count + sample_id} mse_loss: {mse_loss:.6f} vel: {vel_loss:.6f} \n\n[LEFT] {label['left'][0]} \n[RIGHT] {label['right'][0]} \n[RELATION] {label['two_hands_relation'][0]}"
                            else:
                                label_str = label[0]

                            if self.repr == 'joint_rot':
                                from ..utils.mics import rot_motion_to_dict

                            def process_motion(motion, title=None):
                                left_motion, right_motion = np.split(
                                    motion.reshape(motion.shape[0], self.model_without_ddp.njoints, self.model_without_ddp.nfeats),
                                    indices_or_sections=[self.model_without_ddp.njoints // 2],
                                    axis=1
                                )  # (T, J_single, D), (T, J_single, D)
                                cur_motion_to_visualize = dict()
                                if self.repr == 'joint_pos':
                                    cur_motion_to_visualize.update(
                                        dict(
                                            type='skeleton',
                                            left_motion=left_motion,
                                            right_motion=right_motion,
                                        )
                                    )
                                elif self.repr in ['joint_pos_w_scalar_rot', 'joint_pos_w_axisangle_rot']:
                                    cur_motion_to_visualize.update(
                                        dict(
                                            type='skeleton',
                                            left_motion=left_motion[:, :, :3],
                                            right_motion=right_motion[:, :, :3],
                                        )
                                    )

                                elif self.repr == 'joint_rot':
                                    left_motion = rot_motion_to_dict(left_motion)
                                    right_motion = rot_motion_to_dict(right_motion)

                                    cur_motion_to_visualize.update(
                                        type='mano',
                                        left_motion=left_motion,
                                        right_motion=right_motion,
                                    )
                                if title is not None:
                                    cur_motion_to_visualize['title'] = title

                                return cur_motion_to_visualize

                            # GIF 1: Original - GT + all denoising steps
                            motions_to_visualize = []
                            motions_to_visualize.append(process_motion(gt_single[:length], title='Ground Truth'))
                            for i in range(len(samples)):
                                cur_sample = samples[i, :length]
                                if self.cfg.train.viz_cfg.denoising_steps and len(self.cfg.train.viz_cfg.denoising_steps) > 0:
                                    title = f"Denoising Step: {self.cfg.train.viz_cfg.denoising_steps[i]}"
                                else:
                                    title = None

                                motions_to_visualize.append(
                                    process_motion(cur_sample, title=title)
                                )

                            MultiMotionVisualizer.create_3d_animation(
                                motions=motions_to_visualize,
                                text=label_str,
                                save_path=pjoin(self.save_dir, f'step_{self.step}_{dist_utils.get_rank() * self.viz_samples_count + sample_id}_all_steps.gif'),
                                fps=30
                            )

                            dist_utils.barrier()

                            for rank in range(dist_utils.get_world_size()):
                                self.train_platform.report_video(
                                    name=f'video_{rank}',
                                    video_path=pjoin(self.save_dir, f'step_{self.step}_{rank * self.viz_samples_count + sample_id}_all_steps.gif'),
                                    video_format='gif',
                                    group_name='Train'
                                )


                self.step += 1



        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.cfg.train.eval_during_training:
            return
        eval_result = self.evaluate_helper.evaluate(split='train')
        eval_result_on_val = self.evaluate_helper.evaluate(split='val')

        if dist_utils.is_main_process():
            eval_result.update(eval_result_on_val)
            for key in eval_result.keys():
                mylogger.debug(f"EVALUATION {key}: [{eval_result[key]:0.5f}]")
                self.train_platform.report_scalar(
                    name=key,
                    value=eval_result[key],
                    iteration=self.step,
                    group_name='Evaluation'
                )
        dist_utils.barrier()

        # start_eval = time.time()
        # if self.eval_wrapper is not None:
        #     print('Running evaluation loop: [Should take about 90 min]')
        #     log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step):09d}.log')
        #     diversity_times = 300
        #     mm_num_times = 0  # mm is super slow hence we won't run it during training
        #     eval_dict = eval_humanml.evaluation(
        #         self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
        #         replication_times=self.cfg.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
        #     print(eval_dict)
        #     for k, v in eval_dict.items():
        #         if k.startswith('R_precision'):
        #             for i in range(len(v)):
        #                 self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
        #                                                   iteration=self.step,
        #                                                   group_name='Eval')
        #         else:
        #             self.train_platform.report_scalar(name=k, value=v, iteration=self.step,
        #                                               group_name='Eval')

        # elif self.dataset in ['humanact12', 'uestc']:
        #     eval_args = SimpleNamespace(num_seeds=self.cfg.eval_rep_times, num_samples=self.cfg.eval_num_samples,
        #                                 batch_size=self.cfg.eval_batch_size, device=self.device, guidance_param = 1,
        #                                 dataset=self.dataset, unconstrained=self.cfg.unconstrained,
        #                                 model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
        #     eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.train_dataloader.dataset)
        #     print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
        #     for k, v in eval_dict["feats"].items():
        #         if 'unconstrained' not in k:
        #             self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
        #         else:
        #             self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        # end_eval = time.time()
        # print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    def compute_reconstruction_loss(self, samples, gt, lengths):
        """
        Compute MSE loss and velocity loss between samples and ground truth.

        Args:
            samples: numpy array of shape (B, T, D) or (N_samples, T, D)
            gt: numpy array of shape (B, T, D)
            lengths: tensor or numpy array of shape (B,)

        Returns:
            mse_losses: numpy array of shape (B,) containing per-sample MSE loss
            vel_losses: numpy array of shape (B,) containing per-sample velocity MSE loss
        """
        # If samples has multiple denoising steps, take the last one
        if samples.ndim == 3 and gt.ndim == 3:
            # samples: (N_samples, T, D), gt: (B, T, D)
            # This happens when we have multiple denoising steps but batch_size=1
            samples = samples[-1:, :, :]  # Take the final sample, shape: (1, T, D)
        elif samples.ndim == 4:
            # samples: (N_samples, B, T, D), gt: (B, T, D)
            samples = samples[-1]  # Take the final sample, shape: (B, T, D)

        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu().numpy()

        # Ensure samples is at least 2D
        if samples.ndim == 2:
            samples = samples[np.newaxis, :, :]  # (1, T, D)

        batch_size = gt.shape[0]
        mse_losses = []
        vel_losses = []

        for i in range(batch_size):
            length = int(lengths[i])
            sample = samples[i, :length]  # (T, D)
            gt_seq = gt[i, :length]  # (T, D)

            # MSE loss
            mse = np.mean((sample - gt_seq) ** 2)
            mse_losses.append(mse)

            # Velocity MSE loss
            if length > 1:
                sample_vel = sample[1:] - sample[:-1]  # (T-1, D)
                gt_vel = gt_seq[1:] - gt_seq[:-1]  # (T-1, D)
                vel_mse = np.mean((sample_vel - gt_vel) ** 2)
            else:
                vel_mse = 0.0
            vel_losses.append(vel_mse)

        return np.array(mse_losses), np.array(vel_losses)

    def sample(self):
        sample_fn = self.diffusion.p_sample_loop
        # x, model_kwargs = next(iter(self.viz_data))
        try:
            x, model_kwargs = next(self.iter_viz_data)
        except:
            self.iter_viz_data = iter(self.viz_dataloader)
            x, model_kwargs = next(self.iter_viz_data)
        # set model to eval mode
        self.sample_model.eval()

        dump_steps = None
        if self.cfg.train.viz_cfg.denoising_steps and len(self.cfg.train.viz_cfg.denoising_steps) > 0:
            dump_steps = self.cfg.train.viz_cfg.denoising_steps

        samples = sample_fn(
            self.sample_model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (x.shape[0], self.model_without_ddp.njoints, self.model_without_ddp.nfeats, self.sample_length),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=self.device,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=dump_steps,
            noise=None,
            const_noise=False,
        )

        if isinstance(samples, list):
            samples = torch.concat(samples, dim=0)

        gt = x
        samples = rearrange(samples, 'b j f t -> b t (j f)')
        gt = rearrange(gt, 'b j f t -> b t (j f)')

        # Compute losses in NORMALIZED space (before inv_transform)
        samples_normalized = samples.detach().cpu().numpy()
        gt_normalized = gt.detach().cpu().numpy()
        mse_losses, vel_losses = self.compute_reconstruction_loss(samples_normalized, gt_normalized, model_kwargs['y']['lengths'])

        # Then apply inv_transform for visualization
        samples = self.train_dataloader.dataset.inv_transform(samples_normalized)
        gt = self.train_dataloader.dataset.inv_transform(gt_normalized)

        self.model.train()
        action_label = model_kwargs['y'].get('action', None)
        if action_label is not None:
            action_label = [gesture_list[i] for i in action_label.cpu().numpy().reshape(-1)]
        if action_label is None:
            action_label = model_kwargs['y'].get('text', None)
        return samples, gt, model_kwargs['y']['lengths'], action_label, mse_losses, vel_losses

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        param_norm, grad_norm = 0.0, 0.0
        for p in self.model.parameters():
            with torch.no_grad():
                param_norm += torch.norm(p).item() ** 2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad).item() ** 2

        param_norm = np.sqrt(param_norm)
        grad_norm = np.sqrt(grad_norm)
        mylogger.record_metric("param_norm", param_norm)
        mylogger.record_metric("grad_norm", grad_norm)
        self.opt.step()

        clip_grad_norm_(self.model.parameters(), 1.0)

        # self._anneal_lr()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = self.diffusion.training_losses(
                self.model,
                batch,
                t,
                model_kwargs=cond,
                dataset=self.train_dataloader.dataset
            )
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses['loss'].detach()
            )

        loss = (losses['loss'] * weights).mean()

        log_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}, key_prefix="train_"
        )
        loss.backward()

    def validation_forward(self, batch, cond, split):
        self.model.eval()
        with torch.no_grad():
            t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                metrics = self.diffusion.evaluate_metrics(
                    self.model,
                    batch,
                    t,
                    model_kwargs=cond,
                    inv_transform=self.train_dataloader.dataset.inv_transform
                )

            log_dict(
                self.diffusion, t, {k: v * weights for k, v in metrics.items()}, key_prefix=f"val_on_{split}_"
            )
        self.model.train()


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr



    def ckpt_file_name(self):
        return f"model{(self.step):09d}.pt"


    def save(self):
        def save_checkpoint(model_state_dict):
            # Do not save text encoder weights
            checkpoint = dict()
            checkpoint['state_dict'] = dict()

            trainable_param_names = {
                name for name, param in self.model_without_ddp.named_parameters() if param.requires_grad
            }

            for key, value in model_state_dict.items():
                if not key.startswith("_text_model."):
                    checkpoint['state_dict'][key] = value
                elif key in trainable_param_names:
                    checkpoint['state_dict'][key] = value

            mylogger.info(f"saving model...")
            filename = self.ckpt_file_name()
            with open(os.path.join(self.save_dir, filename), "wb") as f:
                torch.save(checkpoint, f)


        dist_utils.barrier()
        if dist_utils.is_main_process():
            save_checkpoint(self.model_without_ddp.state_dict())

            with open(os.path.join(self.save_dir, f"opt{self.step:09d}.pt"), "wb") as f:
                torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0



def log_dict(diffusion : GaussianDiffusion, ts : torch.Tensor, losses : Dict[str, torch.Tensor], key_prefix: str = ""):
    for key, values in losses.items():
        mylogger.record_metric(key_prefix + key, values.mean().item())

        for sub_t, sub_loss in zip(ts.detach().cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            mylogger.record_metric(f"{key_prefix}{key}_q{quartile}", sub_loss)