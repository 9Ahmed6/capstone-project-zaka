import os
import wandb
from omegaconf import OmegaConf

from .. import dist as dist_utils

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass

class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f"{group_name}/{name}" if group_name else name, value, iteration)

    def close(self):
        self.writer.close()

class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir):
        if not dist_utils.is_main_process():
            return
        self.run = wandb.init(
            dir=save_dir,
            project='BihandMotionGeneration_FINAL'
        )

        self.run.log_code(name='source_code')

    def report_scalar(self, name, value, iteration, group_name=None):
        if not dist_utils.is_main_process():
            return
        log_key = f"{group_name}/{name}" if group_name else name
        wandb.log({log_key:value}, step=iteration)

    def report_args(self, args, name):
        if not dist_utils.is_main_process():
            return
        try:
            args_dict = OmegaConf.to_container(args, resolve=True)

            wandb.config.update({name: args_dict})
        except Exception as e:
            raise ValueError(f'WandbPlatform: Could not log args {name}. Error: {e}')

    def report_video(self, name, video_path, video_format, group_name=None):
        if not dist_utils.is_main_process():
            return
        log_key = f"{group_name}/{name}" if group_name else name
        wandb.log({
            log_key: wandb.Video(video_path, format=video_format)
        })

    def close(self):
        if not dist_utils.is_main_process():
            return
        if self.run:
            wandb.finish()
            self.run = None


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass

