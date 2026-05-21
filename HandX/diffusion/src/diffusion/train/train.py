import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil, sys, logging, yaml, hydra
from pathlib import Path
from os.path import join as pjoin
from omegaconf import OmegaConf
from copy import deepcopy

from ..utils.mics import fixseed
from ..config import Config
from ...utils import start_debug
from .train_platforms import *
from ..logger_new import mylogger
from ..utils.model_utils import create_model_and_diffusion
from .training_loop import TrainLoop
from .. import dist as dist_utils


@hydra.main(config_path=pjoin(os.getcwd(), 'conf'), config_name='config', version_base=None)
def main(cfg: Config):
    # start_debug()
    dist_utils.init_distributed_mode(backend='nccl')

    if cfg.train.resume_checkpoint != '':
        old_config_path = Path(cfg.train.resume_checkpoint).parent / "config.yaml"
        with open(old_config_path.as_posix(), "r") as f:
            old_config:Config = OmegaConf.load(f)

        if cfg.train.save_dir == old_config.train.save_dir:
            raise ValueError("Cannot resume from the same save_dir as the old config.")
        old_config.train.save_dir = cfg.train.save_dir
        old_config.train.resume_checkpoint = cfg.train.resume_checkpoint

        cfg = deepcopy(old_config)

    fixseed(cfg.seed)

    if dist_utils.is_main_process():

        if cfg.train.save_dir is None:
            raise FileNotFoundError("save_dir was not specified.")
        elif Path(cfg.train.save_dir).exists() and not cfg.train.overwrite:
            raise FileExistsError(f"save_dir {cfg.train.save_dir} already exists.")
        elif Path(cfg.train.save_dir).exists() and cfg.train.overwrite:
            while True:
                # user_input = input(f"Do you want to overwrite the existing exp dir {args.save_dir}? (yes/no): ")
                user_input = 'yes'
                if user_input.lower() == 'yes':
                    for entry in Path(cfg.train.save_dir).iterdir():
                        if entry.is_dir() and entry.name == 'hydra_log':
                            continue
                        if entry.is_file():
                            os.remove(entry.as_posix())
                        elif entry.is_dir():
                            shutil.rmtree(entry.as_posix())
                    break
                elif user_input.lower() == 'no':
                    sys.exit(0)

        elif not Path(cfg.train.save_dir).exists():
            Path(cfg.train.save_dir).mkdir(parents=True, exist_ok=True)


    dist_utils.barrier()
    if dist_utils.is_main_process():
        mylogger.setup(
            logger_name='MyLogger_MASTER',
            file_log_level=logging.DEBUG,
            stream_log_level=logging.DEBUG,
            log_file=(Path(cfg.train.save_dir) / 'train_master.log').as_posix(),
        )
    else:
        mylogger.setup(
            logger_name=f"MyLogger_RANK{dist_utils.get_rank()}",
            file_log_level=logging.INFO,
            stream_log_level=logging.INFO,
            log_file=(Path(cfg.train.save_dir) / f'train_rank{dist_utils.get_rank()}.log').as_posix(),
        )



    if dist_utils.is_main_process():
        args_path = (Path(cfg.train.save_dir) / 'config.yaml').as_posix()
        with open(args_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    try:
        train_platform_type = eval(cfg.train.train_platform_type)
        train_platform = train_platform_type(cfg.train.save_dir)
        train_platform.report_args(cfg, name='Args')

        mylogger.info("Creating model and diffusion ...")
        model, diffusion = create_model_and_diffusion(cfg.model)
        mylogger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

        # Print treble mask probability
        if hasattr(model, 'treble_mask_prob'):
            mylogger.info(f"Treble mask probability: {model.treble_mask_prob}")
            if model.treble_mask_prob == 1.0:
                mylogger.info("  -> No masking (keep all treble branches)")
            elif model.treble_mask_prob == 0.5:
                mylogger.info("  -> 50% masking (each branch has 50% chance to be masked)")
            else:
                mylogger.info(f"  -> {(1-model.treble_mask_prob)*100:.0f}% masking probability")

        mylogger.info("Training...")
        TrainLoop(cfg, train_platform, model, diffusion).run_loop()
        train_platform.close()

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Cleaning up ...")

    except Exception as e:
        mylogger.error(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if dist_utils.is_dist_avail_and_initialized():
            dist_utils.cleanup()


if __name__ == "__main__":
    main()
