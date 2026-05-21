import os
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import hydra
from hydra.utils import instantiate
import torch
from einops import rearrange
from omegaconf import OmegaConf

from .model.cls_free_sampler import ClassifierFreeSampleWrapper
from .utils.mics import get_device, rot_motion_to_dict, fixseed
from .utils.model_utils import create_model_and_diffusion
from ..constant import gesture_list
from .config import DataConfig, ModelConfig, TextConditionModelConfig, ActionConditionModelConfig, GenerateConfig, Config
from ..visualize.visualize import MultiMotionVisualizer


def generate(
    gen_cfg: GenerateConfig,
    model_cfg: ModelConfig | ActionConditionModelConfig | TextConditionModelConfig,
    data_cfg: DataConfig,
    seed: int | None = None,
):
    if seed:
        fixseed(seed)
    torch.set_float32_matmul_precision("high")
    model, diffusion = create_model_and_diffusion(model_cfg)
    state_dict = torch.load(gen_cfg.model_path, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model = ClassifierFreeSampleWrapper(model, scale=gen_cfg.sample.guidance_param)
    device = get_device()
    model.to(device)
    model.eval()

    dataset = instantiate(data_cfg, split='train')

    batch_size:int = gen_cfg.num_samples
    if data_cfg.repr == 'joint_pos':
        njoints = 42
        nfeats = 3
    elif data_cfg.repr == 'joint_rot':
        njoints = 34
        nfeats = 6
    elif data_cfg.repr == 'joint_pos_w_scalar_rot':
        njoints = 42
        nfeats = 4
    shape = (batch_size, njoints, nfeats, gen_cfg.motion_length)

    model_kwargs = dict(
        y=dict(
            lengths=torch.as_tensor([gen_cfg.motion_length] * batch_size, device=device)
        )
    )
    if model_cfg.cond_mode == 'text':
        if model_cfg.arch in ['trans_enc', "trans_dec"]:
            model_kwargs['y'].update(
                text=[gen_cfg.text_prompt] * batch_size
            )
        elif model_cfg.arch == "trans_dec_treble_concat" or model_cfg.arch == 'trans_dec_treble_residual':
            print(type(gen_cfg.text_prompt))
            assert len(gen_cfg.text_prompt) == 3, \
                "For treble models, text_prompt should be a tuple/list of 3 strings (left, right, two hands relation)."
            model_kwargs['y'].update(
                text=dict(
                    left=[gen_cfg.text_prompt[0]] * batch_size,
                    right=[gen_cfg.text_prompt[1]] * batch_size,
                    two_hands_relation=[gen_cfg.text_prompt[2]] * batch_size,
                )
            )

    elif model_cfg.cond_mode == 'action':
        assert gen_cfg.action_name in gesture_list, f"action {gen_cfg.action_name} not in {gesture_list}"
        action_id = gesture_list.index(gen_cfg.action_name)
        actions = torch.ones(batch_size, dtype=torch.long, device=device) * action_id
        model_kwargs['y'].update(
            actions=actions
        )
    elif model_cfg.cond_mode != 'no_cond':
        raise ValueError(f"cond_mode {model_cfg.cond_mode} not recognized.")

    samples = diffusion.p_sample_loop(
        model, shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        noise=None,
        const_noise=False
    )

    samples = rearrange(samples, 'b j f t -> b t (j f)')
    samples = dataset.inv_transform(samples.detach().cpu().numpy())

    def process_motion(motion, title=None):
        '''
        motion: (B, T, J*F)
        '''
        left_motion, right_motion = np.split(
            motion.reshape(-1, njoints, nfeats),
            indices_or_sections=[njoints // 2],
            axis=1
        )
        cur_motion_to_visualize = dict()
        if data_cfg.repr == 'joint_pos':
            cur_motion_to_visualize.update(
                dict(
                    type='skeleton',
                    left_motion=left_motion,
                    right_motion=right_motion
                )
            )
        elif data_cfg.repr == 'joint_rot':
            left_motion = rot_motion_to_dict(left_motion)
            right_motion = rot_motion_to_dict(right_motion)

            cur_motion_to_visualize.update(
                type='mano',
                left_motion=left_motion,
                right_motion=right_motion
            )
        elif data_cfg.repr == 'joint_pos_w_scalar_rot':
            cur_motion_to_visualize.update(
                dict(
                    type='skeleton',
                    left_motion=left_motion[:, :, :3],
                    right_motion=right_motion[:, :, :3],
                )
            )

        if title is not None:
            cur_motion_to_visualize.update(title=title)

        return cur_motion_to_visualize

    motions_to_visualize = []
    for i in range(batch_size):
        cur_sample = samples[i]
        title = f"Sample {i + 1} - Length: {cur_sample.shape[0]}"
        motions_to_visualize.append(process_motion(cur_sample, title=title))


    text_to_show = ""
    if model_cfg.cond_mode == 'text':
        if isinstance(gen_cfg.text_prompt, str):
            text_to_show = gen_cfg.text_prompt
        else:
            text_to_show = "[LEFT] " + gen_cfg.text_prompt[0] + "  [RIGHT] " + gen_cfg.text_prompt[1] + "  [TWO HANDS RELATION] " + gen_cfg.text_prompt[2]
    elif model_cfg.cond_mode == 'action':
        text_to_show = gen_cfg.action_name
    MultiMotionVisualizer.create_3d_animation(
        motions=motions_to_visualize,
        text=text_to_show,
        save_path=pjoin(gen_cfg.output_dir, "generated_motion.gif"),
        fps=30
    )

@hydra.main(config_path=pjoin(os.getcwd(), "conf"), config_name='generate', version_base=None)
def main(gen_cfg: GenerateConfig):
    exp_config_path = Path(gen_cfg.model_path).parent / "config.yaml"
    with open(exp_config_path.as_posix(), "r") as f:
        exp_config:Config = OmegaConf.load(f)

    generate(
        gen_cfg,
        exp_config.model,
        exp_config.data,
        exp_config.seed
    )

if __name__ == "__main__":
    main()