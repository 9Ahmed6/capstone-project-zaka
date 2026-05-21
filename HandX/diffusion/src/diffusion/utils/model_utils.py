import torch

from ..model.mdm import MotionDiffusionModel
from .. import gaussian_diffusion as gd
from ..respace import SpacedDiffusion, space_timesteps

from ..config import ModelConfig, ActionConditionModelConfig, TextConditionModelConfig, DiffusionConfig


def get_model_args(cfg:ModelConfig | ActionConditionModelConfig | TextConditionModelConfig):
    ret = dict(
        arch=cfg.arch,
        latent_dim=cfg.latent_dim,
        num_heads=cfg.num_heads,
        ff_size=cfg.ff_size,
        dropout=cfg.dropout,
        activation=cfg.activation,
        num_layers=cfg.layers,
        cond_mode=cfg.cond_mode,
        cond_mask_prob=cfg.cond_mask_prob if hasattr(cfg, 'cond_mask_prob') else None,
        contact_prediction=cfg.contact_prediction if hasattr(cfg, 'contact_prediction') else None,
        treble_mask_prob=cfg.treble_mask_prob if hasattr(cfg, 'treble_mask_prob') else 1.0
    )
    if cfg.repr == 'joint_pos':
        ret.update(
            njoints=42,
            nfeats=3,
        )
    elif cfg.repr == 'joint_pos_w_scalar_rot':
        ret.update(
            njoints=42,
            nfeats=4
        )
    elif cfg.repr == 'joint_pos_w_axisangle_rot':
        ret.update(
            njoints=42,
            nfeats=6
        )
    elif cfg.repr == 'joint_rot':
        ret.update(
            njoints=34,
            nfeats=6,
        )


    if cfg.cond_mode == 'text':
        ret.update(
            text_model_name=cfg.text_model,
            text_max_length=cfg.max_text_length
        )
    elif cfg.cond_mode == 'action':
        ret.update(num_actions=cfg.num_actions)
    return ret

def create_model_and_diffusion(cfg:ModelConfig | ActionConditionModelConfig | TextConditionModelConfig):
    model = MotionDiffusionModel(
        **get_model_args(cfg)
    )

    diffusion = create_gaussian_diffusion(cfg.diffusion, predict_contact=cfg.contact_prediction)
    return model, diffusion

def create_gaussian_diffusion(cfg:DiffusionConfig, predict_contact:bool=False):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = cfg.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(cfg.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE
    if cfg.repr == 'joint_rot':
        repr_type = gd.ReprType.JOINT_ROT_6D
        metric_types = [gd.MetricType.JOINT_POS_ERROR, gd.MetricType.JOINT_ROT_ERROR]
    elif cfg.repr == 'joint_pos':
        repr_type = gd.ReprType.JOINT_POS
        metric_types = [gd.MetricType.JOINT_POS_ERROR]
    elif cfg.repr == 'joint_pos_w_scalar_rot':
        repr_type = gd.ReprType.JOINT_POS_W_SCALAR_ROT
        metric_types = [gd.MetricType.JOINT_POS_ERROR, gd.MetricType.JOINT_ROT_ERROR]
    elif cfg.repr == 'joint_pos_w_axisangle_rot':
        repr_type = gd.ReprType.JOINT_POS_W_AXISANGLE_ROT
        metric_types = [gd.MetricType.JOINT_POS_ERROR, gd.MetricType.JOINT_ROT_ERROR]
    else:
        raise ValueError(f"Unknown representation type: {cfg.repr}")

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        repr_type=repr_type,
        metric_types=metric_types,
        rescale_timesteps=rescale_timesteps,
        lambda_repr=cfg.lambda_repr,
        lambda_vel=cfg.lambda_vel,
        lambda_acce=cfg.lambda_acce,
        contact_loss=cfg.contact_loss,
        lambda_contact=cfg.lambda_contact if cfg.contact_loss else 0.,
        contact_predict_loss= predict_contact,
        lambda_contact_predict=cfg.lambda_contact_predict if predict_contact else 0.,
        lambda_rcxyz=cfg.lambda_rcxyz,
        lambda_fc=cfg.lambda_fc,
        lambda_ig=cfg.lambda_ig,
        lambda_w_ig=cfg.lambda_w_ig,
    )
