# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

from typing import List

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List
import torch
import torch as th
from copy import deepcopy

from smplx.utils import MANOOutput
from einops import rearrange

from .nn import mean_flat, sum_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .utils import rotation_conversion

from ..visualize.mano2mesh import left_manomodel, right_manomodel
from .metric.interaction import intra_tip_pairs, all_tips, palm_peripheral_joints
# from data_loaders.humanml.scripts import motion_process

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class ReprType(enum.Enum):
    JOINT_POS = enum.auto()
    JOINT_POS_W_SCALAR_ROT = enum.auto()
    JOINT_POS_W_AXISANGLE_ROT = enum.auto()
    JOINT_ROT_6D = enum.auto()

class MetricType(enum.Enum):
    JOINT_ROT_ERROR = enum.auto()  # error in joint rotation
    JOINT_POS_ERROR = enum.auto() # error in joint position
    JOINT_SCALAR_ROT_ERROR = enum.auto() # error in joint scalar rotation
    JOINT_AXISANGLE_ROT_ERROR = enum.auto() # error in joint axis-angle rotation


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas : NDArray[np.floating],
        model_mean_type : ModelMeanType,
        model_var_type : ModelVarType,
        loss_type : LossType,
        repr_type : ReprType,
        metric_types : List[MetricType],
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_repr=0.,
        lambda_vel=0.,
        lambda_acce=0.,
        contact_loss=False,
        lambda_contact=0.,
        contact_predict_loss=False,
        lambda_contact_predict=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
        lambda_ig=0.,
        lambda_w_ig=0.,
    ):
        '''
        betas: 一个一维数组，定义了在每个时间步t添加的噪声量（方差）。这是扩散过程最核心的超参数。

        model_mean_type: 定义了神经网络模型预测的目标。通常是预测噪声 EPSILON 或预测原始数据 START_X。

        model_var_type: 定义了模型如何处理反向过程的方差。方差可以是固定的 (FIXED_SMALL/FIXED_LARGE) 或由模型学习 (LEARNED)。

        loss_type: 定义了训练时使用的损失函数类型，通常是均方误差 MSE。

        lambda_*: 一系列的权重系数，用于组合不同的损失项。例如 lambda_pose, lambda_loc, lambda_ig 分别代表姿态、位置和“交互图”（Interaction Graph）损失的权重。这表明该模型被用于特定领域（如人体动作生成），其损失函数是多个子目标的加权和。
        '''
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.repr_type = repr_type
        self.metric_types = metric_types
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_repr = lambda_repr
        self.lambda_vel = lambda_vel
        self.lambda_acce = lambda_acce
        self.contact_loss = contact_loss
        self.lambda_contact = lambda_contact
        self.contact_predict_loss = contact_predict_loss
        self.lambda_contact_predict = lambda_contact_predict
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        self.lambda_ig = lambda_ig # add by xiyan, interaction graph loss
        self.lambda_w_ig = lambda_w_ig

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0. or self.lambda_ig > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    def masked_average(self, x, mask):
        '''
        mask: (B, 1, 1, T)
        x: (B, J, 1, T)

        return: (B,)
        '''
        sum_x = sum_flat(x * mask.float()) # (B,)
        n_entries = x.shape[1]
        non_zero_elements = sum_flat(mask) * n_entries # (B,)

        average_x = sum_x / non_zero_elements
        return average_x

    def masked_ig(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        mask = mask.unsqueeze(1) # bs, 1, 1, 1, seqlen
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[1] * a.shape[2] * a.shape[3]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def masked_weighted_ig(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        # assuming a is the ground truth
        loss = self.l2_loss(a, b)
        mask = mask.unsqueeze(1) # bs, 1, 1, 1, seqlen

        # weight term of remos
        weight = torch.exp(-torch.norm(a, dim=-2, keepdim=True)) # bs, J, J, 1, seqlen

        loss = sum_flat(weight * loss * mask.float())
        n_entries = a.shape[1] * a.shape[2] * a.shape[3]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def compute_contact_loss(self, motion:th.Tensor, gt_contact_label:th.Tensor) -> torch.Tensor:
        '''
        motion: (B, 2J, 3, T) from model's output
        gt_contact_label: (B, T, P)
        '''
        B, J, _, T = motion.shape
        J //= 2
        motion = motion.reshape(B, 2, J, 3, T).permute(0, 4, 1, 2, 3) # (B, T, 2, J, 3)
        ret_terms = dict()

        # intra tip contact loss
        intra_tip_pairs_distance = motion[:, :, :, intra_tip_pairs[:, 0], :] - motion[:, :, :, intra_tip_pairs[:, 1], :] # (B, T, 2, P_tip, 3)
        intra_tip_pairs_distance = torch.sum(intra_tip_pairs_distance**2, dim=-1) # (B, T, 2, P_tip)
        gt_intra_tip_contact_label = gt_contact_label[:, :, :intra_tip_pairs.shape[0] * 2].reshape(B, T, 2, intra_tip_pairs.shape[0]) # (B, T, 2, P_tip)
        ret_terms['intra_tip_contact_loss'] = th.mean(intra_tip_pairs_distance * gt_intra_tip_contact_label)

        # inter tip palm loss
        left_palm_center = torch.mean(motion[:, :, 0, palm_peripheral_joints, :], dim=2) # (B, T, 3)
        right_palm_center = torch.mean(motion[:, :, 1, palm_peripheral_joints, :], dim=2) # (B, T, 3)
        left_tip_right_palm_distance = motion[:, :, 0, all_tips, :] - right_palm_center.unsqueeze(2) # (B, T, N_tips, 3)
        left_tip_right_palm_distance = torch.sum(left_tip_right_palm_distance**2, dim=-1) # (B, T, N_tips)
        right_tip_left_palm_distance = motion[:, :, 1, all_tips, :] - left_palm_center.unsqueeze(2) # (B, T, N_tips, 3)
        right_tip_left_palm_distance = torch.sum(right_tip_left_palm_distance**2, dim=-1) # (B, T, N_tips)
        gt_inter_tip_palm_contact_label = gt_contact_label[:, :, intra_tip_pairs.shape[0] * 2:intra_tip_pairs.shape[0] * 2 + all_tips.shape[0] * 2].reshape(B, T, 2, all_tips.shape[0]) # (B, T, 2, N_tips)
        ret_terms['inter_tip_palm_contact_loss'] = th.mean(gt_inter_tip_palm_contact_label * torch.stack([left_tip_right_palm_distance, right_tip_left_palm_distance], dim=2))

        # palm palm contact loss
        palm_palm_distance = left_palm_center - right_palm_center # (B, T, 3)
        palm_palm_distance = torch.sum(palm_palm_distance**2, dim=-1) # (B, T)
        gt_palm_palm_contact_label = gt_contact_label[:, :, -1] # (B, T)
        ret_terms['palm_palm_contact_loss'] = th.mean(palm_palm_distance * gt_palm_palm_contact_label)

        ret_terms['contact_loss'] = (ret_terms['intra_tip_contact_loss'] + ret_terms['inter_tip_palm_contact_loss'] + ret_terms['palm_palm_contact_loss'])
        for key in ret_terms:
            ret_terms[key] = ret_terms[key] * self.lambda_contact

        return ret_terms



    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            # print('model_output', model_output.shape, model_output)
            # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
            # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            # print('model_variance', model_variance)
            # print('model_log_variance',model_log_variance)
            # print('self.posterior_variance', self.posterior_variance)
            # print('self.posterior_log_variance_clipped', self.posterior_log_variance_clipped)
            # print('self.model_var_type', self.model_var_type)


            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def _create_mask_from_frames(self, shape, fixed_frames, device):
        """
        Create inpainting mask from list of frame indices.

        Args:
            shape: (B, J, F, T) - batch, joints, features, time
            fixed_frames: list of int, frame indices to fix, e.g., [0, 1, 2, 3, 4]
            device: torch device

        Returns:
            mask: (B, J, F, T) boolean tensor, True = keep GT, False = generate
        """
        B, J, F, T = shape
        mask = th.zeros(B, J, F, T, dtype=th.bool, device=device)

        if fixed_frames is not None and len(fixed_frames) > 0:
            for frame_idx in fixed_frames:
                if 0 <= frame_idx < T:
                    mask[:, :, :, frame_idx] = True

        return mask

    def _create_mask_from_regions(self, shape, mask_regions, device):
        """
        Create spatiotemporal inpainting mask from list of mask regions.

        Args:
            shape: (B, J, F, T) - batch, joints, features, time
            mask_regions: list of dicts with 'temporal' and 'spatial' keys
                         e.g., [{'temporal': [0,1,2], 'spatial': [0,1,2,3,4]}, ...]
            device: torch device

        Returns:
            mask: (B, J, F, T) boolean tensor, True = keep GT, False = generate
        """
        B, J, F, T = shape
        mask = th.zeros(B, J, F, T, dtype=th.bool, device=device)

        if mask_regions is not None and len(mask_regions) > 0:
            for region in mask_regions:
                temporal_indices = region.get('temporal', [])
                spatial_indices = region.get('spatial', [])

                for t_idx in temporal_indices:
                    if 0 <= t_idx < T:
                        for j_idx in spatial_indices:
                            if 0 <= j_idx < J:
                                mask[:, j_idx, :, t_idx] = True

        return mask

    def _create_soft_mask_from_regions(self, shape, mask_regions, device):
        """
        Create soft spatiotemporal inpainting mask with smooth transitions.

        Args:
            shape: (B, J, F, T) - batch, joints, features, time
            mask_regions: list of dicts:
                - 'temporal': list of frame indices (core fixed region)
                - 'spatial': list of joint indices
                - 'temporal_transition_width': int (default 5) - transition band width (OUTSIDE mask region)
                - 'core_mask_value': float (default 0.85) - mask value for core region
                - 'edge_mask_value': float (default 0.1) - mask value for edge/generation region
            device: torch device

        Returns:
            mask: (B, J, F, T) float32 tensor in [0, 1]
                - Transition bands are placed OUTSIDE the specified temporal region
                - Example: temporal=[10,15], width=5
                    -> frames [10-15]: core_mask_value (0.85)
                    -> frames [5-10): transition from edge_val to core_val
                    -> frames (15-20]: transition from core_val to edge_val
        """
        B, J, F, T = shape
        mask = th.zeros(B, J, F, T, dtype=th.float32, device=device)

        if mask_regions is not None and len(mask_regions) > 0:
            for region in mask_regions:
                temporal_indices = region.get('temporal', [])
                spatial_indices = region.get('spatial', [])

                if len(temporal_indices) == 0 or len(spatial_indices) == 0:
                    continue

                # Get configuration
                t_width = region.get('temporal_transition_width', 5)
                core_val = region.get('core_mask_value', 0.85)
                edge_val = region.get('edge_mask_value', 0.1)

                # Sort and find range
                temporal_indices = sorted(temporal_indices)
                t_min, t_max = temporal_indices[0], temporal_indices[-1]

                # 1. Core fixed region: [t_min, t_max]
                for t_idx in temporal_indices:
                    if 0 <= t_idx < T:
                        for j_idx in spatial_indices:
                            if 0 <= j_idx < J:
                                mask[:, j_idx, :, t_idx] = core_val

                # 2. Front transition band: [t_min - t_width, t_min)
                #    Transition happens OUTSIDE the mask region
                for offset in range(1, t_width + 1):
                    t_idx = t_min - offset
                    if 0 <= t_idx < T:
                        # Linear interpolation: offset=1 (closest to core) -> core_val
                        #                       offset=t_width (farthest) -> edge_val
                        alpha = (t_width - offset) / t_width
                        t_mask_val = edge_val + alpha * (core_val - edge_val)

                        for j_idx in spatial_indices:
                            if 0 <= j_idx < J:
                                # Take max to avoid overwriting higher mask values
                                mask[:, j_idx, :, t_idx] = th.maximum(mask[:, j_idx, :, t_idx], th.tensor(t_mask_val, device=device))

                # 3. Back transition band: (t_max, t_max + t_width]
                for offset in range(1, t_width + 1):
                    t_idx = t_max + offset
                    if 0 <= t_idx < T:
                        # Linear interpolation: offset=1 (closest to core) -> core_val
                        #                       offset=t_width (farthest) -> edge_val
                        alpha = (t_width - offset) / t_width
                        t_mask_val = edge_val + alpha * (core_val - edge_val)

                        for j_idx in spatial_indices:
                            if 0 <= j_idx < J:
                                # Take max to avoid overwriting higher mask values
                                mask[:, j_idx, :, t_idx] = th.maximum(mask[:, j_idx, :, t_idx], th.tensor(t_mask_val, device=device))

        return mask

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        gt_motion=None,
        inpaint_mask=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param gt_motion: if not None, the ground truth motion (x_0) for inpainting.
                         Shape: (B, J, F, T)
        :param inpaint_mask: if not None, a boolean mask indicating which parts to keep from GT.
                            Shape: (B, J, F, T), True = keep GT, False = generate
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        # print('const_noise', const_noise)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # print('mean', out["mean"].shape, out["mean"])
        # print('log_variance', out["log_variance"].shape, out["log_variance"])
        # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # Partially denoising (inpainting): mix generated sample with GT
        if gt_motion is not None and inpaint_mask is not None:
            # Auto-detect mask type: float32 (soft) or bool (hard)
            is_soft_mask = (inpaint_mask.dtype == th.float32)

            if is_soft_mask:
                # === Soft Blending ===
                if t[0].item() == 0:
                    # Final step: soft blend clean GT and generated result
                    sample = inpaint_mask * gt_motion + (1.0 - inpaint_mask) * sample
                else:
                    # Intermediate steps: soft blend noisy GT and generated result
                    t_prev = th.clamp(t - 1, min=0)
                    gt_noisy = self.q_sample(gt_motion, t_prev, noise=noise)
                    sample = inpaint_mask * gt_noisy + (1.0 - inpaint_mask) * sample
            else:
                # === Hard Blending (original logic) ===
                if t[0].item() == 0:
                    # At final step (t=0), use clean GT directly without adding noise
                    sample = th.where(inpaint_mask, gt_motion, sample)
                else:
                    # At intermediate steps (t>0), add noise to GT to match the noise level at t-1
                    t_prev = th.clamp(t - 1, min=0)
                    # q_sample: x_0 -> x_t using forward diffusion
                    gt_noisy = self.q_sample(gt_motion, t_prev, noise=noise)
                    # Mix: where mask is True, use GT; where False, use generated
                    sample = th.where(inpaint_mask, gt_noisy, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        gt_motion=None,
        inpaint_mask=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param const_noise: if True, use constant noise across batch.
        :param gt_motion: if not None, the ground truth motion (x_0) for inpainting.
                         Shape: (B, J, F, T)
        :param inpaint_mask: if not None, a boolean mask indicating which parts to keep from GT.
                            Shape: (B, J, F, T), True = keep GT, False = generate
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            if const_noise:
                noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1))) # (B, *x.shape[1:])
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # Partially denoising (inpainting): mix generated sample with GT
        if gt_motion is not None and inpaint_mask is not None:
            # Auto-detect mask type: float32 (soft) or bool (hard)
            is_soft_mask = (inpaint_mask.dtype == th.float32)

            if is_soft_mask:
                # === Soft Blending ===
                if t[0].item() == 0:
                    # Final step: soft blend clean GT and generated result
                    sample = inpaint_mask * gt_motion + (1.0 - inpaint_mask) * sample
                else:
                    # Intermediate steps: soft blend noisy GT and generated result
                    t_prev = th.clamp(t - 1, min=0)
                    gt_noisy = self.q_sample(gt_motion, t_prev, noise=noise)
                    sample = inpaint_mask * gt_noisy + (1.0 - inpaint_mask) * sample
            else:
                # === Hard Blending (original logic) ===
                if t[0].item() == 0:
                    # At final step (t=0), use clean GT directly without adding noise
                    sample = th.where(inpaint_mask, gt_motion, sample)
                else:
                    # At intermediate steps (t>0), add noise to GT to match the noise level at t-1
                    t_prev = th.clamp(t - 1, min=0)
                    # q_sample: x_0 -> x_t using forward diffusion
                    gt_noisy = self.q_sample(gt_motion, t_prev, noise=noise)
                    # Mix: where mask is True, use GT; where False, use generated
                    sample = th.where(inpaint_mask, gt_noisy, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model:torch.nn.Module,
        shape:Tuple[int, ...] | List[int],
        noise:th.Tensor=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        gt_motion=None,
        fixed_frames=None,
        mask_regions=None,
    ):
        """
        Generate samples from the model.

        :param model: 实际的去噪模型
        :param shape: 生成样本的预期形状 (N, C, H, W).
        :param noise: 如果提供，将使用这个张量作为初始噪声；否则会随机生成。
        :param clip_denoised: 如果为真，去噪后的样本会被裁剪到[-1, 1].
        :param denoised_fn: 用于对去噪后的样本进行后处理 if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: 条件函数，用于引导采样过程。if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: 传递给去噪模型的额外关键字参数。if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: 如果为真，显示一个 tqdm 进度条。if True, show a tqdm progress bar.
        :param skip_timesteps: 调过扩散过程的初始时间步。如果 skip_timesteps 为 0，则从最大的时间步开始；如果为 N，
            则从 N 个时间步之后开始。
        :param init_image: 当 skip_timesteps > 0 时，通常会提供一个初始图像（带有 batch 维）。
        :param randomize_class: 如果为 True 且 model_kwargs 中包含 'y'（类别标签），则会在每个时间步随机化类别标签。这可能用于生成多样化的样本。
        :param cond_fn_with_grad: 如果为 True，则在采样时会使用需要梯度的条件函数（self.p_sample_with_grad），
            否则使用 self.p_sample。
        :param dump_steps: 可选参数，一个步数列表。如果在这些步数时需要保存中间样本，则提供此参数。
        :param const_noise: 如果为 True，则在采样过程中使用的噪声项是固定的。
        :param gt_motion: Ground truth motion for inpainting. Shape: (B, J, F, T). None for full generation.
        :param fixed_frames: List of frame indices to fix from GT, e.g., [0, 1, 2, 3, 4] to fix first 5 frames.
                            None to not fix any frames.
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = [0] * len(dump_steps)

        # if 'text' in model_kwargs['y'].keys():
            # encoding once instead of each iteration saves lots of time
            # model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
            # model_kwargs['y']['text'] = model_kwargs['y']['text'].to(device)
        model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        # Create inpaint mask from fixed_frames or mask_regions if provided
        inpaint_mask = None
        if device is None:
            device = next(model.parameters()).device

        if mask_regions is not None and gt_motion is not None:
            # Check if any region uses soft mask
            use_soft = any(region.get('use_soft_mask', False) for region in mask_regions)

            if use_soft:
                # Use soft mask
                inpaint_mask = self._create_soft_mask_from_regions(shape, mask_regions, device)
            else:
                # Use hard mask
                inpaint_mask = self._create_mask_from_regions(shape, mask_regions, device)

            # Move gt_motion to device if needed
            if not isinstance(gt_motion, th.Tensor):
                gt_motion = th.tensor(gt_motion, device=device, dtype=th.float32)
            elif gt_motion.device != device:
                gt_motion = gt_motion.to(device)
        elif fixed_frames is not None and gt_motion is not None:
            # Use old fixed_frames (temporal only, all joints)
            inpaint_mask = self._create_mask_from_frames(shape, fixed_frames, device)
            # Move gt_motion to device if needed
            if not isinstance(gt_motion, th.Tensor):
                gt_motion = th.tensor(gt_motion, device=device, dtype=th.float32)
            elif gt_motion.device != device:
                gt_motion = gt_motion.to(device)

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            gt_motion=gt_motion,
            inpaint_mask=inpaint_mask,
        )):
            cur_denoise_step = self.num_timesteps - i - 1
            if dump_steps is not None and cur_denoise_step in dump_steps:
                # dump.append(deepcopy(sample["sample"]))
                dump[dump_steps.index(cur_denoise_step)] = deepcopy(sample['sample'])
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model:torch.nn.Module,
        shape:Tuple[int, ...] | List[int],
        noise:torch.Tensor=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        gt_motion=None,
        inpaint_mask=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().

        :param model: 实际的去噪模型
        :param shape: 生成样本的预期形状 (N, C, H, W).
        :param noise: 如果提供，将使用这个张量作为初始噪声；否则会随机生成。
        :param clip_denoised: 如果为真，去噪后的样本会被裁剪到[-1, 1].
        :param denoised_fn: 用于对去噪后的样本进行后处理 if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: 条件函数，用于引导采样过程。if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: 传递给去噪模型的额外关键字参数。if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: 如果为真，显示一个 tqdm 进度条。if True, show a tqdm progress bar.
        :param skip_timesteps: 调过扩散过程的初始时间步。如果 skip_timesteps 为 0，则从最大的时间步开始；如果为 N，
            则从 N 个时间步之后开始。
        :param init_image: 当 skip_timesteps > 0 时，通常会提供一个初始图像（带有 batch 维）。
        :param randomize_class: 如果为 True 且 model_kwargs 中包含 'y'（类别标签），则会在每个时间步随机化类别标签。这可能用于生成多样化的样本。
        :param cond_fn_with_grad: 如果为 True，则在采样时会使用需要梯度的条件函数（self.p_sample_with_grad），
            否则使用 self.p_sample。
        :param const_noise: 如果为 True，则在采样过程中使用的噪声项是固定的。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1] # 生成需要采样的时间步列表（从大到小）

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img) # 将 init_image 噪声化到 my_t 时间步

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, desc='Denoising')

        for i in indices:
            t = th.tensor([i] * shape[0], device=device) # (B,)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(
                    low=0, high=model.num_classes,
                    size=model_kwargs['y'].shape,
                    device=model_kwargs['y'].device
                )
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                    gt_motion=gt_motion,
                    inpaint_mask=inpaint_mask,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']
        enc = model.model
        mask = model_kwargs['y']['mask']
        # get_xyz = lambda sample: enc.rot2xyz(sample, mask=None, pose_rep=enc.pose_rep, translation=enc.translation,
        #                                      glob=enc.glob,
        #                                      # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
        #                                      jointstype='smpl',  # 3.4 iter/sec
        #                                      vertstrans=False)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs) # (B, J, C, T)


            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]

            target = {
                # ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                #     x_start=x_start, x_t=x_t, t=t
                # )[0],
                ModelMeanType.START_X: x_start,
                # ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]

            terms["mse"] = self.masked_l2(
                target[:, :, :3, :], model_output[:, :, :3, :], mask
            )

            terms["repr_mse"] = self.masked_l2(target, model_output, mask) * self.lambda_repr

            terms["vel_mse"] = self.masked_l2(
                target[:, :, :3, 1:] - target[:, :, :3, :-1],
                model_output[:, :, :3, 1:] - model_output[:, :, :3, :-1],
                mask[:, :, :, :-1]
            ) * self.lambda_vel

            terms['acce_mse'] = self.masked_l2(
                model_output[:, :, :3, 2:] - 2 * model_output[:, :, :3, 1:-1] + model_output[:, :, :3, :-2],
                torch.zeros_like(model_output[:, :, :3, 2:]),
                mask[:, :, :, 1:-1]
            ) * self.lambda_acce

            # target_xyz, model_output_xyz = None, None

            if self.lambda_ig > 0 or self.lambda_w_ig > 0:
                target_ig = enc.repr2ig(target) # b j j c t
                model_output_ig = enc.repr2ig(model_output) # b j j c t
                terms["ig"] = self.masked_ig(target_ig, model_output_ig, mask)
                terms["w_ig"] = self.masked_weighted_ig(target_ig, model_output_ig, mask)

            if self.contact_loss:
                gt_contact_label = model_kwargs['y']['contact_label'] # (B, T, P)
                contact_loss_terms = self.compute_contact_loss(
                    model_output[:, :, :3, :], gt_contact_label
                )
                terms.update(contact_loss_terms)

            if self.contact_predict_loss:
                pred_contact = model(x_t, self._scale_timesteps(t), **model_kwargs, predict_contact=True) # (B, T, P)
                gt_contact_label = model_kwargs['y']['contact_label'] # (B, T, P)
                terms['contact_predict_bce'] = th.nn.functional.binary_cross_entropy_with_logits(
                    pred_contact, gt_contact_label.float(), reduction='mean'
                ) * self.lambda_contact_predict

            terms["loss"] = terms["mse"] + terms['repr_mse'] +\
                            terms.get('vb', 0.) +\
                            terms.get('vel_mse', 0.) +\
                            terms.get('acce_mse', 0.) +\
                            terms.get('contact_loss', 0.) +\
                            terms.get('contact_predict_bce', 0.)

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _compute_joint_rotation_error(self, gt_joint_rotation_matrices : th.Tensor, sampled_joint_rotation_matrices : th.Tensor):
        '''
        gt_joint_rotation_matrices: [B, J, 3, 3, T]
        sampled_joint_rotation_matrices: [B, J, 3, 3, T]

        return: [B, J, 1, T]
        '''
        gt_joint_rotation_matrices = gt_joint_rotation_matrices.permute(0, 1, 4, 2, 3) # [B, J, T, 3, 3]
        sampled_joint_rotation_matrices = sampled_joint_rotation_matrices.permute(0, 1, 4, 2, 3) # [B, J, T, 3, 3]
        R_rel = th.matmul(
            gt_joint_rotation_matrices, # [B, J, T, 3, 3]
            sampled_joint_rotation_matrices.permute(0, 1, 2, 4, 3) # [B, J, T, 3, 3] -> [B, J, T, 3, 3] (transpose)
        ) # [B, J, T, 3, 3]
        trace = th.sum(th.diagonal(R_rel, dim1=-2, dim2=-1), dim=-1) # [B, J, T]
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        angle_rad = torch.acos(cos_theta)[:, :, th.newaxis, :] # [B, J, 1, T]
        return th.rad2deg(angle_rad) # [B, J, 1, T] - convert to degrees

    def _compute_joint_scalar_rotation_error(self, gt_joint_scalar_rotations: th.Tensor, sampled_joint_scalar_rotations: th.Tensor):
        '''
        gt_joint_scalar_rotations: [B, J, 1, T]
        sampled_joint_scalar_rotations: [B, J, 1, T]
        '''
        delta = th.abs(gt_joint_scalar_rotations - sampled_joint_scalar_rotations) # [B, J, 1, T]
        return delta

    def _compute_joint_position_error(self, gt_joint_positions: th.Tensor, sampled_joint_positions: th.Tensor):
        '''
        gt_joint_positions: [B, J, 3, T]
        sampled_joint_positions: [B, J, 3, T]

        return: [B, J, 1, T]
        '''
        delta = gt_joint_positions - sampled_joint_positions # [B, J, 3, T]
        return th.linalg.norm(delta, dim=2, keepdim=True) # [B, J, 1, T]


    def joint_6d_rotations_to_rotation_matrices(self, joint_6d_rotations:th.Tensor):
        '''
        joint_6d_rotations: [B, J, 6, T]

        return: [B, J, 3, 3, T]
        '''
        joint_6d_rotations = joint_6d_rotations.permute(0, 1, 3, 2) # [B, J, T, 6]
        joint_rotation_matrices = rotation_conversion.rotation_6d_to_matrix(joint_6d_rotations) # [B, J, T, 3, 3]
        return joint_rotation_matrices.permute(0, 1, 3, 4, 2) # [B, J, 3, 3, T]

    def joint_axisangle_rotations_to_rotation_matrices(self, joint_axisangle_rotations: th.Tensor):
        '''
        joint_axisangle_rotations: [B, J, 3, T]

        return: [B, J, 3, 3, T]
        '''
        joint_axisangle_rotations = joint_axisangle_rotations.permute(0, 1, 3, 2) # [B, J, T, 3]
        joint_rotation_matrices = rotation_conversion.axis_angle_to_matrix(joint_axisangle_rotations, fast=True) # [B, J, T, 3, 3]
        return joint_rotation_matrices.permute(0, 1, 3, 4, 2) # [B, J, 3, 3, T]

    def single_joint_6d_rotations_to_joint_positions(self, single_motion:th.Tensor, hand:str):
        '''
        single_motion: [B, 17, 6, T]
            0:1 - trans
            1:2 - global ori
            2:17 - joint rotations

        return: [B, J, 3, T]
        '''
        single_motion = single_motion.permute(0, 3, 1, 2) # [B, T, 17, 6]
        B, T, _, _ = single_motion.shape
        trans = single_motion[:, :, 0, :3].reshape(-1, 3) # [BT, 3]
        mano_params = single_motion[:, :, 1:] # [B, T, 16, 6]
        mano_params = rotation_conversion.matrix_to_axis_angle(rotation_conversion.rotation_6d_to_matrix(mano_params)) # [B, T, 16, 3]

        global_orient = mano_params[:, :, 0, :].reshape(-1, 3) # [BT, 3]
        hand_pose = mano_params[:, :, 1:, :].reshape(-1, 45) # [BT, 45]

        mano_model = left_manomodel if hand == 'left' else right_manomodel
        mano_model = mano_model.to(single_motion.device)

        # print(f'global_orient shape: {global_orient.shape}, hand_pose shape: {hand_pose.shape} trans shape: {trans.shape}')

        mano_output:MANOOutput = mano_model(
            betas=th.zeros((B * T, 10), device=single_motion.device), # [BT, 10]
            global_orient=global_orient, # [BT, 3]
            hand_pose=hand_pose, # [BT, 45]
            transl=trans, # [BT, 3]
        )

        joints = mano_output.joints # [BT, J, 3]

        joints = joints.reshape(B, T, -1, 3) # [B, T, J, 3]
        return joints.permute(0, 2, 3, 1) # [B, J, 3, T]


    def joint_6d_rotations_to_joint_positions(self, joint_6d_rotations:th.Tensor):
        '''
        joint_6d_rotations: [B, 34, 6, T]
            0:1 - left trans
            1:2 - left global ori
            2:17 - left joint rotations
            17:18 - right trans
            18:19 - right global ori
            19:34 - right joint rotations
        '''
        left_motion, right_motion = th.split(joint_6d_rotations, 17, dim=1) # [B, 17, 6, T], [B, 17, 6, T]

        left_positions = self.single_joint_6d_rotations_to_joint_positions(left_motion, 'left') # [B, J, 3, T]
        right_positions = self.single_joint_6d_rotations_to_joint_positions(right_motion, 'right') # [B, J, 3, T]

        joint_positions = th.cat([
            left_positions,
            right_positions
        ], dim=1) # [B, 2J, 3, T]

        return joint_positions



    def evaluate_metrics(self, model, x_start, t, inv_transform, model_kwargs=None, noise=None):
        """
        Compute evaluation metrics for a single timestep.

        :param model: the model to evaluate metrics on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "metrics" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        mask = model_kwargs['y']['mask'] # (B, 1, 1, T)

        if model_kwargs is None:
            model_kwargs = dict()

        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)


        terms = dict()

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs) # (B, J, C, T)
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise
        }[self.model_mean_type]

        _, J, _, _ = model_output.shape
        model_output = rearrange(model_output, 'b j c t -> b t (j c)') # [B, T, J*C]
        target = rearrange(target, 'b j c t -> b t (j c)') # [B, T, J*C]

        model_output = inv_transform(model_output) # [B, T, J*C]
        target = inv_transform(target) # [B, T, J*C]

        model_output = rearrange(model_output, 'b t (j c) -> b j c t', j=J) # [B, J, C, T]
        target = rearrange(target, 'b t (j c) -> b j c t', j=J)

        for metric_type in self.metric_types:
            if metric_type == MetricType.JOINT_ROT_ERROR:
                if self.repr_type == ReprType.JOINT_ROT_6D:
                    mano_indices = list(range(1, 17)) + list(range(18, 34))
                    model_output_matrices = self.joint_6d_rotations_to_rotation_matrices(
                        model_output[:, mano_indices],
                    ) # [B, J, 3, 3, T]
                    target_matrices = self.joint_6d_rotations_to_rotation_matrices(
                        target[:, mano_indices]
                    ) # [B, J, 3, 3, T]


                    rotation_error = self._compute_joint_rotation_error(
                        gt_joint_rotation_matrices=target_matrices,
                        sampled_joint_rotation_matrices=model_output_matrices
                    ) # [B, J, 1, T]

                    terms['joint_rot_error'] = self.masked_average(rotation_error, mask)

                    model_output_global_trans = model_output[:, [0, 17], :3, :] # [B, 2, 3, T]
                    target_global_trans = target[:, [0, 17], :3, :] # [B, 2, 3, T]

                    position_error = self._compute_joint_position_error(
                        model_output_global_trans, target_global_trans
                    )

                    terms['global_trans_error'] = self.masked_average(position_error, mask)
                elif self.repr_type == ReprType.JOINT_POS_W_SCALAR_ROT:
                    model_output_scalar_rot = model_output[:, :, 3:, :].clone() # [B, J, 1, T]
                    target_scalar_rot = target[:, :, 3:, :].clone() # [B, J, 1, T]

                    scalar_rotation_error = self._compute_joint_scalar_rotation_error(
                        gt_joint_scalar_rotations=target_scalar_rot,
                        sampled_joint_scalar_rotations=model_output_scalar_rot
                    )

                    terms['joint_scalar_rot_error'] = self.masked_average(scalar_rotation_error, mask)
                elif self.repr_type == ReprType.JOINT_POS_W_AXISANGLE_ROT:
                    model_output_axisangle_rot = model_output[:, :, 3:, :].clone() # [B, J, 3, T]
                    target_axisangle_rot = target[:, :, 3:, :].clone() # [B, J, 3, T]

                    model_output_matrices = self.joint_axisangle_rotations_to_rotation_matrices(model_output_axisangle_rot) # [B, J, 3, 3, T]
                    target_matrices = self.joint_axisangle_rotations_to_rotation_matrices(target_axisangle_rot) # [B, J, 3, 3, T]

                    rotation_error = self._compute_joint_rotation_error(target_matrices, model_output_matrices) # [B, J, 1, T]

                    terms['joint_axisangle_rot_error'] = self.masked_average(rotation_error, mask)
                else:
                    raise NotImplementedError(f"Representation type {self.repr_type} is not supported for joint rotation error.")


            elif metric_type == MetricType.JOINT_POS_ERROR:
                if self.repr_type == ReprType.JOINT_ROT_6D:
                    model_output_positions = self.joint_6d_rotations_to_joint_positions(model_output) # [B, J, 3, T]
                    target_positions = self.joint_6d_rotations_to_joint_positions(target) # [B, J, 3, T]
                elif self.repr_type == ReprType.JOINT_POS:
                    model_output_positions = model_output.clone().detach()
                    target_positions = target.clone().detach()
                elif self.repr_type in [ReprType.JOINT_POS_W_SCALAR_ROT, ReprType.JOINT_POS_W_AXISANGLE_ROT]:
                    '''
                    model_output: [B, J, 3+1, T]
                    '''
                    model_output_positions = model_output[:, :, :3, :].clone() # [B, J, 3, T]
                    target_positions = target[:, :, :3, :].clone() # [B, J, 3, T]

                else:
                    raise NotImplementedError(f"Representation type {self.repr_type} is not supported for joint position error.")

                position_error = self._compute_joint_position_error(
                    gt_joint_positions=target_positions,
                    sampled_joint_positions=model_output_positions
                )

                terms['joint_pos_error'] = self.masked_average(position_error, mask)

            else:
                raise NotImplementedError(f"Metric type {metric_type} is not implemented.")

        return terms


    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()
        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11
        """ Contact calculated by 'Kfir Method' Commented code)"""
        # contact_signal = torch.zeros((pose_xyz.shape[0], pose_xyz.shape[3], 2), device=pose_xyz.device) # [BatchSize, Frames, 2]
        # left_xyz = 0.5 * (pose_xyz[:, l_ankle_idx, :, :] + pose_xyz[:, l_foot_idx, :, :]) # [BatchSize, 3, Frames]
        # right_xyz = 0.5 * (pose_xyz[:, r_ankle_idx, :, :] + pose_xyz[:, r_foot_idx, :, :])
        # left_z, right_z = left_xyz[:, 2, :], right_xyz[:, 2, :] # [BatchSize, Frames]
        # left_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)  # [BatchSize, Frames]
        # right_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)
        #
        # left_z_mask = left_z <= torch.mean(torch.sort(left_z)[0][:, :left_z.shape[1] // 5], axis=-1)
        # left_z_mask = torch.stack([left_z_mask, left_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # left_z_mask[:, :, 1] = False  # Blank right side
        # contact_signal[left_z_mask] = 0.4
        #
        # right_z_mask = right_z <= torch.mean(torch.sort(right_z)[0][:, :right_z.shape[1] // 5], axis=-1)
        # right_z_mask = torch.stack([right_z_mask, right_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # right_z_mask[:, :, 0] = False  # Blank left side
        # contact_signal[right_z_mask] = 0.4
        # contact_signal[left_z <= (torch.mean(torch.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        # contact_signal[right_z <= (torch.mean(torch.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1

        # plt.plot(to_np_cpu(left_z[0]), label='left_z')
        # plt.plot(to_np_cpu(left_velocity[0]), label='left_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 0]), label='left_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.plot(to_np_cpu(right_z[0]), label='right_z')
        # plt.plot(to_np_cpu(right_velocity[0]), label='right_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 1]), label='right_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()

        gt_joint_xyz = gt_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        fc_mask = (gt_joint_vel <= 0.01)
        pred_joint_xyz = pred_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        pred_joint_vel[~fc_mask] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        """DEBUG CODE"""
        # print(f'mask: {mask.shape}')
        # print(f'pred_joint_vel: {pred_joint_vel.shape}')
        # plt.title(f'Joint: {joint_idx}')
        # plt.plot(to_np_cpu(gt_joint_vel[0]), label='velocity')
        # plt.plot(to_np_cpu(fc_mask[0]), label='fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        return self.masked_l2(pred_joint_vel, torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
                              mask[:, :, :, 1:])
    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def foot_contact_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259
        contact_mask_gt = contact > 0.5  # contact mask order for indexes are fid_l [7, 10], fid_r [8, 11]
        vel_lf_7 = local_velocity[:, 7 * 3:8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3:9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3:11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3:12 * 3, :, :]

        calc_vel_lf_7 = ric_data[:, 6 * 3:7 * 3, :, 1:] - ric_data[:, 6 * 3:7 * 3, :, :-1]
        calc_vel_rf_8 = ric_data[:, 7 * 3:8 * 3, :, 1:] - ric_data[:, 7 * 3:8 * 3, :, :-1]
        calc_vel_lf_10 = ric_data[:, 9 * 3:10 * 3, :, 1:] - ric_data[:, 9 * 3:10 * 3, :, :-1]
        calc_vel_rf_11 = ric_data[:, 10 * 3:11 * 3, :, 1:] - ric_data[:, 10 * 3:11 * 3, :, :-1]

        # vel_foots = torch.stack([vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], dim=1)
        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
                [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11],
                [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11],
                [7, 10, 8, 11],
                [0, 1, 2, 3]):
            tmp_mask_gt = contact_mask_gt[:, contact_mask_idx, :, :].cpu().detach().numpy().reshape(-1).astype(int)
            chosen_vel_norm = np.linalg.norm(chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0)
            chosen_vel_calc_norm = np.linalg.norm(chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)),
                                                  axis=0)

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt
            plt.plot(tmp_mask_gt, label='FC mask')
            plt.plot(chosen_vel_norm, label='Vel. XYZ norm (from vector)')
            plt.plot(chosen_vel_calc_norm, label='Vel. XYZ norm (calculated diff XYZ)')

            plt.title(f'FC idx {contact_mask_idx}, Joint Index {joint_idx}')
            plt.legend()
            plt.show()
        # print(vel_foots.shape)
        return 0
    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def velocity_consistency_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]  # Slicing out root
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(target.permute(0, 2, 3, 1).type(th.FloatTensor))
        print(f'r_rot_quat: {r_rot_quat.shape}')
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape}')
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(th.FloatTensor)
        r_rot_quat_adapted = r_rot_quat[..., :-1, None, :].repeat((1,1,1,21,1)).to(calc_vel_from_xyz.device)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')
        print(f'r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}')

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')

        import matplotlib.pyplot as plt
        for i in range(21):
            plt.plot(np.linalg.norm(calc_vel_from_xyz[:,i*3:(i+1)*3,:,:].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Calc Vel')
            plt.plot(np.linalg.norm(velocity_from_vector[:,i*3:(i+1)*3,:,:].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Vector Vel')
            plt.title(f'Joint idx: {i}')
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz-velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
