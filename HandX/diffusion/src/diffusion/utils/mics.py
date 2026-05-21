import numpy as np
import torch
import random
import os
from typing import Dict
from .. import dist as dist_utils

from .rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle

def fixseed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)

def get_device():
    if not dist_utils.is_dist_avail_and_initialized():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        return torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')


def rot_motion_to_dict(motion:np.ndarray) -> Dict[str, np.ndarray]:
    '''
    motion: (T, 17, 6)
    '''
    trans = motion[:, 0, :3] # (T, 3)
    pose = motion[:, 1:] # (T, 16, 6)

    with torch.no_grad():
        pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(
                torch.from_numpy(pose)
            )
        ).cpu().numpy() # (T, 16, 3)

    T = pose.shape[0]
    pose = pose.reshape(T, -1) # (T, 48)

    return dict(
        trans=trans,
        pose=pose
    )

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