import numpy as np
from scipy.spatial.transform import Rotation as R

from ..constant import JOINT_NAME_INDEX_MAP

def axis_angle2matrix(r_in_axis_angle:np.ndarray):
    '''
    r_in_axis_angle: (..., 3)
    return: (..., 3, 3)
    '''
    r = R.from_rotvec(r_in_axis_angle.reshape(-1, 3))
    return r.as_matrix().reshape(r_in_axis_angle.shape[:-1] + (3, 3))

def get_local_coordinate_axis(skeleton_motion:np.ndarray, global_orient:np.ndarray):
    wrist_index = JOINT_NAME_INDEX_MAP['wrist']
    origin = skeleton_motion[:, wrist_index] # (F, 3)
    axis = axis_angle2matrix(global_orient) # (F, 3, 3)
    return axis, origin # (F, 3, 3), (F, 3)

def convert_coordinate(skeleton_motion:np.ndarray, new_axis:np.ndarray | None=None, new_origin:np.ndarray | None=None):
    '''
    skeleton_motion: (F, J, 3)
    new_axis: (F, 3, 3)
    new_origin: (F, 3)
    '''
    if new_origin is not None:
        skeleton_motion = skeleton_motion - new_origin[:, np.newaxis, :] # (F, J, 3)
    if new_axis is not None:
        skeleton_motion = np.matmul(
            np.linalg.inv(new_axis)[:, np.newaxis, :, :], # (F, 1, 3, 3)
            skeleton_motion[:, :, :, np.newaxis] # (F, J, 3, 1)
        ).squeeze(-1) # (F, J, 3)

    return skeleton_motion