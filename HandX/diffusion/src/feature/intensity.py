import numpy as np

from ..constant import SKELETON_CHAIN, JOINT_NAME_INDEX_MAP

DELTA_T = 1 / 30.0
def compute_joint_angular_velocity(skeleton_motion: np.ndarray):
    DELTA_T = 1/30.0
    weights_per_middle = [9.0, 3.0, 1.0]

    joint_means = []
    joint_weights = []

    for chain in SKELETON_CHAIN:
        for i, w in zip(range(1, len(chain)-1), weights_per_middle):
            v1 = skeleton_motion[:, chain[i]]   - skeleton_motion[:, chain[i-1]]
            v2 = skeleton_motion[:, chain[i+1]] - skeleton_motion[:, chain[i]]

            n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
            n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
            v1 = np.divide(v1, n1, out=np.zeros_like(v1), where=n1!=0)
            v2 = np.divide(v2, n2, out=np.zeros_like(v2), where=n2!=0)

            theta = np.rad2deg(np.arccos(np.clip(np.sum(v1*v2, axis=-1), -1.0, 1.0)))
            vel = np.abs(theta[1:] - theta[:-1]) / DELTA_T  # (F-1,)
            joint_means.append(vel.mean())
            joint_weights.append(w)

    return float(np.average(joint_means, weights=joint_weights))

def compute_joint_velocity(skeleton_motion: np.ndarray):
    valid_joint = [index for name, index in JOINT_NAME_INDEX_MAP.items() if name != 'wrist']
    skeleton_motion = skeleton_motion[:, valid_joint, :] # (F, J, 3)
    velocity = np.linalg.norm(skeleton_motion[1:] - skeleton_motion[:-1], axis=-1) / DELTA_T # (F-1, J)
    return np.mean(velocity)

def compute_std(skeleton_motion: np.ndarray):
    '''
    skeleton_motion: (F, J, 3)
    '''
    motion_std = np.std(skeleton_motion, axis=0) # (J, 3)
    return np.mean(np.linalg.norm(motion_std, axis=-1))