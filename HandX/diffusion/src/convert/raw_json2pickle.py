import json
import pickle
import numpy as np

from ..utils import load_json, save_pkl

op2ma_index = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

def process_sequence(seq_data):
    motion = np.zeros((len(seq_data), 2, 21, 3))
    for frame_idx, frame_data in enumerate(seq_data):
        for joint_idx in range(21):
            motion[frame_idx, 0, op2ma_index[joint_idx], :] = np.array(frame_data['left_hand'][joint_idx])
            motion[frame_idx, 1, op2ma_index[joint_idx], :] = np.array(frame_data['right_hand'][joint_idx])
    return motion

