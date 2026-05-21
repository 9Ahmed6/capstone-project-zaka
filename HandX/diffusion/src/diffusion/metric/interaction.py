import numpy as np
import json
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.ndimage import convolve1d

from ...feature.single_motioncode import MotionCoder
from ...constant import JOINT_NAME_INDEX_MAP

intra_tip_pairs = [
    ["thumb_tip", "index_tip"],
    ["thumb_tip", "middle_tip"],
    ["thumb_tip", "ring_tip"],
    ["thumb_tip", "pinky_tip"]
]

all_tips = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

palm_peripheral_joints = ["wrist", "thumb_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]

for i in range(len(intra_tip_pairs)):
    intra_tip_pairs[i] = [
        JOINT_NAME_INDEX_MAP[intra_tip_pairs[i][0]],
        JOINT_NAME_INDEX_MAP[intra_tip_pairs[i][1]]
    ]
for i in range(len(all_tips)):
    all_tips[i] = JOINT_NAME_INDEX_MAP[all_tips[i]]
for i in range(len(palm_peripheral_joints)):
    palm_peripheral_joints[i] = JOINT_NAME_INDEX_MAP[palm_peripheral_joints[i]]
intra_tip_pairs = np.array(intra_tip_pairs)  # (P, 2)
all_tips = np.array(all_tips)  # (N,)
palm_peripheral_joints = np.array(palm_peripheral_joints)  # (M,)

def min_duration_restriction(bool_data: np.ndarray, min_duration: int) -> np.ndarray:
    '''
    bool_data: (T, P)
    return:(P,)
    '''
    int_data = bool_data.astype(np.int8) # (T, P)
    kernel = np.ones((min_duration,), dtype=np.int8) # (K,)
    convolved = convolve1d(int_data, kernel, axis=0, mode='constant', cval=0) # (T, P)
    return np.any(convolved == min_duration, axis=0) # (P,)

def tip_tip_contact_occurs(p1: np.ndarray, p2: np.ndarray, threshold: float) -> np.ndarray:
    '''
    p1: (T, P, 3)
    p2: (T, P, 3)
    return: (T, P)
    '''
    dists = np.linalg.norm(p1 - p2, axis=-1) # (T, P)
    return dists < threshold

def tip_palm_contact_occurs(p: np.ndarray, palm: np.ndarray, threshold: float) -> np.ndarray:
    '''
    p: (T, N, 3)
    palm: (T, M, 3)
    return: (T, N)
    '''
    diff = p[:, :, np.newaxis, :] - palm[:, np.newaxis, :, :] # (T, N, M, 3)
    dists = np.linalg.norm(diff, axis=-1) # (T, N, M)
    min_dists = np.min(dists, axis=-1) # (T, N)
    return min_dists < threshold

def palm_palm_contact_occurs(palm1: np.ndarray, palm2: np.ndarray, threshold: float) -> np.ndarray:
    '''
    palm1: (T, M, 3)
    palm2: (T, M, 3)
    return (T,)
    '''
    diff = palm1[:, :, np.newaxis, :] - palm2[:, np.newaxis, :, :] # (T, M, M, 3)
    dists = np.linalg.norm(diff, axis=-1) # (T, M, M)
    min_dists = np.min(dists, axis=(-1, -2)) # (T,)
    return min_dists < threshold

def get_palm_glob(motion: np.ndarray, is_right: bool) -> np.ndarray:
    '''
    motion: (T, J, 3)
    return: (T, N, 3)
    '''
    motion_coder = MotionCoder(motion, is_right)
    motion_coder.get_palm()
    return motion_coder.palm_points_glob

def intra_contact(motion:np.ndarray, threshold: float, min_duration: int) -> np.ndarray:
    '''
    motion: (T, J, 3)
    return: (P,)
    '''
    contacts = tip_tip_contact_occurs(
        p1=motion[:, intra_tip_pairs[:, 0], :],
        p2=motion[:, intra_tip_pairs[:, 1], :],
        threshold=threshold,
    ) # (T, P)
    contacts = min_duration_restriction(contacts, min_duration) # (P,)
    return contacts  # (P,)

def inter_contact(motion:np.ndarray, another_motion:np.ndarray, another_is_right: bool, threshold: float, min_duration: int) -> np.ndarray:
    '''
    motion: (T, J, 3)
    another_motion: (T, J, 3)
    return: (N_tips,)
    '''
    palm_points = get_palm_glob(another_motion, another_is_right) # (T, M, 3)
    contacts = tip_palm_contact_occurs(
        p=motion[:, all_tips, :],
        palm=palm_points,
        threshold=threshold,
    ) # (T, N_tips)
    contacts = min_duration_restriction(contacts, min_duration) # (N_tips,)
    return contacts  # (N_tips,)

def lenient_contact_occurs(motion:np.ndarray, another_motion:np.ndarray, another_is_right: bool, threshold: float, min_duration: int) -> np.ndarray:
    '''
    motion: (T, J, 3)
    another_motion: (T, J, 3)

    return: (T,)
    '''
    points = get_palm_glob(motion, not another_is_right) # (T, M, 3)
    anothter_points = get_palm_glob(another_motion, another_is_right) # (T, M, 3)

    points = np.concatenate([points, motion], axis=1) # (T, M+J, 3)
    anothter_points = np.concatenate([anothter_points, another_motion], axis=1) # (T, M+J, 3)

    diff = points[:, :, np.newaxis, :] - anothter_points[:, np.newaxis, :, :] # (T, M+J, M+J, 3)
    dists = np.linalg.norm(diff, axis=-1) # (T, M+J, M+J)
    min_dists = np.min(dists, axis=(-1, -2)) # (T,)
    return min_dists < threshold

def lenient_contact(motion:np.ndarray, another_motion:np.ndarray, another_is_right: bool, threshold: float, min_duration: int) -> bool:
    '''
    motion: (T, J, 3)
    another_motion: (T, J, 3)
    return: bool
    '''
    contacts = lenient_contact_occurs(
        motion,
        another_motion,
        another_is_right,
        threshold,
        min_duration
    ) # (T,)
    contacts = min_duration_restriction(contacts[:, np.newaxis], min_duration) # (1,)
    return contacts[0]

def count_label(gt_arr: np.ndarray, pred_arr: np.ndarray) -> Tuple[int, int, int]:
    '''
    gt_arr: (N,)
    pred_arr: (N,)
    return: tp, fp, fn
    '''
    tp = np.sum(np.logical_and(gt_arr, pred_arr))
    fp = np.sum(np.logical_and(np.logical_not(gt_arr), pred_arr))
    fn = np.sum(np.logical_and(gt_arr, np.logical_not(pred_arr)))
    return tp, fp, fn

def compute_metric(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    '''
    return: precision, recall, f1
    '''
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def compute_intra_metric(gt_motion:np.ndarray, pred_motion:np.ndarray, threshold: float, min_duration: int) -> Tuple[float, float, float]:
    '''
    gt_motion: (T, 2, J, 3)
    pred_motion: (T, 2, J, 3)
    return: tp, fp, fn
    '''
    gt_left_contacts = intra_contact(
        gt_motion[:, 0, :, :],
        threshold, min_duration
    ) # (P,)
    gt_right_contacts = intra_contact(
        gt_motion[:, 1, :, :],
        threshold, min_duration
    ) # (P,)
    gt_contacts = np.concatenate([gt_left_contacts, gt_right_contacts], axis=0) # (2P,)

    pred_left_contacts = intra_contact(
        pred_motion[:, 0, :, :],
        threshold, min_duration
    ) # (P,)
    pred_right_contacts = intra_contact(
        pred_motion[:, 1, :, :],
        threshold, min_duration
    ) # (P,)
    pred_contacts = np.concatenate([pred_left_contacts, pred_right_contacts], axis=0) # (2P,)

    return count_label(gt_contacts, pred_contacts)

def compute_inter_metric(gt_motion:np.ndarray, pred_motion:np.ndarray, threshold: float, min_duration: int) -> Tuple[float, float, float]:
    '''
    gt_motion: (T, 2, J, 3)
    pred_motion: (T, 2, J, 3)
    return: tp, fp, fn
    '''
    gt_left_contacts = inter_contact(
        gt_motion[:, 0, :, :],
        gt_motion[:, 1, :, :],
        another_is_right=True,
        threshold=threshold,
        min_duration=min_duration
    ) # (N_tips,)
    gt_right_contacts = inter_contact(
        gt_motion[:, 1, :, :],
        gt_motion[:, 0, :, :],
        another_is_right=False,
        threshold=threshold,
        min_duration=min_duration
    ) # (N_tips,)
    gt_contacts = np.concatenate([gt_left_contacts, gt_right_contacts], axis=0) # (2N_tips,)

    pred_left_contacts = inter_contact(
        pred_motion[:, 0, :, :],
        pred_motion[:, 1, :, :],
        another_is_right=True,
        threshold=threshold,
        min_duration=min_duration
    ) # (N_tips,)
    pred_right_contacts = inter_contact(
        pred_motion[:, 1, :, :],
        pred_motion[:, 0, :, :],
        another_is_right=False,
        threshold=threshold,
        min_duration=min_duration
    ) # (N_tips,)
    pred_contacts = np.concatenate([pred_left_contacts, pred_right_contacts], axis=0) # (2N_tips,)

    return count_label(gt_contacts, pred_contacts)

def compute_lenient_inter_metric(gt_motion:np.ndarray, pred_motion:np.ndarray, threshold: float, min_duration: int) -> Tuple[float, float, float]:
    '''
    gt_motion: (T, 2, J, 3)
    pred_motion: (T, 2, J, 3)
    return: tp, fp, fn
    '''
    gt_contacts = lenient_contact(
        gt_motion[:, 0, :, :],
        gt_motion[:, 1, :, :],
        another_is_right=True,
        threshold=threshold,
        min_duration=min_duration
    ) # bool
    pred_contacts = lenient_contact(
        pred_motion[:, 0, :, :],
        pred_motion[:, 1, :, :],
        another_is_right=True,
        threshold=threshold,
        min_duration=min_duration
    ) # bool

    gt_arr = np.array([gt_contacts], dtype=bool)  # (1,)
    pred_arr = np.array([pred_contacts], dtype=bool)  # (1,)

    return count_label(gt_arr, pred_arr)

def give_contact_label(motion:np.ndarray, tip_tip_threshold: float, tip_palm_threshold: float, palm_palm_threshold: float) -> np.ndarray:
    '''
    motion: (T, 2, J, 3)
    return: (T, total_pairs)
    '''
    left_motion = motion[:, 0, :, :] # (T, J, 3)
    right_motion = motion[:, 1, :, :] # (T, J, 3)
    left_palm = get_palm_glob(left_motion, is_right=False) # (T, M, 3)
    right_palm = get_palm_glob(right_motion, is_right=True) # (T, M, 3)

    left_intra_contacts = tip_tip_contact_occurs(
        p1=left_motion[:, intra_tip_pairs[:, 0], :],
        p2=left_motion[:, intra_tip_pairs[:, 1], :],
        threshold=tip_tip_threshold,
    ) # (T, intra_contact_pairs)
    right_intra_contacts = tip_tip_contact_occurs(
        p1=right_motion[:, intra_tip_pairs[:, 0], :],
        p2=right_motion[:, intra_tip_pairs[:, 1], :],
        threshold=tip_tip_threshold,
    ) # (T, intra_contact_pairs)

    left_tip_right_palm_contacts = tip_palm_contact_occurs(
        p=left_motion[:, all_tips, :],
        palm=right_palm,
        threshold=tip_palm_threshold,
    ) # (T, N_tips)
    right_tip_left_palm_contacts = tip_palm_contact_occurs(
        p=right_motion[:, all_tips, :],
        palm=left_palm,
        threshold=tip_palm_threshold,
    ) # (T, N_tips)

    palm_palm_contacts = palm_palm_contact_occurs(
        left_palm,
        right_palm,
        threshold=palm_palm_threshold
    )[:, np.newaxis] # (T, 1)

    contact_labels = np.concatenate(
        [left_intra_contacts, right_intra_contacts, left_tip_right_palm_contacts, right_tip_left_palm_contacts, palm_palm_contacts],
        axis=1
    ) # (T, total_pairs)
    return contact_labels