import numpy as np
import json

from ..constant import JOINT_NAME_INDEX_MAP
from .single_motioncode import MotionCoder, split_events, split_contact_events

def compute_topk_closest_vectors(A, B, k=10):
    '''
    A: (T, 3)
    B: (T, n, 3)
    return: (T, k, 3)
    '''
    A_expanded = A[:, np.newaxis, :] # (T, 1, 3)
    diff = B - A_expanded # (T, n, 3)
    dists = np.linalg.norm(diff, axis=2) # (T, n)

    topk_indices = np.argpartition(dists, kth=k, axis=1)[:, :k] # (T, k)

    row_indices = np.arange(A.shape[0])[:, np.newaxis] # (T, 1)
    sorted_order = np.argsort(dists[row_indices, topk_indices], axis=1)
    topk_sorted_indices = topk_indices[row_indices, sorted_order]

    C = diff[row_indices, topk_sorted_indices] # (T, k, 3)
    return C

def compute_closest_distances(A, B):
    '''
    A: (T, n, 3)
    B: (T, n, 3)
    return
        - C: (T, n)
        - C_vec: (T, n, 3)
    '''
    T, n, _ = A.shape

    A_exp = A[:, :, np.newaxis, :] # (T, n, 1, 3)
    B_exp = B[:, np.newaxis, :, :] # (T, 1, n, 3)

    diff = A_exp - B_exp # (T, n, n, 3)
    dists = np.linalg.norm(diff, axis=-1) # (T, n, n)

    C = np.min(dists, axis=2) # (T, n)

    indices = np.argmin(dists, axis=2) # (T, n)
    C_vec = np.empty((T, n, 3))
    for t in range(T):
        for i in range(n):
            j = indices[t, i]
            C_vec[t, i] = -A[t, i] + B[t, j]

    return C, C_vec

def topk_smallest_elements(C, k):
    topk_part = np.partition(C, kth=k-1, axis=1)[:, :k]
    D = np.sort(topk_part, axis=1)
    return D

class BihandMotionCoder(object):
    def __init__(self, motion:np.ndarray):
        self.lft_motion = motion[:, 0] # (T, J, 3)
        self.rgt_motion = motion[:, 1] # (T, J, 3)
        # print(f"lft motion z mean: {np.mean(self.lft_motion[:, :, 2])}")
        # print(f"rgt motion z mean: {np.mean(self.rgt_motion[:, :, 2])}")
        self.lft_motioncode = MotionCoder(self.lft_motion, isright=False)
        self.rgt_motioncode = MotionCoder(self.rgt_motion, isright=True)
        self.tip_index = [
            JOINT_NAME_INDEX_MAP['thumb_tip'],
            JOINT_NAME_INDEX_MAP['index_tip'],
            JOINT_NAME_INDEX_MAP['middle_tip'],
            JOINT_NAME_INDEX_MAP['ring_tip'],
            JOINT_NAME_INDEX_MAP['pinky_tip'],
        ]
        self.unit_len = self.lft_motioncode.unit_len

    def get_finger_finger_distance(self):
        finger_finger_distance = []
        tips = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        for i in range(len(tips)):
            for j in range(len(tips)):
                joint_pair = [JOINT_NAME_INDEX_MAP[tips[i]], JOINT_NAME_INDEX_MAP[tips[j]]]
                dist = np.linalg.norm(self.lft_motion[:, joint_pair[0], :] - self.rgt_motion[:, joint_pair[1], :], axis=1) # (T,)
                finger_finger_distance.append((
                    ('left_' + tips[i], "right_" + tips[j]),
                    dist
                ))

                # if tips[i] == 'index_tip' and tips[j] == 'index_tip':
                #     print(f"{joint_pair}:\n{dist}")
        self.finger_finger_distance = finger_finger_distance

    def get_finger_palm_distance(self):
        finger_palm_distance, palm_finger_distance = [], []
        tips = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        for i in range(len(tips)):
            closest_vectors = compute_topk_closest_vectors(
                self.lft_motion[:, JOINT_NAME_INDEX_MAP[tips[i]], :], # (T, 3)
                self.rgt_motioncode.palm_points_glob, # (T, n, 3)
                k=5
            ) # (T, k, 3)
            closest_mean_dist = np.mean(np.linalg.norm(closest_vectors, axis=2), axis=1) # (T,)
            finger_palm_distance.append((
                ('left_' + tips[i], "right_palm"),
                closest_mean_dist
            ))

        for i in range(len(tips)):
            closest_vectors = compute_topk_closest_vectors(
                self.rgt_motion[:, JOINT_NAME_INDEX_MAP[tips[i]], :], # (T, 3)
                self.lft_motioncode.palm_points_glob, # (T, n, 3)
                k=5
            )
            closest_mean_dist = np.mean(np.linalg.norm(closest_vectors, axis=2), axis=1)
            palm_finger_distance.append((
                ('right_' + tips[i], "left_palm"),
                closest_mean_dist
            ))

        self.finger_palm_distance = finger_palm_distance
        self.palm_finger_distance = palm_finger_distance

    def get_palm_palm_distance(self):
        closest_dist, closest_vec = compute_closest_distances(
            self.lft_motioncode.palm_points_glob,
            self.rgt_motioncode.palm_points_glob
        )
        self.palm_palm_vec = np.mean(self.rgt_motioncode.palm_points_glob, axis=1) - np.mean(self.lft_motioncode.palm_points_glob, axis=1) # (T, 3)
        # print(f"palm_palm_vec: {self.palm_palm_vec}")
        closest_dist_topk = topk_smallest_elements(closest_dist, k=30)
        closest_dist_mean = np.mean(closest_dist_topk, axis=-1) # (T,)
        self.palm_palm_distance = [(
            ('left_palm', 'right_palm'),
            closest_dist_mean
        )]

    def extract_feats(self):
        # print("GET LEFT MOTION FEATS")
        self.lft_motioncode.extract_feats()
        # print("GET RIGHT MOTION FEATS")
        self.rgt_motioncode.extract_feats()
        self.get_finger_finger_distance()
        self.get_finger_palm_distance()
        self.get_palm_palm_distance()

    def split_finger_finger_events(self):
        contact_thre = 0.020

        self.finger_finger_distance_events = dict()
        for pair, spacing in self.finger_finger_distance:
            events = split_contact_events(spacing, contact_thre)
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'}
                    for x in events
                ]
                self.finger_finger_distance_events[', '.join(pair)] = events

    def split_finger_palm_events(self):
        spacing_thre = [
            (0, 0.025, 'Contact'),
            (0.025, 0.035, 'Near'),
            (0.035, 100, 'Far')
        ]
        vel_thre = [
            (0, 0.025, 'Slow'),
            (0.025, 0.07, 'Medium'),
            (0.07, 100, 'Fast')
        ]
        self.finger_palm_distance_events = dict()
        for pair, spacing in self.finger_palm_distance:
            events = split_events(
                spacing, spacing_thre, vel_thre,
                delta_thre=0.03,
                min_duration=6,
                diff_state=True,
                unit_len=self.unit_len,
            )
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'}
                    for x in events
                ]
                self.finger_palm_distance_events[', '.join(pair)] = events

        for pair, spacing in self.palm_finger_distance:
            events = split_events(spacing, spacing_thre, vel_thre)
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'}
                    for x in events
                ]
                self.finger_palm_distance_events[', '.join(pair)] = events


    def split_palm_palm_events(self):
        spacing_thre = [
            (0, 0.04, 'Contact'),
            (0.04, 0.08, 'Near'),
            (0.08, 0.12, 'Medium'),
            (0.12, 100, 'Far')
        ]
        vel_thre = [
            (0, 0.025, 'Slow'),
            (0.025, 0.07, 'Medium'),
            (0.07, 100, 'Fast')
        ]
        self.palm_palm_distance_events = dict()
        for pair, spacing in self.palm_palm_distance:
            events = split_events(
                x=spacing,
                x_intervals=spacing_thre,
                v_intervals=vel_thre,
                delta_thre=0.05,
                min_duration=6,
                diff_state=True,
                unit_len=self.unit_len,
            )
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'}
                    for x in events
                ]
                self.palm_palm_distance_events[', '.join(pair)] = events

        relation_thre = [
            [
                (-100, -0.02, 'right hand is to the LEFT of the left hand.'),
                (-0.02, 0.02, 'right hand is ALIGNED with the left hand.'),
                (0.02, 100, 'right hand is to the RIGHT of the left hand.')
            ],
            [
                (-100, -0.02, 'right hand is to the BACK of the left hand.'),
                (-0.02, 0.02, 'right hand is ALIGNED with the left hand.'),
                (0.02, 100, 'right hand is to the FRONT of the left hand.')],
            [
                (-100, -0.02, 'right hand is to the DOWN of the left hand.'),
                (-0.02, 0.02, 'right hand is ALIGNED with the left hand.'),
                (0.02, 100, 'right hand is to the UP of the left hand.')
            ]
        ]
        self.palm_palm_relative_position_events = dict()
        for i, axis in enumerate(['left-right', 'front-back', 'up-down']):
            component = self.palm_palm_vec[:, i]
            # print(f"axis: {axis} component:\n{component}")
            events = split_events(
                x=component,
                x_intervals=relation_thre[i],
                v_intervals=vel_thre,
                delta_thre=0.05,
                min_duration=4,
                diff_state=True,
                unit_len=self.unit_len,
            )
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'}
                    for x in events
                ]
                self.palm_palm_relative_position_events[axis] = events

    def extract_events(self):
        self.lft_motioncode.extract_events()
        self.rgt_motioncode.extract_events()
        self.split_finger_finger_events()
        self.split_finger_palm_events()
        self.split_palm_palm_events()

    def get_json(self):
        events_summary = {
            'frame_count': self.lft_motion.shape[0],
            'left_hand_events': self.lft_motioncode.print_json(),
            'right_hand_events': self.rgt_motioncode.print_json(),
            'two_hand_relationships': {
                'finger_tip_contact': self.finger_finger_distance_events,
                'finger_palm_distance': self.finger_palm_distance_events,
                'palm_palm_distance': self.palm_palm_distance_events,
                'palm_palm_relative_position': self.palm_palm_relative_position_events
            }
        }
        return events_summary

    def generate_motion_codes(self):
        self.extract_feats()
        self.extract_events()