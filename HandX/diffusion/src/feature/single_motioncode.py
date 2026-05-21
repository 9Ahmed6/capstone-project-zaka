import numpy as np
import json
from scipy.spatial.transform import Rotation as R

from ..constant import SKELETON_CHAIN, JOINT_NAME_INDEX_MAP, JOINT_INDEX_NAME_MAP, SKELETON_CHAIN_NAME


class InvalidJointDataError(Exception):
    """Exception raised when joint positions are invalid (e.g., zero vectors in coordinate calculation)."""
    pass

def consistent_sign(x:np.ndarray, tolerance:float):
    signs = np.sign(x)
    total = len(signs)
    pos_count = np.sum(signs == 1)
    neg_count = np.sum(signs == -1)
    zero_count = np.sum(signs == 0)

    if (pos_count + zero_count) / total >= tolerance:
        return 1, True
    elif (neg_count + zero_count) / total >= tolerance:
        return -1, True
    return 0, False

def process_wrist_traj(axis, events):
    vel_dir = {
        'x': {'pos': 'left-to-right', 'neg': 'right-to-left'},
        'y': {'pos': 'back-to-front', 'neg': 'front-to-back'},
        'z': {'pos': 'down-to-up', 'neg': 'up-to-down'},
    }
    res = []
    for e in events:
        new_e = {
            'start': e['start'],
            'end': e['end'],
            'direction': vel_dir[axis][e['direction']],
            'v_des': e['v_des'],
        }
        res.append(new_e)
    return res, {'x': 'left-right', 'y': 'front-back', 'z': 'down-up'}[axis]



def match_interval(x, interval):
    for l, r, label in interval:
        if l <= x < r:
            return label
    return None

def split_contact_events(
    spacing, contact_threshold
):
    events = []
    T = spacing.shape[0]
    i = 0
    while i < T:
        if spacing[i] > contact_threshold:
            i += 1
            continue
        j = i + 1
        while j < T and spacing[j] <= contact_threshold:
            j += 1
        events.append({
            'start': i,
            'end': j - 1,
            'constant_des': 'Contact'
        })
        i = j
    return events


def split_events(
    x, x_intervals, v_intervals,
    v_0thre=None, delta_thre=None, min_duration=6,
    diff_state=True,
    unit_len=6, fps=30, v_abs=True,
    debug=False, debug_l=None, debug_r=None
):
    events = []
    v = np.concatenate([np.zeros((1,)), x[1:] - x[:-1]], axis=0) * fps # (T,)


    if v_0thre is not None:
        v[v < v_0thre] = 0

    T = x.shape[0]
    i = 0
    while (i < T):
        j = T
        while(j > i):
            if delta_thre is not None and np.abs(x[j - 1] - x[i]) < delta_thre:
                j -= unit_len
                continue
            if j - i < min_duration:
                j = i
                break
            sign, consistent = consistent_sign(v[i : j], tolerance=0.8)
            if consistent:
                v_ave = np.mean(v[i : j], axis=0)
                if v_abs:
                    v_des = match_interval(np.abs(v_ave), v_intervals)

                else:
                    v_des = match_interval(v_ave, v_intervals)

                start_des = match_interval(x[i], x_intervals)
                end_des = match_interval(x[j - 1], x_intervals)

                if start_des != end_des or not diff_state:
                    events.append({
                        'start': i, 'end': j,
                        'direction': 'pos' if sign > 0 else 'neg',
                        'start_des': start_des, 'end_des': end_des,
                        'v_des': v_des
                    })
                    break
            j -= unit_len
        i = j if j > i else i + unit_len

    if len(events) == 0 and diff_state:
        x_mean = np.mean(x)
        des = match_interval(x_mean, x_intervals)

        events.append({
            'start': 0, 'end': T,
            'constant_des': des
        })
    return events


def signed_angle_ab_batch(a_whole, b_whole, isright=False, is_thumb=False, ignore_sign=False, debug=False, debug_l=None, debug_r=None):
    a, b = a_whole.copy(), b_whole.copy()
    if not is_thumb:
        a[:, 1] = 0
        b[:, 1] = 0
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True) # (B, 3)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True) # (B, 3)

    n = np.cross(a_norm, b_norm) # (B, 3)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True) # (B, 1)

    cos_theta = np.sum(a_norm * b_norm, axis=1) # (B,)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # (B,)
    angle = np.arccos(cos_theta) # (B,)
    zero_mask = (angle / np.pi * 180) < 11 # (1, B, 1)

    n_unit = np.zeros_like(n) # (B, 3)
    non_zero_mask = ~zero_mask
    n_unit[non_zero_mask] = n[non_zero_mask] / n_norm[non_zero_mask] # (B, 3)

    if not ignore_sign:
        if not is_thumb:
            sign = np.sign(n_unit[:, 1]) # (B,)
            if not isright:
                sign = -sign
        else:
            sign = -np.sign(n_unit[:, 2]) # (B,)
    else:
        sign = np.ones_like(angle)

    signed_angle = sign * angle # (B,)
    signed_angle[zero_mask] = 0.0


    return signed_angle


class MotionCoder:
    def __init__(self, motion, isright):
        self.motion = motion.reshape(-1, 21, 3)
        self.unit_len = 1
        self.fps = 30
        self.spf = 1 / self.fps
        self.isright = isright

    def get_palm(self):
        self.n_palm_point = 100
        self.palm_points = []
        for i in range(len(SKELETON_CHAIN) - 1):
            n = self.n_palm_point // (len(SKELETON_CHAIN) - 1)
            chain_i = SKELETON_CHAIN[i]
            chain_nxt = SKELETON_CHAIN[i + 1]
            u = np.random.rand(n)
            v = np.random.rand(n)
            mask = (u + v) > 1
            u[mask] = 1 - u[mask]
            v[mask] = 1 - v[mask]
            w = 1 - u - v
            self.palm_points.append(
                [u, v, w, 0, chain_i[1], chain_nxt[1]]
            )

        self.palm_points_glob = []
        for points in self.palm_points:
            u, v, w, a, b, c = points
            a, b, c = self.motion[:, [a]], self.motion[:, [b]], self.motion[:, [c]]
            u, v, w = u.reshape(1, -1, 1), v.reshape(1, -1, 1), w.reshape(1, -1, 1)
            glob = u * a + v * b + w * c
            self.palm_points_glob.append(glob)
        self.palm_points_glob = np.concatenate(self.palm_points_glob, axis=1)

    def get_local_coordinate(self):
        wrist_pos = self.motion[:, JOINT_NAME_INDEX_MAP['wrist']] # (T, 3)
        middle_finger_mcp = self.motion[:, JOINT_NAME_INDEX_MAP['middle_mcp']] # (T, 3)
        x_axis = middle_finger_mcp - wrist_pos
        x_axis_norm = np.linalg.norm(x_axis, axis=1, keepdims=True)

        # Check for zero vectors (invalid joint data)
        # if np.any(x_axis_norm < 1e-8):
        #     invalid_frames = np.where(x_axis_norm.flatten() < 1e-8)[0]
        #     raise InvalidJointDataError(
        #         f"Zero vector detected in x_axis (middle_mcp - wrist) at frame(s): {invalid_frames.tolist()[:5]}... "
        #         f"({len(invalid_frames)} total invalid frames)"
        #     )
        x_axis = x_axis / x_axis_norm

        index_finger_mcp = self.motion[:, JOINT_NAME_INDEX_MAP['index_mcp']]
        y_axis = index_finger_mcp - wrist_pos
        y_axis_norm = np.linalg.norm(y_axis, axis=1, keepdims=True)

        # Check for zero vectors (invalid joint data)
        # if np.any(y_axis_norm < 1e-8):
        #     invalid_frames = np.where(y_axis_norm.flatten() < 1e-8)[0]
        #     raise InvalidJointDataError(
        #         f"Zero vector detected in y_axis (index_mcp - wrist) at frame(s): {invalid_frames.tolist()[:5]}... "
        #         f"({len(invalid_frames)} total invalid frames)"
        #     )
        y_axis = y_axis / y_axis_norm

        z_axis = np.cross(x_axis, y_axis)
        z_axis_norm = np.linalg.norm(z_axis, axis=1, keepdims=True)

        # Check for zero vectors (invalid joint data)
        if np.any(z_axis_norm < 1e-8):
            invalid_frames = np.where(z_axis_norm.flatten() < 1e-8)[0]
            raise InvalidJointDataError(
                f"Zero vector detected in z_axis (cross product) at frame(s): {invalid_frames.tolist()[:5]}... "
                f"({len(invalid_frames)} total invalid frames)"
            )
        z_axis = z_axis / z_axis_norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.linalg.norm(y_axis, axis=1, keepdims=True)

        # Check for zero vectors (invalid joint data)
        if np.any(y_axis_norm < 1e-8):
            invalid_frames = np.where(y_axis_norm.flatten() < 1e-8)[0]
            raise InvalidJointDataError(
                f"Zero vector detected in final y_axis (cross product) at frame(s): {invalid_frames.tolist()[:5]}... "
                f"({len(invalid_frames)} total invalid frames)"
            )
        y_axis = y_axis / y_axis_norm

        self.local_axis = np.stack([x_axis, y_axis, z_axis], axis=2) # (T, 3, 3)

        pos_diff = self.motion - wrist_pos[:, np.newaxis, :] # (T, J, 3)

        local_x = np.sum(pos_diff * x_axis[:, np.newaxis, :], axis=2) # (T, J)
        local_y = np.sum(pos_diff * y_axis[:, np.newaxis, :], axis=2) # (T, J)
        local_z = np.sum(pos_diff * z_axis[:, np.newaxis, :], axis=2) # (T, J)

        local_motion = np.stack([local_x, local_y, local_z], axis=2) # (T, J, 3)
        self.local_motion = local_motion

    def get_wrist_traj(self):
        self.wrist_traj = self.motion[:, 0] # (T, 3)

    def get_finger_flexing(self):
        finger_flexing = []
        for finger_chain_index, finger_chain in enumerate(SKELETON_CHAIN):
            finger_chain_name = SKELETON_CHAIN_NAME[finger_chain_index]
            for i in range(3):
                j1, j2, j3 = finger_chain[i:i+3]
                v12 = self.local_motion[:, j2] - self.local_motion[:, j1] # (T, 3)
                v23 = self.local_motion[:, j3] - self.local_motion[:, j2] # (T, 3)

                if finger_chain_name == 'thumb':
                    if i <= 0:
                        continue
                    elif i == 1:
                        theta = signed_angle_ab_batch(
                            v12, v23,
                            isright=self.isright,
                            is_thumb=True,
                            ignore_sign=False,
                            debug=False,
                        )
                    else:
                        theta = signed_angle_ab_batch(
                            v12, v23,
                            isright=self.isright,
                            is_thumb=True,
                            ignore_sign=True,
                            debug=False,
                        )

                else:
                    theta = signed_angle_ab_batch(
                        v12, v23,
                        isright=self.isright,
                        is_thumb=False,
                        debug=False,
                    )

                finger_flexing.append((JOINT_INDEX_NAME_MAP[j2], theta))

        self.finger_flexing = finger_flexing

    def get_finger_spacing(self):
        finger_spacing = []
        spacing_pairs = [
            ('index_pip', 'index_mcp', 'middle_pip', 'middle_mcp'),
            ('middle_pip',  'middle_mcp', 'ring_pip', 'ring_mcp'),
            ('ring_pip', 'ring_mcp', 'pinky_pip', 'pinky_mcp'),
        ]
        for pair in spacing_pairs:
            finger1_segment = [JOINT_NAME_INDEX_MAP[pair[0]], JOINT_NAME_INDEX_MAP[pair[1]]]
            finger2_segment = [JOINT_NAME_INDEX_MAP[pair[2]], JOINT_NAME_INDEX_MAP[pair[3]]]

            finger1_vector = self.local_motion[:, finger1_segment[0]] - self.local_motion[:, finger1_segment[1]] # (T, 3)
            finger2_vector = self.local_motion[:, finger2_segment[0]] - self.local_motion[:, finger2_segment[1]] # (T, 3)

            # # projection onto the palm plane
            # finger1_vector = finger1_vector[:, :2]# (T, 2)
            # finger2_vector = finger2_vector[:, :2]# (T, 2)

            finger1_vector_length = np.linalg.norm(finger1_vector, axis=1) # (T)
            finger2_vector_length = np.linalg.norm(finger2_vector, axis=1) # (T)

            # zero_mask1 = finger1_vector_length < 0.02
            # zero_mask2 = finger2_vector_length < 0.02

            finger1_vector_norm = np.zeros_like(finger1_vector) # (T, 2)
            finger2_vector_norm = np.zeros_like(finger2_vector) # (T, 2)
            # finger1_vector_norm[~zero_mask1] = finger1_vector[~zero_mask1] / finger1_vector_length[~zero_mask1][:, np.newaxis]
            # finger2_vector_norm[~zero_mask2] = finger2_vector[~zero_mask2] / finger2_vector_length[~zero_mask2][:, np.newaxis]
            finger1_vector_norm = finger1_vector / finger1_vector_length[:, np.newaxis]
            finger2_vector_norm = finger2_vector / finger2_vector_length[:, np.newaxis]


            angle = np.arccos(np.clip(np.sum(finger1_vector_norm * finger2_vector_norm, axis=1), -1.0, 1.0)) # (T,)
            # angle[zero_mask1 | zero_mask2] = 0.0


            finger_spacing.append((
                (pair[0].split('_')[0], pair[2].split('_')[0]),
                angle
            ))

            # print(f"{(pair[0].split('_')[0], pair[2].split('_')[0])}:\n{angle * 180 / np.pi}")

        self.finger_spacing = finger_spacing

    def get_finger_distance(self):
        finger_distance = []
        tips = ['index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']

        for i in range(len(tips)):
            joint_pair = [JOINT_NAME_INDEX_MAP[tips[i]], JOINT_NAME_INDEX_MAP['thumb_tip']]
            finger_distance.append((
                (tips[i], 'thumb_tip'),
                np.linalg.norm(
                    self.local_motion[:, joint_pair[0]] - self.local_motion[:, joint_pair[1]], # (T, 3)
                    axis=1
                )
            ))
        self.finger_distance = finger_distance

    def extract_feats(self):
        self.get_palm()
        self.get_local_coordinate()
        self.get_wrist_traj()
        self.get_finger_flexing()
        self.get_finger_spacing()
        self.get_finger_distance()

    def split_wrist_traj_events(self, x):
        vel_thre = [(0, 0.03, "Slow"), (0.03, 0.15, "Medium"), (0.15, 100, 'Fast')]
        self.wrist_traj_events = dict()

        for i, axis in enumerate(['x', 'y', 'z']):
            traj = x[:, i]
            events = split_events(
                traj,
                x_intervals=[], v_intervals=vel_thre, delta_thre=0.03,
                min_duration=4,
                diff_state=False, unit_len=self.unit_len, fps=self.fps
            )

            if len(events) > 0:
                res, dir = process_wrist_traj(axis, events)
                self.wrist_traj_events[dir] = res

    def split_finger_flexing_events(self, x):
        flex_thre = [
            (-np.pi, -np.pi , "Hyper extend"),
            (-np.pi , np.pi / 6, 'Fully extended'),
            (np.pi / 6, np.pi / 3, 'Partially bent'),
            (np.pi / 3, np.pi, "Fully bent")
        ]
        vel_thre = [
            (0, 0.15, "Slow"),
            (0.15, 1.3, "Medium"),
            (1.3, 100, 'Fast')
        ]
        self.finger_flexing_events = dict()
        for joint, flex in x:
            events = split_events(
                flex,
                flex_thre,
                vel_thre, delta_thre=np.pi/6,
                min_duration=6,
                unit_len=self.unit_len, fps=self.fps,
                debug=False,
                # debug=False,
                debug_l=0,
                debug_r=52
            )
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'} for x in events
                ]
                self.finger_flexing_events[joint] = events

    def split_finger_spacing_events(self, x):
        angle_thre = [
            (0, np.pi / 9, 'Closed'),
            (np.pi / 9, np.inf, "Open")
        ]
        vel_thre = [
            (0, 0.03, "Slow"),
            (0.03, 1.3, "Medium"),
            (1.3, 100, 'Fast')
        ]
        self.finger_spacing_events = dict()
        for pair, spacing in x:
            events = split_events(spacing, angle_thre, vel_thre, delta_thre=np.pi/10, min_duration=6, unit_len=self.unit_len, fps=self.fps)
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'} for x in events
                ]
                pair = [x.split('_')[0] for x in pair]
                self.finger_spacing_events[', '.join(pair)] = events

    def split_finger_distance_events(self, x):
        contact_thre = 0.02
        self.finger_distance_events = dict()
        for pair, spacing in x:
            events = split_contact_events(spacing, contact_thre)
            if len(events) > 0:
                events = [
                    {k: v for k, v in x.items() if k != 'direction'} for x in events
                ]
                self.finger_distance_events[', '.join(pair)] = events

    def extract_events(self):
        self.split_wrist_traj_events(self.wrist_traj)
        self.split_finger_flexing_events(self.finger_flexing)
        self.split_finger_spacing_events(self.finger_spacing)
        self.split_finger_distance_events(self.finger_distance)


    def print_json(self, json_file=None):
        events_summary = {
            'finger_flexing': self.finger_flexing_events,
            'finger_spacing': self.finger_spacing_events,
            'finger_tip_contact': self.finger_distance_events,
            'wrist_trajectory': self.wrist_traj_events,
        }

        if json_file is not None:
            with open(json_file, 'w') as f:
                json.dump(events_summary, f, indent=4)

        else:
            return events_summary

    def generate_motion_codes(self):
        self.extract_feats()
        self.extract_events()