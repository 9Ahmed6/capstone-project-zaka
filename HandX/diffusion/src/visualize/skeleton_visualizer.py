import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ..constant import SKELETON_CHAIN, JOINT_NAME_INDEX_MAP
from ..utils import get_motion_data_boundary


FINGER_COLOR_MAP = {
    'wrist': 'black',
    'thumb': 'red',
    'index': 'green',
    'middle': 'blue',
    'ring': 'orange',
    'pinky': 'purple'
}

INDEX_TO_FINGER = {}
for name, index in JOINT_NAME_INDEX_MAP.items():
    if 'thumb' in name:
        INDEX_TO_FINGER[index] = 'thumb'
    elif 'index' in name:
        INDEX_TO_FINGER[index] = 'index'
    elif 'middle' in name:
        INDEX_TO_FINGER[index] = 'middle'
    elif 'ring' in name:
        INDEX_TO_FINGER[index] = 'ring'
    elif 'pinky' in name:
        INDEX_TO_FINGER[index] = 'pinky'
    elif 'wrist' in name:
        INDEX_TO_FINGER[index] = 'wrist'

FINGER_INDICES = {finger: [] for finger in FINGER_COLOR_MAP.keys()}
for index, finger in INDEX_TO_FINGER.items():
    FINGER_INDICES[finger].append(index)



class Skeleton_Visualize_Helper:
    hand_link_colors = {'left': 'cyan', 'right': 'magenta'}
    def __init__(self, ax:Axes3D, left_motion:np.ndarray|None=None, right_motion:np.ndarray|None=None, title:str | None = None):
        assert left_motion is not None or right_motion is not None, "At least one hand motion should be provided"
        self.ax = ax
        self.left_motion = left_motion # (T, J, 3)
        self.right_motion = right_motion # (T, J, 3)
        self.title = title

    def initialize_ax(self):
        if self.left_motion is None:
            motion_data = self.right_motion
        elif self.right_motion is None:
            motion_data = self.left_motion
        else:
            motion_data = np.concatenate([self.left_motion, self.right_motion], axis=1)

        xmin, xmax, ymin, ymax, zmin, zmax = get_motion_data_boundary(motion_data)
        x_range, y_range, z_range = xmax - xmin, ymax - ymin, zmax - zmin
        max_range = max(x_range, y_range, z_range)
        x_mid, y_mid, z_mid = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2

        self.ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        self.ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        self.ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        if self.title:
            self.ax.set_title(self.title)

        self.link_plots = dict()
        self.joint_scatters = dict()

        for hand in ['left', 'right']:
            self.link_plots[hand] = self.ax.plot(
                [], [], [],
                color=self.hand_link_colors[hand],
                label=f'{hand.capitalize()} Hand Links'
            )[0]

            self.joint_scatters[hand] = {}
            for finger, color in FINGER_COLOR_MAP.items():
                label = finger.upper() if hand == 'left' else None
                self.joint_scatters[hand][finger] = self.ax.scatter(
                    [], [], [],
                    color=color,
                    label=label
                )

    def draw_single_hand(self, hand, frame):
        motion = self.left_motion if hand == 'left' else self.right_motion
        if motion is None or frame >= motion.shape[0]:
            return

        current_pos = motion[frame] # (J, 3)

        all_x, all_y, all_z = [], [], []
        for i, chain in enumerate(SKELETON_CHAIN):
            all_x.extend(current_pos[chain, 0].tolist() + [np.nan])
            all_y.extend(current_pos[chain, 1].tolist() + [np.nan])
            all_z.extend(current_pos[chain, 2].tolist() + [np.nan])
        self.link_plots[hand].set_data(all_x, all_y)
        self.link_plots[hand].set_3d_properties(all_z)


        for finger, indices in FINGER_INDICES.items():
            if not indices:
                continue

            points = current_pos[indices] # (num_points_in_finger, 3)
            self.joint_scatters[hand][finger]._offsets3d = (
                points[:, 0],
                points[:, 1],
                points[:, 2]
            )

    def draw(self, frame):
        self.draw_single_hand('left', frame)
        self.draw_single_hand('right', frame)