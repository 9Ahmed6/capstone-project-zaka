import warnings, math
from os.path import join as pjoin
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from typing import Literal

from torch.utils.data import Dataset
from .. import dist as dist_utils
from .tensors import motion_action_collate, motion_text_collate, motion_text_treble_collate
from ..metric.interaction import give_contact_label
from ...feature.single_motioncode import MotionCoder
from ...constant import INTRA_TIP_CONTACT_THRESH, TIP_PALM_CONTACT_THRESH, PALM_PALM_CONTACT_THRESH, SKELETON_CHAIN

class HandXDataset(Dataset):
    contact_label: bool
    normalize: bool
    repr: Literal['joint_pos', 'joint_rot', 'joint_pos_w_scalar_rot', "joint_pos_w_axisangle_rot"]
    data_file_name: str
    data_dir: str

    def __init__(self, split: str, debug=False, *args, **kwargs):
        super(HandXDataset, self).__init__()
        self.debug = debug
        self.data_dir = kwargs['data_dir']
        self.data_file_name = kwargs['data_file_name']
        self.repr = kwargs['repr']
        self.normalize = kwargs['normalize']
        self.contact_label = kwargs.get("contact_label", False)
        self.ratio = kwargs.get("ratio", 1.0)

        if split == 'train':
            self.data_file_name = "train_" + self.data_file_name
            self.mano_file_name = 'train_mano.npz'
        elif split == 'val':
            self.data_file_name = 'test_' + self.data_file_name
            self.mano_file_name = 'test_mano.npz'

        self.split = split
        self.load_motion()
        if self.normalize:
            self._calc_mean_std()

        self.collate_fn = motion_text_treble_collate

    def _calc_mean_std(self):
        if self.split == 'val':
            dist_utils.barrier()
            print("Val split doesn't need create mean and std files.")
            mean_std_path = pjoin(self.data_dir, f'mean_std_{self.repr}')
            assert Path(mean_std_path).exists(), f"Mean and std not found at {mean_std_path}. Please run the dataset preparation script."
            self.mean = np.load(pjoin(mean_std_path, 'mean.npy'))
            self.std = np.load(pjoin(mean_std_path, 'std.npy'))
            return

        total_frames = 0
        first_motion = next(iter(self.data_dict.values()))['motion']
        feature_shape = first_motion.shape[1:]

        sum_of_data = np.zeros(feature_shape, dtype=np.float64)
        sum_of_squares = np.zeros(feature_shape, dtype=np.float64)

        for data in self.data_dict.values():
            motion_data = data['motion']
            if motion_data.ndim != 3:
                raise ValueError(f"Expected motion data to be 3D, but got {motion_data.ndim}D shape.")

            num_frames_in_batch = motion_data.shape[0]

            total_frames += num_frames_in_batch
            sum_of_data += np.sum(motion_data, axis=0)
            sum_of_squares += np.sum(np.square(motion_data), axis=0)

        if total_frames == 0:
            warnings.warn("No frames found in the dataset. Mean and std will be zero.")
            self.mean = np.zeros(feature_shape)
            self.std = np.zeros(feature_shape)
        else:
            self.mean = sum_of_data / total_frames

            variance = (sum_of_squares / total_frames) - np.square(self.mean)
            variance[variance < 0] = 0
            self.std = np.sqrt(variance)
            self.std[self.std < 1e-4] = 1.0

        if dist_utils.is_main_process():
            save_path = (Path(self.data_dir) / f'mean_std_{self.repr}').as_posix()
            Path(save_path).mkdir(parents=True, exist_ok=True)

            # mylogger.info(f"Saving mean and std to {save_path}")
            np.save((Path(save_path) / 'mean.npy').as_posix(), self.mean)
            np.save((Path(save_path) / 'std.npy').as_posix(), self.std)

    def inv_transform(self, data):
        if isinstance(data, torch.Tensor):
            tmp_mean = torch.from_numpy(self.mean).to(data.device).float()
            tmp_std = torch.from_numpy(self.std).to(data.device).float()
            ret = data * tmp_std.reshape(-1) + tmp_mean.reshape(-1) # [B, T, J*C]
            return ret
        elif isinstance(data, np.ndarray):
            ret = data * self.std.reshape(-1) + self.mean.reshape(-1) # [B, T, J*C]
            return ret
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or np.ndarray.")

    def _get_axisangle_rotation(self, mano_pose:np.ndarray) -> np.ndarray:
        '''
        mano_pose: (T, 48)
        return: (T, J, 3)
        '''
        mano_pose = mano_pose.reshape(mano_pose.shape[0], -1, 3) # (T, 16, 3)
        zero_padding = np.zeros((mano_pose.shape[0], 21-16, 3)) # (T, 5, 3)
        return np.concatenate([mano_pose, zero_padding], axis=1) # (T, 21, 3)

    def _get_scalar_rotation(self, single_motion_seq:np.ndarray, side:Literal['left', 'right']):
        '''
        single_motion_seq: (T, J, 3)
        '''
        temp_motioncoder = MotionCoder(single_motion_seq, isright=(side=='right'))
        temp_motioncoder.get_local_coordinate()
        local_motion = temp_motioncoder.local_motion # (T, J, 3)

        additional_scalar_rotation = np.zeros((single_motion_seq.shape[0], single_motion_seq.shape[1])) # (T, J)
        for skeleton_chain in SKELETON_CHAIN:
            additional_scalar_rotation[:, skeleton_chain[0]] = 0
            additional_scalar_rotation[:, skeleton_chain[-1]] = 0
            for s in range(1, len(skeleton_chain) - 1):
                j = skeleton_chain[s]
                pre_j = skeleton_chain[s - 1]
                nxt_j = skeleton_chain[s + 1]
                v1 = (local_motion[:, j, :] - local_motion[:, pre_j, :])[:, [0, 2]] # (T, 2)
                v2 = (local_motion[:, nxt_j, :] - local_motion[:, j, :])[:, [0, 2]] # (T, 2)

                v1_direction_angle = np.arctan2(v1[:, 1], v1[:, 0]) # (T,)
                v2_direction_angle = np.arctan2(v2[:, 1], v2[:, 0]) # (T,)
                angle_diff = v2_direction_angle - v1_direction_angle # (T,)

                if side == 'right':
                    angle_diff = -angle_diff

                additional_scalar_rotation[:, j] = angle_diff

        return additional_scalar_rotation # (T, J)

    def load_motion(self):
        motion_data = dict(np.load(pjoin(self.data_dir, self.data_file_name), allow_pickle=True))
        for key in motion_data:
            motion_data[key] = motion_data[key].item()

        # mano_data = dict(np.load(pjoin(self.data_dir, self.mano_file_name), allow_pickle=True))
        # for key in tqdm(mano_data, desc=f"RANK {dist_utils.get_rank()} | loading mano data for {self.split} split"):
        #     mano_data[key] = mano_data[key].item()

        self.data_dict = dict()
        self.name_list = []
        self.length_list = []

        for clip_name in tqdm(list(motion_data.keys()), desc=f"RANK {dist_utils.get_rank()} | processing {self.split} data"):
            motion = motion_data[clip_name]['motion'] # (T, 2, J, 3)

            if self.repr == 'joint_pos_w_axisangle_rot':
                left_mano_pose = mano_data[clip_name]['left_pose'] # (T, 48)
                right_mano_pose = mano_data[clip_name]['right_pose'] # (T, 48)

                left_rot = self._get_axisangle_rotation(left_mano_pose) # (T, J, 3)
                right_rot = self._get_axisangle_rotation(right_mano_pose) # (T, J, 3)

                motion = np.concatenate([
                    motion,
                    np.stack([left_rot, right_rot], axis=1) # (T, 2, J, 3)
                ], axis=-1) # (T, 2, J, 6)

                motion = motion.reshape(motion.shape[0], -1, 6) # (T, 2J, 6)

                transl = np.mean((motion[:, 0, :3] + motion[:, 21, :3]) / 2, axis=0) # (3,)
                motion[:, :, :3] -= transl

            elif self.repr == 'joint_pos_w_scalar_rot':
                left_rot_scalar = self._get_scalar_rotation(motion[:, 0], side='left') # (T, J)
                left_rot_scalar = np.nan_to_num(left_rot_scalar, nan=0.0, posinf=0.0, neginf=0.0)
                right_rot_scalar = self._get_scalar_rotation(motion[:, 1], side='right') # (T, J)
                right_rot_scalar = np.nan_to_num(right_rot_scalar, nan=0.0, posinf=0.0, neginf=0.0)
                motion = np.concatenate([
                    motion,
                    np.stack([left_rot_scalar, right_rot_scalar], axis=1)[:, :, :, np.newaxis] # (T, 2, J, 1)
                ], axis=-1) # (T, 2, J, 4)

                motion = motion.reshape(motion.shape[0], -1, 4) # (T, 2J, 4)

                transl = np.mean((motion[:, 0, :3] + motion[:, 21, :3]) / 2, axis=0) # (3,)
                motion[:, :, :3] -= transl

            elif self.repr == 'joint_pos':
                transl = np.mean((motion[:, 0] + motion[:, 21]) / 2, axis=0) # (3,)
                motion[:, :, :3] -= transl
                pass

            else:
                raise NotImplementedError(f"Representation {self.repr} not implemented.")

            assert len(motion_data[clip_name]['left_annotation']) == len(motion_data[clip_name]['right_annotation']) == len(motion_data[clip_name]['interaction_annotation']), f"Annotation length mismatch for clip {clip_name}"
            annotation_count = len(motion_data[clip_name]['left_annotation'])
            for j in range(annotation_count):
                name = f"{clip_name}_ann{j}"
                self.name_list.append(name)
                self.length_list.append(motion.shape[0])
                self.data_dict[name] = dict(
                    motion=motion,
                    text={
                        'left': motion_data[clip_name]['left_annotation'][j],
                        'right': motion_data[clip_name]['right_annotation'][j],
                        'two_hands_relation': motion_data[clip_name]['interaction_annotation'][j]
                    }
                )

        self.length_list = np.array(self.length_list)

        if self.split == 'train' and self.ratio < 1.0:
            samples_count = int(math.ceil(len(self.name_list) * self.ratio))
            random_indices = np.random.choice(len(self.name_list), size=samples_count, replace=False)
            self.name_list = [self.name_list[i] for i in random_indices]
            self.length_list = self.length_list[random_indices]
            self.data_dict = {name: self.data_dict[name] for name in self.name_list}

    def __getitem__(self, index):
        if self.debug:
            print(f"motion name: {self.name_list[index]}")
            print(f"self.name_list is UNIQUE: {len(self.name_list) == len(set(self.name_list))}")
        motion = self.data_dict[self.name_list[index]]['motion'] # (T, 2J, C)
        m_length = self.length_list[index]
        text = self.data_dict[self.name_list[index]]['text']

        if self.contact_label == True:
            contact_label = give_contact_label(
                motion[:, :, :3].reshape(motion.shape[0], 2, -1, 3), # (T, 2, J, 3)
                tip_tip_threshold=INTRA_TIP_CONTACT_THRESH,
                tip_palm_threshold=TIP_PALM_CONTACT_THRESH,
                palm_palm_threshold=PALM_PALM_CONTACT_THRESH
            )

        if self.normalize:
            motion = (motion - self.mean[np.newaxis]) / self.std

        if self.contact_label:
            return motion.transpose((1, 2, 0)), m_length, text, contact_label
        else:
            return motion.transpose((1, 2, 0)), m_length, text

    def __len__(self):
        return len(self.data_dict)

