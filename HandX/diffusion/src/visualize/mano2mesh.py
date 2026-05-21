from typing import Optional, Dict, Union
import numpy as np
import torch
from smplx import MANO
from smplx.utils import MANOOutput, Tensor
from smplx.lbs import lbs

from ..constant import MANO_MODEL_DIR


class ModifiedMANO(MANO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
    ) -> MANOOutput:
        '''
        betas: (B, 10)
        global_orient: (B, 3)
        hand_pose: (B, 45)
        transl: (B, 3)
        '''
        global_orient = global_orient if global_orient is not None else self.global_orient
        betas = betas if betas is not None else self.betas
        hand_pose = hand_pose if hand_pose is not None else self.hand_pose

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(
            betas, full_pose, self.v_template,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, pose2rot=True
        )

        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1).to(joints.device)
            vertices = vertices + transl.unsqueeze(dim=1).to(vertices.device)

        output = MANOOutput(
            vertices=vertices,
            joints=joints,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose,
        )
        return output

left_manomodel = ModifiedMANO(
    model_path=MANO_MODEL_DIR,
    is_rhand=False,
    model_type='mano',
    use_pca=False,
    ext='pkl'
)

right_manomodel = ModifiedMANO(
    model_path=MANO_MODEL_DIR,
    is_rhand=True,
    model_type='mano',
    use_pca=False,
    ext='pkl'
)

def sequential_single_mano2mesh(
    hand:str,
    betas:np.ndarray | None,
    global_orient:np.ndarray | None,
    hand_pose:np.ndarray | None,
    transl:np.ndarray | None
):
    if hand == 'left':
        mano_model = left_manomodel
    else:
        mano_model = right_manomodel

    mano_model.to('cpu')

    output = mano_model(
        betas=torch.from_numpy(betas).float() if betas is not None else None,
        global_orient=torch.from_numpy(global_orient).float() if global_orient is not None else None,
        hand_pose=torch.from_numpy(hand_pose).float() if hand_pose is not None else None,
        transl=torch.from_numpy(transl).float() if transl is not None else None
    )

    vertices = output.vertices.clone().detach().cpu().numpy()
    faces = np.tile(mano_model.faces, (vertices.shape[0], 1, 1))

    return vertices, faces

def merge_two_meshes(
    left_vertices:np.ndarray,
    right_vertices:np.ndarray,
    left_faces:np.ndarray,
    right_faces:np.ndarray
) -> Dict[str, Union[np.ndarray, np.ndarray]]:
    '''
    Merge two meshes into one.
    left_vertices: (T, N, 3)
    right_vertices: (T, M, 3)
    left_faces: (T, F, 3)
    right_faces: (T, G, 3)
    '''
    vertices = np.concatenate([left_vertices, right_vertices], axis=1)
    faces = np.concatenate([
        left_faces,
        right_faces + left_vertices.shape[1]
    ], axis=1)

    return vertices, faces

def bihand_mano2mesh(
    left_motion:Dict[str, np.ndarray],
    right_motion:Dict[str, np.ndarray],
):
    '''
    motion:
        shape: (T, 10)
        pose: (T, 48)
        trans: (T, 3)
    '''
    T = left_motion['pose'].shape[0]
    left_vertices, left_faces = sequential_single_mano2mesh(
        'left',
        betas=left_motion['shape'] if 'shape' in left_motion else np.zeros((T, 10)),
        global_orient=left_motion['pose'][:, :3] if 'pose' in left_motion else np.zeros((T, 3)),
        hand_pose=left_motion['pose'][:, 3:] if 'pose' in left_motion else np.zeros((T, 45)),
        transl=left_motion['trans'] if 'trans' in left_motion else np.zeros((T, 3))
    )
    right_vertices, right_faces = sequential_single_mano2mesh(
        'right',
        betas=right_motion['shape'] if 'shape' in right_motion else np.zeros((T, 10)),
        global_orient=right_motion['pose'][:, :3] if 'pose' in right_motion else np.zeros((T, 3)),
        hand_pose=right_motion['pose'][:, 3:] if 'pose' in right_motion else np.zeros((T, 45)),
        transl=right_motion['trans'] if 'trans' in right_motion else np.zeros((T, 3))
    )
    vertices, faces = merge_two_meshes(
        left_vertices=left_vertices,
        right_vertices=right_vertices,
        left_faces=left_faces,
        right_faces=right_faces
    )
    return vertices, faces
