import numpy as np
import rerun as rr

from ..constant import SKELETON_CHAIN


class RerunVisualizer(object):
    left_mesh_color = [107, 174, 214]
    right_mesh_color = [252, 146, 114]
    left_skeleton_color = [8, 81, 156]
    right_skeleton_color = [203, 24, 29]
    @staticmethod
    def visualize_singlemotion(
        motion:np.ndarray=None, vertices:np.ndarray=None, faces:np.ndarray=None,
        application_id:str="Visualize Single Hand", hand:str="left",
    ):

        rr.log(
            '3D/axis',
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3, 3),
                labels=["X", "Y", "Z"],
            ),
            static=True
        )

        motion_frames = motion.shape[0] if motion is not None else 0
        mano_frames = vertices.shape[0] if vertices is not None else 0
        frames = max(motion_frames, mano_frames)
        for frame in range(frames):
            rr.set_time("stable_time", duration=frame / 30)
            rr.log(
                'text',
                rr.TextDocument(
                    text=f'frame {frame}',
                )
            )
            if motion is not None and frame < motion.shape[0]:
                rr.log(
                    '3D/skeleton',
                    rr.LineStrips3D(
                        strips=motion[frame, np.array(SKELETON_CHAIN)],
                        colors=RerunVisualizer.left_skeleton_color if hand == "left" else RerunVisualizer.right_skeleton_color
                    )
                )
            if vertices is not None and faces is not None:
                if frame < len(vertices) and frame < len(faces):
                    rr.log(
                        '3D/mesh',
                        rr.Mesh3D(
                            vertex_positions=vertices[frame],
                            triangle_indices=faces[frame],
                            vertex_colors=RerunVisualizer.left_skeleton_color if hand == "left" else RerunVisualizer.right_skeleton_color,
                            albedo_factor=RerunVisualizer.left_mesh_color if hand == "left" else RerunVisualizer.right_mesh_color
                        )
                    )

    @staticmethod
    def visualize_bihandmotion(
        motion:np.ndarray=None, left_vertices:np.ndarray=None, right_vertices:np.ndarray=None,
        left_faces:np.ndarray=None, right_faces:np.ndarray=None,
    ):
        rr.log(
            '3D/axis',
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3, 3),
                labels=["X", "Y", "Z"],
            ),
            static=True
        )

        assert motion is not None or (left_vertices is not None and right_vertices is not None), \
            "Either motion or left/right vertices must be provided."

        print("Visualizing bi-hand motion...")

        motion_frames = motion.shape[0] if motion is not None else 0
        mano_frames = max(
            left_vertices.shape[0] if left_vertices is not None else 0,
            right_vertices.shape[0] if right_vertices is not None else 0
        )
        frames = max(motion_frames, mano_frames)
        for frame in range(frames):
            rr.set_time("stable_time", duration=frame / 30)
            rr.log(
                'text',
                rr.TextDocument(
                    text=f'frame {frame}',
                )
            )
            if motion is not None and frame < motion.shape[0]:
                rr.log(
                    '3D/left_skeleton',
                    rr.LineStrips3D(
                        strips=motion[frame, 0, np.array(SKELETON_CHAIN)],
                        colors=RerunVisualizer.left_skeleton_color
                    )
                )
                rr.log(
                    '3D/right_skeleton',
                    rr.LineStrips3D(
                        strips=motion[frame, 1, np.array(SKELETON_CHAIN)],
                        colors=RerunVisualizer.right_skeleton_color
                    )
                )
            if left_vertices is not None and left_faces is not None:
                if frame < len(left_vertices) and frame < len(left_faces):
                    rr.log(
                        '3D/left_mesh',
                        rr.Mesh3D(
                            vertex_positions=left_vertices[frame],
                            triangle_indices=left_faces[frame],
                            vertex_colors=RerunVisualizer.left_skeleton_color,
                            albedo_factor=RerunVisualizer.left_mesh_color
                        )
                    )
            if right_vertices is not None and right_faces is not None:
                if frame < len(right_vertices) and frame < len(right_faces):
                    rr.log(
                        '3D/right_mesh',
                        rr.Mesh3D(
                            vertex_positions=right_vertices[frame],
                            triangle_indices=right_faces[frame],
                            vertex_colors=RerunVisualizer.right_skeleton_color,
                            albedo_factor=RerunVisualizer.right_mesh_color
                        )
                    )

