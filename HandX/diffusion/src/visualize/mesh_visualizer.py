import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

from ..utils import get_motion_data_boundary

class Mesh_Visualize_Helper:
    def __init__(self, ax:Axes3D, vertices:np.ndarray, faces:np.ndarray, title:str | None = None):
        self.ax = ax
        self.vertices = vertices
        self.faces = faces
        self.title = title

    def initialize_ax(self):
        xmin, xmax, ymin, ymax, zmin, zmax = get_motion_data_boundary(self.vertices)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmin, zmax)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.axis('off')
        # self.ax.view_init(azim=-64, elev=-18, roll=172)
        # self.ax.view_init(azim=120)
        if self.title:
            self.ax.set_title(self.title)

        single_face_num = self.faces.shape[1] // 2
        left_face_color = np.tile(np.array([107, 174, 214])[np.newaxis, :], (single_face_num, 1)) / 255
        right_face_color = np.tile(np.array([252, 146, 114])[np.newaxis, :], (single_face_num, 1)) / 255
        left_edge_color = np.tile(np.array([8, 81, 156])[np.newaxis, :], (single_face_num, 1)) / 255
        right_edge_color = np.tile(np.array([203, 24, 29])[np.newaxis, :], (single_face_num, 1)) / 255
        self.mesh_collection = Poly3DCollection(
            [],
            facecolors=np.concatenate([left_face_color, right_face_color], axis=0),
            edgecolors=np.concatenate([left_edge_color, right_edge_color], axis=0),
            linewidths=0.1,
            shade=True,
            alpha=0.8
        )
        self.ax.add_collection3d(self.mesh_collection)

    def draw_mesh(self, frame):
        if frame >= self.vertices.shape[0]:
            return
        mesh_faces = [[self.vertices[frame, idx] for idx in face] for face in self.faces[frame]]
        self.mesh_collection.set_verts(mesh_faces)
        self.mesh_collection.set_verts(mesh_faces)