from typing import Dict, List
import numpy as np
from tqdm import tqdm
import math, json
import textwrap
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatter import Formatter
from pygments.styles import get_style_by_name
from pygments.token import Token

from .mano2mesh import bihand_mano2mesh, bihand_mano2mesh
from .mesh_visualizer import Mesh_Visualize_Helper
from .skeleton_visualizer import Skeleton_Visualize_Helper
from ..utils import smart_wrap

class MatplotlibFormatter(Formatter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    def format(self, tokensource, outfile):
        self.data = []
        for ttype, value in tokensource:
            style = self.style.style_for_token(ttype)
            color = style['color']
            if color:
                color = f"#{color}"
            else:

                style_bg_is_dark = self.style.background_color < '#777'
                color = '#ffffff' if style_bg_is_dark else '#000000'

            self.data.append((value, color, ttype))


class JsonPlotter:
    @staticmethod
    def plot_json(
        data_dict: dict | list,
        ax: plt.Axes,
        width: int = 80,
        style_name: str = 'monokai',
        font_family: str = 'monospace',
        font_size: int = 10
    ):

        json_str = json.dumps(data_dict, indent=4, ensure_ascii=False)
        lexer = JsonLexer()
        style = get_style_by_name(style_name)
        formatter = MatplotlibFormatter(style=style)
        highlight(json_str, lexer, formatter)

        ax.set_facecolor(style.background_color)

        fig = ax.get_figure()
        font = FontProperties(family=font_family, size=font_size)


        sample_text_obj = ax.text(0, 0, 'M', fontproperties=font, visible=False)

        try:
            fig.canvas.draw()

            text_bbox = sample_text_obj.get_window_extent()
            ax_bbox = ax.get_window_extent()

            if ax_bbox.width == 0 or text_bbox.width == 0:
                raise ValueError("Canvas has zero width, cannot calculate font metrics.")

            char_width = text_bbox.width / ax_bbox.width
            line_height = text_bbox.height / ax_bbox.height * 1.6

            print(f"Char Width: {char_width}, Line Height: {line_height}")
        except Exception:
            char_width = 0.006 * (font_size / 10)
            line_height = 0.025 * (font_size / 10)
        finally:
            sample_text_obj.remove()


        lines_of_tokens = []
        current_line = []
        current_char_count = 0

        wrapper = textwrap.TextWrapper(
            width=width,
            break_long_words=True,
            break_on_hyphens=False
        )

        for text, color, ttype in formatter.data:
            if text == '\n':
                lines_of_tokens.append(current_line)
                current_line = []
                current_char_count = 0
                continue

            is_string = ttype in Token.Literal.String
            if is_string and (current_char_count + len(text)) > width:
                leading_spaces = current_char_count
                wrapper.initial_indent = " " * 0

                wrapper.subsequent_indent = " " * (leading_spaces + 4)

                content = text[1:-1]
                wrapped_lines = wrapper.wrap(content)

                current_line.append((text[0], color))

                for i, line_content in enumerate(wrapped_lines):
                    if i == 0:

                        current_line.append((line_content, color))
                    else:
                        lines_of_tokens.append(current_line)

                        current_line = [ (line_content.lstrip(), color) ]


                current_line.append((text[-1], color))
            else:
                current_line.append((text, color))
                current_char_count += len(text)

        if current_line:
            lines_of_tokens.append(current_line)

        x_pos, y_pos = 0.02, 0.98
        left_margin = x_pos

        for line_tokens in lines_of_tokens:
            x_pos = left_margin
            for text, color in line_tokens:
                if not text: continue
                ax.text(x_pos, y_pos, text, color=color, fontproperties=font, ha='left', va='top')
                x_pos += len(text) * char_width
            y_pos -= line_height

        ax.axis('off')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

class Visualize_Helper:
    @staticmethod
    def initialize():
        fig = plt.figure(figsize=(10, 10), layout='constrained')

        gs = gridspec.GridSpec(2, 2, figure=fig)
        skeleton_ax = fig.add_subplot(gs[0, 0], projection='3d')
        mano_ax = fig.add_subplot(gs[0, 1], projection='3d')
        annotation_ax = fig.add_subplot(gs[1, :])
        annotation_ax.axis("off")
        return fig, skeleton_ax, mano_ax, annotation_ax

    @staticmethod
    def format_plain_string(text:str, width:int=90):
        return '\n'.join(textwrap.wrap(text, width=width))


    @staticmethod
    def create_3d_animation(
        left_motion:np.ndarray, right_motion:np.ndarray,
        left_mano:Dict[str, np.ndarray]|None=None, right_mano:Dict[str, np.ndarray]|None=None,
        annotation:str | dict | list="",
        save_path:str|None=None, fps:int=30
    ):
        '''
        left_motion: (T, J, 3)
        right_motion: (T, J, 3)

        left_mano:
            shape: (T, 10)
            pose: (T, 48),
            trans: (T, 3)
        right_mano: the same as left_mano
        '''
        fig, skeleton_ax, mano_ax, annotation_ax = Visualize_Helper.initialize()

        print(f"Annotation Type: {type(annotation)}")

        if isinstance(annotation, dict) or isinstance(annotation, list):
            JsonPlotter.plot_json(annotation, annotation_ax)
        elif isinstance(annotation, str):
            print(f"annotation: {annotation.__repr__()}")
            annotation = '\n'.join(textwrap.wrap(annotation, width=90, replace_whitespace=False, tabsize=4))
            annotation_ax.text(
                x=0.1, y=0.95, s=annotation, ha='left', va='top', fontsize=14
            )
        frame_num = fig.text(
            x=0.5, y=0.95, s="", ha='center', va='top', fontsize=16
        )
        skeleton_helper = Skeleton_Visualize_Helper(skeleton_ax, left_motion, right_motion)
        skeleton_helper.initialize_ax()

        if left_mano is not None and right_mano is not None:
            vertices, faces = bihand_mano2mesh(left_mano, right_mano)
            mesh_helper = Mesh_Visualize_Helper(mano_ax, vertices, faces)
            mesh_helper.initialize_ax()

        def update(frame):
            frame_num.set_text(f"Frame: {frame} / {left_motion.shape[0]}")
            skeleton_helper.draw(frame)
            if left_mano is not None and right_mano is not None:
                mesh_helper.draw_mesh(frame)

        ani = FuncAnimation(fig, update, frames=left_motion.shape[0])

        if save_path:
            pbar = tqdm(total=left_motion.shape[0], desc='Exporting animation')
            ani.save(save_path, writer='ffmpeg', fps=fps, progress_callback=lambda x, y: pbar.update(1))


        plt.close(fig)

class MultiMotionVisualizer:

    def initialize(number_of_motions:int):
        rows = int(math.floor(math.sqrt(number_of_motions)))
        columns = int(math.ceil(number_of_motions / rows))
        fig = plt.figure(figsize=(4 * columns, 4 * rows + 8))
        gs = gridspec.GridSpec(rows + 1, columns, figure=fig, height_ratios=[4] * rows + [8])
        motion_axes = []
        for i in range(number_of_motions):
            ax = fig.add_subplot(gs[i // columns, i % columns], projection='3d')
            motion_axes.append(ax)
        motion_axes = np.array(motion_axes)

        text_ax = fig.add_subplot(gs[rows, :])
        text_ax.axis("off")

        return fig, motion_axes, text_ax

    @staticmethod
    def create_3d_animation(
        motions:List[Dict[str, np.ndarray | str | dict]],
        text:str="",
        save_path:str=None, fps:int=30
    ):
        '''
        if motion['type'] == 'skeleton'
            motion = {
                'type': 'skeleton',
                'left_motion': (T, J, 3),
                'right_motion': (T, J, 3)
            }
        elif motion['type'] == 'mano'
            motion = {
                'type': 'mano',
                'left_motion': dict
                'right_motion': dict'
            }
        '''
        fig, axes, text_ax = MultiMotionVisualizer.initialize(len(motions))

        drawers = []
        frame_num = 0
        for motion_id, motion in enumerate(motions):
            if motion['type'] == 'skeleton':
                helper = Skeleton_Visualize_Helper(
                    axes[motion_id], motion['left_motion'], motion['right_motion'], motion.get('title', None)
                )
                frame_num = max(frame_num, motion['left_motion'].shape[0])
            elif motion['type'] == 'mano':
                vertices, faces = bihand_mano2mesh(
                    motion['left_motion'], motion['right_motion']
                )
                helper = Mesh_Visualize_Helper(
                    axes[motion_id], vertices, faces, motion.get('title', None)
                )
                frame_num = max(vertices.shape[0], frame_num)
            helper.initialize_ax()
            drawers.append(helper)

        frame_num_text = fig.text(
            x=0.5, y=0.95, s="", ha='center', va='top', fontsize=16
        )

        # text = '\n'.join(textwrap.wrap(text, width=90))
        text = smart_wrap(text, width=90)

        text_ax.text(
            x=0.01, y=0.5, s=text, ha='left', va='center', fontsize=14
        )

        def update(frame):
            frame_num_text.set_text(f"Frame: {frame} / {frame_num}")
            for drawer in drawers:
                if isinstance(drawer, Skeleton_Visualize_Helper):
                    drawer.draw(frame)
                elif isinstance(drawer, Mesh_Visualize_Helper):
                    drawer.draw_mesh(frame)

        ani = FuncAnimation(fig, update, frames=frame_num)

        if save_path:
            name = save_path.split('/')[-1]
            pbar = tqdm(total=frame_num, desc=f'Exporting animation to {name}')
            ani.save(save_path, writer='ffmpeg', fps=fps, progress_callback=lambda x, y: pbar.update(1))

        plt.close(fig)


