from textwrap import wrap
import matplotlib
from matplotlib.patches import Rectangle
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import inspect 
from typing import Dict, List, Tuple
import textwrap
import random

class TimelineVisualizer(object):
    LEFT_MARGIN = -50
    def __init__(
        self, 
        ax:Axes, 
        data:Dict[str, List[Tuple[int, int, str]]],
        total_frames:int,
        title:str|None=None
    ):
        '''
        data:
        set_name: [(start_frame, end_frame, text)]
        '''
        self.ax = ax 
        self.data = data 
        self.title = title
        self.total_frames = total_frames

    def get_text_wrapping_width(self, fontsize, x_left, x_right):
        fig:Figure = self.ax.get_figure()
        dpi = fig.get_dpi()

        pixel_coords = self.ax.transData.transform([(x_left, 0), (x_right, 0)])
        available_width_pixels = pixel_coords[1, 0] - pixel_coords[0, 0]

        avg_char_width_pixels = fontsize * 0.5 * (dpi / 72.0) 

        if avg_char_width_pixels <= 0: 
            return 1 

        wrap_width = int(available_width_pixels / avg_char_width_pixels)
        return wrap_width
    @staticmethod
    def get_random_light_color():
        min_val = 180
    
        # 在阈值和255之间生成随机整数
        r = random.randint(min_val, 255)
        g = random.randint(min_val, 255)
        b = random.randint(min_val, 255)
        
        # Matplotlib 的颜色元组格式要求值为 0-1 的浮点数，
        # 因此需要将 0-255 的整数进行归一化处理（除以255）
        return (r / 255.0, g / 255.0, b / 255.0)
        
    def initialize_ax(self):
        self.ax.set_xlim(self.LEFT_MARGIN, self.total_frames)
        self.ax.set_ylim(-0.8 * len(self.data), 1)
        self.ax.set_xticks(range(0, self.total_frames + 1, 50))
        self.ax.set_yticks([])
        self.ax.set_xlabel("Frames", fontsize=10)
        self.ax.spines[['left', 'right', 'top']].set_visible(False)

        self.vline = self.ax.axvline(0, color='k', lw=2)

        for set_index, (set_name, set_segments) in enumerate(self.data.items()):
            wrapped_set_name = textwrap.fill(set_name, width=13)
            self.ax.text(self.LEFT_MARGIN + 1, -0.8 * set_index + 0.4, wrapped_set_name, ha='left', va='center', weight='bold', fontsize=9)
            for start_frame, end_frame, text in set_segments:
                face_color = TimelineVisualizer.get_random_light_color()
                self.ax.add_patch(Rectangle((start_frame, -0.8 * set_index), end_frame - start_frame, 0.8, ec='k', lw=0.5, facecolor=face_color))
                auto_width = self.get_text_wrapping_width(9, start_frame, end_frame)
                wrapped_text = textwrap.fill(text, width=auto_width)
                self.ax.text((start_frame + end_frame) / 2, -0.8 * set_index + 0.4, wrapped_text, ha='center', va='center', fontsize=9)

    def update(self, frame):
        self.vline.set_xdata((frame, frame))


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print(inspect.getfile(type(ax)))
    print(isinstance(ax, Axes))
