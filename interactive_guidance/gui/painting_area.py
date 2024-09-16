import io
import tkinter as tk

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

from .utils import is_point_within_area


class PaintingArea(tk.Canvas):
    """
    Painting Area
    TODO Comments; Types; Camera Stream
    """

    def __init__(self, bg_image) -> None:
        self._disp_size = np.array([640, 480])
        self._mask_size = np.array([bg_image.width, bg_image.height])
        self._upsampler = torch.nn.Upsample(size=[bg_image.width, bg_image.height], mode="nearest")
        self._plot_density = 0.05
        self._plot_offset = -0.0575 * self._disp_size
        self._plot_scale = 0.0142 * self._disp_size

        self._bg_image = ImageTk.PhotoImage(bg_image.resize(self._disp_size))

        self.brush_size = tk.IntVar()
        self.brush_size.set(50)

        self.brush_angle = tk.DoubleVar()
        self.brush_angle.set(-np.pi / 2)
        self._brush_dir = [np.cos(self.brush_angle.get()), -np.sin(self.brush_angle.get())]

        super().__init__(width=self._disp_size[0], height=self._disp_size[1])
        self.pack()

        self.bind("<Button-1>", self._on_mouse_press)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_release)
        self._old_event = [None, None]

        self.create_image(0, 0, anchor=tk.NW, image=self._bg_image)
        self.clear()

    @property
    def mask(self) -> torch.Tensor:
        disp_mask = einops.rearrange(torch.tensor(self._mask), "w h d -> 1 d w h")

        interp_mask = self._upsampler(disp_mask)
        interp_mask = einops.rearrange(interp_mask, "1 d w h -> w h d")

        mask = torch.zeros((*self._mask_size, 3))
        mask[:, :, :2] = interp_mask
        return mask

    def clear(self) -> None:
        self._mask = np.zeros((*self._disp_size, 2))
        # self._mask[:, :, 1] = 1
        self._update_plot()

    def _update_plot(self) -> None:
        self._quiver_photo = ImageTk.PhotoImage(self._create_plot())
        self.create_image(*self._plot_offset, anchor=tk.NW, image=self._quiver_photo)

    def _create_plot(self):
        mesh_num = self._disp_size * self._plot_density
        x, y = np.meshgrid(
            np.linspace(0, self._disp_size[0] - 1, num=int(mesh_num[0]), dtype=int),
            np.linspace(0, -(self._disp_size[1] - 1), num=int(mesh_num[1]), dtype=int),
        )
        u = self._mask[x, -y, 0]
        v = self._mask[x, -y, 1]

        fig, ax = plt.subplots(figsize=self._plot_scale)
        ax.quiver(x, y, u, v, scale=75, headwidth=5, headlength=5, color="r")
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        plt.close(fig)
        return Image.open(buf)

    def _update_mask(self, x, y) -> None:
        grid = np.stack(np.meshgrid(np.arange(self._disp_size[0]), np.arange(self._disp_size[1]))).T
        circle = is_point_within_area(
            grid,
            [x, y],
            self._old_event,
            self.brush_size.get(),
        )

        self._mask[circle] = self._brush_dir
        self._update_plot()

    def _on_mouse_press(self, event) -> None:
        self._brush_dir = [np.cos(self.brush_angle.get()), -np.sin(self.brush_angle.get())]
        self._old_event = [event.x, event.y]
        self._update_mask(event.x, event.y)

    def _on_mouse_drag(self, event) -> None:
        if self._old_event:
            self._update_mask(event.x, event.y)
            self._old_event = [event.x, event.y]

    def _on_mouse_release(self, _) -> None:
        self._old_event = [None, None]
