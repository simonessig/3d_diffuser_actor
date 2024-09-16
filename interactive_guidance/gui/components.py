import tkinter as tk
from tkinter import ttk

import numpy as np


class IGUIButton(ttk.Button):
    """
    TODO Comments; Types
    """

    def __init__(self, master, **kwargs):
        ttk.Button.__init__(self, master, style="TButton", **kwargs)
        self.pack(fill=tk.X, pady=5)


class IGUILabel(ttk.Label):
    """
    TODO Comments; Types
    """

    def __init__(self, master, **kwargs):
        ttk.Label.__init__(self, master, style="TLabel", **kwargs)
        self.pack(fill=tk.X)


class IGUIScale(ttk.Scale):
    """
    TODO Comments; Types
    """

    def __init__(self, master, **kwargs):
        ttk.Scale.__init__(self, master, style="TScale", **kwargs)
        self.pack(fill=tk.X)


class IGUIDirectionChooser(tk.Canvas):
    """
    TODO Comments; Types
    """

    def __init__(self, master, angle, radius=100, **kwargs):
        super().__init__(master, width=2 * radius, height=2 * radius, **kwargs)
        self._radius = radius
        self._angle = angle
        # self._set_angle = set_angle

        self.create_oval(0, 0, 2 * radius, 2 * radius, outline="black", width=3)
        self._arrow = self.create_line(
            radius,
            radius,
            radius,
            0,
            arrow=tk.LAST,
            fill="red",
            width=3,
        )

        self.bind("<Button-1>", self._update_arrow)
        self.bind("<B1-Motion>", self._update_arrow)

    def _update_arrow(self, event):
        dx = event.x - self._radius
        dy = event.y - self._radius
        self._angle.set(np.arctan2(dy, dx))
        end_x = self._radius + self._radius * np.cos(self._angle.get())
        end_y = self._radius + self._radius * np.sin(self._angle.get())
        self.coords(self._arrow, self._radius, self._radius, end_x, end_y)
        # self._set_angle(self._angle)
