import tkinter as tk
from tkinter import ttk

import torch
from PIL import Image

from .controls import ControlsFrame
from .painting_area import PaintingArea

__all__ = ["start_gui"]


FONT = ("fixed", 10)


class App(tk.Tk):
    """
    App
    TODO Comments Types
    """

    def __init__(self, bg_image):
        super().__init__()

        style = ttk.Style()

        style.configure("TLabel", font=FONT)
        style.configure("TButton", font=FONT)

        self.title("IGUI")

        self.canvas = PaintingArea(bg_image)
        self.controls = ControlsFrame(self, self.canvas)

    @property
    def mask(self) -> torch.Tensor:
        return self.canvas.mask


def start_gui(bg_image) -> torch.Tensor:
    """TODO"""
    app = App(bg_image)
    app.mainloop()
    return app.mask
