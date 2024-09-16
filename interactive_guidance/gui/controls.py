import tkinter as tk
from tkinter import ttk

from .components import IGUIButton, IGUIDirectionChooser, IGUILabel, IGUIScale
from .painting_area import PaintingArea


class ControlsFrame(ttk.Frame):
    """
    Controls
    TODO Comments; Types
    """

    def __init__(self, root: tk.Widget, painting_area: PaintingArea) -> None:
        super().__init__(root)

        self._painting_area = painting_area

        self.pack()
        self.pen_width = 5
        self.pen_color = "black"

        self._init_components()

    def _init_components(self):
        default_padding = 10

        # Brush size label and slider
        size_frame = ttk.Frame(self)
        size_frame.pack(side=tk.LEFT, fill=tk.Y, padx=default_padding)

        IGUILabel(size_frame, text="Brush Size")

        IGUIScale(
            size_frame,
            from_=20,
            to=100,
            variable=self._painting_area.brush_size,
            orient=tk.HORIZONTAL,
        )

        # Direction
        dir_frame = ttk.Frame(self)
        dir_frame.pack(side=tk.LEFT, fill=tk.Y, padx=default_padding)
        direction_chooser = IGUIDirectionChooser(
            dir_frame,
            self._painting_area.brush_angle,
            radius=25,
        )
        direction_chooser.pack()

        # Clear button
        IGUIButton(self, text="Clear", command=self._painting_area.clear)

        # Save
        # save_frame = ttk.Frame(self)
        # save_frame.pack(side=tk.LEFT, fill=tk.Y, padx=default_padding)

        # IGUIButton(save_frame, text="Save", command=self.clear_canvas)
        # IGUIButton(save_frame, text="Save as ...", command=self.clear_canvas)

    def clear_canvas(self):
        pass
