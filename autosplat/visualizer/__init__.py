"""
AutoSplat Visualizer
====================

Visualization utilities for Gaussian Splatting renders.

This subpackage provides tools to visualize BlenderLayer and RenderObject
outputs by generating rotating camera animations and other visual previews.

Available functions
-------------------
- **render_rotation_gif** : Generates a 360° rotating GIF around the object,
  rendering frames from different camera angles using the given renderer.
- **plot_cameras_plotly** : Plots multiple cameras in 3D space using Plotly,
  showing their positions and orientations.

Example
-------
>>> from autosplat.visualizer import render_rotation_gif, plot_cameras_plotly
>>> render_rotation_gif(renderer, blend_id=0, res=256, steps=90, save_path="spin.gif", fps=15)
🎞️ GIF saved at: spin.gif
>>> plot_cameras_plotly(cameras)
📸 Displays interactive 3D view of cameras.
"""

from .render_rotation import render_rotation_gif
from .plot_cameras import plot_cameras_plotly

__all__ = [
    "render_rotation_gif",
    "plot_cameras_plotly",
]
