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

Example
-------
>>> from autosplat.visualizer import render_rotation_gif
>>> render_rotation_gif(renderer, blend_id=0, res=256, steps=90, save_path="spin.gif", fps=15)
🎞️ GIF saved at: spin.gif
"""

from .render_rotation import render_rotation_gif

__all__ = [
    "render_rotation_gif",
]

