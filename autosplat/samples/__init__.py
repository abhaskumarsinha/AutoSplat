"""
AutoSplat Samples
=================

Procedural data generators for testing and visualizing the AutoSplat 3D Gaussian
rendering pipeline.

This subpackage provides ready-to-use sample scenes and textures to help you
quickly evaluate the renderer and camera pipeline without needing real-world
datasets.

Available Samples
-----------------
- **generate_textured_ball_images** : Creates a procedurally textured sphere
  viewed from four canonical camera angles (front, back, left, right).

Usage
-----
>>> from autosplat.samples import generate_textured_ball_images
>>> imgs = generate_textured_ball_images(res=128, save=True)
🎨 Generated sphere sample images saved as front.png, back.png, left.png, right.png
>>> imgs.shape
(4, 128, 128, 3)
"""

from .sphere import generate_textured_ball_images

__all__ = [
    "generate_textured_ball_images",
]
