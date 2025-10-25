
# autosplat/io/__init__.py

"""
I/O module for AutoSplat.

Provides functions to load/save cameras and Gaussian3D layers in JSON format,
and a main utility to batch process folders containing these files.
"""

from .camera import save_cameras_to_json, load_cameras_from_json
from .gaussians import save_gaussians_to_json, load_gaussians_from_json
from .main import main

__all__ = [
    "save_cameras_to_json",
    "load_cameras_from_json",
    "save_gaussians_to_json",
    "load_gaussians_from_json",
    "main",
]
